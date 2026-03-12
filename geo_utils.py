"""
geo_utils.py
------------
Handles loading the NYC street centerline GeoJSON, building a spatial index,
filtering by bounding box, computing street segment midpoints, and converting
geographic coordinates to viewport pixel positions using Web Mercator projection.
"""

import json
import math
from shapely.geometry import shape, box
from shapely.strtree import STRtree

DATA_PATH = "data/nycstreetcenterline.geojson"

ZOOM = 16
VIEWPORT_W = 1920
VIEWPORT_H = 1080
TILE_PX = 256

# Module-level cache so the 205MB file is parsed only once
_features: list[dict] | None = None
_strtree: STRtree | None = None


def load_features() -> list[dict]:
    """
    Parse the NYC street centerline GeoJSON and cache it in memory.
    On subsequent calls, returns the cached result immediately.

    Each returned dict contains:
        physicalid       str  — unique segment ID
        street_name      str  — display street name
        trafdir          str  — source traffic direction code (FT/TF/TW/...)
        directions       list[str] — flow direction labels for output (NB/SB/EB/WB)
        midpoint         shapely.Point  — centroid along segment (lng, lat)
        sample_points    list[dict] — sampled points with local tangent vectors
        original_geometry dict — raw GeoJSON geometry dict for output
    """
    global _features, _strtree
    if _features is not None:
        return _features

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # More sample positions reduce misses on thin rendered traffic polylines.
    SAMPLE_FRACS = [0.15, 0.3, 0.5, 0.7, 0.85]
    tangent_eps = 0.02

    cache: list[dict] = []
    for feat in data["features"]:
        props = feat["properties"] or {}
        geom_dict = feat["geometry"]
        if not geom_dict:
            continue
        try:
            geom = shape(geom_dict)
        except Exception:
            continue

        midpoint = geom.interpolate(0.5, normalized=True)
        sample_points = []
        for frac in SAMPLE_FRACS:
            pt = geom.interpolate(frac, normalized=True)
            a = geom.interpolate(max(0.0, frac - tangent_eps), normalized=True)
            b = geom.interpolate(min(1.0, frac + tangent_eps), normalized=True)
            dx = b.x - a.x
            dy = b.y - a.y
            if abs(dx) + abs(dy) < 1e-12:
                dx = 1e-9
                dy = 0.0
            sample_points.append({"point": pt, "tan_dx": dx, "tan_dy": dy})

        start_lng_lat, end_lng_lat = _geometry_endpoints(geom_dict)
        forward_dir = _cardinal_direction(start_lng_lat, end_lng_lat)
        reverse_dir = _opposite_direction(forward_dir)
        trafdir = (props.get("trafdir") or "").upper()

        if trafdir == "TW":
            directions = [forward_dir, reverse_dir]
        elif trafdir == "TF":
            directions = [reverse_dir]
        else:
            # FT and unknown values fall back to geometry-forward direction.
            directions = [forward_dir]

        cache.append(
            {
                "physicalid": str(props.get("physicalid", "")),
                "street_name": props.get("stname_label") or props.get("full_street_name", ""),
                "trafdir": trafdir,
                "directions": directions,
                "midpoint": midpoint,           # shapely Point (lng, lat)
                "sample_points": sample_points, # list of dicts with point + tangent
                "original_geometry": geom_dict,  # raw dict — written to output GeoJSON
            }
        )

    # Build R-tree spatial index over midpoints for fast bbox queries
    midpoints = [f["midpoint"] for f in cache]
    _strtree = STRtree(midpoints)
    _features = cache
    return _features


def filter_by_bbox(bbox: list[float]) -> list[dict]:
    """
    Return all street segments whose midpoint falls inside bbox.

    bbox: [min_lng, min_lat, max_lng, max_lat]
    """
    features = load_features()
    query_box = box(bbox[0], bbox[1], bbox[2], bbox[3])

    candidate_indices = _strtree.query(query_box)
    result = []
    for i in candidate_indices:
        feat = features[i]
        if query_box.contains(feat["midpoint"]):
            result.append(feat)
    return result


# ---------------------------------------------------------------------------
# Web Mercator projection helpers
# ---------------------------------------------------------------------------

def _lat_lng_to_world_px(lat_deg: float, lng_deg: float, zoom: int = ZOOM) -> tuple[float, float]:
    """
    Convert geographic coordinates to world pixel space using Web Mercator.

    At zoom level z the world is (2^z * TILE_PX) pixels square.
    Origin is at the top-left (NW corner: lat=85.05°, lng=-180°).
    """
    scale = (2 ** zoom) * TILE_PX
    world_x = (lng_deg + 180.0) / 360.0 * scale
    sin_lat = math.sin(math.radians(lat_deg))
    sin_lat = max(-0.9999, min(0.9999, sin_lat))
    world_y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * scale
    return world_x, world_y


def latlng_to_viewport_pixel(
    lat: float,
    lng: float,
    center_lat: float,
    center_lng: float,
    zoom: int = ZOOM,
    vp_w: int = VIEWPORT_W,
    vp_h: int = VIEWPORT_H,
) -> tuple[int, int]:
    """
    Convert a lat/lng point to pixel coordinates within a viewport.

    The viewport center corresponds to (center_lat, center_lng).
    Returns (px, py) as integers; may be outside the viewport bounds.
    """
    wx, wy = _lat_lng_to_world_px(lat, lng, zoom)
    cwx, cwy = _lat_lng_to_world_px(center_lat, center_lng, zoom)
    px = wx - cwx + vp_w / 2.0
    py = wy - cwy + vp_h / 2.0
    return int(round(px)), int(round(py))


def viewport_covers_point(
    px: int,
    py: int,
    vp_w: int = VIEWPORT_W,
    vp_h: int = VIEWPORT_H,
    margin: int = 10,
) -> bool:
    """Return True if (px, py) is inside the viewport with given margin."""
    return margin <= px < vp_w - margin and margin <= py < vp_h - margin


# ---------------------------------------------------------------------------
# Tiling strategy
# ---------------------------------------------------------------------------

def compute_tiles(
    bbox: list[float],
    zoom: int = ZOOM,
    vp_w: int = VIEWPORT_W,
    vp_h: int = VIEWPORT_H,
    overlap_frac: float = 0.10,
) -> list[tuple[float, float]]:
    """
    Divide a bounding box into a grid of viewport tile centers.

    Each tile center is a (lat, lng) tuple that, when used as the Google Maps
    viewport center at *zoom*, displays a vp_w×vp_h viewport covering a
    distinct (slightly overlapping) portion of the bbox.

    overlap_frac: fractional overlap between adjacent tiles (e.g. 0.10 = 10%).
    Always returns at least one tile.
    """
    min_lng, min_lat, max_lng, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2.0

    # Metres per pixel at the centre latitude
    mpp = 156543.03392 * math.cos(math.radians(center_lat)) / (2 ** zoom)

    # Step size in degrees (accounting for overlap)
    step_lng = (vp_w * mpp * (1.0 - overlap_frac)) / (
        111320.0 * math.cos(math.radians(center_lat))
    )
    step_lat = (vp_h * mpp * (1.0 - overlap_frac)) / 111000.0

    bbox_w = max_lng - min_lng
    bbox_h = max_lat - min_lat

    tiles: list[tuple[float, float]] = []

    # When the bbox fits inside a single viewport, use the bbox center
    if bbox_w <= step_lng and bbox_h <= step_lat:
        tiles.append(((min_lat + max_lat) / 2.0, (min_lng + max_lng) / 2.0))
    else:
        # Grid of tile centers; clamp each center within the bbox
        lat = min_lat + step_lat / 2.0
        while True:
            clamped_lat = min(lat, max_lat - step_lat / 2.0)
            clamped_lat = max(clamped_lat, min_lat + step_lat / 2.0)
            lng = min_lng + step_lng / 2.0
            while True:
                clamped_lng = min(lng, max_lng - step_lng / 2.0)
                clamped_lng = max(clamped_lng, min_lng + step_lng / 2.0)
                tiles.append((clamped_lat, clamped_lng))
                lng += step_lng
                if lng >= max_lng:
                    break
            lat += step_lat
            if lat >= max_lat:
                break

    # Guarantee at least one tile
    if not tiles:
        tiles.append(((min_lat + max_lat) / 2.0, (min_lng + max_lng) / 2.0))

    return tiles


def assign_segments_to_tiles(
    segments: list[dict],
    tiles: list[tuple[float, float]],
    zoom: int = ZOOM,
    vp_w: int = VIEWPORT_W,
    vp_h: int = VIEWPORT_H,
) -> dict[int, list[tuple[dict, dict[str, list[tuple[int, int]]]]]]:
    """
    Assign each segment to its best tile.

    Returns dict: tile_index → list of (segment_dict, sample_pixels_by_direction)
    where sample_pixels_by_direction is:
      direction_label -> list of (px, py), one per sample point.

    Each segment is assigned to the *first* tile (in iteration order) whose
    viewport contains the midpoint.  If no tile viewport contains the midpoint,
    falls back to the geographically nearest tile center to guarantee every
    segment is assigned.
    """
    if not tiles:
        return {}

    assignment: dict[int, list] = {i: [] for i in range(len(tiles))}

    for seg in segments:
        mid_lat = seg["midpoint"].y  # shapely Point: .y = lat (GeoJSON [lng,lat])
        mid_lng = seg["midpoint"].x

        assigned = False
        for i, (tlat, tlng) in enumerate(tiles):
            px, py = latlng_to_viewport_pixel(mid_lat, mid_lng, tlat, tlng, zoom, vp_w, vp_h)
            if viewport_covers_point(px, py, vp_w, vp_h):
                # Convert all sample points to pixel coords in this tile
                sample_pixels = _sample_points_to_pixels(
                    seg["sample_points"], seg["directions"], tlat, tlng, zoom, vp_w, vp_h,
                )
                assignment[i].append((seg, sample_pixels))
                assigned = True
                break  # Assign to first covering tile only

        if not assigned:
            # Fallback: assign to the geographically nearest tile center
            nearest_i = min(
                range(len(tiles)),
                key=lambda idx: (tiles[idx][0] - mid_lat) ** 2 + (tiles[idx][1] - mid_lng) ** 2,
            )
            tlat, tlng = tiles[nearest_i]
            sample_pixels = _sample_points_to_pixels(
                seg["sample_points"], seg["directions"], tlat, tlng, zoom, vp_w, vp_h,
            )
            assignment[nearest_i].append((seg, sample_pixels))

    return assignment


def _sample_points_to_pixels(
    sample_points,
    segment_directions: list[str],
    center_lat: float,
    center_lng: float,
    zoom: int,
    vp_w: int,
    vp_h: int,
) -> dict[str, list[tuple[int, int]]]:
    """
    Convert sampled geometry points to per-direction pixel coordinates.

    For two-way streets, applies a small left/right perpendicular offset so
    opposite directions can sample different carriageways when visible.
    """
    result: dict[str, list[tuple[int, int]]] = {}
    if not segment_directions:
        segment_directions = ["NB"]

    for d in segment_directions:
        result[d] = []

    twoway = len(segment_directions) == 2
    lateral_offset_px = 3.0
    step = 1e-4

    for sample in sample_points:
        pt = sample["point"]
        tan_dx = sample["tan_dx"]
        tan_dy = sample["tan_dy"]

        px, py = latlng_to_viewport_pixel(pt.y, pt.x, center_lat, center_lng, zoom, vp_w, vp_h)
        px = max(0, min(px, vp_w - 1))
        py = max(0, min(py, vp_h - 1))

        if not twoway:
            result[segment_directions[0]].append((px, py))
            continue

        norm = math.hypot(tan_dx, tan_dy)
        if norm < 1e-12:
            norm = 1.0
            tan_dx, tan_dy = 1.0, 0.0
        ux = tan_dx / norm
        uy = tan_dy / norm

        ref_lng = pt.x + ux * step
        ref_lat = pt.y + uy * step
        rx, ry = latlng_to_viewport_pixel(ref_lat, ref_lng, center_lat, center_lng, zoom, vp_w, vp_h)
        vx, vy = rx - px, ry - py
        if vx == 0 and vy == 0:
            vx, vy = 1, 0

        nx, ny = -vy, vx
        nlen = math.hypot(nx, ny)
        if nlen < 1e-9:
            nx, ny, nlen = 0.0, 1.0, 1.0
        nx /= nlen
        ny /= nlen

        # Direction 0 follows geometry order, direction 1 is opposite flow.
        p0x = int(round(max(0, min(vp_w - 1, px + nx * lateral_offset_px))))
        p0y = int(round(max(0, min(vp_h - 1, py + ny * lateral_offset_px))))
        p1x = int(round(max(0, min(vp_w - 1, px - nx * lateral_offset_px))))
        p1y = int(round(max(0, min(vp_h - 1, py - ny * lateral_offset_px))))
        result[segment_directions[0]].append((p0x, p0y))
        result[segment_directions[1]].append((p1x, p1y))

    return result


def _geometry_endpoints(geom_dict: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return (start_lng_lat, end_lng_lat) from LineString/MultiLineString geometry."""
    gtype = geom_dict.get("type")
    coords = geom_dict.get("coordinates") or []
    if gtype == "LineString" and len(coords) >= 2:
        return tuple(coords[0]), tuple(coords[-1])
    if gtype == "MultiLineString" and coords:
        first = coords[0][0]
        last = coords[-1][-1]
        return tuple(first), tuple(last)
    return (0.0, 0.0), (0.0, 0.0)


def _cardinal_direction(start_lng_lat: tuple[float, float], end_lng_lat: tuple[float, float]) -> str:
    """Map segment delta to one of NB/SB/EB/WB."""
    slng, slat = start_lng_lat
    elng, elat = end_lng_lat
    dlat = elat - slat
    dlng = elng - slng
    mean_lat = (slat + elat) * 0.5
    east = dlng * math.cos(math.radians(mean_lat))
    north = dlat
    if abs(north) >= abs(east):
        return "NB" if north >= 0 else "SB"
    return "EB" if east >= 0 else "WB"


def _opposite_direction(direction: str) -> str:
    return {"NB": "SB", "SB": "NB", "EB": "WB", "WB": "EB"}.get(direction, "SB")
