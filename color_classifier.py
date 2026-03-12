"""
color_classifier.py
-------------------
Classifies Google Maps traffic overlay colors from screenshots.

This version avoids "median of saturated pixels" because saturated map blues
often dominate local neighborhoods and lead to false unknowns. Instead:
  1) classify many pixels in a local region with traffic-specific HSV models
  2) weight by distance to segment sample point and class confidence
  3) aggregate votes across all sample points on the segment
"""

import colorsys
import math
from collections import Counter
from collections import defaultdict


def rgb_to_hsv_360(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Convert 0-255 RGB to HSV with H in [0,360], S and V in [0,100]."""
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h * 360.0, s * 100.0, v * 100.0


# Google Maps traffic ramp (user-provided, 2026):
# fast      #11d68f
# moderate  #ffcf43
# slow      #f24e42
# very_slow #a92727
_CLASS_RGB: dict[str, tuple[int, int, int]] = {
    "fast": (0x11, 0xD6, 0x8F),
    "moderate": (0xFF, 0xCF, 0x43),
    "slow": (0xF2, 0x4E, 0x42),
    "very_slow": (0xA9, 0x27, 0x27),
}
_CLASS_PROTOTYPES: dict[str, list[tuple[float, float, float]]] = {
    cls: [rgb_to_hsv_360(*rgb)] for cls, rgb in _CLASS_RGB.items()
}
_UNKNOWN_RGB = (128, 128, 128)


def _hue_distance_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _prototype_similarity(
    h: float, s: float, v: float, ph: float, ps: float, pv: float
) -> float:
    dh = _hue_distance_deg(h, ph) / 34.0
    ds = abs(s - ps) / 34.0
    dv = abs(v - pv) / 34.0
    d2 = dh * dh + ds * ds + dv * dv
    return math.exp(-0.5 * d2)


def _pixel_class_scores(r: int, g: int, b: int) -> dict[str, float]:
    """Return per-class similarity scores for one RGB pixel."""
    h, s, v = rgb_to_hsv_360(r, g, b)

    # Base rejector for road/background/labels.
    if s < 22.0 or v < 18.0:
        return {}

    # Restrict to hue families consistent with the 4-color traffic ramp;
    # rejects most map blue/purple UI elements.
    in_red = h <= 18.0 or h >= 346.0
    in_orange_yellow = 24.0 <= h <= 64.0
    in_green_cyan = 138.0 <= h <= 176.0
    if not (in_red or in_orange_yellow or in_green_cyan):
        return {}

    scores: dict[str, float] = {}
    for cls, prototypes in _CLASS_PROTOTYPES.items():
        scores[cls] = max(
            _prototype_similarity(h, s, v, ph, ps, pv) for ph, ps, pv in prototypes
        )

    # Red family split: darker reds are more likely very_slow.
    if in_red and v < 50.0:
        scores["very_slow"] *= 1.16
        scores["slow"] *= 0.9
    if in_red and v >= 50.0:
        scores["slow"] *= 1.08
        scores["very_slow"] *= 0.92

    return scores


def _nearest_ramp_class_loose(r: int, g: int, b: int) -> tuple[str, float]:
    """
    Looser nearest-class matcher for fallback when strict scoring finds too
    little evidence. Returns (class_name, confidence_0_to_1).
    """
    h, s, v = rgb_to_hsv_360(r, g, b)
    if s < 16.0 or v < 16.0:
        return "unknown", 0.0

    best_cls = "unknown"
    best_score = 0.0
    for cls, (cr, cg, cb) in _CLASS_RGB.items():
        ch, cs, cv = _CLASS_PROTOTYPES[cls][0]
        dr = r - cr
        dg = g - cg
        db = b - cb
        rgb_dist = math.sqrt(dr * dr + dg * dg + db * db)

        # Broad traffic hue affinity to one of the 4 anchors.
        hue_d = _hue_distance_deg(h, ch)
        if hue_d > 34.0 and s < 35.0:
            continue

        rgb_sim = math.exp(-(rgb_dist / 82.0) ** 2)
        hue_sim = math.exp(-((hue_d / 36.0) ** 2))
        sv_sim = math.exp(-(((abs(s - cs) / 45.0) ** 2) + ((abs(v - cv) / 55.0) ** 2)))
        score = 0.55 * rgb_sim + 0.3 * hue_sim + 0.15 * sv_sim
        if score > best_score:
            best_score = score
            best_cls = cls

    return best_cls, best_score


def classify_color(r: int, g: int, b: int) -> str:
    """
    Map an RGB pixel value to a traffic condition string.
    Returns one of: fast, moderate, slow, very_slow, unknown.
    """
    scores = _pixel_class_scores(r, g, b)
    if not scores:
        cls, conf = _nearest_ramp_class_loose(r, g, b)
        return cls if conf >= 0.52 else "unknown"
    cls, score = max(scores.items(), key=lambda kv: kv[1])
    if score >= 0.30:
        return cls
    cls2, conf2 = _nearest_ramp_class_loose(r, g, b)
    return cls2 if conf2 >= 0.56 else "unknown"


def find_traffic_color_in_region(
    image,  # PIL.Image.Image in RGB mode
    px: int,
    py: int,
    radius: int = 12,
) -> tuple[int, int, int]:
    """
    Return a representative RGB color around a sample point.

    Uses confidence-weighted pixel voting across a local neighborhood and
    returns the color of the strongest traffic class.
    """
    w, h = image.size
    sigma = max(2.0, radius * 0.55)
    class_scores: dict[str, float] = defaultdict(float)
    class_rgb_weighted: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    class_weight: dict[str, float] = defaultdict(float)
    total_spatial_weight = 0.0
    total_traffic_weight = 0.0

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = px + dx, py + dy
            if 0 <= x < w and 0 <= y < h:
                p = image.getpixel((x, y))
                r, g, b = p[0], p[1], p[2]
                spatial_w = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
                total_spatial_weight += spatial_w
                scores = _pixel_class_scores(r, g, b)
                if not scores:
                    continue

                cls, cls_score = max(scores.items(), key=lambda kv: kv[1])
                if cls_score < 0.36:
                    continue

                vote_w = spatial_w * (0.35 + 0.65 * cls_score)
                class_scores[cls] += vote_w
                class_rgb_weighted[cls][0] += vote_w * r
                class_rgb_weighted[cls][1] += vote_w * g
                class_rgb_weighted[cls][2] += vote_w * b
                class_weight[cls] += vote_w
                total_traffic_weight += vote_w

    if total_traffic_weight >= 1.35 and (
        total_spatial_weight <= 0 or (total_traffic_weight / total_spatial_weight) >= 0.01
    ):
        best_cls = max(class_scores.items(), key=lambda kv: kv[1])[0]
        wsum = max(class_weight[best_cls], 1e-6)
        r = int(round(class_rgb_weighted[best_cls][0] / wsum))
        g = int(round(class_rgb_weighted[best_cls][1] / wsum))
        b = int(round(class_rgb_weighted[best_cls][2] / wsum))
        return (r, g, b)

    # Fallback pass: nearest-ramp matching in a slightly larger neighbourhood.
    fallback_scores: dict[str, float] = defaultdict(float)
    extra = 6
    r2 = radius + extra
    sigma2 = max(2.5, r2 * 0.6)
    for dy in range(-r2, r2 + 1):
        for dx in range(-r2, r2 + 1):
            x, y = px + dx, py + dy
            if 0 <= x < w and 0 <= y < h:
                rr, gg, bb = image.getpixel((x, y))
                cls, conf = _nearest_ramp_class_loose(rr, gg, bb)
                if cls == "unknown" or conf < 0.56:
                    continue
                spatial_w = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma2 * sigma2))
                fallback_scores[cls] += spatial_w * conf

    if fallback_scores:
        fallback_cls, fallback_score = max(fallback_scores.items(), key=lambda kv: kv[1])
        if fallback_score >= 1.1:
            return _CLASS_RGB[fallback_cls]

    return _UNKNOWN_RGB


def classify_segment_from_samples(
    image,
    sample_pixels: list[tuple[int, int]],
) -> tuple[str, str]:
    """
    Sample traffic color at multiple pixel positions along a street segment.

    For each position, scans a wide neighbourhood for traffic-coloured pixels
    and classifies.  Returns the majority-vote (condition, hex_color) pair,
    preferring non-"unknown" results.

    Returns:
        (traffic_condition, traffic_color_hex)
    """
    votes: list[tuple[str, int, int, int]] = []

    for px, py in sample_pixels:
        r, g, b = find_traffic_color_in_region(image, px, py, radius=14)
        condition = classify_color(r, g, b)
        if condition != "unknown":
            r, g, b = _CLASS_RGB[condition]
        else:
            r, g, b = _UNKNOWN_RGB
        votes.append((condition, r, g, b))

    # Prefer non-unknown votes
    non_unknown = [(c, r, g, b) for c, r, g, b in votes if c != "unknown"]
    if not non_unknown:
        # Segment-level fallback: combine weak signals across sample points.
        soft_scores: dict[str, float] = defaultdict(float)
        for _c, r, g, b in votes:
            cls, conf = _nearest_ramp_class_loose(r, g, b)
            if cls != "unknown" and conf >= 0.52:
                soft_scores[cls] += conf
        if soft_scores:
            cls, agg = max(soft_scores.items(), key=lambda kv: kv[1])
            if agg >= 1.15:
                cr, cg, cb = _CLASS_RGB[cls]
                return cls, rgb_to_hex(cr, cg, cb)

    pool = non_unknown if non_unknown else votes

    # Majority vote on condition; if tie, keep first encountered.
    counts = Counter(v[0] for v in pool)
    best_condition = counts.most_common(1)[0][0]

    # Pick the representative color from the winning condition
    for c, r, g, b in pool:
        if c == best_condition:
            return best_condition, rgb_to_hex(r, g, b)

    # Fallback
    c, r, g, b = votes[0]
    return c, rgb_to_hex(r, g, b)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"
