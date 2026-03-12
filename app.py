"""
app.py
------
Flask web application for the NYC Google Maps traffic condition recorder.

Run with:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import datetime
import json
import logging
import os
import threading

from flask import Flask, jsonify, render_template, request, send_from_directory

import color_classifier
import geo_utils
import scraper as scraper_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["OUTPUT_DIR"] = "output"
app.config["GEOJSON_DIR"] = os.path.join(app.config["OUTPUT_DIR"], "geojson")
app.config["IMAGE_DIR"] = os.path.join(app.config["OUTPUT_DIR"], "image")

# ---------------------------------------------------------------------------
# Global recording state (protected by _state_lock)
# ---------------------------------------------------------------------------

_state: dict = {
    "active": False,
    "bbox": None,
    "capture_count": 0,
    "last_capture": None,
    "next_capture": None,
    "error": None,
    "segment_count": 0,
    "tile_count": 0,
}
_state_lock = threading.Lock()

_scheduler = None          # APScheduler BackgroundScheduler instance


# ---------------------------------------------------------------------------
# Startup: pre-load GeoJSON in a background thread so the first request
# does not block for 5–10 seconds while the 205 MB file is parsed.
# ---------------------------------------------------------------------------

def _preload_geojson() -> None:
    try:
        logger.info("Pre-loading GeoJSON data (this may take a few seconds)...")
        feats = geo_utils.load_features()
        logger.info("GeoJSON loaded: %d street segments cached.", len(feats))
    except Exception as exc:
        logger.error("GeoJSON pre-load failed: %s", exc)
        with _state_lock:
            _state["error"] = f"GeoJSON load failed: {exc}"


threading.Thread(target=_preload_geojson, daemon=True, name="geojson-preload").start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/preview_bbox")
def preview_bbox():
    """Return segment_count and tile_count for a bbox without starting recording."""
    try:
        raw = request.args.get("bbox", "")
        parts = [float(p) for p in raw.split(",")]
        if len(parts) != 4:
            raise ValueError("bbox must have exactly 4 values")
        bbox = parts  # [min_lng, min_lat, max_lng, max_lat]
        segments = geo_utils.filter_by_bbox(bbox)
        tiles = geo_utils.compute_tiles(bbox)
        return jsonify({"segment_count": len(segments), "tile_count": len(tiles)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/start", methods=["POST"])
def start_recording():
    """Start a new recording session with the provided bounding box."""
    global _scheduler

    from apscheduler.schedulers.background import BackgroundScheduler

    with _state_lock:
        if _state["active"]:
            return jsonify({"error": "A recording session is already active."}), 400

    data = request.get_json(silent=True) or {}
    bbox = data.get("bbox")
    if not bbox or len(bbox) != 4:
        return jsonify({"error": "Request body must contain bbox: [minLng, minLat, maxLng, maxLat]"}), 400

    try:
        bbox = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return jsonify({"error": "bbox values must be numbers"}), 400

    # Validate bbox order
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return jsonify({"error": "Invalid bbox: minLng must be < maxLng and minLat must be < maxLat"}), 400

    segments = geo_utils.filter_by_bbox(bbox)
    if not segments:
        return jsonify({"error": "No street segments found in the selected area."}), 400

    tiles = geo_utils.compute_tiles(bbox)
    tile_map = geo_utils.assign_segments_to_tiles(segments, tiles)

    total_assigned = sum(len(v) for v in tile_map.values())
    logger.info(
        "Recording setup: %d segments in bbox, %d tiles, %d segment-tile assignments",
        len(segments), len(tiles), total_assigned,
    )
    for ti, segs in tile_map.items():
        logger.info("  Tile %d (%.5f, %.5f): %d segments", ti, tiles[ti][0], tiles[ti][1], len(segs))

    now = datetime.datetime.now()
    next_capture = now + datetime.timedelta(minutes=5)

    with _state_lock:
        _state.update(
            {
                "active": True,
                "bbox": bbox,
                "capture_count": 0,
                "last_capture": None,
                "next_capture": next_capture.isoformat(timespec="seconds"),
                "error": None,
                "segment_count": len(segments),
                "tile_count": len(tiles),
            }
        )

    # Schedule captures every 5 minutes; run the first one immediately.
    # The scraper (Playwright browser) is created lazily inside _do_capture
    # so it runs in the APScheduler thread — avoids cross-thread issues.
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        func=_do_capture,
        args=[bbox, tiles, tile_map],
        trigger="interval",
        minutes=5,
        next_run_time=now,
        id="traffic_capture",
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()

    return jsonify(
        {
            "status": "started",
            "segment_count": len(segments),
            "tile_count": len(tiles),
        }
    )


@app.route("/api/stop", methods=["POST"])
def stop_recording():
    """Stop the active recording session and release browser resources."""
    global _scheduler

    with _state_lock:
        _state["active"] = False
        _state["next_capture"] = None

    if _scheduler and _scheduler.running:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            pass
        _scheduler = None

    # Clean up the scraper if it was created
    _cleanup_scraper()

    return jsonify({"status": "stopped"})


@app.route("/api/status")
def get_status():
    with _state_lock:
        return jsonify(dict(_state))


@app.route("/api/captures")
def list_captures():
    geojson_dir = app.config["GEOJSON_DIR"]
    if not os.path.isdir(geojson_dir):
        return jsonify({"files": []})
    files = sorted(f for f in os.listdir(geojson_dir) if f.endswith(".geojson"))
    return jsonify({"files": files})


@app.route("/api/captures/<path:filename>")
def download_capture(filename: str):
    geojson_dir = os.path.abspath(app.config["GEOJSON_DIR"])
    return send_from_directory(
        geojson_dir,
        filename,
        as_attachment=True,
        mimetype="application/geo+json",
    )


# ---------------------------------------------------------------------------
# Scraper lifecycle — created lazily inside APScheduler thread
# ---------------------------------------------------------------------------

_scraper: scraper_module.TrafficScraper | None = None
_scraper_lock = threading.Lock()


def _get_or_create_scraper() -> scraper_module.TrafficScraper:
    """Return the existing scraper or create one in the current thread."""
    global _scraper
    with _scraper_lock:
        if _scraper is None:
            logger.info("Launching headless Chromium browser...")
            _scraper = scraper_module.TrafficScraper()
            _scraper.start()
            logger.info("Headless browser ready.")
        return _scraper


def _cleanup_scraper() -> None:
    global _scraper
    with _scraper_lock:
        if _scraper:
            try:
                _scraper.close()
            except Exception:
                pass
            _scraper = None


# ---------------------------------------------------------------------------
# Capture worker (runs inside the APScheduler background thread)
# ---------------------------------------------------------------------------

def _do_capture(bbox: list, tiles: list, tile_map: dict) -> None:
    """
    Execute one traffic capture sweep:
    - For each tile, take a screenshot and sample colour at each street midpoint.
    - Write the results to a timestamped GeoJSON file under output/.
    """
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    features: list[dict] = []
    geojson_dir = app.config["GEOJSON_DIR"]
    image_dir = app.config["IMAGE_DIR"]

    total_expected = sum(len(v) for v in tile_map.values())
    logger.info(
        "Capture started at %s — %d tiles, %d expected segments",
        timestamp, len(tiles), total_expected,
    )

    try:
        scraper = _get_or_create_scraper()

        for tile_idx, (clat, clng) in enumerate(tiles):
            segs_in_tile = tile_map.get(tile_idx, [])
            if not segs_in_tile:
                logger.info("  Tile %d: 0 segments, skipping.", tile_idx)
                continue

            logger.info("  Tile %d: capturing (%.5f, %.5f), %d segments...",
                        tile_idx, clat, clng, len(segs_in_tile))

            try:
                image = scraper.capture_tile(clat, clng)
                logger.info("  Tile %d: screenshot captured (%dx%d).",
                            tile_idx, image.width, image.height)
            except Exception as exc:
                logger.error("  Tile %d: capture FAILED: %s", tile_idx, exc)
                with _state_lock:
                    _state["error"] = f"Tile {tile_idx} capture failed: {exc}"
                continue  # Skip this tile but continue with the rest

            # Save tile screenshot for this capture
            os.makedirs(image_dir, exist_ok=True)
            safe_ts_dbg = timestamp.replace(":", "-")
            image.save(os.path.join(image_dir, f"tile_{tile_idx}_{safe_ts_dbg}.png"))

            for seg, sample_pixels_by_direction in segs_in_tile:
                for direction, sample_pixels in sample_pixels_by_direction.items():
                    condition, hex_color = color_classifier.classify_segment_from_samples(
                        image, sample_pixels,
                    )

                    features.append(
                        {
                            "type": "Feature",
                            "geometry": seg["original_geometry"],
                            "properties": {
                                "physicalid": seg["physicalid"],
                                "street_name": seg["street_name"],
                                "timestamp": timestamp,
                                "Direction": direction,
                                "traffic_color": hex_color,
                                "traffic_condition": condition,
                            },
                        }
                    )

        # Write GeoJSON output
        os.makedirs(geojson_dir, exist_ok=True)
        safe_ts = timestamp.replace(":", "-")
        filename = f"traffic_{safe_ts}.geojson"
        geojson = {
            "type": "FeatureCollection",
            "metadata": {
                "bbox": bbox,
                "captured_at": timestamp,
                "segment_count": len(features),
            },
            "features": features,
        }
        out_path = os.path.join(geojson_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False)

        logger.info(
            "Capture complete: %d features written to %s", len(features), filename,
        )

        next_capture = (
            datetime.datetime.now() + datetime.timedelta(minutes=5)
        ).isoformat(timespec="seconds")

        with _state_lock:
            _state["capture_count"] += 1
            _state["last_capture"] = timestamp
            _state["next_capture"] = next_capture
            _state["error"] = None  # Clear any previous non-fatal error

    except Exception as exc:
        logger.error("Capture failed with exception: %s", exc, exc_info=True)
        with _state_lock:
            _state["error"] = str(exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(app.config["OUTPUT_DIR"], exist_ok=True)
    os.makedirs(app.config["GEOJSON_DIR"], exist_ok=True)
    os.makedirs(app.config["IMAGE_DIR"], exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
