/* -----------------------------------------------------------------------
   app.js  —  NYC Traffic Recorder frontend logic
   ----------------------------------------------------------------------- */

// ---- Map initialisation ------------------------------------------------

const map = L.map("map", { preferCanvas: true }).setView([40.7128, -74.006], 13);

// Google Maps road map with live traffic overlay
L.tileLayer("https://mt{s}.google.com/vt/lyrs=m,traffic&x={x}&y={y}&z={z}", {
  subdomains: "0123",
  maxZoom: 20,
  attribution: "Map data © Google",
}).addTo(map);

// ---- Draw control (rectangle only) ------------------------------------

const drawnItems = new L.FeatureGroup().addTo(map);

const drawControl = new L.Control.Draw({
  draw: {
    rectangle: {
      shapeOptions: { color: "#1565c0", weight: 2, fillOpacity: 0.08 },
    },
    polygon:       false,
    circle:        false,
    marker:        false,
    polyline:      false,
    circlemarker:  false,
  },
  edit: { featureGroup: drawnItems, remove: true },
});
map.addControl(drawControl);

// ---- State ------------------------------------------------------------

let currentBbox = null;   // [minLng, minLat, maxLng, maxLat]
let pollTimer   = null;

// ---- DOM references ---------------------------------------------------

const bboxDisplay   = document.getElementById("bbox-display");
const previewDisplay = document.getElementById("preview-display");
const btnStart      = document.getElementById("btn-start");
const btnStop       = document.getElementById("btn-stop");
const stActive      = document.getElementById("st-active");
const stSegments    = document.getElementById("st-segments");
const stTiles       = document.getElementById("st-tiles");
const stCount       = document.getElementById("st-count");
const stLast        = document.getElementById("st-last");
const stNext        = document.getElementById("st-next");
const stError       = document.getElementById("st-error");
const fileList      = document.getElementById("file-list");

// ---- Draw events -------------------------------------------------------

map.on(L.Draw.Event.CREATED, async (e) => {
  drawnItems.clearLayers();
  drawnItems.addLayer(e.layer);

  const bounds = e.layer.getBounds();
  currentBbox = [
    bounds.getWest(),
    bounds.getSouth(),
    bounds.getEast(),
    bounds.getNorth(),
  ];

  bboxDisplay.textContent =
    `W:${currentBbox[0].toFixed(4)}  S:${currentBbox[1].toFixed(4)}\n` +
    `E:${currentBbox[2].toFixed(4)}  N:${currentBbox[3].toFixed(4)}`;
  bboxDisplay.classList.remove("muted");
  previewDisplay.textContent = "Fetching preview…";
  btnStart.disabled = true;

  try {
    const params = new URLSearchParams({ bbox: currentBbox.join(",") });
    const res    = await fetch(`/api/preview_bbox?${params}`);
    const data   = await res.json();
    if (data.error) {
      previewDisplay.textContent = `Error: ${data.error}`;
    } else {
      previewDisplay.textContent =
        `${data.segment_count} street segments  ·  ${data.tile_count} tile(s)`;
      previewDisplay.classList.remove("muted");
      btnStart.disabled = false;
    }
  } catch (err) {
    previewDisplay.textContent = `Preview failed: ${err.message}`;
  }
});

map.on(L.Draw.Event.DELETED, () => {
  currentBbox = null;
  bboxDisplay.textContent = "No area selected";
  bboxDisplay.classList.add("muted");
  previewDisplay.textContent = "";
  btnStart.disabled = true;
});

// ---- Start recording --------------------------------------------------

btnStart.addEventListener("click", async () => {
  if (!currentBbox) return;

  btnStart.disabled = true;
  clearError();

  try {
    const res  = await fetch("/api/start", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ bbox: currentBbox }),
    });
    const data = await res.json();

    if (data.error) {
      showError(data.error);
      btnStart.disabled = false;
      return;
    }

    btnStop.disabled = false;
    startPolling();
  } catch (err) {
    showError(`Request failed: ${err.message}`);
    btnStart.disabled = false;
  }
});

// ---- Stop recording ---------------------------------------------------

btnStop.addEventListener("click", async () => {
  btnStop.disabled = true;
  stopPolling();
  clearError();

  try {
    await fetch("/api/stop", { method: "POST" });
  } catch (_) { /* ignore */ }

  setBadge("idle");
  stNext.textContent = "—";
  btnStart.disabled = (currentBbox === null);
  await refreshStatus();
  await refreshFileList();
});

// ---- Polling ----------------------------------------------------------

function startPolling() {
  refreshStatus();
  refreshFileList();
  pollTimer = setInterval(async () => {
    await refreshStatus();
    await refreshFileList();
  }, 10_000);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function refreshStatus() {
  try {
    const res  = await fetch("/api/status");
    const data = await res.json();

    if (data.active) {
      setBadge("recording");
      btnStop.disabled  = false;
      btnStart.disabled = true;
    } else {
      setBadge("idle");
    }

    stSegments.textContent = data.segment_count != null ? data.segment_count : "—";
    stTiles.textContent    = data.tile_count    != null ? data.tile_count    : "—";
    stCount.textContent    = data.capture_count != null ? data.capture_count : "0";
    stLast.textContent     = data.last_capture  ? formatTs(data.last_capture) : "—";
    stNext.textContent     = data.next_capture  ? formatTs(data.next_capture) : "—";

    if (data.error) {
      showError(data.error);
    } else {
      clearError();
    }
  } catch (_) { /* ignore transient errors */ }
}

async function refreshFileList() {
  try {
    const res  = await fetch("/api/captures");
    const data = await res.json();
    const files = data.files || [];

    if (files.length === 0) {
      fileList.innerHTML = '<li class="muted">No captures yet.</li>';
      return;
    }

    fileList.innerHTML = files
      .slice()
      .reverse()   // Most recent first
      .map(
        (f) =>
          `<li><a href="/api/captures/${encodeURIComponent(f)}" download="${f}">${f}</a></li>`
      )
      .join("");
  } catch (_) { /* ignore */ }
}

// ---- Helpers ----------------------------------------------------------

function setBadge(state) {
  stActive.className = "badge";
  if (state === "recording") {
    stActive.classList.add("badge-recording");
    stActive.textContent = "Recording";
  } else {
    stActive.classList.add("badge-idle");
    stActive.textContent = "Idle";
  }
}

function showError(msg) {
  stError.textContent = msg;
  stError.style.display = "block";
}

function clearError() {
  stError.textContent = "";
  stError.style.display = "none";
}

function formatTs(iso) {
  // "2024-01-15T14:30:00" → "2024-01-15 14:30:00"
  return iso.replace("T", " ");
}

// ---- Initial status load on page open ---------------------------------
refreshStatus();
refreshFileList();
