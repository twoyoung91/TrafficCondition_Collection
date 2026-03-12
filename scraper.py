"""
scraper.py
----------
Playwright-based headless browser automation for capturing Google Maps
traffic screenshots.

Usage:
    scraper = TrafficScraper()
    scraper.start()
    image = scraper.capture_tile(40.7580, -73.9855)   # PIL.Image
    scraper.close()
"""

import io
import time

from PIL import Image

ZOOM = 16
VIEWPORT_W = 1920
VIEWPORT_H = 1080

# Google Maps URL with traffic layer enabled (!5m1!1e1 parameter)
MAPS_URL = "https://www.google.com/maps/@{lat},{lng},{zoom}z/data=!5m1!1e1"

# Realistic Chrome user agent to reduce bot-detection heuristics
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Seconds to wait after networkidle for the traffic overlay tiles to render.
# The traffic layer tiles are fetched from a separate subdomain and render
# 1–3 s after the base map reaches networkidle.
TRAFFIC_RENDER_DELAY = 4.0

# Maximum time (ms) to wait for a page navigation
NAV_TIMEOUT_MS = 35_000


class TrafficScraper:
    """Manages a single headless Chromium session for tile capture."""

    def __init__(self) -> None:
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._consent_dismissed = False

    def start(self) -> None:
        """Launch the headless browser.  Must be called before capture_tile()."""
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--lang=en-US",
                f"--window-size={VIEWPORT_W},{VIEWPORT_H}",
            ],
        )
        self._context = self._browser.new_context(
            viewport={"width": VIEWPORT_W, "height": VIEWPORT_H},
            device_scale_factor=1,
            user_agent=USER_AGENT,
            locale="en-US",
            timezone_id="America/New_York",
        )
        # Remove the navigator.webdriver fingerprint
        self._context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        self._page = self._context.new_page()

    def capture_tile(self, center_lat: float, center_lng: float) -> Image.Image:
        """
        Navigate to a Google Maps tile centered on (center_lat, center_lng),
        wait for the traffic overlay to render, then return a PIL RGB image.
        """
        if self._page is None:
            raise RuntimeError("TrafficScraper not started; call start() first.")

        url = MAPS_URL.format(
            lat=f"{center_lat:.6f}",
            lng=f"{center_lng:.6f}",
            zoom=ZOOM,
        )
        self._page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)

        # Dismiss cookie / consent popup on the very first navigation
        if not self._consent_dismissed:
            self._try_dismiss_consent()
            self._consent_dismissed = True

        # Wait for base map tiles to finish loading
        try:
            self._page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT_MS)
        except Exception:
            pass  # Proceed even if networkidle times out

        # Extra delay for traffic overlay tiles (served from separate origin)
        time.sleep(TRAFFIC_RENDER_DELAY)

        screenshot_bytes = self._page.screenshot(full_page=False, type="png")
        return Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")

    def _try_dismiss_consent(self) -> None:
        """
        Dismiss Google's cookie / consent dialog if it appears.
        Tries several button selectors that Google has used over time.
        """
        selectors = [
            'button:has-text("Accept all")',
            'button:has-text("I agree")',
            'button:has-text("Agree")',
            'button[aria-label*="Accept"]',
            'form[action*="consent"] button',
            '#L2AGLb',   # "Accept all" button ID used in some regions
        ]
        for sel in selectors:
            try:
                btn = self._page.query_selector(sel)
                if btn and btn.is_visible():
                    btn.click()
                    self._page.wait_for_timeout(1_500)
                    return
            except Exception:
                continue

    def close(self) -> None:
        """Shut down the browser and Playwright runtime."""
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        finally:
            try:
                if self._pw:
                    self._pw.stop()
            except Exception:
                pass
            self._pw = None
            self._browser = None
            self._context = None
            self._page = None
            self._consent_dismissed = False
