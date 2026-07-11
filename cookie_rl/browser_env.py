"""Playwright wrapper that runs the real Cookie Clicker headless with a virtual clock.

One CookieBrowser = one Chromium page running the game, driven synchronously via
page.evaluate calls into the injected harness (harness.js). Game files are served
from vendor/cookieclicker over a per-process local HTTP server.
"""

from __future__ import annotations

import functools
import http.server
import json
import socketserver
import threading
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = REPO_ROOT / "vendor" / "cookieclicker"
HARNESS_PATH = Path(__file__).resolve().parent / "harness.js"

# Feb 2 2026 12:00 UTC — outside every seasonal event window
DEFAULT_START_MS = 1770033600000


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass


_server: socketserver.ThreadingTCPServer | None = None
_server_port: int | None = None
_server_lock = threading.Lock()


def _ensure_server(game_dir: Path) -> int:
    """Start (once per process) a local HTTP server serving the game files."""
    global _server, _server_port
    with _server_lock:
        if _server_port is not None:
            return _server_port
        handler = functools.partial(_QuietHandler, directory=str(game_dir))
        _server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
        _server.daemon_threads = True
        _server_port = _server.server_address[1]
        threading.Thread(target=_server.serve_forever, daemon=True).start()
        return _server_port


class CookieBrowser:
    def __init__(
        self,
        game_dir: Path = GAME_DIR,
        seed: int = 1,
        start_time_ms: int = DEFAULT_START_MS,
        headless: bool = True,
    ) -> None:
        if not (game_dir / "main.js").exists():
            raise FileNotFoundError(
                f"Game files not found in {game_dir}. Run scripts/setup_game.sh first."
            )
        port = _ensure_server(game_dir)
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=headless,
            args=[
                "--mute-audio",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
            ],
        )
        self._ctx = self._browser.new_context(viewport={"width": 1280, "height": 800})
        # block everything that is not our local server (fonts, analytics, ads)
        self._ctx.route(
            "**/*",
            lambda route: route.continue_()
            if route.request.url.startswith(f"http://127.0.0.1:{port}/")
            else route.abort(),
        )
        cfg = {"seed": seed, "startTimeMs": start_time_ms}
        self._ctx.add_init_script(f"window.__CC_CONFIG = {json.dumps(cfg)};")
        self._ctx.add_init_script(path=str(HARNESS_PATH))
        self.page = self._ctx.new_page()
        self.page.goto(f"http://127.0.0.1:{port}/index.html", wait_until="domcontentloaded")
        self.page.wait_for_function("window.Game && Game.ready", timeout=120_000)
        ok = self.page.evaluate("__cc.init()")
        if not ok:
            raise RuntimeError("harness init failed (Game not ready?)")

    # ---- core API -----------------------------------------------------------

    def step(self, frames: int) -> dict:
        """Advance N logic frames (N/30 game-seconds); returns observation."""
        return self.page.evaluate("(n) => __cc.step(n)", frames)

    def reset_game(self, seed: int | None = None) -> dict:
        """Full save wipe + RNG reseed, without reloading the page."""
        return self.page.evaluate("(s) => __cc.reset(s)", seed)

    def observe(self) -> dict:
        return self.page.evaluate("__cc.observe()")

    def set_state(self, clicks_per_sec: float | None = None, auto_pop: bool | None = None,
                  pop_wrath: bool | None = None, burst_clicks_per_frame: int | None = None) -> None:
        s: dict[str, Any] = {}
        if clicks_per_sec is not None:
            s["clicksPerSec"] = clicks_per_sec
        if auto_pop is not None:
            s["autoPop"] = auto_pop
        if pop_wrath is not None:
            s["popWrath"] = pop_wrath
        if burst_clicks_per_frame is not None:
            s["burstClicksPerFrame"] = burst_clicks_per_frame
        if s:
            self.page.evaluate("(s) => __cc.setState(s)", s)

    def buy_building(self, building_id: int, amount: int = 1) -> bool:
        return self.page.evaluate("([i, n]) => __cc.buyBuilding(i, n)", [building_id, amount])

    def buy_upgrade_slot(self, slot: int) -> bool:
        return self.page.evaluate("(k) => __cc.buyUpgradeSlot(k)", slot)

    def pop_shimmers(self) -> None:
        self.page.evaluate("__cc.popNow()")

    def click_lump(self) -> None:
        self.page.evaluate("__cc.clickLump()")

    def ascend(self) -> bool:
        return self.page.evaluate("__cc.ascend()")

    def evaluate(self, expr: str, arg: Any = None) -> Any:
        """Escape hatch for tests/benchmarks."""
        return self.page.evaluate(expr, arg)

    def close(self) -> None:
        for closer in (self._ctx.close, self._browser.close, self._pw.stop):
            try:
                closer()
            except Exception:
                pass

    def __enter__(self) -> "CookieBrowser":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
