"""The task's Chromium, shared by the grader (this process) and the agent (over CDP).

WebArena's ScriptBrowserEnv launches its browser inside Playwright with no
debugging port, so the agent process could not reach it. Here Chromium runs
as its own process with ``--remote-debugging-port``; this process attaches
with WebArena's Playwright to set up the task (cookies, start pages) and later
to grade it, and agent-core's browser agent attaches to the same browser in
``remote`` mode. Both use the browser's default context, the one a CDP client
is given on connect, so the agent sees the task's cookies and tabs and the
graders see the pages the agent left.

The setup mirrors ScriptBrowserEnv.setup: 1280x720 viewport, the task's
storage_state, one tab per start URL with the first in front.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from typing import Any, Optional

from playwright.sync_api import Browser, BrowserContext, CDPSession, Page, Playwright, sync_playwright

VIEWPORT = {"width": 1280, "height": 720}
_CDP_READY_S = 30


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _version_key(path: str) -> int:
    m = re.search(r"chromium-(\d+)", path)
    return int(m.group(1)) if m else -1


def find_chromium(explicit: str = "") -> str:
    """The Chromium to run: --chrome-path / CHROME_PATH, else the newest Playwright download.

    The newest matters: agent-core's Playwright MCP is far newer than
    WebArena's Playwright 1.32, and an old client on a new browser works
    where a new client on WebArena's Chromium 112 may not.
    """
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"no Chromium at {explicit}")
        return explicit
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""), os.path.expanduser("~/.cache/ms-playwright")]
    found = []
    for root in filter(None, roots):
        for pattern in ("chromium-*/chrome-linux*/chrome", "chromium-*/chrome-win*/chrome.exe",
                        "chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium"):
            found += glob.glob(os.path.join(root, pattern))
    if not found:
        raise SystemExit("no Chromium found; run `npx -y playwright install chromium` "
                         "or pass --chrome-path")
    return max(found, key=_version_key)


class TaskBrowser:
    """A Chromium process for one task, with this process attached to it over CDP."""

    def __init__(self, chrome_path: str = "", headless: bool = True, port: int = 0) -> None:
        self.chrome_path = find_chromium(chrome_path or os.environ.get("CHROME_PATH", ""))
        self.headless = headless
        self.port = port or free_port()
        self.user_data_dir = tempfile.mkdtemp(prefix="jiuwen_arena_profile_")
        self.proc: Optional[subprocess.Popen] = None
        self._pw_cm = None
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None

    @property
    def cdp_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        args = [self.chrome_path, f"--remote-debugging-port={self.port}", "--remote-debugging-address=127.0.0.1",
                f"--user-data-dir={self.user_data_dir}", f"--window-size={VIEWPORT['width']},{VIEWPORT['height']}",
                "--force-device-scale-factor=1", "--no-first-run", "--no-default-browser-check",
                "--disable-features=Translate", "about:blank"]
        if self.headless:
            args.insert(1, "--headless=new")
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            args.insert(1, "--no-sandbox")
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._wait_for_cdp()
        self._pw_cm = sync_playwright()
        self.playwright = self._pw_cm.__enter__()
        self.browser = self.playwright.chromium.connect_over_cdp(self.cdp_url)
        self.context = self.browser.contexts[0]

    def _wait_for_cdp(self) -> None:
        deadline = time.monotonic() + _CDP_READY_S
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"Chromium exited at startup (code {self.proc.returncode}): {self.chrome_path}")
            try:
                with urllib.request.urlopen(f"{self.cdp_url}/json/version", timeout=2) as r:
                    json.load(r)
                return
            except OSError:
                time.sleep(0.25)
        raise RuntimeError(f"Chromium's debugging port {self.port} did not open in {_CDP_READY_S}s")

    def close(self) -> None:
        try:
            if self.browser is not None:
                self.browser.close()  # over CDP this only disconnects
        except Exception:  # noqa: BLE001 - closing must not hide the run's own result
            pass
        try:
            if self._pw_cm is not None:
                self._pw_cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        shutil.rmtree(self.user_data_dir, ignore_errors=True)

    # ------------------------------------------------------------------ task setup

    def setup_task(self, task: dict[str, Any], urls: list[str]) -> None:
        """Cookies, geolocation and start tabs, as ScriptBrowserEnv.setup gives them."""
        assert self.context is not None
        state = task.get("storage_state")
        if state:
            self._apply_storage_state(state)
        geolocation = task.get("geolocation")
        if geolocation:
            self.context.grant_permissions(["geolocation"])
            self.context.set_geolocation(geolocation)
        blank = list(self.context.pages)
        for url in urls or ["about:blank"]:
            page = self.context.new_page()
            page.set_viewport_size(VIEWPORT)
            page.goto(url)
        for page in blank:  # the about:blank tab Chromium opened at launch
            page.close()
        self.context.pages[0].bring_to_front()

    def _apply_storage_state(self, path: str) -> None:
        # Playwright's storage_state= only exists for new contexts; the default
        # context a CDP client gets takes the cookies and localStorage by hand.
        assert self.context is not None
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        if state.get("cookies"):
            self.context.add_cookies(state["cookies"])
        for origin in state.get("origins") or []:
            items = {item["name"]: item["value"] for item in origin.get("localStorage") or []}
            if items:
                self.context.add_init_script(
                    "if (location.origin === %s) { const items = %s; "
                    "for (const k in items) { localStorage.setItem(k, items[k]); } }"
                    % (json.dumps(origin["origin"]), json.dumps(items)))

    # ------------------------------------------------------------------ grading

    def grading_page(self) -> Page:
        """The tab the agent ended on: WebArena grades env.page, its current tab.

        The tab with focus if the browser reports one, else the last one opened.
        """
        assert self.context is not None
        pages = [p for p in self.context.pages if not p.is_closed()]
        if not pages:
            page = self.context.new_page()
            page.set_viewport_size(VIEWPORT)
            return page
        for page in reversed(pages):
            try:
                if page.evaluate("document.visibilityState === 'visible' && document.hasFocus()"):
                    return page
            except Exception:  # noqa: BLE001 - a page mid-navigation just is not a candidate
                continue
        return pages[-1]

    def cdp_session(self, page: Page) -> CDPSession:
        assert self.context is not None
        client = self.context.new_cdp_session(page)
        client.send("Accessibility.enable")
        return client
