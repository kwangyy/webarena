"""jiuwen_arena parent side, without the WebArena sites or a model.

Run with WebArena's .venv: ``.venv/bin/python -m pytest tests/test_jiuwen_arena.py``.
The browser tests start a real Chromium and a local HTTP server, and grade
with WebArena's own evaluators, so the CDP hand-over and the grading path are
checked end to end.
"""
from __future__ import annotations

import http.server
import json
import os
import threading

import pytest

# browser_env/env_config.py asserts these at import (the graders import it).
for _var in ("REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"):
    os.environ.setdefault(_var, f"http://{_var.lower()}.test:1234")

from jiuwen_arena import tasks  # noqa: E402
from jiuwen_arena.browser_host import TaskBrowser  # noqa: E402
from jiuwen_arena.run_task import answer_of, grade  # noqa: E402
from jiuwen_arena.protocol import ANSWER_FILE  # noqa: E402


def test_load_task_fills_every_placeholder(monkeypatch):
    monkeypatch.setenv("SHOPPING_ADMIN", "http://host:7780/admin/")
    monkeypatch.setenv("SHOPPING", "http://host:7770")
    task = tasks.load_task(0)
    assert task["start_url"] == "http://host:7780/admin"  # not "<SHOPPING>_ADMIN__"
    filled = json.dumps([tasks.load_task(i) for i in range(812)])
    assert not [p for p, _ in tasks.SITE_VARS if p in filled]


def test_missing_site_vars_stop_the_run(monkeypatch):
    monkeypatch.delenv("GITLAB")
    with pytest.raises(SystemExit, match="GITLAB"):
        tasks.load_task(0)


def test_storage_state_sites_and_start_urls():
    assert tasks.sites_of_storage_state("./.auth/gitlab.reddit_state.json") == ["gitlab", "reddit"]
    assert tasks.sites_of_storage_state("./.auth/shopping_admin_state.json") == ["shopping_admin"]
    assert tasks.start_urls({"start_url": "http://a |AND| http://b"}) == ["http://a", "http://b"]
    assert tasks.start_urls({"start_url": None}) == []


def test_answer_of(tmp_path):
    assert answer_of(str(tmp_path)) == ("", False)
    (tmp_path / ANSWER_FILE).write_text(json.dumps({"answer": "42"}), encoding="utf-8")
    assert answer_of(str(tmp_path)) == ("42", True)


# ---------------------------------------------------------------------- live browser

PAGES = {
    "/a": "<html><head><title>A</title></head><body><p id='who'>nobody</p></body></html>",
    "/b": "<html><head><title>B</title></head><body><p id='who'>"
          "<script>document.write(document.cookie)</script></p></body></html>",
}


@pytest.fixture(scope="module")
def site():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = PAGES.get(self.path.split("?")[0], "<html><body>other</body></html>").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture()
def browser():
    b = TaskBrowser()
    b.start()
    yield b
    b.close()


def _config(tmp_path, site, eval_block):
    cfg = {"task_id": 9999, "sites": ["test"], "intent": "test", "storage_state": None,
           "start_url": f"{site}/a", "geolocation": None, "eval": eval_block}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return str(path)


def test_setup_opens_start_tabs_with_cookies(browser, site, tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"cookies": [{"name": "session", "value": "s3cr3t", "domain": "127.0.0.1",
                                              "path": "/", "expires": -1, "httpOnly": False, "secure": False,
                                              "sameSite": "Lax"}], "origins": []}), encoding="utf-8")
    browser.setup_task({"storage_state": str(state)}, [f"{site}/a", f"{site}/b"])
    urls = [p.url for p in browser.context.pages]
    assert urls == [f"{site}/a", f"{site}/b"]  # the launch tab is gone
    assert "session=s3cr3t" in browser.context.pages[1].inner_text("#who")


def test_a_second_cdp_client_shares_the_tabs(browser, site):
    """What the agent's Playwright MCP sees when it attaches: the same context and tabs."""
    from playwright.sync_api import sync_playwright

    browser.setup_task({}, [f"{site}/a"])
    seen = []

    def agent():  # its own thread: one sync Playwright per thread
        with sync_playwright() as p:
            other = p.chromium.connect_over_cdp(browser.cdp_url)
            page = other.contexts[0].pages[0]
            seen.append(page.url)
            page.goto(f"{site}/b")  # the agent navigates ...
            other.close()

    t = threading.Thread(target=agent)
    t.start()
    t.join(60)
    assert seen == [f"{site}/a"]
    assert browser.grading_page().url == f"{site}/b"  # ... and the grader sees it


def test_grade_string_url_and_html(browser, site, tmp_path):
    browser.setup_task({}, [f"{site}/a"])
    string_cfg = _config(tmp_path, site, {"eval_types": ["string_match"],
                                          "reference_answers": {"exact_match": "Quest Band"},
                                          "reference_url": "", "program_html": [], "string_note": ""})
    assert grade(browser, string_cfg, "Quest Band", str(tmp_path)) == 1.0
    assert grade(browser, string_cfg, "Other", str(tmp_path)) == 0.0

    url_cfg = _config(tmp_path, site, {"eval_types": ["url_match"], "reference_answers": None,
                                       "reference_url": f"{site}/a", "url_note": "GOLD in PRED",
                                       "program_html": []})
    assert grade(browser, url_cfg, "", str(tmp_path)) == 1.0

    html_cfg = _config(tmp_path, site, {"eval_types": ["program_html"], "reference_answers": None,
                                        "reference_url": "", "program_html": [
                                            {"url": "last", "locator": "document.querySelector('#who').outerText",
                                             "required_contents": {"must_include": ["nobody"]}}]})
    assert grade(browser, html_cfg, "", str(tmp_path)) == 1.0
