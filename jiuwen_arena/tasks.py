"""WebArena task configs for the jiuwen_arena runners.

Parent side only (WebArena's own .venv): it fills the site placeholders of
``config_files/test.raw.json`` from the same environment variables as
``scripts/generate_test_data.py`` and renews the login cookies the way
``run.py`` does, without writing into ``config_files/``. Each run gets its own
copy of the task config in its results directory, and that copy is what the
graders read.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CONFIG = os.path.join(REPO_ROOT, "config_files", "test.raw.json")

# Placeholder -> environment variable, as in scripts/generate_test_data.py.
# __SHOPPING_ADMIN__ goes before __SHOPPING__, which is a prefix of it.
SITE_VARS = (
    ("__SHOPPING_ADMIN__", "SHOPPING_ADMIN"),
    ("__SHOPPING__", "SHOPPING"),
    ("__GITLAB__", "GITLAB"),
    ("__REDDIT__", "REDDIT"),
    ("__WIKIPEDIA__", "WIKIPEDIA"),
    ("__MAP__", "MAP"),
)
# Real-world hosts that test.raw.json writes where it means a placeholder.
# Task 66's reference answers spell the reddit clone as http://www.reddit.com,
# so they could never match a URL on the live host.
HOST_ALIASES = (
    ("http://www.reddit.com", "__REDDIT__"),
)
# browser_env/env_config.py asserts every one of these is set, HOMEPAGE included.
REQUIRED_VARS = tuple(var for _, var in SITE_VARS) + ("HOMEPAGE",)


def missing_site_vars() -> list[str]:
    return [var for var in REQUIRED_VARS if not os.environ.get(var, "").strip()]


def load_task(task_id: int) -> dict[str, Any]:
    """Task ``task_id`` of test.raw.json with the site URLs filled in."""
    missing = missing_site_vars()
    if missing:
        raise SystemExit(f"Set the WebArena site URLs first; missing: {', '.join(missing)} "
                         "(see jiuwen_arena/README.md).")
    with open(RAW_CONFIG, encoding="utf-8") as f:
        raw = f.read()
    for host, placeholder in HOST_ALIASES:
        raw = raw.replace(host, placeholder)
    for placeholder, var in SITE_VARS:
        raw = raw.replace(placeholder, os.environ[var].strip().rstrip("/"))
    tasks = json.loads(raw)
    by_id = {task["task_id"]: task for task in tasks}
    if task_id not in by_id:
        raise SystemExit(f"no task {task_id} in {RAW_CONFIG} (ids 0-{len(tasks) - 1})")
    return by_id[task_id]


def sites_of_storage_state(path: str) -> list[str]:
    """``./.auth/gitlab.reddit_state.json`` -> ``["gitlab", "reddit"]`` (as auto_login names them)."""
    name = os.path.basename(path)
    if not name.endswith("_state.json"):
        raise ValueError(f"unexpected storage_state name: {path}")
    return name[: -len("_state.json")].split(".")


def renew_login(task: dict[str, Any], auth_dir: str) -> dict[str, Any]:
    """Log in afresh for the task's sites, as run.py does before every task.

    WebArena's cookies expire, and a task that changes the account (a reddit
    post, a gitlab setting) must not inherit another run's session. Runs
    browser_env/auto_login.py with this interpreter, so it is WebArena's own
    login code and Playwright.
    """
    state = task.get("storage_state")
    if not state:
        return task
    os.makedirs(auth_dir, exist_ok=True)
    sites = sites_of_storage_state(state)
    proc = subprocess.run(
        [sys.executable, os.path.join("browser_env", "auto_login.py"), "--auth_folder", auth_dir,
         "--site_list", *sites],
        cwd=REPO_ROOT, capture_output=True, text=True)
    renewed = os.path.join(auth_dir, os.path.basename(state))
    if proc.returncode != 0 or not os.path.isfile(renewed):
        raise SystemExit(f"auto_login for {sites} failed (exit {proc.returncode}):\n{proc.stdout}{proc.stderr}")
    return {**task, "storage_state": renewed}


def write_task_config(task: dict[str, Any], results_dir: str) -> str:
    path = os.path.join(results_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)
    return path


def start_urls(task: dict[str, Any]) -> list[str]:
    raw = task.get("start_url") or ""
    return [url.strip() for url in raw.split(" |AND| ") if url.strip()]
