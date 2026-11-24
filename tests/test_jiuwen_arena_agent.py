"""jiuwen_arena agent side: the agent-core seams it relies on, without a model or a browser.

Run with .venv-agent: ``.venv-agent/bin/python -m pytest -p no:cacheprovider tests/test_jiuwen_arena_agent.py
--noconftest`` (WebArena's conftest imports its own Playwright stack). Rerun after moving the agent-core pin.
"""
from __future__ import annotations

import asyncio
import json
import os

os.environ.setdefault("OPENROUTER_API_KEY", "test")
os.environ.setdefault("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
os.environ.setdefault("MODEL_NAME", "anthropic/claude-opus-5")

from jiuwen_arena import openjiuwen_agent as agent_mod  # noqa: E402
from jiuwen_arena.protocol import ANSWER_FILE  # noqa: E402
from jiuwen_arena.tools.submit_answer import SubmitAnswerTool, write_fallback_answer  # noqa: E402


def test_submit_answer_writes_the_answer(tmp_path):
    tool = SubmitAnswerTool(str(tmp_path))
    out = asyncio.run(tool.invoke({"answer": "  Quest Lumaflex™ Band "}))
    assert out.success
    saved = json.loads((tmp_path / ANSWER_FILE).read_text(encoding="utf-8"))
    assert saved == {"answer": "Quest Lumaflex™ Band", "by": "submit_answer"}
    assert not asyncio.run(tool.invoke({"answer": 3})).success


def test_fallback_answer(tmp_path):
    write_fallback_answer(str(tmp_path), " N/A\n")
    assert json.loads((tmp_path / ANSWER_FILE).read_text(encoding="utf-8"))["answer"] == "N/A"


def test_browser_agent_builds_in_remote_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYWRIGHT_RUNTIME_MCP_CWD", str(tmp_path / "mcp"))
    agent = agent_mod.create_browser_agent(
        agent_mod.build_model(), system_prompt=agent_mod.DEFAULT_BROWSER_AGENT_SYSTEM_PROMPT["en"],
        tools=[SubmitAnswerTool(str(tmp_path))], language="en", max_iterations=3, workspace=str(tmp_path / "ws"),
        browser_instance={"key": "webarena-1", "driver_mode": "remote", "cdp_url": "http://127.0.0.1:9222"})
    assert hasattr(agent, "ensure_initialized") and hasattr(agent, "get_current_context")


def test_playwright_mcp_gets_the_cdp_url():
    from openjiuwen.harness.tools.browser_move.playwright_runtime.config import (BrowserInstanceConfig,
                                                                                  build_playwright_mcp_config)

    cfg = build_playwright_mcp_config(BrowserInstanceConfig(key="webarena-1", driver_mode="remote",
                                                            cdp_url="http://127.0.0.1:9222"))
    params = cfg.params if hasattr(cfg, "params") else cfg.model_dump().get("params", {})
    assert params["env"]["PLAYWRIGHT_MCP_CDP_ENDPOINT"] == "http://127.0.0.1:9222"


def test_query_lists_start_tabs():
    q = agent_mod.build_query("Do X", ["http://a", "http://b"])
    assert "Do X" in q and "- http://a" in q and "- http://b" in q
