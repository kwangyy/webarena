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
from jiuwen_arena.tools import submit_answer as submit_mod  # noqa: E402
from jiuwen_arena.tools.submit_answer import SubmitAnswerTool, write_fallback_answer  # noqa: E402


STORE = "http://157.230.85.206:7770"


def test_cat_filtered_store_page():
    check = submit_mod.is_cat_filtered_store_page
    assert check(STORE + "/clothing-shoes-jewelry.html?cat=145", STORE)
    assert check(STORE + "/video-games.html?product_list_order=price&cat=236", STORE)
    assert not check(STORE + "/clothing-shoes-jewelry/men/shoes.html", STORE)
    assert not check(STORE + "/catalogsearch/result/?q=cat+toy", STORE)  # "cat" in a value, not a key
    assert not check("http://157.230.85.206:7780/admin/x?cat=1", STORE)  # other site
    assert not check(None, STORE) and not check(STORE + "/a.html?cat=1", None)


def test_submit_answer_refuses_done_on_a_cat_page_once(tmp_path, monkeypatch):
    monkeypatch.setattr(submit_mod, "front_tab_url", lambda cdp: STORE + "/video-games.html?cat=236")
    tool = SubmitAnswerTool(str(tmp_path), cdp_url="http://127.0.0.1:1", shopping_base=STORE)
    first = asyncio.run(tool.invoke({"answer": "Done"}))
    assert not first.success and "?cat=" in first.error and tool.answer is None
    assert not (tmp_path / ANSWER_FILE).exists()
    second = asyncio.run(tool.invoke({"answer": "Done"}))
    assert second.success and tool.answer == "Done"


def test_submit_answer_checks_only_done(tmp_path, monkeypatch):
    monkeypatch.setattr(submit_mod, "front_tab_url", lambda cdp: STORE + "/video-games.html?cat=236")
    tool = SubmitAnswerTool(str(tmp_path), cdp_url="http://127.0.0.1:1", shopping_base=STORE)
    assert asyncio.run(tool.invoke({"answer": "9.99"})).success


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


def test_browser_deadline_is_the_same_for_every_task(monkeypatch):
    from openjiuwen.harness.tools.browser_move.playwright_runtime import runtime as browser_runtime

    monkeypatch.setattr(browser_runtime, "_BROWSER_SIMPLE_TASK_DEADLINE_S", 240.0)
    monkeypatch.setattr(browser_runtime, "_BROWSER_COMPLEX_TASK_DEADLINE_S", 600.0)
    monkeypatch.delenv("BROWSER_TIMEOUT_S", raising=False)
    agent_mod.set_browser_deadline(600)
    assert browser_runtime._BROWSER_SIMPLE_TASK_DEADLINE_S == 600.0
    assert browser_runtime._BROWSER_COMPLEX_TASK_DEADLINE_S == 600.0
    assert "BROWSER_TIMEOUT_S" not in os.environ  # 600 is agent-core's own default
    agent_mod.set_browser_deadline(900)
    assert browser_runtime._BROWSER_SIMPLE_TASK_DEADLINE_S == 900.0
    assert os.environ["BROWSER_TIMEOUT_S"] == "900"


def test_browser_agent_keeps_our_output_limit():
    from openjiuwen.harness.subagents import browser_agent

    ours = browser_agent._browser_model_with_temperature(agent_mod.build_model(65536), 0.0)
    assert ours.model_config.max_tokens == 65536
    default = browser_agent._browser_model_with_temperature(agent_mod.build_model(), 0.0)
    assert default.model_config.max_tokens == browser_agent.DEFAULT_BROWSER_AGENT_MAX_OUTPUT_TOKENS


def test_feasibility_gate_verdict_tools_record_once():
    from jiuwen_arena.feasibility_gate import _VerdictTool
    holder = {}
    infeasible = _VerdictTool("report_infeasible", True, holder)
    feasible = _VerdictTool("report_feasible", False, holder)
    assert asyncio.run(infeasible.invoke({"reason": "no such filter"})).success
    assert holder == {"infeasible": True, "reason": "no such filter"}
    asyncio.run(feasible.invoke({"reason": "changed my mind"}))  # a second verdict is ignored
    assert holder["infeasible"] is True


def test_submit_answer_record_by(tmp_path):
    tool = SubmitAnswerTool(str(tmp_path))
    tool.record("N/A", by="feasibility_gate")
    assert json.loads((tmp_path / ANSWER_FILE).read_text()) == {"answer": "N/A", "by": "feasibility_gate"}
