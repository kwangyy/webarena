#!/usr/bin/env python3
"""The agent process: agent-core's browser agent on a WebArena task.

Started by jiuwen_arena/run_task.py with .venv-agent's Python. It never
touches WebArena's code or graders: it gets the CDP URL of the browser the
parent set up and the task's instruction, drives the browser with
``create_browser_agent`` in ``remote`` driver mode (Playwright MCP attached
over CDP), and leaves its answer in ``<results-dir>/answer.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO_ROOT, ".env"))

from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig  # noqa: E402
from openjiuwen.core.runner.runner import Runner  # noqa: E402
from openjiuwen.harness.subagents.browser_agent import (  # noqa: E402
    DEFAULT_BROWSER_AGENT_SYSTEM_PROMPT, create_browser_agent)

from jiuwen_arena.rails import TrajectoryStreamRail, dump_trajectory  # noqa: E402
from jiuwen_arena.tools.submit_answer import SubmitAnswerTool, write_fallback_answer  # noqa: E402

EXIT_AGENT_ERROR = 3  # agent returned result_type=error (e.g. max iterations); run_common reads it

# Same routing as OSWorld's jiuwen agent: OpenRouter, Anthropic's own endpoint first.
PROVIDER_PREFERENCE = {"order": ["anthropic"], "allow_fallbacks": True}
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "").strip().strip('"')

# agent-core's browser prompt is written for a subagent working for a parent
# agent; this says what is particular to WebArena.
WEBARENA_RULES = """

# This task

You work alone on one WebArena task: there is no parent agent and no user to ask. The browser is already open on \
the task's start page(s) and logged in where the task needs it. The websites are self-hosted copies (an online \
store, its admin panel, a GitLab, a Reddit-like forum, a map, a wiki); stay on them -- the answer is never on the \
public internet, and their data, not your own knowledge, decides it.

- Do exactly what the task asks, with the exact values it gives; do not change anything it did not ask for.
- Dates, counts and rankings come from the site's own data (its reports, filters, order history, search).
- Zero is an answer. When a search, filter or page shows no matching items (no results, "0 records found", an \
empty list), check once that you searched in the right place with the right terms -- for example that the same \
search finds something for a term you know is there -- and then submit 0 (or 'None', as the task asks). Do not keep \
searching other places for a different result.
- Answer from what the site shows, not from what you think the task's author expected. A simple or empty-looking \
answer (0, None, a single item) is as likely to be right as any other; do not doubt it for being simple. Once you \
have checked something, do not check it again: decide and act.
- When the task is to change something, do it completely and leave the browser on the page that shows the result.
- End by calling submit_answer exactly once. For a question, the answer is only the requested information, \
exactly as the site shows it. If the task is impossible here, submit 'N/A'.
"""


def set_browser_deadline(seconds: int) -> None:
    """One deadline for every task, instead of agent-core's 240 s / 600 s split.

    agent-core's browser runtime calls a task "complex" (600 s) when its text
    contains words such as filter, sort or cart, and "simple" (240 s) otherwise,
    so most WebArena questions got 240 s. Both module constants are read when a
    task starts, so setting them here is enough. The run as a whole is also
    capped by BROWSER_TIMEOUT_S (agent-core default 600), which is raised to
    match when the deadline is longer. Raises if agent-core renames them.
    """
    from openjiuwen.harness.tools.browser_move.playwright_runtime import runtime as browser_runtime

    names = ("_BROWSER_SIMPLE_TASK_DEADLINE_S", "_BROWSER_COMPLEX_TASK_DEADLINE_S")
    missing = [n for n in names if not hasattr(browser_runtime, n)]
    if missing:
        raise RuntimeError(f"agent-core no longer has {', '.join(missing)}; update set_browser_deadline")
    for name in names:
        setattr(browser_runtime, name, float(seconds))
    if seconds > int(os.environ.get("BROWSER_TIMEOUT_S") or 600):
        os.environ["BROWSER_TIMEOUT_S"] = str(seconds)


def build_model(max_output_tokens: int = 0) -> Model:
    """The agent's model on OpenRouter.

    max_output_tokens caps one model call, thinking included. agent-core's
    browser agent uses an explicit cap as given and otherwise sets 8192, which
    a thinking model can spend on one step without producing a tool call.
    """
    client = ModelClientConfig(client_provider="OpenRouter", api_key=os.environ["OPENROUTER_API_KEY"],
                               api_base=os.environ["OPENROUTER_API_BASE"])
    name = os.environ["MODEL_NAME"].strip().strip('"')
    # Extra fields go to the OpenAI SDK as keyword arguments; extra_body is the
    # one it accepts, and puts them in the request body for OpenRouter.
    extra_body = {"provider": PROVIDER_PREFERENCE}
    if REASONING_EFFORT:
        extra_body["reasoning"] = {"effort": REASONING_EFFORT}
    request = ModelRequestConfig(model=name, extra_body=extra_body)
    if max_output_tokens > 0:
        request.max_tokens = max_output_tokens
    return Model(model_client_config=client, model_config=request)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--cdp-url", required=True)
    p.add_argument("--instruction-file", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--max-iterations", type=int, default=100)
    p.add_argument("--browser-deadline", type=int, default=900)
    p.add_argument("--max-output-tokens", type=int, default=16384)
    p.add_argument("--start-url", action="append", default=[])
    return p.parse_args()


def build_query(instruction: str, start_urls: list[str]) -> str:
    tabs = "\n".join(f"- {url}" for url in start_urls) or "- about:blank"
    return f"Task: {instruction}\n\nOpen tabs at the start (the first is in front):\n{tabs}"


def final_text(result) -> str:
    if isinstance(result, dict):
        for key in ("output", "result", "content"):
            if isinstance(result.get(key), str):
                return result[key]
    return result if isinstance(result, str) else ""


async def run(args: argparse.Namespace) -> None:
    with open(args.instruction_file, encoding="utf-8") as f:
        instruction = f.read().strip()
    os.makedirs(args.workspace, exist_ok=True)
    set_browser_deadline(args.browser_deadline)

    submit = SubmitAnswerTool(args.results_dir)
    trajectory_path = os.path.join(args.results_dir, "trajectory.jsonl")
    trajectory_rail = TrajectoryStreamRail(trajectory_path)
    agent = create_browser_agent(
        build_model(args.max_output_tokens),
        system_prompt=DEFAULT_BROWSER_AGENT_SYSTEM_PROMPT["en"] + WEBARENA_RULES,
        tools=[submit],
        rails=[trajectory_rail],
        language="en",
        max_iterations=args.max_iterations or sys.maxsize,  # 0 = no cap
        workspace=args.workspace,
        # The parent's browser, not one agent-core launches: Playwright MCP
        # attaches over CDP and works in its default context, where the task's
        # cookies and start tabs are.
        browser_instance={"key": f"webarena-{args.task_id}", "driver_mode": "remote", "cdp_url": args.cdp_url},
    )
    trajectory_rail.agent = agent

    session_id = f"webarena-{args.task_id}-{uuid.uuid4().hex[:6]}"
    print(f"Running browser agent (model={os.environ['MODEL_NAME']}, cdp={args.cdp_url}, "
          f"max_iterations={args.max_iterations or 'unlimited'}, deadline={args.browser_deadline}s, max_output_tokens={args.max_output_tokens})...", flush=True)
    result = None
    await Runner.start()
    try:
        await agent.ensure_initialized()
        result = await Runner.run_agent(agent, {"query": build_query(instruction, args.start_url)},
                                        session=session_id)
        print(f"Agent result: {result}", flush=True)
    finally:
        dump_trajectory(agent, trajectory_path, session_id)
        await Runner.stop()
    if submit.answer is None:
        text = final_text(result)
        print("WARNING: submit_answer was never called; using the final message as the answer.", flush=True)
        write_fallback_answer(args.results_dir, text)
    if isinstance(result, dict) and result.get("result_type") == "error":
        sys.exit(EXIT_AGENT_ERROR)


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
