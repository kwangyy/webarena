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
- When the task is to change something, do it completely and leave the browser on the page that shows the result.
- End by calling submit_answer exactly once. For a question, the answer is only the requested information, \
exactly as the site shows it. If the task is impossible here, submit 'N/A'.
"""


def build_model() -> Model:
    client = ModelClientConfig(client_provider="OpenRouter", api_key=os.environ["OPENROUTER_API_KEY"],
                               api_base=os.environ["OPENROUTER_API_BASE"])
    name = os.environ["MODEL_NAME"].strip().strip('"')
    # Extra fields go to the OpenAI SDK as keyword arguments; extra_body is the
    # one it accepts, and puts them in the request body for OpenRouter.
    extra_body = {"provider": PROVIDER_PREFERENCE}
    if REASONING_EFFORT:
        extra_body["reasoning"] = {"effort": REASONING_EFFORT}
    return Model(model_client_config=client, model_config=ModelRequestConfig(model=name, extra_body=extra_body))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--cdp-url", required=True)
    p.add_argument("--instruction-file", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--max-iterations", type=int, default=100)
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

    submit = SubmitAnswerTool(args.results_dir)
    trajectory_path = os.path.join(args.results_dir, "trajectory.jsonl")
    trajectory_rail = TrajectoryStreamRail(trajectory_path)
    agent = create_browser_agent(
        build_model(),
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
          f"max_iterations={args.max_iterations or 'unlimited'})...", flush=True)
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
