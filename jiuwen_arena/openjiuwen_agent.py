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
import shutil
import sys
import urllib.parse
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO_ROOT, ".env"))

from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig  # noqa: E402
from openjiuwen.core.runner.runner import Runner  # noqa: E402
from openjiuwen.harness.subagents.browser_agent import (  # noqa: E402
    DEFAULT_BROWSER_AGENT_SYSTEM_PROMPT, create_browser_agent)

from openjiuwen.harness.rails._multimodal import should_enable_read_image_multimodal  # noqa: E402
from jiuwen_arena.feasibility_gate import page_tab_ids, restore_start_tabs, run_gate  # noqa: E402
from jiuwen_arena.rails import TrajectoryStreamRail, dump_trajectory  # noqa: E402
from jiuwen_arena.tools.view_page import ViewPageTool  # noqa: E402
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
search finds something for a term you know is there -- and then submit 0. Do not keep searching other places for \
a different result.
- 0 and N/A are different answers. Submit 0 when the task asks for a number and the count is zero. Submit 'N/A' \
when the task cannot be answered as asked because what it asks for does not exist here -- for example it asks you \
to find or list items and there are none, or it names something the site does not have. Do not answer 'None', \
'nothing' or an empty list instead of 'N/A'.
- Answer from what the site shows, not from what you think the task's author expected. A simple or empty-looking \
answer (0, None, a single item) is as likely to be right as any other; do not doubt it for being simple. Once you \
have checked something, do not check it again: decide and act.
- Do not deliberate at length. When the task's wording allows more than one reading, choose the most natural one in \
a single sentence and act on it; do not weigh the readings again later. Never reason about the benchmark, the task's \
author, how the answer will be graded or what answer is expected. Keep the thinking for each step short, and as soon \
as you have the information the task asks for, submit it.
- Judge what a review, post or comment says by its meaning, not by matching a word: a review that calls the print \
quality "okay", "so-so" or "nothing special" mentions average print quality as much as one that says "average". \
Read every review or post in scope before answering.
- Stay on the site the task starts on. Each start tab is the site the task is about; do not switch to another site \
(for example from the online store to its admin panel, or back) unless the task names that site or starts a tab \
on it.
- If browser_probe_cards fails twice on a page, or the listing stays marked stale (listing_stale) after you have \
tried once to refresh it, stop using browser_probe_cards on that page: reload the page by navigating to its URL, \
then read it with browser_page_action read_text, and work from that text. A page you just loaded and read is \
current; do not wait for the card listing to recover before using what you read.
- Search a list iteratively, one page at a time. When what you are looking for may be in a list of many entries \
(order history, reviews, posts, search results, an admin grid), start at the first page and read every entry on it, \
opening an entry's own page (such as View Order) when the list does not show what you need, before moving on. Then \
go to the next page with the list's own Next or page-number links, and repeat until you find it or reach the last \
page. Do not conclude that something is missing until you have gone through every page.
- When the task is to change something, do it completely and leave the browser on the page that shows the result.
- When the task is to show, list, look up or browse something on the site, the page the browser ends on is the \
answer: open the page that shows exactly what was asked (the item, the category, the sorted or filtered listing) \
using the site's own links, menus, sort and filter controls, and leave the browser on that page. After you have \
reached it, do not navigate anywhere else, not even to check something; submit 'Done'.
- When the answer is a number, an amount or a size, give the bare value without units or unit symbols: no $ or \
other currency signs, no inch marks (") and no words such as inches, dollars or cm (347.25, not $347.25; 16x24, not \
16"x24" or 16 x 24 inches). Write a size as the numbers joined by a lowercase x without spaces. Product and other \
names are not values: copy them exactly as the site writes them, including any units in them (Sprite Stasis Ball \
65 cm). When you calculate a number yourself (a total, a discount, a price per item), give the full result of the \
calculation and do not round it, not even to the nearest cent: 12.99 / 2 is 6.495, not 6.50.
- When a task asks you to draft something (a price rule, a message, a form), fill in every field the task \
gives and then stop with the filled form on screen: do not press Save or Submit. The filled, unsaved form must be \
the final page.
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


def build_model(max_output_tokens: int = 0, vision: bool = False) -> Model:
    """The agent's model on OpenRouter.

    max_output_tokens caps one model call, thinking included. agent-core's
    browser agent uses an explicit cap as given and otherwise sets 8192, which
    a thinking model can spend on one step without producing a tool call.

    vision declares the model as taking image input (ModelClientConfig.supports_vision,
    read by openjiuwen's should_enable_read_image_multimodal). Without it the browser
    runtime assumes text only: it removes the screenshot tools and tells the model
    "Image input is unavailable or unverified for this run".
    """
    client = ModelClientConfig(client_provider="OpenRouter", api_key=os.environ["OPENROUTER_API_KEY"],
                               api_base=os.environ["OPENROUTER_API_BASE"],
                               supports_vision=vision)
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


def enable_runtime_image_input() -> None:
    """Make openjiuwen's browser runtime treat the model as taking images.

    supports_vision on the model is not enough: BrowserRuntimeRail decides with
    should_enable_read_image_multimodal(ctx.agent), and for the system-prompt
    section and the batch screenshot check ctx.agent is the inner ReActAgent,
    which has no deep_config and so no model to read supports_vision from: it
    always resolves to False ("Image input is unavailable or unverified for this
    run"). Until openjiuwen passes the setting down, answer the rail's question
    here, for this process only.
    """
    from openjiuwen.harness.tools.browser_move.playwright_runtime.runtime import BrowserRuntimeRail
    BrowserRuntimeRail._image_input_supported = staticmethod(lambda agent: True)


# Site guidance lives in jiuwen_arena/skills/<site>/SKILL.md, one skill per site. With --site-guidance skills
# (the default) the skills of the task's sites are copied into the agent's workspace and offered through
# openjiuwen's SkillUseRail (the model sees each skill's name and description and loads it with skill_tool);
# with --site-guidance inline their text is appended to the system prompt instead.
SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
SITE_SKILL_BY_PORT = {"7770": "shopping", "7780": "shopping_admin", "9999": "reddit"}
SKILLS_RULE = """
- Each site has a guide, offered to you as a skill named after the site (shopping, shopping_admin, reddit). \
Before your first browser action, call skill_tool for the skill of each site in the start tabs, and follow it.
"""


def site_skills(start_urls: list[str]) -> list[str]:
    """Skill names for the sites in the start tabs, in order, without repeats."""
    names: list[str] = []
    for url in start_urls:
        port = urllib.parse.urlparse(url).port
        name = SITE_SKILL_BY_PORT.get(str(port)) if port else None
        if name and name not in names and os.path.isfile(os.path.join(SKILLS_DIR, name, "SKILL.md")):
            names.append(name)
    return names


def skill_body(name: str) -> str:
    """A skill's SKILL.md without its front matter."""
    with open(os.path.join(SKILLS_DIR, name, "SKILL.md"), encoding="utf-8") as f:
        text = f.read()
    if text.startswith("---"):
        text = text.split("---", 2)[2]
    return text.strip()


def site_guidance(mode: str, names: list[str], workspace: str):
    """(text to add to the system prompt, rails to add) for the task's site skills."""
    if not names:
        return "", []
    if mode == "inline":
        return "\n\n" + "\n\n".join(skill_body(n) for n in names) + "\n", []
    from openjiuwen.harness.rails.skills.skill_use_rail import SkillUseRail
    dest = os.path.join(workspace, "skills")
    os.makedirs(dest, exist_ok=True)
    for n in names:
        shutil.copytree(os.path.join(SKILLS_DIR, n), os.path.join(dest, n), dirs_exist_ok=True)
    # include_tools=False: only skill_tool, not the read_file / bash fallbacks.
    return SKILLS_RULE, [SkillUseRail(skills_dir=[dest], skill_mode="all", include_tools=False)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--cdp-url", required=True)
    p.add_argument("--instruction-file", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--max-iterations", type=int, default=100)
    p.add_argument("--browser-deadline", type=int, default=900)
    p.add_argument("--max-output-tokens", type=int, default=4096)
    p.add_argument("--start-url", action="append", default=[])
    p.add_argument("--site-guidance", choices=["skills", "inline"], default="skills",
                   help="give the site guides (jiuwen_arena/skills) as openjiuwen skills, or inline in the system prompt")
    p.add_argument("--vision", action="store_true",
                   help="declare the model as taking image input, which enables the screenshot tools")
    p.add_argument("--feasibility-gate", action="store_true",
                   help="run the pre-task feasibility gate first (jiuwen_arena/feasibility_gate.py)")
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

    submit = SubmitAnswerTool(args.results_dir, cdp_url=args.cdp_url, shopping_base=os.environ.get("SHOPPING"))
    trajectory_path = os.path.join(args.results_dir, "trajectory.jsonl")
    trajectory_rail = TrajectoryStreamRail(trajectory_path)
    skill_names = site_skills(args.start_url)
    guidance_text, guidance_rails = site_guidance(args.site_guidance, skill_names, args.workspace)
    print(f"Site guidance ({args.site_guidance}): {', '.join(skill_names) or 'none'}", flush=True)
    agent = create_browser_agent(
        build_model(args.max_output_tokens, args.vision),
        system_prompt=DEFAULT_BROWSER_AGENT_SYSTEM_PROMPT["en"] + WEBARENA_RULES + guidance_text,
        tools=[submit] + ([ViewPageTool(args.cdp_url, args.results_dir)] if args.vision else []),
        rails=[trajectory_rail] + guidance_rails,
        skills=skill_names if guidance_rails else None,
        language="en",
        max_iterations=args.max_iterations or sys.maxsize,  # 0 = no cap
        workspace=args.workspace,
        # The parent's browser, not one agent-core launches: Playwright MCP
        # attaches over CDP and works in its default context, where the task's
        # cookies and start tabs are.
        browser_instance={"key": f"webarena-{args.task_id}", "driver_mode": "remote", "cdp_url": args.cdp_url},
    )
    trajectory_rail.agent = agent

    if args.vision:
        enable_runtime_image_input()
    session_id = f"webarena-{args.task_id}-{uuid.uuid4().hex[:6]}"
    print(f"Running browser agent (model={os.environ['MODEL_NAME']}, cdp={args.cdp_url}, "
          f"max_iterations={args.max_iterations or 'unlimited'}, deadline={args.browser_deadline}s, max_output_tokens={args.max_output_tokens}, vision={args.vision})...", flush=True)
    result = None
    await Runner.start()
    if args.feasibility_gate:
        start_tabs = page_tab_ids(args.cdp_url)
        try:
            verdict = await run_gate(build_model(args.max_output_tokens, args.vision), instruction, args.start_url, args)
        except Exception:
            await Runner.stop()
            raise
        if verdict["infeasible"]:
            # WebArena's answer for an unachievable task, as OSWorld's FAIL action: the agent never starts.
            submit.record("N/A", by="feasibility_gate")
            await Runner.stop()
            return
        await restore_start_tabs(args.cdp_url, args.start_url, start_tabs)
    try:
        await agent.ensure_initialized()
        print(f"Image input per openjiuwen: {should_enable_read_image_multimodal(agent)}", flush=True)
        result = await run_until_submitted(
            Runner.run_agent(agent, {"query": build_query(instruction, args.start_url)}, session=session_id),
            submit)
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


async def run_until_submitted(agent_run, submit: SubmitAnswerTool):
    """Run the agent until it ends or submit_answer succeeds, whichever is first.

    submit_answer is WebArena's stop action: the answer and the page it leaves are
    what is graded, so the agent is cancelled as soon as the answer is recorded,
    and cannot act on the browser afterwards.
    """
    run_task = asyncio.ensure_future(agent_run)
    stop_wait = asyncio.ensure_future(submit.submitted.wait())
    try:
        await asyncio.wait({run_task, stop_wait}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop_wait.cancel()
    if run_task.done():
        return run_task.result()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass
    except Exception as exc:  # noqa: BLE001 - the answer is recorded; an error while cancelling does not matter
        print(f"Agent raised while being stopped after submit_answer: {exc!r}", flush=True)
    print("Stopped the agent: submit_answer recorded the answer.", flush=True)
    return {"result_type": "answer", "output": submit.answer, "stopped_by": "submit_answer"}


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
