"""Pre-task feasibility gate, after OSWorld's Pointer agent (mm_agents/pointer/agent_feasibility_gate.py).

Before the agent works on a task, a separate browser agent with its own prompt
probes the sites read-only and ends by calling report_feasible or
report_infeasible exactly once. An infeasible verdict ends the task with the
answer 'N/A', WebArena's answer for an unachievable task (as OSWorld's FAIL);
a feasible verdict, or a gate that runs out of steps, lets the agent start, as
in Pointer ("Budget exhausted - defaulting to feasible"). Afterwards the start
tabs are restored, so the agent begins exactly where it would have without the gate.

Run by openjiuwen_agent.py when --feasibility-gate is given. The verdict and
its reason are written to <results-dir>/feasibility_gate.json, the gate's own
messages to gate_trajectory.jsonl.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.parse
import urllib.request
import uuid
from typing import Any, AsyncIterator, Dict

from openjiuwen.core.foundation.tool.base import Tool, ToolCard
from openjiuwen.core.runner.runner import Runner
from openjiuwen.harness.subagents.browser_agent import DEFAULT_BROWSER_AGENT_SYSTEM_PROMPT, create_browser_agent
from openjiuwen.harness.tools.base_tool import ToolOutput

from jiuwen_arena.rails import TrajectoryStreamRail, dump_trajectory

GATE_FILE = "feasibility_gate.json"
GATE_MAX_STEPS = int(os.environ.get("GATE_MAX_STEPS", 20))  # Pointer: POINTER_GATE_AGENT_MAX_STEPS=20

GATE_RULES = """

# Feasibility gate

You are a Feasibility Gate. Your job is to determine whether a task CAN be completed on these websites before \
another agent attempts it. You do NOT complete the task -- you only verify whether it is possible. There is no \
user to ask. The websites are self-hosted copies (an online store, its admin panel, a GitLab, a Reddit-like forum, a \
map, a wiki), already open and logged in; the answer is never on the public internet.

## Probing is READ-ONLY

Open pages, read them, search, sort, filter, hover and expand menus. NEVER change anything: do not submit, save, \
post, comment, vote, subscribe, follow, add to cart or wishlist, place or cancel an order, edit a setting, or send a \
form. If checking something would need a change, judge from what you can read instead.

## Mandatory checks

Before deciding, you MUST:
1. VERIFY RESOURCES -- if the task names something specific (a product, order, customer, user, forum, post, \
repository, issue, report, review, date range, place), look it up and confirm it exists on the site.
2. VERIFY THE FEATURE -- if the task needs a site feature (a filter, sort, report, setting, kind of page or \
action), confirm the site has it, for example by opening the menu, page or form that offers it (without submitting).
3. PARSE CONSTRAINTS -- read the task for conditions that restrict what counts (within a budget, at least N \
reviews, only X, a date range) and check that something on the site can meet them.

## Infeasibility criteria

Call report_infeasible when ANY of these hold:
1. MISSING FEATURE -- the site does not offer the feature the task needs (for example the store has no way to show \
only discounted items, the forum has no such setting).
2. MISSING RESOURCES -- what the task names does not exist on the site (no such product, user, forum, order, \
repository), and it is not just named slightly differently.
3. HARD CONSTRAINT BLOCKING -- the task's own conditions rule out every candidate on the site.

## Not infeasible

- A question whose correct answer is zero, none found, or an empty list is FEASIBLE: the answer is 0 or a statement \
of what was found. Only a task that asks for something the site cannot provide is infeasible.
- Ambiguous wording is FEASIBLE: the agent will choose the natural reading.
- Hard to find is not missing: search, browse the menus and use the site's own search before deciding something is \
absent.
- If, after checking, you are not sure, report FEASIBLE. Wrongly stopping a feasible task loses it; the agent can \
still answer 'N/A' later if it finds the task impossible.

## Decision

Call report_feasible or report_infeasible exactly once, with a short reason citing the specific evidence from your \
probing (the page, search or menu you checked). Do not deliberate at length; keep the thinking for each step short. \
You have at most {max_steps} steps.
"""


class _VerdictTool(Tool):
    """report_feasible / report_infeasible: record the gate's verdict once."""

    def __init__(self, name: str, infeasible: bool, holder: dict) -> None:
        what = "IS" if not infeasible else "is NOT"
        super().__init__(ToolCard(
            id=f"webarena_gate_{name}", name=name,
            description=f"Report that the task {what} feasible on these websites. Call exactly once, with your reason "
                        "citing the evidence from your probing. This ends the feasibility check.",
            input_params={"type": "object",
                          "properties": {"reason": {"type": "string", "description": "Evidence-based reason."}},
                          "required": ["reason"]},
        ))
        self._infeasible = infeasible
        self._holder = holder

    async def invoke(self, inputs: Dict[str, Any], **kwargs: Any) -> ToolOutput:
        if "infeasible" in self._holder:
            return ToolOutput(success=True, data={"status": "a verdict is already recorded; stop now"})
        self._holder.update(infeasible=self._infeasible, reason=str(inputs.get("reason", "")).strip())
        return ToolOutput(success=True, data={"status": "verdict recorded",
                                              "next": "The feasibility check is over: stop now, no further actions."})

    async def stream(self, inputs: Dict[str, Any], **kwargs: Any) -> AsyncIterator[Any]:
        return
        yield


def _cdp_json(cdp_url: str, path: str, method: str = "GET") -> Any:
    req = urllib.request.Request(cdp_url.rstrip("/") + path, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read()
    try:
        return json.loads(body)
    except ValueError:
        return body.decode("utf-8", "replace")


def page_tab_ids(cdp_url: str) -> list[str]:
    """IDs of the browser's page tabs, front first."""
    return [t["id"] for t in _cdp_json(cdp_url, "/json/list") if t.get("type") == "page"]


async def _navigate(ws_url: str, url: str) -> None:
    import websockets
    async with websockets.connect(ws_url, max_size=None, open_timeout=10) as ws:
        await ws.send(json.dumps({"id": 1, "method": "Page.navigate", "params": {"url": url}}))
        while json.loads(await asyncio.wait_for(ws.recv(), 30)).get("id") != 1:
            pass


async def restore_start_tabs(cdp_url: str, start_urls: list[str], start_tabs: list[str]) -> None:
    """Put the browser back as it was before the gate: the original tabs on the start URLs.

    The original tabs are navigated in place, never closed and reopened: the
    parent process holds Playwright handles to them and grades through those,
    and a tab replaced behind Playwright's back leaves it a dead handle ("page:
    no object with guid"). Only tabs the gate opened itself are closed.
    """
    targets = {t["id"]: t for t in _cdp_json(cdp_url, "/json/list") if t.get("type") == "page"}
    kept = [tab for tab in start_tabs if tab in targets]
    for tab_id in targets:
        if tab_id not in kept:
            _cdp_json(cdp_url, f"/json/close/{tab_id}")
    for tab_id, url in zip(kept, start_urls or ["about:blank"]):
        await _navigate(targets[tab_id]["webSocketDebuggerUrl"], url)
    if kept:
        _cdp_json(cdp_url, f"/json/activate/{kept[0]}")


async def run_gate(model, instruction: str, start_urls: list[str], args) -> dict:
    """Run the gate; return {"infeasible": bool, "reason": str, "steps_exhausted": bool}."""
    holder: dict = {}
    tools = [_VerdictTool("report_feasible", False, holder), _VerdictTool("report_infeasible", True, holder)]
    trajectory_path = os.path.join(args.results_dir, "gate_trajectory.jsonl")
    rail = TrajectoryStreamRail(trajectory_path)
    workspace = os.path.join(args.workspace, "gate")
    os.makedirs(workspace, exist_ok=True)
    gate = create_browser_agent(
        model,
        system_prompt=DEFAULT_BROWSER_AGENT_SYSTEM_PROMPT["en"] + GATE_RULES.format(max_steps=GATE_MAX_STEPS),
        tools=tools, rails=[rail], language="en",
        max_iterations=GATE_MAX_STEPS,
        workspace=workspace,
        browser_instance={"key": f"webarena-{args.task_id}-gate", "driver_mode": "remote", "cdp_url": args.cdp_url},
    )
    rail.agent = gate
    tabs = "\n".join(f"- {url}" for url in start_urls) or "- about:blank"
    query = (f"TASK: {instruction}\n\nOpen tabs at the start (the first is in front):\n{tabs}\n\n"
             "Probe the sites read-only, then call report_feasible or report_infeasible exactly once.")
    session_id = f"webarena-gate-{args.task_id}-{uuid.uuid4().hex[:6]}"
    print(f"Feasibility gate: up to {GATE_MAX_STEPS} steps...", flush=True)
    try:
        await gate.ensure_initialized()
        await Runner.run_agent(gate, {"query": query}, session=session_id)
    except Exception as exc:  # noqa: BLE001 - a broken gate must not lose the task: default to feasible
        print(f"Feasibility gate error, defaulting to feasible: {exc!r}", file=sys.stderr, flush=True)
    finally:
        dump_trajectory(gate, trajectory_path, session_id)
    exhausted = "infeasible" not in holder
    verdict = {"infeasible": bool(holder.get("infeasible", False)),
               "reason": holder.get("reason", "Budget exhausted - defaulting to feasible."),
               "steps_exhausted": exhausted, "max_steps": GATE_MAX_STEPS}
    with open(os.path.join(args.results_dir, GATE_FILE), "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=1)
    print(f"Feasibility gate verdict: {'INFEASIBLE' if verdict['infeasible'] else 'feasible'} - {verdict['reason']}",
          flush=True)
    return verdict
