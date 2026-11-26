#!/usr/bin/env python3
"""Run one WebArena task with the openjiuwen browser agent, then grade it.

Run with WebArena's own .venv from the repository root::

    .venv/bin/python jiuwen_arena/run_task.py 27 --tag smoke

1. Fill the task's site URLs and renew its login (jiuwen_arena/tasks.py).
2. Start Chromium with a debugging port, set the cookies and start tabs
   (jiuwen_arena/browser_host.py).
3. Run jiuwen_arena/openjiuwen_agent.py in .venv-agent, attached to that
   browser over CDP; it records its answer with submit_answer.
4. Grade with WebArena's own evaluator_router on the tab the agent ended on,
   with the answer as the trajectory's final stop action, as run.py does.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
load_dotenv(os.path.join(REPO_ROOT, ".env"))

from jiuwen_arena.browser_host import TaskBrowser  # noqa: E402
from jiuwen_arena.protocol import ANSWER_FILE, FINAL_URL_FILE  # noqa: E402
from jiuwen_arena.run_common import AGENT_PYTHON, RESULTS_ROOT, run_agent_child, save_run_record  # noqa: E402
from jiuwen_arena.tasks import load_task, renew_login, start_urls, write_task_config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task_id", type=int, help="task_id in config_files/test.raw.json (0-811)")
    p.add_argument("--tag", default="run", help="results go to results/jiuwen_arena/<task_id>_<tag>/")
    p.add_argument("--max-iterations", type=int, default=100, help="agent model calls; 0 = no cap")
    p.add_argument("--time-budget", type=int, default=1800, help="seconds for the agent; 0 = no limit")
    p.add_argument("--browser-deadline", type=int, default=900,
                   help="agent-core's own per-task deadline in seconds, the same for every task "
                        "(agent-core gives 240 to tasks it judges simple, 600 to complex ones)")
    p.add_argument("--max-output-tokens", type=int, default=4096,
                   help="output tokens per model call, thinking included (agent-core's browser agent "
                        "uses 8192 when none is given)")
    p.add_argument("--vision", action="store_true",
                   help="give the agent's model image input: the screenshot tools are enabled")
    p.add_argument("--feasibility-gate", action="store_true",
                   help="run the pre-task feasibility gate before the agent (jiuwen_arena/feasibility_gate.py)")
    p.add_argument("--chrome-path", default="", help="Chromium binary (default: CHROME_PATH or newest Playwright one)")
    p.add_argument("--headed", action="store_true", help="show the browser window")
    p.add_argument("--no-renew-login", action="store_true",
                   help="use the task's .auth/ cookie file as is instead of logging in afresh")
    return p.parse_args(argv)


def answer_of(results_dir: str) -> tuple[str, bool]:
    """The submitted answer, and whether the agent submitted one at all."""
    path = os.path.join(results_dir, ANSWER_FILE)
    if not os.path.isfile(path):
        return "", False
    with open(path, encoding="utf-8") as f:
        return str(json.load(f).get("answer", "")), True


def grade(browser: TaskBrowser, config_file: str, answer: str, results_dir: str) -> float:
    # Imported here: evaluation_harness pulls in nltk/transformers, slow and only needed now.
    from browser_env.actions import create_stop_action
    from evaluation_harness import evaluator_router

    page = browser.grading_page()
    with open(os.path.join(results_dir, FINAL_URL_FILE), "w", encoding="utf-8") as f:
        f.write(page.url)
    # The graders read the last action (the answer) and nothing earlier; a
    # placeholder state stands in for the observations run.py would keep.
    trajectory = [{"observation": {}, "info": {}}, create_stop_action(answer)]
    evaluator = evaluator_router(config_file)
    return float(evaluator(trajectory=trajectory, config_file=config_file, page=page,
                           client=browser.cdp_session(page)))


def run(args: argparse.Namespace) -> int:
    if not os.path.isfile(AGENT_PYTHON):
        raise SystemExit(f"{AGENT_PYTHON} not found; create .venv-agent (jiuwen_arena/README.md).")
    results_dir = os.path.join(RESULTS_ROOT, f"{args.task_id}_{args.tag}")
    os.makedirs(results_dir, exist_ok=True)
    for stale in (ANSWER_FILE, FINAL_URL_FILE, "result.json", "feasibility_gate.json"):
        if os.path.exists(os.path.join(results_dir, stale)):
            os.remove(os.path.join(results_dir, stale))

    task = load_task(args.task_id)
    print(f"Task {args.task_id} ({', '.join(task['sites'])}): {task['intent']}", flush=True)
    if not args.no_renew_login:
        task = renew_login(task, os.path.join(results_dir, "auth"))
    config_file = write_task_config(task, results_dir)
    urls = start_urls(task)

    record: dict = {"task_id": args.task_id, "sites": task["sites"], "intent": task["intent"],
                    "eval_types": task["eval"]["eval_types"], "benchmark": "webarena",
                    "browser_deadline_s": args.browser_deadline, "max_output_tokens": args.max_output_tokens,
                    "vision": args.vision}
    browser = TaskBrowser(args.chrome_path, headless=not args.headed)
    try:
        browser.start()
        print(f"Chromium up: {browser.cdp_url} ({browser.chrome_path})", flush=True)
        browser.setup_task(task, urls)

        started = time.monotonic()
        status = run_agent_child(args, browser.cdp_url, task, urls, results_dir)
        agent_seconds = round(time.monotonic() - started)
        print(f"Agent {status} after {agent_seconds}s", flush=True)
        answer, submitted = answer_of(results_dir)
        print(f"Answer: {answer!r}" if submitted else "Agent submitted no answer; grading with an empty one.",
              flush=True)

        print("Evaluating...", flush=True)
        try:
            score = grade(browser, config_file, answer, results_dir)
            grade_error = None
        except Exception as exc:  # noqa: BLE001 - a grader crash is a recorded 0, not a lost run
            score, grade_error = 0.0, f"{exc!r}\n{traceback.format_exc()}"
            print(f"Grader error: {exc!r}", flush=True)
        print(f"RESULT score={score} agent={status}", flush=True)
        gate_path = os.path.join(results_dir, "feasibility_gate.json")
        if os.path.exists(gate_path):
            with open(gate_path, encoding="utf-8") as f:
                record["feasibility_gate"] = json.load(f)
        record.update({"score": score, "agent_status": status, "agent_seconds": agent_seconds,
                       "answer": answer, "answer_submitted": submitted, "grade_error": grade_error})
        save_run_record(results_dir, record)
        return 0
    finally:
        browser.close()


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
