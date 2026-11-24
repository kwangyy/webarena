#!/usr/bin/env python3
"""Run several WebArena tasks, each in its own run_task.py process, and print a scoreboard.

    .venv/bin/python jiuwen_arena/run_parallel.py --tag smoke 0 27 44
    .venv/bin/python jiuwen_arena/run_parallel.py --tag shop --site shopping --max-parallel 2

The WebArena sites are shared and many tasks change them (orders, posts,
settings), so parallel runs can disturb one another, and the official numbers
come from sites reset between task groups. --max-parallel defaults to 1.
Options this script does not know are passed to run_task.py as they are.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from jiuwen_arena.run_common import RESULTS_ROOT  # noqa: E402
from jiuwen_arena.tasks import RAW_CONFIG  # noqa: E402

RUN_TASK = os.path.join(REPO_ROOT, "jiuwen_arena", "run_task.py")


def select_tasks(ids: list[str], site: str, task_file: str) -> list[int]:
    chosen = [int(i) for i in ids]
    if task_file:
        with open(task_file, encoding="utf-8") as f:
            chosen += [int(line.split("#")[0]) for line in f if line.split("#")[0].strip()]
    if site:
        with open(RAW_CONFIG, encoding="utf-8") as f:
            tasks = json.load(f)
        chosen += [t["task_id"] for t in tasks if site in t["sites"]]
    return list(dict.fromkeys(chosen))


def run_one(task_id: int, tag: str, passthrough: list[str], log_dir: str) -> dict:
    log_path = os.path.join(log_dir, f"{task_id}.log")
    started = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log:
        code = subprocess.call([sys.executable, RUN_TASK, str(task_id), "--tag", tag, *passthrough],
                               cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    result_path = os.path.join(RESULTS_ROOT, f"{task_id}_{tag}", "result.json")
    record = {"task_id": task_id, "exit": code, "seconds": round(time.monotonic() - started)}
    if os.path.isfile(result_path):
        with open(result_path, encoding="utf-8") as f:
            record.update(json.load(f))
    print(f"[{task_id}] score={record.get('score', 'none')} agent={record.get('agent_status', '-')} "
          f"({record['seconds']}s)", flush=True)
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task_ids", nargs="*", help="task ids (0-811)")
    p.add_argument("--tag", required=True)
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument("--site", default="", help="add every task on this site (shopping, gitlab, reddit, ...)")
    p.add_argument("--task-file", default="", help="file with one task id per line (# comments)")
    args, passthrough = p.parse_known_args()

    task_ids = select_tasks(args.task_ids, args.site, args.task_file)
    if not task_ids:
        p.error("no tasks given")
    log_dir = os.path.join(RESULTS_ROOT, f"_parallel_logs_{args.tag}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"{len(task_ids)} task(s), max-parallel {args.max_parallel}; logs in {log_dir}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as pool:
        records = list(pool.map(lambda t: run_one(t, args.tag, passthrough, log_dir), task_ids))

    scored = [r for r in records if isinstance(r.get("score"), (int, float))]
    total = sum(r["score"] for r in scored)
    board = {"tag": args.tag, "tasks": len(records), "scored": len(scored), "score_sum": total,
             "success_rate": total / len(records), "runs": records}
    with open(os.path.join(log_dir, "scoreboard.json"), "w", encoding="utf-8") as f:
        json.dump(board, f, indent=2, ensure_ascii=False)
    print(f"\n{total:g}/{len(records)} passed ({100 * total / len(records):.1f}%); "
          f"scoreboard: {os.path.join(log_dir, 'scoreboard.json')}", flush=True)
    unscored = [r["task_id"] for r in records if r not in scored]
    if unscored:
        print(f"No score for: {' '.join(map(str, unscored))}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
