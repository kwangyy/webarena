"""Parent-side helpers: start the agent process and write the run record.

The agent runs in .venv-agent (Python 3.12, agent-core); this process runs in
WebArena's .venv (Python 3.10, Playwright 1.32, the graders). They share
nothing but the browser's CDP URL and files in the run's results directory.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_PYTHON = os.path.join(REPO_ROOT, ".venv-agent",
                            *(("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")))
CHILD = os.path.join(REPO_ROOT, "jiuwen_arena", "openjiuwen_agent.py")
EXIT_AGENT_ERROR = 3  # openjiuwen_agent.EXIT_AGENT_ERROR
RESULTS_ROOT = os.path.join(REPO_ROOT, "results", "jiuwen_arena")


def run_agent_child(args: argparse.Namespace, cdp_url: str, task: dict[str, Any], start_urls: list[str],
                    results_dir: str) -> str:
    """Run the agent process; return 'completed', 'timeout' or why it stopped."""
    instruction_file = os.path.join(results_dir, "instruction.txt")
    with open(instruction_file, "w", encoding="utf-8") as f:
        f.write(task["intent"])
    cmd = [AGENT_PYTHON, CHILD, "--task-id", str(task["task_id"]), "--cdp-url", cdp_url,
           "--instruction-file", instruction_file, "--results-dir", results_dir,
           "--max-iterations", str(args.max_iterations),
           "--browser-deadline", str(args.browser_deadline),
           "--max-output-tokens", str(args.max_output_tokens),
           # One workspace per run: openjiuwen locks its workspace across processes.
           "--workspace", os.path.join(results_dir, "workspace")]
    for url in start_urls:
        cmd += ["--start-url", url]
    if getattr(args, "feasibility_gate", False):
        cmd.append("--feasibility-gate")
    if getattr(args, "vision", False):
        cmd.append("--vision")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8",
           # openjiuwen opens <cwd>/logs/browser_agent.log with mode "w": one per run.
           "OPENJIUWEN_BROWSER_AGENT_LOG_FILE": os.path.join(results_dir, "browser_agent.log"),
           # Playwright MCP's own working directory (profiles, snapshots) per run.
           "PLAYWRIGHT_RUNTIME_MCP_CWD": os.path.join(results_dir, "mcp")}
    os.makedirs(env["PLAYWRIGHT_RUNTIME_MCP_CWD"], exist_ok=True)
    log_path = os.path.join(results_dir, "agent.log")
    print(f"Agent log: {log_path}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=os.name != "nt")
        try:
            code = proc.wait(timeout=args.time_budget or None)  # 0 = no limit
        except subprocess.TimeoutExpired:
            # The whole tree: the agent also runs npx / Playwright MCP.
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
            else:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return "timeout"
    if code == EXIT_AGENT_ERROR:
        return "stopped with an agent error (e.g. max iterations)"
    return "completed" if code == 0 else f"failed (exit {code})"


def save_run_record(results_dir: str, record: dict[str, Any]) -> None:
    with open(os.path.join(results_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
