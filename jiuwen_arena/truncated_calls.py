"""Show the model calls of a run that stopped at the output limit (finish_reason=length).

    python3 jiuwen_arena/truncated_calls.py results/jiuwen_arena/147_fmtrule5 [more run dirs ...]

browser_agent.log keeps every model response, its reasoning included, in its
llm_call_end events; this prints the reasoning and content of the ones that
were cut off, and writes them to truncated_calls.txt in the run directory.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys

FIELD = re.compile(r"(reasoning|content)=('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|None)")


def unquote(text: str) -> str:
    if text == "None":
        return ""
    # The response is the repr of the SDK's ChatCompletion: each field is a Python string literal.
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text[1:-1]


def truncated_calls(log_path: str):
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"llm_call_end"' not in line or "finish_reason='length'" not in line:
                continue
            event = json.loads(line[line.index("{"):])
            metadata = event.get("metadata", "")
            response = str(metadata.get("response", "") if isinstance(metadata, dict) else metadata)
            fields = {name: unquote(value) for name, value in FIELD.findall(response)}
            yield event.get("timestamp", ""), fields.get("reasoning", ""), fields.get("content", "")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for run_dir in sys.argv[1:]:
        log = os.path.join(run_dir, "browser_agent.log")
        if not os.path.exists(log):
            print(f"{run_dir}: no browser_agent.log")
            continue
        parts = []
        for i, (stamp, reasoning, content) in enumerate(truncated_calls(log), 1):
            parts.append(f"=== call {i} cut off at {stamp} ===\n--- reasoning ({len(reasoning)} chars) ---\n"
                         f"{reasoning}\n--- content ({len(content)} chars) ---\n{content}\n")
        out = os.path.join(run_dir, "truncated_calls.txt")
        if parts:
            with open(out, "w", encoding="utf-8") as f:
                f.write("\n".join(parts))
        print(f"{run_dir}: {len(parts)} truncated call(s)" + (f" -> {out}" if parts else ""))


if __name__ == "__main__":
    main()
