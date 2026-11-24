# openjiuwen agent on WebArena

This folder runs agent-core's (openjiuwen) browser agent on WebArena. It follows the same structure as
OSWorld's `jiuwen/`: nothing outside `jiuwen_arena/` (and two test files in `tests/`) is changed, and
WebArena's tasks, login code and graders are used as they are.

## How it fits together

There are two processes, as in OSWorld, because WebArena pins Python 3.10 and Playwright 1.32 while agent-core
needs Python 3.11 or newer:

| | Parent: `run_task.py` | Agent: `openjiuwen_agent.py` |
|---|---|---|
| venv | `.venv` (WebArena's own) | `.venv-agent` (agent-core) |
| Owns | the browser, the task setup and the grading | the model and the browser tools |
| Talks to the other through | the browser's CDP URL, plus files in the results dir | the same |

For each task:

1. `tasks.py` fills the task's site URLs from the same environment variables as `scripts/generate_test_data.py`.
   It then logs in afresh with `browser_env/auto_login.py`, as `run.py` does.
2. `browser_host.py` starts Chromium as its own process with a debugging port. It sets up the task the way
   `ScriptBrowserEnv.setup` does: 1280x720, the task's cookies, and one tab per start URL with the first in front.
3. The agent process calls `create_browser_agent(..., browser_instance={"driver_mode": "remote", "cdp_url": ...})`.
   agent-core's Playwright MCP then attaches to that same browser and works in its default context, where the
   cookies and tabs already are.
   The agent ends by calling `submit_answer`, which writes `answer.json`. This plays the role of WebArena's
   `stop [answer]` action.
4. The parent grades the task with `evaluator_router(config_file)` on the tab the agent ended on. The answer is
   given as the final stop action of the trajectory, as in `run.py`.

Unlike OSWorld, no part of agent-core is patched: its browser agent already supports attaching to an existing
browser over CDP. The OSWorld-only extras are left out for now: the gate, the completion confirmation,
the benchmark-integrity guard, the ablations and the other rails.
`rails/trajectory_rail.py` is copied from OSWorld.

## One-time setup

All commands run from the repository root, in WSL or Linux.

1. **WebArena's environment** (the parent and the graders):

   ```
   uv venv -p 3.10 .venv
   uv pip install --python .venv/bin/python -r requirements.txt python-dotenv pytest
   uv pip install --python .venv/bin/python -e .
   .venv/bin/python -c "import nltk, sys; [nltk.download(r, download_dir=sys.prefix + '/nltk_data') for r in ('punkt', 'punkt_tab')]"
   ```

   The last line fetches NLTK's tokenizer data. WebArena's `must_include` grader splits the answer into words
   with it whenever the expected answer is a single item (for example `"2"`); without it the grader raises
   `LookupError: Resource 'punkt_tab' not found`, which `run_task.py` records as a score of 0 with the error in
   `grade_error`.

2. **The agent's environment**, plus Node.js on the PATH (agent-core starts Playwright MCP with `npx`):

   ```
   uv venv -p 3.12 .venv-agent
   uv pip install --python .venv-agent/bin/python -r jiuwen_arena/requirements-agent.txt
   npx -y playwright install chromium
   ```

   The last step downloads a recent Chromium, which `browser_host.py` picks by default (the newest under
   `~/.cache/ms-playwright`). A recent Chromium is needed because Playwright MCP is much newer than WebArena's
   Playwright. To use another browser, pass `--chrome-path` or set `CHROME_PATH`.

   `requirements-agent.txt` pins agent-core to commit `45dc1f0ec` of `kwangyy/agent-core` `develop`. To move
   the pin, change the commit and rerun `tests/test_jiuwen_arena_agent.py`.

3. **The WebArena sites.** Host them yourself (`environment_docker/README.md`) or use the AMI.
   Then put the following, with your hosts, in `.env` in the repository root (never commit it):

   ```
   SHOPPING=http://<host>:7770
   SHOPPING_ADMIN=http://<host>:7780/admin
   REDDIT=http://<host>:9999
   GITLAB=http://<host>:8023
   MAP=http://<host>:3000
   WIKIPEDIA=http://<host>:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
   HOMEPAGE=http://<host>:4399

   OPENROUTER_API_KEY=<your key>
   OPENROUTER_API_BASE=https://openrouter.ai/api/v1
   MODEL_NAME=anthropic/claude-opus-5
   # Optional: OpenRouter reasoning effort (none | minimal | low | medium | high | xhigh | max)
   # REASONING_EFFORT=high

   # WebArena's fuzzy_match / ua_match graders call OpenAI directly with gpt-4-1106-preview:
   OPENAI_API_KEY=<openai key>
   ```

## Running

The task ids are the `task_id`s in `config_files/test.raw.json` (0 to 811).

```
.venv/bin/python jiuwen_arena/run_task.py 27 --tag smoke
.venv/bin/python jiuwen_arena/run_parallel.py --tag smoke 0 27 44
.venv/bin/python jiuwen_arena/run_parallel.py --tag gitlab --site gitlab --max-parallel 2
```

- `run_task.py` options:
  - `--max-iterations` caps the agent's model calls (default 100; 0 means no cap).
  - `--time-budget` limits the agent's time in seconds (default 1800; 0 means no limit).
  - `--headed` shows the browser.
  - `--no-renew-login` uses the existing `.auth/` cookie files instead of logging in afresh.

  `run_parallel.py` passes any option it doesn't recognise on to `run_task.py`.
- Results for each run go to `results/jiuwen_arena/<task_id>_<tag>/`:

  | File | Contents |
  |---|---|
  | `result.json` | score, agent status, answer |
  | `config.json` | the task config that was graded |
  | `answer.json` | the submitted answer |
  | `final_url.txt` | the URL of the tab that was graded |
  | `agent.log`, `browser_agent.log` | the agent's logs |
  | `trajectory.jsonl`, `trajectory.pretty.json` | the agent's full message history |

  `run_parallel.py` puts each run's log and a `scoreboard.json` in `results/jiuwen_arena/_parallel_logs_<tag>/`,
  and exits with code 1 if any run produced no score.
- Many tasks change the sites (orders, posts, settings), and the sites are shared. Runs in parallel can therefore
  disturb each other, and a site that earlier runs have changed can make later tasks fail. Reset the sites
  between batches (`environment_docker/README.md`). `--max-parallel` defaults to 1.

## Tests

```
.venv/bin/python -m pytest tests/test_jiuwen_arena.py
.venv-agent/bin/python -m pytest -p no:cacheprovider --noconftest tests/test_jiuwen_arena_agent.py
```

- The first set uses a local HTTP server instead of the WebArena sites. It starts a real Chromium, attaches a
  second CDP client the way the agent does, and grades with WebArena's own `string_match`, `url_match` and
  `program_html` evaluators.
- The second set checks the agent-core interfaces the agent process relies on.

Neither set needs a model or the sites.

## Known gaps

- The graded tab is the one that has focus, or failing that the last one opened. WebArena's own environment
  tracks the current tab exactly. A run that ends with several tabs open could be graded on the wrong one.
  The graded URL is saved in `final_url.txt` so this can be checked.
- If the agent never calls `submit_answer`, its final message is used as the answer instead, and `answer.json`
  records `"by": "final_message"`.
