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
- Read "author", "reviewer" and similar words by what they are attached to. The author of a book, article, song or \
other work is the person the post or page names as its author or maker (a post titled "Dune by Frank Herbert" \
gives Frank Herbert), not the forum user who posted about it; copy that name from the post's title or text, and use your \
own knowledge only when the post names no one. For an audiobook or a reading, the narrator or reader the post names \
counts as an author too, even when the task says "the author name": answer with both names, the narrator and \
the writer, in one line (for "a free audiobook of Emma narrated by Jane Doe": Emma by Jane Doe (narrator) and \
Jane Austen). This is settled: do not weigh it again. Write such answers as "BOOK — AUTHOR" items joined by "; ", \
with the post's own names first and the writer you know added after them (The Hobbit — J. R. R. Tolkien; Emma — \
Jane Doe (narrator), Jane Austen), and submit as soon as you have found the posts. When you write a name with initials, put a space after \
each initial's period, also when the site writes it without spaces (C. S. Lewis, not C.S. Lewis). The author of a post, comment or \
review is the account that wrote it on this site.
- In the online store's product reviews, each review is laid out as its title, its rating, its text and then a \
"Review by NAME" / "Posted on DATE" line: the name comes after the text it belongs to, not before. A review's \
reviewer is the name in the "Review by" line below its text, written as the store shows it.
- Judge what a review, post or comment says by its meaning, not by matching a word: a review that calls the print \
quality "okay", "so-so" or "nothing special" mentions average print quality as much as one that says "average". \
Read every review or post in scope before answering.
- In the store's admin panel, product reviews are under Marketing > All Reviews (/admin/review/product/). \
Type the product's name in the grid's Product filter (or its SKU in the SKU filter), leave Status on its blank \
"any" choice, and press Search: the "records found" count is every review of that product, whatever its status. \
The grid cuts each review's text short, so open each review with Edit and read its full text from the Review \
box, the #detail textarea (read_text shows a textarea as "undefined"; read it with browser_evaluate, e.g. \
document.querySelector('#detail').value). These are the same reviews the storefront shows, so there is no \
need to check the storefront as well. When you have read them all, answer from what they say.
- Order status (Canceled, refunded, Closed) matters only in some questions. Decide which kind the task is:
  1. Which orders: "my latest order", "the most recent N orders", "the last order", or an order number. Pick the \
newest orders by date, whatever their status: a Canceled order is still one of them. Then report or add up \
what those orders contain (items, quantities, totals) as shown, the canceled one included; a canceled order's \
status is reported as it is (it has no delivery date and will never arrive).
  2. Money the customer spent or paid (how much I spent, total spend in a period). Count only orders that went \
through: leave out every Canceled order, and any refunded one, since it was never paid for or was paid back.
  3. Counts of successful, fulfilled or completed orders. Leave out Canceled orders and refunded ones, even \
when a refunded order's status shows Complete or Closed. When you count from a report (such as Reports > Sales \
> Orders), read its Refunded column for every row you use: a non-zero Refunded amount in a row means that \
period holds a refunded order. Find that order in the order grid (a Closed order whose total matches the \
refunded amount) and subtract it from that row's order count.
  4. The task names a status ("completed orders", "pending orders", "cancelled orders", "non-cancelled orders"). \
Use exactly the orders with that status.
- Stay on the site the task starts on. Each start tab is the site the task is about; do not switch to another site \
(for example from the online store to its admin panel, or back) unless the task names that site or starts a tab \
on it. On the store, "my" orders, purchases, reviews and account mean the logged-in customer's own pages \
(My Account, My Orders at /sales/order/history/), not the admin panel's grids, which list every customer.
- If browser_probe_cards fails twice on a page, or the listing stays marked stale (listing_stale) after you have \
tried once to refresh it, stop using browser_probe_cards on that page: reload the page by navigating to its URL, \
then read it with browser_page_action read_text, and work from that text. A page you just loaded and read is \
current; do not wait for the card listing to recover before using what you read.
- In the store's admin panel, questions about how many orders customers have placed or completed (most, fewest, \
the Nth most, exactly N orders, over the whole history or a period) are answered from the Order Count report: \
Reports > Customers > Order Count (/admin/reports/report_customer/orders/). Set From to a date well before the \
store's first orders and To to today, set Show By to Year (Day or Month makes the report too long to read), press \
Refresh, then read the whole table with browser_page_action read_text. The report lists each customer once per \
year: a year label is shown only on that year's first row, and the rows under it belong to the same year. For the \
whole history, add up each customer's orders across all the years before ranking them.
- Search a list iteratively, one page at a time. When what you are looking for may be in a list of many entries \
(order history, reviews, posts, search results, an admin grid), start at the first page and read every entry on it, \
opening an entry's own page (such as View Order) when the list does not show what you need, before moving on. Then \
go to the next page with the list's own Next or page-number links, and repeat until you find it or reach the last \
page. Do not conclude that something is missing until you have gone through every page.
- When the task is to change something, do it completely and leave the browser on the page that shows the result.
- To order, buy, purchase or reorder something in the online store, putting it in the cart is not enough: the task \
is done only when the order is placed. Add the item to the cart, open the checkout (Proceed to Checkout), keep the \
saved shipping address and choose a shipping method, keep the default payment method, press the Place Order button, \
and submit 'Done' only on the confirmation page that says the order was placed and gives its order number.
- When the task is to show, list, look up or browse something on the site, the page the browser ends on is the \
answer: open the page that shows exactly what was asked (the item, the category, the sorted or filtered listing) \
using the site's own links, menus, sort and filter controls, and leave the browser on that page. After you have \
reached it, do not navigate anywhere else, not even to check something; submit 'Done'.
- To browse or list a category of the online store, open that exact category's own page, such as \
/clothing-shoes-jewelry/men/shoes.html -- not its parent, and not a page with ?cat=<number> added (the \
sidebar's category filter). Be specific, for example, "Woman clothing" is the Clothing category under Women (/women/clothing.html), not the \
Women page; "PS4 accessories" is Accessories under PlayStation 4 (/playstation-4/accessories.html), not the \
PlayStation 4 page. The top menu has several levels: hover over each level in turn until the named category's own \
link appears and open it, or follow a breadcrumb link above a product of that category. Apply any sort or price \
filter on that page, and before submitting check that the last part of the URL path names the category the task \
asked for.
- When the answer is a number, an amount or a size, give the bare value without units or unit symbols: no $ or \
other currency signs, no inch marks (") and no words such as inches, dollars or cm (347.25, not $347.25; 16x24, not \
16"x24" or 16 x 24 inches). Write a size as the numbers joined by a lowercase x without spaces. Product and other \
names are not values: copy them exactly as the site writes them, including any units in them (Sprite Stasis Ball \
65 cm). When you calculate a number yourself (a total, a discount, a price per item), give the full result of the \
calculation and do not round it, not even to the nearest cent: 12.99 / 2 is 6.495, not 6.50.
- On the forum, "the newest post" in a forum means the first post the forum's own page lists \
(/f/<forum>, as it opens, without changing the sort). Do not switch the list to New: sorted that way the list \
starts from a different post. In /f/explainlikeimfive this first post is "ELI5: How does pinching a ribbon and \
sliding your finger…". Upvote or open that one.
- In the store's admin, a configurable product (one product with size and colour variants, such as a hoodie) has a \
parent product and one child product per variant. To mark all of it out of stock, start with the parent: in \
Catalog > Products search the name, open the row whose Type is Configurable Product (its name has no size or \
colour), set Stock Status to Out of Stock, and press Save. Then open each child variant (the rows whose names add a \
size and colour, or the parent's Configurations section) and set each to Out of Stock as well, saving each one. \
Do not use the grid's mass "Update attributes" action for this: it only queues the change.
- To change an order's address in the admin, open the order (Sales > Orders > View), go to its Address Information \
section and click the Edit link next to Shipping Address (and next to Billing Address too, when the task does not \
say which). That opens the address form (/admin/sales/order/address/address_id/N/): change the street lines, \
city, State/Province (pick the full state name from the list, e.g. Florida for FL) and zip, and press Save Order \
Address. Do not use the Edit button at the top of the order page: it cancels the order and creates a new one.
- When a task asks you to draft something (a price rule, a message, a form), fill in every field the task \
gives and then stop with the filled form on screen: do not press Save or Submit. The filled, unsaved form must be \
the final page. A marketing "price rule" for a sale or a discount for customers is a Cart Price Rule: \
Marketing > Cart Price Rules > Add New Rule (/admin/sales_rule/promo_quote/new/), not a Catalog Price Rule. In \
that form, always select Main Website in the Websites list: it starts with nothing selected even though it has only \
one choice, so select it yourself (e.g. a select_option on the website_ids list) and confirm it shows as selected. \
And for "all customers" \
select the customer groups General, Wholesale and Retailer (customers with accounts; leave NOT LOGGED IN out unless \
the task mentions guests or visitors who are not logged in).
- When a task asks you to show or find something for the user, make the final page one that shows the task's own \
key words as well as the item: for example, search the store for the problem in the task's words ("jaw bruxism") \
and end on that search's results page if it lists a fitting item (a page showing "jaw bruxism" and "mouth guard"), \
rather than on a product page that does not repeat those words.
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
    agent = create_browser_agent(
        build_model(args.max_output_tokens, args.vision),
        system_prompt=DEFAULT_BROWSER_AGENT_SYSTEM_PROMPT["en"] + WEBARENA_RULES,
        tools=[submit] + ([ViewPageTool(args.cdp_url, args.results_dir)] if args.vision else []),
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
