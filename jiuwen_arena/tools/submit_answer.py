"""submit_answer: the agent's final answer, the ``stop [answer]`` action of WebArena.

WebArena grades the answer of the trajectory's last (stop) action with its
string_match graders, and ``N/A`` is how an unachievable task is answered. The
agent runs in a child process with no trajectory of WebArena's, so the tool
writes the answer into the results directory and run_task.py puts it into the
stop action it grades.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.parse
import urllib.request
from typing import Any, AsyncIterator, Dict

from openjiuwen.core.foundation.tool.base import Tool, ToolCard
from openjiuwen.harness.tools.base_tool import ToolOutput

from jiuwen_arena.protocol import ANSWER_FILE

DESCRIPTION = (
    "Submit your final answer to the task and end it. Call it exactly once, when you are done.\n"
    "- If the task asks for information (a name, number, date, list, yes/no ...), the answer is only that "
    "information, with no explanation around it: e.g. 'Quest Lumaflex™ Band', '25.99', '16x24', '2022-03-04', "
    "'Yes'. Names are copied exactly as the website shows them; numbers, amounts and sizes are bare values with no "
    "units or symbols ($, \", inches, dollars) and are not rounded. Several items: separated by commas.\n"
    "- If the task asks you to do something on the website (post, edit, order, configure, navigate to a page), "
    "do it completely first, leave the browser on the page that shows the result, then submit a short answer "
    "such as 'Done'.\n"
    "- If the task cannot be done or the information does not exist on these websites, submit 'N/A'. Check "
    "properly first: hard to find is not missing.\n"
    "- A count of zero is a valid answer: when the task asks for a number and the right search shows no matching "
    "items, submit '0'. When it asks you to find or list items and there are none, submit 'N/A', not 'None'.\n"
    "- For a task to show, list, look up or browse something, call this only while the browser is on the page that "
    "shows the result; do not navigate after reaching it."
)


CAT_FILTER_REFUSAL = (
    "Not recorded: the browser is on {url}, a parent category page filtered with the sidebar's ?cat= "
    "filter, not the category's own page. Open the category's own page instead (its link in the top menu, "
    "shown by hovering over each menu level, or the breadcrumb above one of its products), such as "
    "/clothing-shoes-jewelry/men/shoes.html, apply any sort or filter there, then call submit_answer again. "
    "This check is made once: the next submit_answer is recorded as it is."
)


def front_tab_url(cdp_url: str) -> str | None:
    """URL of the most recently used page tab, from the browser's DevTools /json/list.

    Chromium lists targets most recently activated first; the grader reads the
    tab with focus, which is that one.
    """
    try:
        with urllib.request.urlopen(cdp_url.rstrip("/") + "/json/list", timeout=5) as resp:
            targets = json.load(resp)
    except Exception:  # noqa: BLE001 - no URL means no check, never a lost answer
        return None
    for target in targets:
        if target.get("type") == "page":
            return target.get("url")
    return None


def is_cat_filtered_store_page(url: str | None, shopping_base: str | None) -> bool:
    """A storefront page whose query has the sidebar category filter, cat=<id>."""
    if not url or not shopping_base:
        return False
    page, store = urllib.parse.urlparse(url), urllib.parse.urlparse(shopping_base)
    if page.netloc != store.netloc:
        return False
    return "cat" in urllib.parse.parse_qs(page.query)


class SubmitAnswerTool(Tool):
    def __init__(self, results_dir: str, cdp_url: str | None = None,
                 shopping_base: str | None = None) -> None:
        super().__init__(ToolCard(
            id="webarena_submit_answer",
            name="submit_answer",
            description=DESCRIPTION,
            input_params={
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "The final answer, or 'N/A'."}},
                "required": ["answer"],
            },
        ))
        self._path = os.path.join(results_dir, ANSWER_FILE)
        self._cdp_url = cdp_url
        self._shopping_base = shopping_base
        self._cat_check_done = False
        self.answer: str | None = None
        self.submitted = asyncio.Event()  # set by a successful submit_answer: the run is stopped on it

    async def invoke(self, inputs: Dict[str, Any], **kwargs: Any) -> ToolOutput:
        answer = inputs.get("answer")
        if not isinstance(answer, str):
            return ToolOutput(success=False, error="answer must be a string")
        answer = answer.strip()
        # A "Done" on a ?cat= filtered page is the store's usual wrong final page
        # for a browse/list task: refuse it once so the agent can open the real one.
        if answer.lower() == "done" and not self._cat_check_done and self._cdp_url:
            url = front_tab_url(self._cdp_url)
            if is_cat_filtered_store_page(url, self._shopping_base):
                self._cat_check_done = True
                return ToolOutput(success=False, error=CAT_FILTER_REFUSAL.format(url=url))
        self.record(answer)
        self.submitted.set()
        return ToolOutput(success=True, data={"status": "recorded", "answer": self.answer,
                                              "next": "The task is over: stop now, no further browser actions."})

    def record(self, answer: str, by: str = "submit_answer") -> None:
        """Write the answer for run_task.py to grade."""
        self.answer = answer
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"answer": self.answer, "by": by}, f, ensure_ascii=False)

    async def stream(self, inputs: Dict[str, Any], **kwargs: Any) -> AsyncIterator[Any]:
        return
        yield


def write_fallback_answer(results_dir: str, text: str) -> None:
    """The agent's final message as the answer, for a run that never called submit_answer."""
    with open(os.path.join(results_dir, ANSWER_FILE), "w", encoding="utf-8") as f:
        json.dump({"answer": text.strip(), "by": "final_message"}, f, ensure_ascii=False)
