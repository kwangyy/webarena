"""submit_answer: the agent's final answer, the ``stop [answer]`` action of WebArena.

WebArena grades the answer of the trajectory's last (stop) action with its
string_match graders, and ``N/A`` is how an unachievable task is answered. The
agent runs in a child process with no trajectory of WebArena's, so the tool
writes the answer into the results directory and run_task.py puts it into the
stop action it grades.
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Dict

from openjiuwen.core.foundation.tool.base import Tool, ToolCard
from openjiuwen.harness.tools.base_tool import ToolOutput

from jiuwen_arena.protocol import ANSWER_FILE

DESCRIPTION = (
    "Submit your final answer to the task and end it. Call it exactly once, when you are done.\n"
    "- If the task asks for information (a name, number, date, list, yes/no ...), the answer is only that "
    "information, exactly as the website shows it, with no explanation around it: e.g. 'Quest Lumaflex™ Band', "
    "'$25.99', '2022-03-04', 'Yes'. Several items: separated by commas.\n"
    "- If the task asks you to do something on the website (post, edit, order, configure, navigate to a page), "
    "do it completely first, leave the browser on the page that shows the result, then submit a short answer "
    "such as 'Done'.\n"
    "- If the task cannot be done or the information does not exist on these websites, submit 'N/A'. Check "
    "properly first: hard to find is not missing.\n"
    "- A count of zero is a valid answer: when the right search or page shows no matching items, submit '0'."
)


class SubmitAnswerTool(Tool):
    def __init__(self, results_dir: str) -> None:
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
        self.answer: str | None = None

    async def invoke(self, inputs: Dict[str, Any], **kwargs: Any) -> ToolOutput:
        answer = inputs.get("answer")
        if not isinstance(answer, str):
            return ToolOutput(success=False, error="answer must be a string")
        self.answer = answer.strip()
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"answer": self.answer, "by": "submit_answer"}, f, ensure_ascii=False)
        return ToolOutput(success=True, data={"status": "recorded", "answer": self.answer,
                                              "next": "The task is over: stop now, no further browser actions."})

    async def stream(self, inputs: Dict[str, Any], **kwargs: Any) -> AsyncIterator[Any]:
        return
        yield


def write_fallback_answer(results_dir: str, text: str) -> None:
    """The agent's final message as the answer, for a run that never called submit_answer."""
    with open(os.path.join(results_dir, ANSWER_FILE), "w", encoding="utf-8") as f:
        json.dump({"answer": text.strip(), "by": "final_message"}, f, ensure_ascii=False)
