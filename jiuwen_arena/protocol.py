"""What the agent process leaves for the parent. No imports, so both venvs can use it."""

ANSWER_FILE = "answer.json"  # {"answer": str}, written by the submit_answer tool
FINAL_URL_FILE = "final_url.txt"  # not written by the agent; the parent records the graded tab's URL here
