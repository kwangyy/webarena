"""view_page: a screenshot of the front tab, given to the model as an image.

openjiuwen's browser runtime has no route for screenshot pixels to reach the
model: browser_take_screenshot is not offered, and browser_batch_interact
op=screenshot saves a file and returns only metrics. A tool result whose data
carries ``multimodal`` image items is a different matter: the ReAct agent adds
those images to the conversation as a user message after the tool messages
(ReActAgent._add_tool_result_images), the same way read_file attaches an image.
This tool captures the tab over CDP and returns it that way.

Only offered with --vision, for a model that takes image input.
"""
from __future__ import annotations

import base64
import io
import json
import urllib.request
from typing import Any, AsyncIterator, Dict

from openjiuwen.core.foundation.tool.base import Tool, ToolCard
from openjiuwen.harness.tools.base_tool import ToolOutput

JPEG_QUALITY = 70  # A 1280x720 viewport is then roughly 60-120 KB.

DESCRIPTION = (
    "Take a screenshot of the browser's current page (the visible viewport) and see it as an image. "
    "Use it when the page's text or element probes fail, come back empty or look inconsistent, or when the "
    "answer depends on how the page looks. It does not change the page."
)


def _front_page_target(cdp_url: str) -> Dict[str, Any] | None:
    with urllib.request.urlopen(cdp_url.rstrip("/") + "/json/list", timeout=5) as resp:
        targets = json.load(resp)
    return next((t for t in targets if t.get("type") == "page"), None)


def _png_to_jpeg(png: bytes) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(png)) as img:
        out = io.BytesIO()
        img.convert("RGB").save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        return out.getvalue()


async def capture_jpeg(cdp_url: str, full_page: bool = False) -> tuple[bytes, str]:
    """JPEG bytes and URL of the front tab.

    Captured as PNG and re-encoded here: Chromium's own format="jpeg" output
    from this headless setup comes back garbled (two overlapping, scrambled
    copies of the page), while its PNG is clean.
    """
    import websockets

    target = _front_page_target(cdp_url)
    if target is None:
        raise RuntimeError("no page tab is open")
    params: Dict[str, Any] = {"format": "png"}
    if full_page:
        params["captureBeyondViewport"] = True
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=None, open_timeout=10) as ws:
        await ws.send(json.dumps({"id": 1, "method": "Page.captureScreenshot", "params": params}))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("id") == 1:
                break
    if "error" in msg:
        raise RuntimeError(msg["error"].get("message", str(msg["error"])))
    return _png_to_jpeg(base64.b64decode(msg["result"]["data"])), target.get("url", "")


class ViewPageTool(Tool):
    def __init__(self, cdp_url: str, save_dir: str | None = None) -> None:
        super().__init__(ToolCard(
            id="webarena_view_page",
            name="view_page",
            description=DESCRIPTION,
            input_params={
                "type": "object",
                "properties": {"full_page": {"type": "boolean",
                                             "description": "Capture the whole scrollable page, not only the viewport."}},
                "required": [],
            },
        ))
        self._cdp_url = cdp_url
        self._save_dir = save_dir  # each screenshot is also kept as <save_dir>/view_page_<n>.jpg
        self._count = 0

    async def invoke(self, inputs: Dict[str, Any], **kwargs: Any) -> ToolOutput:
        try:
            jpeg, url = await capture_jpeg(self._cdp_url, bool(inputs.get("full_page", False)))
        except Exception as exc:  # noqa: BLE001 - a failed capture is a tool error, not a lost run
            return ToolOutput(success=False, error=f"screenshot failed: {exc}")
        self._count += 1
        if self._save_dir:
            with open(f"{self._save_dir}/view_page_{self._count}.jpg", "wb") as f:
                f.write(jpeg)
        data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        return ToolOutput(success=True, data={
            "content": f"Screenshot of {url} ({len(jpeg)} bytes, JPEG). The image is attached after this result.",
            "multimodal": [{"type": "image", "source": "view_page", "source_path": url,
                            "mime_type": "image/jpeg", "data_url": data_url}],
        })

    async def stream(self, inputs: Dict[str, Any], **kwargs: Any) -> AsyncIterator[Any]:
        return
        yield
