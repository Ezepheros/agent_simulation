"""WebFetch tool — fetches a URL and returns cleaned plain text."""

from __future__ import annotations

import time
import uuid

import httpx
from bs4 import BeautifulSoup

from small_agent.core.types import ToolResult, ToolSchema
from small_agent.tools.base import BaseTool


class WebFetchTool(BaseTool):
    """Fetch a web page and return the visible text content."""

    def __init__(
        self,
        max_chars: int = 8_000,
        timeout_s: float = 15.0,
        user_agent: str = "SmallAgent/0.1 (research bot)",
    ) -> None:
        self.max_chars = max_chars
        self.timeout_s = timeout_s
        self.user_agent = user_agent

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch the content of a web page at a given URL. "
            "Returns the visible text of the page (HTML stripped)."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch (must start with http:// or https://).",
                    },
                },
                "required": ["url"],
            },
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        url: str = arguments.get("url", "")
        t0 = time.monotonic()

        try:
            response = httpx.get(
                url,
                timeout=self.timeout_s,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            )
            response.raise_for_status()
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output="",
                error=f"HTTP error: {exc}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        text = self._extract_text(response.text)
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n[… truncated]"

        return ToolResult(
            call_id=call_id,
            tool_name=self.name,
            output=text,
            latency_ms=(time.monotonic() - t0) * 1000,
            metadata={"final_url": str(response.url), "status_code": response.status_code},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script / style / nav noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        lines = (line.strip() for line in soup.get_text(separator="\n").splitlines())
        return "\n".join(line for line in lines if line)
