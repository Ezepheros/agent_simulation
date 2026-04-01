"""Tavily search tool — AI-optimised web search via the Tavily API."""

from __future__ import annotations

import json
import os
import time
import uuid

from small_agent.core.types import ToolResult, ToolSchema
from small_agent.tools.base import BaseTool


class TavilySearchTool(BaseTool):
    """Search the web using Tavily's AI-optimised search API.

    Returns a list of relevant results with titles, URLs, and content snippets.
    Better suited for research tasks than raw web fetching — results are
    pre-filtered for relevance.
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self._max_results = max_results
        self._search_depth = search_depth  # "basic" or "advanced"

    @property
    def name(self) -> str:
        return "tavily_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for up-to-date information using Tavily. "
            "Returns ranked results with titles, URLs, and content summaries. "
            "Use this to find information on a topic before fetching full pages with web_fetch."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": f"Number of results to return (default {self._max_results}, max 10).",
                    },
                },
                "required": ["query"],
            },
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        t0 = time.monotonic()
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self._api_key)
            query = arguments["query"]
            max_results = min(int(arguments.get("max_results", self._max_results)), 10)

            response = client.search(
                query=query,
                max_results=max_results,
                search_depth=self._search_depth,
            )

            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score"),
                }
                for r in response.get("results", [])
            ]

            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output=json.dumps(results, ensure_ascii=False, indent=2),
                latency_ms=(time.monotonic() - t0) * 1000,
                metadata={"result_count": len(results), "query": query},
            )
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output="",
                error=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000,
            )
