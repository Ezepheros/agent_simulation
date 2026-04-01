"""ClarifyTool — lets the agent pause and ask the user a question."""

from __future__ import annotations

import time
import uuid

from small_agent.core.types import ToolResult, ToolSchema
from small_agent.tools.base import BaseTool


class ClarifyTool(BaseTool):
    """Ask the user a clarifying question and wait for their response.

    Use this when the task is ambiguous and proceeding with an assumption
    would likely produce a wrong result. Do not use it for questions you
    can answer yourself or by using other tools.
    """

    @property
    def name(self) -> str:
        return "clarify"

    @property
    def description(self) -> str:
        return (
            "Ask the user a clarifying question when the task is ambiguous "
            "and you cannot safely proceed without more information. "
            "Use this sparingly — only when an incorrect assumption would lead to a wrong or "
            "harmful action (e.g. modifying the wrong calendar event, deleting something). "
            "Do not use it for questions you can resolve yourself."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The specific question to ask the user.",
                    },
                },
                "required": ["question"],
            },
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        question: str = arguments.get("question", "").strip()
        t0 = time.monotonic()

        print(f"\n[Agent needs clarification]\n{question}\n> ", end="", flush=True)
        try:
            answer = input().strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""

        return ToolResult(
            call_id=call_id,
            tool_name=self.name,
            output=answer if answer else "(no response provided)",
            latency_ms=(time.monotonic() - t0) * 1000,
        )
