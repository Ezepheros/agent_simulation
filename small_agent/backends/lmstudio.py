"""LMStudio backend — wraps the OpenAI-compatible local server."""

from __future__ import annotations

import json
import re
import time
import uuid

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from small_agent.backends.base import BaseLLMBackend
from small_agent.core.types import LLMResponse, Message, ToolCall, ToolSchema


class LMStudioBackend(BaseLLMBackend):
    """Talks to a local LMStudio server via its OpenAI-compatible /v1 endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(base_url=base_url, api_key="lmstudio", timeout=timeout_s)

    # ------------------------------------------------------------------
    # BaseLLMBackend interface
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> LLMResponse:
        t0 = time.monotonic()
        oai_messages = [self._to_oai_message(m) for m in messages]
        oai_tools = [t.to_openai_dict() for t in tools] if tools else None

        try:
            kwargs: dict = dict(
                model=self.model,
                messages=oai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            if oai_tools:
                kwargs["tools"] = oai_tools
                kwargs["tool_choice"] = "auto"

            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            return LLMResponse(
                content=f"[Backend error: {exc}]",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        choice = response.choices[0].message
        reasoning = getattr(choice, "reasoning_content", None) or getattr(choice, "thinking", None)
        tool_calls = self._extract_tool_calls(choice)

        # Fallback: some models (e.g. Ministral with reasoning) write <tool_call> XML tags
        # in the content field instead of using the structured tool_calls format
        if not tool_calls and choice.content:
            tool_calls = self._extract_xml_tool_calls(choice.content)

        usage = response.usage

        return LLMResponse(
            content=choice.content or "",
            reasoning=reasoning or None,
            tool_calls=tool_calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_oai_message(msg: Message) -> dict:
        base: dict = {"role": msg.role, "content": msg.content}
        if msg.tool_call_id:
            base["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            base["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        return base

    @staticmethod
    def _extract_tool_calls(choice: ChatCompletionMessage) -> list[ToolCall]:
        if not choice.tool_calls:
            return []
        results = []
        for tc in choice.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            results.append(
                ToolCall(
                    call_id=tc.id,
                    tool_name=tc.function.name,
                    arguments=args,
                )
            )
        return results

    @staticmethod
    def _extract_xml_tool_calls(content: str) -> list[ToolCall]:
        """Parse <tool_call><function=name><parameter=k>v</parameter></function></tool_call>
        tags that some models emit as plain text instead of structured tool_calls."""
        results = []
        for block in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
            fn_match = re.search(r"<function=(\w+)>(.*?)</function>", block, re.DOTALL)
            if not fn_match:
                continue
            name = fn_match.group(1)
            params_text = fn_match.group(2)
            arguments = {
                m.group(1): m.group(2).strip()
                for m in re.finditer(
                    r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", params_text, re.DOTALL
                )
            }
            results.append(
                ToolCall(
                    call_id=str(uuid.uuid4()),
                    tool_name=name,
                    arguments=arguments,
                )
            )
        return results
