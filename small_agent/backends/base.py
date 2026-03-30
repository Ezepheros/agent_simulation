"""Abstract base for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from small_agent.core.types import LLMResponse, Message, ToolSchema


class BaseLLMBackend(ABC):
    """Abstract interface for an LLM inference backend.

    All backends accept a conversation history (list of Messages) plus an
    optional list of tool schemas, and return a single LLMResponse.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> LLMResponse:
        """Send a completion request and return the model's response.

        Must NOT raise on expected API errors — return an LLMResponse with
        content set to an error string instead.
        """
        ...
