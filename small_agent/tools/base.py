"""Base class for all agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod

from small_agent.core.types import ToolResult, ToolSchema


class BaseTool(ABC):
    """Abstract base for every tool the agent can invoke.

    Subclasses must declare:
    - ``name``        — unique snake_case identifier (used by the LLM)
    - ``description`` — one-sentence description forwarded to the LLM
    - ``schema``      — full JSON-Schema spec of accepted arguments
    - ``execute``     — implementation that runs the tool and returns a ToolResult
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def schema(self) -> ToolSchema: ...

    @abstractmethod
    def execute(self, arguments: dict) -> ToolResult:
        """Run the tool with the given arguments dict.

        Must NOT raise — catch all errors and return a ToolResult with error set.
        """
        ...
