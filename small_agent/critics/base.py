"""Base class for step critics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from small_agent.core.types import AgentStep, ToolCall


@dataclass
class CritiqueResult:
    """Outcome of a critic review on a proposed agent step."""

    has_issues: bool
    feedback: str  # Empty when has_issues is False


@dataclass
class ProposedAction:
    """The agent's current proposed step, before tool execution."""

    thought: str
    tool_calls: list[ToolCall]


class BaseCritic(ABC):
    """Reviews a proposed agent step and returns a critique.

    The critic receives the original task, all completed previous steps
    (without any prior critiques — to avoid critique-of-critique loops),
    and the current proposed action. It returns whether issues were found
    and what they are.
    """

    @abstractmethod
    def review(
        self,
        task: str,
        previous_steps: list[AgentStep],
        proposed: ProposedAction,
        agent_context: str | None = None,
    ) -> CritiqueResult:
        """Review the proposed action and return a CritiqueResult.

        agent_context: the agent's system prompt (already formatted), passed so
        the critic shares the same date/time context as the agent.
        """
        ...
