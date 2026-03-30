"""Abstract base for agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

from small_agent.core.types import AgentRun


class BaseAgent(ABC):
    """Abstract interface for an agent.

    An agent takes a task string and returns a complete AgentRun record
    that captures every step, tool interaction, and the final answer.
    """

    @abstractmethod
    def run(self, task: str, run_id: str | None = None) -> AgentRun:
        """Execute the agent on the given task and return the full run record."""
        ...
