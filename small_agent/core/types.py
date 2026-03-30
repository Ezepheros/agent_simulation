"""Core domain objects for the agent framework.

All types are plain dataclasses — no framework dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# LLM conversation primitives
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """A single turn in the LLM conversation."""

    role: Role
    content: str | None  # None is valid for pure tool-call assistant turns
    # tool_calls populated on assistant messages that contain function calls
    # (must be re-sent in history so the model can link "tool" responses back)
    tool_calls: list["ToolCall"] = field(default_factory=list)
    # tool_call_id populated on role=="tool" messages linking back to a ToolCall
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool schema — the JSON-Schema description sent to the LLM
# ---------------------------------------------------------------------------

@dataclass
class ToolSchema:
    """JSON-Schema description of a tool, forwarded to the LLM as a function spec."""

    name: str
    description: str
    # JSON Schema object for the 'parameters' field (dict with "type", "properties", etc.)
    parameters: dict[str, Any]

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Tool call / result pairs
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """An LLM-requested tool invocation."""

    call_id: str          # Opaque ID echoed back in ToolResult (matches LLM's id)
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """The outcome of executing a single ToolCall."""

    call_id: str
    tool_name: str
    output: str           # Serialised result (text, JSON string, code output, …)
    error: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Agent reasoning step
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """One complete reasoning step inside the agent loop.

    A step corresponds to one LLM call plus any tool executions that follow.
    """

    step_number: int
    # Raw thought text produced by the LLM (may be empty for pure tool-call responses)
    thought: str
    # Internal reasoning/thinking tokens if the backend exposes them
    reasoning: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    # Set only on the final step when the agent decides it is done
    final_answer: str | None = None
    elapsed_ms: float = 0.0

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


# ---------------------------------------------------------------------------
# Full agent run record
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    """Aggregate statistics collected over a complete agent run."""

    total_steps: int = 0
    total_tool_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    wall_time_s: float = 0.0
    # Per-tool call counts: {"web_fetch": 3, "python_sandbox": 1, …}
    tool_call_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class AgentRun:
    """The complete record of a single agent run from task to final answer."""

    run_id: str
    task: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None
    # Why the run ended: "answer", "max_steps", "error"
    termination_reason: str = "unknown"
    metrics: RunMetrics = field(default_factory=RunMetrics)


# ---------------------------------------------------------------------------
# LLM backend response
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Raw response from a BaseLLMBackend.complete() call."""

    content: str | None          # Text content (may be None if response is tool-call only)
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str | None = None  # Thinking tokens if the backend exposes them
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Latency of the backend HTTP call
    latency_ms: float = 0.0
