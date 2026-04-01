"""LLMCritic — uses a language model to review proposed agent steps."""

from __future__ import annotations

import json

from small_agent.backends.base import BaseLLMBackend
from small_agent.core.types import AgentStep, Message
from small_agent.critics.base import BaseCritic, CritiqueResult, ProposedAction
from small_agent.logging import get_logger, truncate

_CRITIC_SYSTEM_PROMPT = """\
You are a precise critic reviewing an AI agent's proposed actions before they are executed.

You will be given:
- The original task
- The agent's completed previous steps (thoughts and tool results)
- The agent's current proposed action (thought and proposed tool calls)

Your job is to identify errors before they cause harm. Look for:
- Wrong parameter values (e.g. incorrect dates, IDs, timezones, units)
- Incorrect reasoning or false assumptions in the thought
- Actions that contradict what previous tool results showed
- Missing required information that the agent assumed it had
- Logical mistakes in multi-step reasoning

Be concise and specific — point to the exact error and what the correct value should be.
If the task or parameters are ambiguous in a way that could lead to a wrong action, flag the ambiguity
and suggest the agent use the clarify tool to ask the user before proceeding.
If everything looks correct and unambiguous, respond with exactly the word: OK
Do not suggest improvements or alternatives unless there is an actual error or ambiguity.
"""

_OK_SIGNAL = "ok"


class LLMCritic(BaseCritic):
    """Critic that uses an LLM backend to review proposed steps."""

    def __init__(self, backend: BaseLLMBackend) -> None:
        self.backend = backend
        self._log = get_logger(__name__)

    def review(
        self,
        task: str,
        previous_steps: list[AgentStep],
        proposed: ProposedAction,
    ) -> CritiqueResult:
        prompt = self._build_prompt(task, previous_steps, proposed)
        messages = [
            Message(role="system", content=_CRITIC_SYSTEM_PROMPT),
            Message(role="user", content=prompt),
        ]

        self._log.debug("Critic reviewing step with %d previous steps", len(previous_steps))
        response = self.backend.complete(messages, tools=None)
        feedback = (response.content or "").strip()

        if not feedback or feedback.lower() == _OK_SIGNAL:
            self._log.info("Critic: OK")
            return CritiqueResult(has_issues=False, feedback="")

        self._log.warning("Critic found issues: %s", feedback)
        return CritiqueResult(has_issues=True, feedback=feedback)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        task: str,
        previous_steps: list[AgentStep],
        proposed: ProposedAction,
    ) -> str:
        parts = [f"## Task\n{task.strip()}\n"]

        if previous_steps:
            parts.append("## Completed previous steps")
            for step in previous_steps:
                parts.append(f"### Step {step.step_number}")
                if step.thought and step.thought.strip():
                    parts.append(f"Thought: {step.thought.strip()}")
                for tc in step.tool_calls:
                    parts.append(
                        f"Tool called: {tc.tool_name}\n"
                        f"Arguments: {json.dumps(tc.arguments, indent=2)}"
                    )
                for tr in step.tool_results:
                    status = "success" if tr.success else f"error: {tr.error}"
                    parts.append(
                        f"Tool result ({tr.tool_name}, {status}):\n"
                        f"{truncate(tr.output, 600) if tr.output else '(no output)'}"
                    )

        parts.append("## Current proposed action (NOT yet executed)")
        if proposed.thought and proposed.thought.strip():
            parts.append(f"Thought: {proposed.thought.strip()}")
        if proposed.tool_calls:
            for tc in proposed.tool_calls:
                parts.append(
                    f"Proposing to call: {tc.tool_name}\n"
                    f"With arguments: {json.dumps(tc.arguments, indent=2)}"
                )
        else:
            parts.append("(No tool calls — agent is proposing a final answer)")

        parts.append(
            "\nIdentify any errors in the thought or proposed tool arguments. "
            "If correct, reply: OK"
        )
        return "\n\n".join(parts)
