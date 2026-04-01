"""ReAct agent — interleaves Reasoning and Acting until done or max steps."""

from __future__ import annotations

import time
import uuid
from datetime import date

from small_agent.agents.base import BaseAgent
from small_agent.backends.base import BaseLLMBackend
from small_agent.core.types import (
    AgentRun,
    AgentStep,
    Message,
    RunMetrics,
    ToolResult,
)
from small_agent.critics.base import BaseCritic, ProposedAction
from small_agent.logging import get_logger, truncate
from small_agent.tools.base import BaseTool

_DEFAULT_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to tools.
Today's date is {today}.
Think step by step. Use tools when needed to gather information or execute code.
When you have enough information to answer the user's task, reply with your final answer.
"""

_DONE_SIGNAL = "[DONE]"


class ReActAgent(BaseAgent):
    """Implements the ReAct (Reason + Act) loop.

    Each iteration:
      1. LLM receives the full conversation history and available tool schemas.
      2. LLM returns a thought (text) and zero or more tool calls.
      3. Tools execute; results are appended to the conversation.
      4. If no tools were called (or the LLM emits the done signal), the run ends.
    """

    def __init__(
        self,
        backend: BaseLLMBackend,
        tools: list[BaseTool],
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 10,
        critic: BaseCritic | None = None,
    ) -> None:
        self.backend = backend
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.critic = critic
        self._log = get_logger(__name__)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, task: str, run_id: str | None = None) -> AgentRun:
        run_id = run_id or str(uuid.uuid4())[:8]
        run = AgentRun(run_id=run_id, task=task)
        tool_schemas = [t.schema for t in self.tools.values()]

        system_prompt = self.system_prompt.format(today=date.today().isoformat())
        messages: list[Message] = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=task),
        ]

        self._log.info("Starting ReAct run | task=%r | max_steps=%d", task, self.max_steps)
        wall_start = time.monotonic()

        for step_num in range(1, self.max_steps + 1):
            self._log.info("--- Step %d ---", step_num)
            step, messages, done = self._step(
                step_num, messages, tool_schemas, run.metrics, list(run.steps)
            )
            run.steps.append(step)

            if done:
                run.final_answer = step.final_answer
                run.termination_reason = "answer"
                break
        else:
            run.termination_reason = "max_steps"
            self._log.warning("Reached max_steps=%d without a final answer.", self.max_steps)

        run.metrics.total_steps = len(run.steps)
        run.metrics.wall_time_s = time.monotonic() - wall_start
        self._log.info(
            "Run complete | steps=%d | reason=%s | wall_time=%.1fs",
            run.metrics.total_steps,
            run.termination_reason,
            run.metrics.wall_time_s,
        )
        return run

    # ------------------------------------------------------------------
    # Internal step logic
    # ------------------------------------------------------------------

    def _step(
        self,
        step_num: int,
        messages: list[Message],
        tool_schemas,
        metrics: RunMetrics,
        previous_steps: list[AgentStep] | None = None,
    ) -> tuple[AgentStep, list[Message], bool]:
        t0 = time.monotonic()

        self._log.debug(
            "Context sent to LLM (%d messages):\n%s",
            len(messages),
            "\n".join(
                f"  [{m.role.upper()}] {truncate(m.content or '(no content)', 300)}"
                + (f" + {len(m.tool_calls)} tool_call(s)" if m.tool_calls else "")
                for m in messages
            ),
        )

        llm_resp = self.backend.complete(messages, tools=tool_schemas)
        metrics.total_prompt_tokens += llm_resp.prompt_tokens
        metrics.total_completion_tokens += llm_resp.completion_tokens

        thought = llm_resp.content or ""
        if llm_resp.reasoning:
            self._log.info("Reasoning: %s", llm_resp.reasoning)
        if thought:
            self._log.info("Thought: %s", thought)
        else:
            self._log.info("Thought: (none — model went straight to tool call)")

        # --- Critic review (before tool execution) -----------------------
        critique_text: str | None = None
        if self.critic is not None:
            proposed = ProposedAction(thought=thought, tool_calls=llm_resp.tool_calls)
            result = self.critic.review(
                task=messages[1].content or "",  # user message is always index 1
                previous_steps=previous_steps or [],
                proposed=proposed,
            )
            if result.has_issues:
                critique_text = result.feedback
                self._log.warning("Critic raised issues — asking agent to revise:\n%s", critique_text)
                # Inject critique as a user message so the agent can revise
                messages.append(Message(role="assistant", content=thought, tool_calls=llm_resp.tool_calls))
                messages.append(Message(
                    role="user",
                    content=(
                        f"[CRITIC REVIEW]\n{critique_text}\n\n"
                        "Please correct the errors above and provide a revised response."
                    ),
                ))
                # One revision attempt
                llm_resp = self.backend.complete(messages, tools=tool_schemas)
                metrics.total_prompt_tokens += llm_resp.prompt_tokens
                metrics.total_completion_tokens += llm_resp.completion_tokens
                thought = llm_resp.content or ""
                self._log.info("Revised thought after critique: %s", truncate(thought, 300))
                # Remove the injected critique exchange — it is captured in the step record,
                # not needed in the ongoing conversation history
                messages.pop()
                messages.pop()

        # Append the assistant's turn — tool_calls must be included so the
        # model can later match "tool" role messages back to its own requests
        messages.append(Message(role="assistant", content=thought, tool_calls=llm_resp.tool_calls))

        tool_results: list[ToolResult] = []
        done = False

        if not llm_resp.tool_calls and not thought.strip():
            if critique_text:
                # Empty response after a critic revision — model failed to act on the feedback.
                # Do NOT treat as final answer; let the loop continue to the next step.
                self._log.warning(
                    "Step %d: empty response after critic revision — model did not produce "
                    "a tool call or answer. Continuing to next step.",
                    step_num,
                )
            else:
                # Empty response with no critique context — likely hit max_tokens mid-reasoning.
                self._log.warning(
                    "Step %d: empty response (no tool calls, no content) — "
                    "model may have hit max_tokens mid-reasoning. "
                    "Consider raising max_tokens in base.yaml.",
                    step_num,
                )
                done = True

        if (
            not llm_resp.tool_calls
            and llm_resp.reasoning
            and "<tool_call>" in llm_resp.reasoning
        ):
            # Model planned a tool call inside its reasoning but never emitted it as structured
            # output — common failure mode with reasoning models under confusion or token pressure
            self._log.warning(
                "Step %d: model wrote a tool call inside its reasoning tokens but did not "
                "emit it as a structured response. The call was NOT executed. "
                "The model may be confused — consider simplifying the task or clarifying the prompt.",
                step_num,
            )

        if llm_resp.tool_calls:
            # Execute each requested tool
            for tc in llm_resp.tool_calls:
                self._log.info(
                    "Tool call: %s | args=%s", tc.tool_name, tc.arguments
                )
                tool = self.tools.get(tc.tool_name)
                if tool is None:
                    result = ToolResult(
                        call_id=tc.call_id,
                        tool_name=tc.tool_name,
                        output="",
                        error=f"Unknown tool '{tc.tool_name}'",
                    )
                else:
                    result = tool.execute(tc.arguments)
                    result.call_id = tc.call_id  # ensure ID matches

                tool_results.append(result)
                metrics.total_tool_calls += 1
                metrics.tool_call_counts[tc.tool_name] = (
                    metrics.tool_call_counts.get(tc.tool_name, 0) + 1
                )

                log_level = "info" if result.success else "warning"
                getattr(self._log, log_level)(
                    "Tool result: %s | ok=%s | latency=%.0fms | preview=%s",
                    result.tool_name,
                    result.success,
                    result.latency_ms,
                    truncate(result.output or result.error or "", 200),
                )

                messages.append(
                    Message(
                        role="tool",
                        content=result.output if result.success else f"Error: {result.error}",
                        tool_call_id=result.call_id,
                    )
                )
        else:
            # No tool calls → agent is declaring its answer
            done = True
            self._log.info("No tool calls — treating response as final answer.")

        final_answer = thought if done else None
        step = AgentStep(
            step_number=step_num,
            thought=thought,
            reasoning=llm_resp.reasoning,
            critique=critique_text,
            tool_calls=llm_resp.tool_calls,
            tool_results=tool_results,
            final_answer=final_answer,
            elapsed_ms=(time.monotonic() - t0) * 1000,
        )
        return step, messages, done
