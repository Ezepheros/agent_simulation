"""AgentRunPipeline — orchestrates a complete agent run from config to saved artefacts."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from small_agent.config.schemas import RunConfig
from small_agent.core.types import AgentRun
from small_agent.logging import get_logger, setup_run_logging, truncate
from small_agent.registry import build


class AgentRunPipeline:
    """Reads a RunConfig, assembles components, runs the agent, and saves outputs.

    Outputs written to ``runs/{run_id}/``:
    - ``logs/{run_id}.log``    — human-readable log
    - ``logs/{run_id}.jsonl``  — structured JSON Lines log
    - ``result.json``          — final answer + metrics summary
    """

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        # Logging is set up immediately so every later call to get_logger() is live
        setup_run_logging(
            run_id=cfg.run_id,
            log_dir=cfg.logging.log_dir,
            level=cfg.logging.level,
        )
        self._log = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> AgentRun:
        cfg = self.cfg
        self._log.info("Pipeline start | run_id=%s | task=%r", cfg.run_id, cfg.task)

        backend = self._build_backend()
        tools = self._build_tools()
        critic = self._build_critic(backend)
        agent = self._build_agent(backend, tools, critic)

        t0 = time.monotonic()
        agent_run = agent.run(cfg.task, run_id=cfg.run_id)
        agent_run.metrics.wall_time_s = time.monotonic() - t0

        self._save_result(agent_run)
        self._log.info(
            "Pipeline done | answer=%r | steps=%d | wall_time=%.1fs",
            truncate(agent_run.final_answer or "", 120),
            agent_run.metrics.total_steps,
            agent_run.metrics.wall_time_s,
        )
        return agent_run

    # ------------------------------------------------------------------
    # Component construction
    # ------------------------------------------------------------------

    def _build_backend(self):
        cfg = self.cfg.llm
        self._log.info("Building backend | type=%s | model=%s", cfg.type, cfg.model)
        params = cfg.to_build_dict()
        params.pop("type")
        return build(cfg.type, params)

    def _build_tools(self) -> list:
        tools = []
        for tc in self.cfg.tools:
            self._log.info("Building tool | type=%s", tc.type)
            params = tc.to_build_dict()
            params.pop("type")
            params.pop("name", None)  # name is a property on the tool, not a ctor arg
            tools.append(build(tc.type, params))
        return tools

    def _build_critic(self, backend):
        cfg = self.cfg.critic
        if not cfg.enabled:
            return None
        critic_backend = backend
        if cfg.llm is not None:
            self._log.info("Building critic backend | model=%s", cfg.llm.model)
            params = cfg.llm.to_build_dict()
            params.pop("type")
            critic_backend = build(cfg.llm.type, params)
        self._log.info("Building critic | type=%s", cfg.type)
        return build(cfg.type, {"backend": critic_backend})

    def _build_agent(self, backend, tools, critic):
        cfg = self.cfg.agent
        self._log.info("Building agent | type=%s | max_steps=%d", cfg.type, cfg.max_steps)
        params = cfg.to_build_dict()
        params.pop("type")
        params["backend"] = backend
        params["tools"] = tools
        params["critic"] = critic
        # Remove None system_prompt so agent uses its own default
        if params.get("system_prompt") is None:
            params.pop("system_prompt")
        return build(cfg.type, params)

    # ------------------------------------------------------------------
    # Output persistence
    # ------------------------------------------------------------------

    def _save_result(self, run: AgentRun) -> None:
        out_dir = Path("runs") / self.cfg.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        result_path = out_dir / "result.json"

        # Serialise — dataclasses need manual conversion
        data = {
            "run_id": run.run_id,
            "task": run.task,
            "final_answer": run.final_answer,
            "termination_reason": run.termination_reason,
            "metrics": asdict(run.metrics),
            "step_count": len(run.steps),
            "steps": [
                {
                    "step_number": s.step_number,
                    "reasoning": s.reasoning,
                    "critique": s.critique,
                    "thought": s.thought,
                    "tool_calls": [
                        {"call_id": tc.call_id, "tool_name": tc.tool_name, "arguments": tc.arguments}
                        for tc in s.tool_calls
                    ],
                    "tool_results": [
                        {
                            "call_id": tr.call_id,
                            "tool_name": tr.tool_name,
                            "success": tr.success,
                            "output_preview": truncate(tr.output or "", 300),
                            "error": tr.error,
                            "latency_ms": tr.latency_ms,
                        }
                        for tr in s.tool_results
                    ],
                    "elapsed_ms": s.elapsed_ms,
                }
                for s in run.steps
            ],
        }

        result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self._log.info("Result saved to %s", result_path)
