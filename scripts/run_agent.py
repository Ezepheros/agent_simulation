"""CLI entry point for running an agent from a YAML config.

Usage:
    python scripts/run_agent.py --config configs/runs/web_research.yaml
    python scripts/run_agent.py --config configs/runs/web_research.yaml \\
        --task "Summarise recent news about open-source LLMs"
    python scripts/run_agent.py --config configs/runs/web_research.yaml \\
        --run-id my_custom_run_id
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure the project root is on sys.path when running as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from small_agent.config.schemas import RunConfig
from small_agent.pipeline.agent_pipeline import AgentRunPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small-model agent.")
    parser.add_argument("--config", required=True, help="Path to a run YAML config file.")
    parser.add_argument("--task", default=None, help="Override the task defined in the config.")
    parser.add_argument("--run-id", default=None, help="Override the run_id in the config.")
    args = parser.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    if args.task:
        cfg.task = args.task
    if args.run_id:
        cfg.run_id = args.run_id
        cfg.logging.log_dir = f"runs/{cfg.run_id}/logs"

    pipeline = AgentRunPipeline(cfg)
    run = pipeline.run()

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(run.final_answer or "(no answer — check logs)")
    print(f"\nTermination reason : {run.termination_reason}")
    print(f"Steps taken        : {run.metrics.total_steps}")
    print(f"Tool calls         : {run.metrics.total_tool_calls}")
    print(f"Wall time          : {run.metrics.wall_time_s:.1f}s")
    print(f"Results saved to   : runs/{cfg.run_id}/")


if __name__ == "__main__":
    main()
