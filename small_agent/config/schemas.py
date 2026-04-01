"""Pydantic v2 configuration schemas for agent runs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Component configs
# ---------------------------------------------------------------------------

class ComponentConfig(BaseModel):
    """Base for any component instantiated via the registry."""

    model_config = {"extra": "allow"}  # pass-through for component-specific fields

    type: str  # dotted class path, e.g. "tools.web_fetch.WebFetchTool"

    def to_build_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data.pop("type")
        return {"type": self.type, **data}


class ToolConfig(ComponentConfig):
    """Config for a single tool."""
    name: str | None = None  # optional override of the tool's default name


class LLMConfig(BaseModel):
    """Config for the LLM backend."""

    type: str = "backends.lmstudio.LMStudioBackend"
    base_url: str = "http://localhost:1234/v1"
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_s: float = 120.0

    model_config = {"extra": "allow"}

    def to_build_dict(self) -> dict[str, Any]:
        return self.model_dump()


class CriticConfig(BaseModel):
    """Optional critic that reviews each proposed step before tool execution.

    By default the critic reuses the same LLM backend as the agent.
    Override llm to use a separate model for critiquing.
    """

    enabled: bool = False
    type: str = "critics.llm_critic.LLMCritic"
    # If set, uses a different LLM for critiquing (e.g. a faster/cheaper model)
    llm: LLMConfig | None = None

    model_config = {"extra": "allow"}


class AgentConfig(BaseModel):
    """Config for the agent implementation."""

    type: str = "agents.react.ReActAgent"
    max_steps: int = 10
    system_prompt: str | None = None

    model_config = {"extra": "allow"}

    def to_build_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Logging config
# ---------------------------------------------------------------------------

class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "runs/{run_id}/logs"

    @model_validator(mode="after")
    def _expand_paths(self) -> "LoggingConfig":
        # Placeholder expansion happens at pipeline build time when run_id is known
        return self


# ---------------------------------------------------------------------------
# Top-level run config
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    """Complete configuration for a single agent run."""

    run_id: str
    task: str
    description: str = ""

    agent: AgentConfig
    llm: LLMConfig
    critic: CriticConfig = Field(default_factory=CriticConfig)
    tools: list[ToolConfig] = Field(default_factory=list)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Internal: base config path (resolved before validation)
    base_config: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _expand_log_dir(self) -> "RunConfig":
        self.logging.log_dir = self.logging.log_dir.replace("{run_id}", self.run_id)
        return self

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        raw = _load_yaml(Path(path))
        if "base_config" in raw:
            base_path = Path(path).parent / raw.pop("base_config")
            base = _load_yaml(base_path)
            raw = _deep_merge(base, raw)
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
