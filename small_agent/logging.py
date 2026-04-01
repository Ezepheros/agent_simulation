"""Structured logging for agent runs.

Usage:
    from small_agent.logging import setup_run_logging, get_logger

    setup_run_logging(run_id="my_run_001", log_dir="runs/my_run_001/logs")
    log = get_logger(__name__)
    log.info("Step started")   # → "[my_run_001] small_agent.agents.react | INFO | Step started"
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_active_run_id: str = "unset"


# ---------------------------------------------------------------------------
# Run-scoped logger adapter
# ---------------------------------------------------------------------------

class _RunAdapter(logging.LoggerAdapter):
    """Injects [run_id] prefix into every log record."""

    def process(self, msg, kwargs):
        return f"[{self.extra['run_id']}] {msg}", kwargs


def get_logger(name: str) -> _RunAdapter:
    """Return a logger adapter for the given module name, scoped to the active run."""
    return _RunAdapter(logging.getLogger(name), {"run_id": _active_run_id})


# ---------------------------------------------------------------------------
# JSON Lines handler
# ---------------------------------------------------------------------------

class _JsonlHandler(logging.FileHandler):
    """Writes one JSON object per log record."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "time": self.formatter.formatTime(record, "%Y-%m-%dT%H:%M:%S"),  # type: ignore[arg-type]
                "level": record.levelname,
                "logger": record.name,
                "run_id": getattr(record, "run_id", _active_run_id),
                "message": record.getMessage(),
            }
            if record.exc_info:
                entry["exc_info"] = self.formatter.formatException(record.exc_info)
            self.stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.stream.flush()
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars and append a marker if it was cut off."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[truncated]"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_run_logging(
    run_id: str,
    log_dir: str | Path,
    level: str = "INFO",
) -> None:
    """Configure logging for a run.

    Creates two sinks:
    - ``<log_dir>/<run_id>.log``   — human-readable plain-text
    - ``<log_dir>/<run_id>.jsonl`` — one JSON object per log line
    - stdout                        — real-time console output
    """
    global _active_run_id
    _active_run_id = run_id

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any pre-existing handlers to avoid duplication
    root.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # Plain-text file
    file_handler = logging.FileHandler(log_path / f"{run_id}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # JSON Lines file
    jsonl_handler = _JsonlHandler(log_path / f"{run_id}.jsonl", encoding="utf-8")
    jsonl_handler.setFormatter(fmt)
    root.addHandler(jsonl_handler)

    # Silence noisy third-party loggers — only show their warnings/errors
    for _noisy in (
        "httpx", "httpcore", "openai", "openai._base_client",
        "google_auth_oauthlib", "googleapiclient", "googleapiclient.discovery_cache",
        "urllib3", "charset_normalizer",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
