"""PythonSandbox tool — executes Python code in an isolated subprocess."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from small_agent.core.types import ToolResult, ToolSchema
from small_agent.tools.base import BaseTool


class PythonSandboxTool(BaseTool):
    """Execute a Python code snippet and return stdout/stderr."""

    def __init__(self, timeout_s: float = 30.0, max_output_chars: int = 4_000) -> None:
        self.timeout_s = timeout_s
        self.max_output_chars = max_output_chars

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "python_sandbox"

    @property
    def description(self) -> str:
        return (
            "Execute a Python code snippet and return stdout, stderr, and exit code. "
            "Use this whenever a task involves: sorting or ranking many items, "
            "counting or aggregating, arithmetic or statistics, string formatting across "
            "many inputs, generating structured output (tables, JSON, markdown), "
            "or any step where doing it by hand would be error-prone or tedious."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source code to execute.",
                    },
                    "timeout_s": {
                        "type": "number",
                        "description": "Execution timeout in seconds (default uses tool default).",
                    },
                },
                "required": ["code"],
            },
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        code: str = arguments.get("code", "")
        timeout = float(arguments.get("timeout_s", self.timeout_s))
        t0 = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="small_agent_sandbox_") as tmpdir:
            script = Path(tmpdir) / "script.py"
            script.write_text(code, encoding="utf-8")

            try:
                result = subprocess.run(
                    [sys.executable, str(script)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmpdir,
                    # Inherit PATH so Python can find stdlib/site-packages on any OS,
                    # but strip secrets from the environment
                    env={
                        k: v for k, v in os.environ.items()
                        if k in ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP",
                                 "PYTHONPATH", "PYTHONHOME", "USERPROFILE", "HOME")
                    },
                )
                stdout = self._truncate(result.stdout)
                stderr = self._truncate(result.stderr)
                output = self._format_output(stdout, stderr, result.returncode)
                error = None if result.returncode == 0 else f"Exit code {result.returncode}"
            except subprocess.TimeoutExpired:
                output = ""
                error = f"Execution timed out after {timeout}s"
            except Exception as exc:
                output = ""
                error = str(exc)

        return ToolResult(
            call_id=call_id,
            tool_name=self.name,
            output=output,
            error=error,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _truncate(self, text: str) -> str:
        if len(text) > self.max_output_chars:
            return text[: self.max_output_chars] + "\n[… truncated]"
        return text

    @staticmethod
    def _format_output(stdout: str, stderr: str, returncode: int) -> str:
        parts = []
        if stdout:
            parts.append(f"[stdout]\n{stdout}")
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        parts.append(f"[exit_code] {returncode}")
        return "\n".join(parts)
