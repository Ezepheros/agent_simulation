"""Google Docs tools — read/write document content via the Docs API v1.

Three tools are provided:
    - ReadDocTool    → read_doc
    - AppendTextTool → append_text
    - CreateDocTool  → create_doc

Auth reuses the same OAuth2 credentials file as the Gmail tool
(GMAIL_CREDENTIALS_PATH) but stores a separate token at DOCS_TOKEN_PATH.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from small_agent.core.types import ToolResult, ToolSchema
from small_agent.tools.base import BaseTool

_SCOPES = ["https://www.googleapis.com/auth/documents"]

_SERVICE_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _resolve_auth(credentials_path: str | None, token_path: str | None) -> tuple[Path, Path]:
    creds = Path(
        credentials_path
        or os.environ.get(
            "GMAIL_CREDENTIALS_PATH",
            "~/.config/small_agent/gmail_credentials.json",
        )
    ).expanduser()
    tok = Path(
        token_path
        or os.environ.get(
            "DOCS_TOKEN_PATH",
            "~/.config/small_agent/docs_token.json",
        )
    ).expanduser()
    return creds, tok


def _build_service(credentials_path: Path, token_path: Path):
    """Build (and cache) a Google Docs API v1 service client."""
    cache_key = str(token_path)
    if cache_key in _SERVICE_CACHE:
        return _SERVICE_CACHE[cache_key]

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), _SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())

    service = build("docs", "v1", credentials=creds)
    _SERVICE_CACHE[cache_key] = service
    return service


def _extract_text(doc: dict) -> str:
    """Extract plain text from a Google Docs document body."""
    parts: list[str] = []
    for element in doc.get("body", {}).get("content", []):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        line_parts: list[str] = []
        for pe in paragraph.get("elements", []):
            text_run = pe.get("textRun")
            if text_run:
                line_parts.append(text_run.get("content", ""))
        parts.append("".join(line_parts))
    return "".join(parts)


def _get_end_index(doc: dict) -> int:
    """Return the index just before the final newline — safe insertion point."""
    content = doc.get("body", {}).get("content", [])
    if not content:
        return 1
    last_end = content[-1].get("endIndex", 1)
    # The body always ends with a trailing newline at endIndex - 1; insert before it.
    return max(last_end - 1, 1)


# ---------------------------------------------------------------------------
# Tool 1 — read_doc
# ---------------------------------------------------------------------------

class ReadDocTool(BaseTool):
    """Read the plain-text content of a Google Doc."""

    def __init__(
        self,
        credentials_path: str | None = None,
        token_path: str | None = None,
        max_chars: int = 10_000,
    ) -> None:
        self._credentials_path, self._token_path = _resolve_auth(credentials_path, token_path)
        self._max_chars = max_chars

    @property
    def name(self) -> str:
        return "read_doc"

    @property
    def description(self) -> str:
        return (
            "Read the plain-text content of a Google Doc. "
            "Returns the document title and body text."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The document ID from the Google Docs URL.",
                    },
                },
                "required": ["document_id"],
            },
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        t0 = time.monotonic()
        try:
            svc = _build_service(self._credentials_path, self._token_path)
            document_id = arguments["document_id"]
            doc = svc.documents().get(documentId=document_id).execute()
            title = doc.get("title", "")
            text = _extract_text(doc)
            if len(text) > self._max_chars:
                text = text[: self._max_chars] + "\n[… truncated]"
            output = {
                "document_id": document_id,
                "title": title,
                "char_count": len(text),
                "content": text,
            }
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output=json.dumps(output, ensure_ascii=False, indent=2),
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output="",
                error=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000,
            )


# ---------------------------------------------------------------------------
# Tool 2 — append_text
# ---------------------------------------------------------------------------

class AppendTextTool(BaseTool):
    """Append text to the end of a Google Doc."""

    def __init__(
        self,
        credentials_path: str | None = None,
        token_path: str | None = None,
    ) -> None:
        self._credentials_path, self._token_path = _resolve_auth(credentials_path, token_path)

    @property
    def name(self) -> str:
        return "append_text"

    @property
    def description(self) -> str:
        return (
            "Append text to the end of a Google Doc. "
            "Use \\n for newlines within the text. "
            "The text is inserted before the document's final newline."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The document ID from the Google Docs URL.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to append. Use \\n for newlines.",
                    },
                },
                "required": ["document_id", "text"],
            },
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        t0 = time.monotonic()
        try:
            svc = _build_service(self._credentials_path, self._token_path)
            document_id = arguments["document_id"]
            text = arguments["text"]

            doc = svc.documents().get(documentId=document_id).execute()
            end_index = _get_end_index(doc)

            requests = [{"insertText": {"location": {"index": end_index}, "text": text}}]
            svc.documents().batchUpdate(
                documentId=document_id, body={"requests": requests}
            ).execute()

            output = {
                "document_id": document_id,
                "chars_appended": len(text),
                "inserted_at_index": end_index,
            }
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output=json.dumps(output, ensure_ascii=False, indent=2),
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output="",
                error=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000,
            )


# ---------------------------------------------------------------------------
# Tool 3 — create_doc
# ---------------------------------------------------------------------------

class CreateDocTool(BaseTool):
    """Create a new Google Doc and optionally write initial content."""

    def __init__(
        self,
        credentials_path: str | None = None,
        token_path: str | None = None,
    ) -> None:
        self._credentials_path, self._token_path = _resolve_auth(credentials_path, token_path)

    @property
    def name(self) -> str:
        return "create_doc"

    @property
    def description(self) -> str:
        return (
            "Create a new Google Doc. "
            "Returns the document ID and URL. "
            "Optionally provide initial text content to write into the document."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the new document.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Optional initial text to write into the document. Use \\n for newlines.",
                    },
                },
                "required": ["title"],
            },
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        t0 = time.monotonic()
        try:
            svc = _build_service(self._credentials_path, self._token_path)
            title = arguments["title"]
            doc = svc.documents().create(body={"title": title}).execute()
            document_id = doc["documentId"]
            doc_url = f"https://docs.google.com/document/d/{document_id}/edit"

            if arguments.get("content"):
                content = arguments["content"]
                requests = [{"insertText": {"location": {"index": 1}, "text": content}}]
                svc.documents().batchUpdate(
                    documentId=document_id, body={"requests": requests}
                ).execute()

            output = {
                "document_id": document_id,
                "document_url": doc_url,
                "title": title,
            }
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output=json.dumps(output, ensure_ascii=False, indent=2),
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output="",
                error=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000,
            )
