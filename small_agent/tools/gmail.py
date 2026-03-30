"""Gmail tool — read emails via the Gmail API using OAuth2."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from small_agent.core.types import ToolResult, ToolSchema
from small_agent.tools.base import BaseTool

# Lazy-imported so the tool is importable even without google-auth installed
_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class GmailTool(BaseTool):
    """Read emails from a Gmail inbox using a Gmail search query."""

    def __init__(
        self,
        credentials_path: str | None = None,
        token_path: str | None = None,
        max_body_chars: int = 2_000,
        max_results: int = 5,
    ) -> None:
        self.credentials_path = Path(
            credentials_path or os.environ.get("GMAIL_CREDENTIALS_PATH", "")
        ).expanduser()
        self.token_path = Path(
            token_path
            or os.environ.get(
                "GMAIL_TOKEN_PATH",
                "~/.config/small_agent/gmail_token.json",
            )
        ).expanduser()
        self.max_body_chars = max_body_chars
        self.max_results = max_results
        self._service = None  # Lazy-loaded

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "gmail_read"

    @property
    def description(self) -> str:
        return (
            "Search and read emails from Gmail. "
            "Accepts a standard Gmail search query (e.g. 'from:alice subject:report'). "
            "Returns a list of matching emails with subject, sender, date, and body."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Gmail search query string (same syntax as the Gmail search box).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of emails to return (default 5, max 20).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        query: str = arguments.get("query", "")
        max_results: int = min(int(arguments.get("max_results", self.max_results)), 20)
        t0 = time.monotonic()

        try:
            service = self._get_service()
            messages_ref = (
                service.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_results)
                .execute()
            )
            items = messages_ref.get("messages", [])
            emails = [self._fetch_email(service, item["id"]) for item in items]
            output = json.dumps(emails, ensure_ascii=False, indent=2)
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output="",
                error=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        return ToolResult(
            call_id=call_id,
            tool_name=self.name,
            output=output,
            latency_ms=(time.monotonic() - t0) * 1000,
            metadata={"email_count": len(emails)},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_service(self):
        if self._service is not None:
            return self._service

        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        creds = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), _SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), _SCOPES
                )
                creds = flow.run_local_server(port=0)
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(creds.to_json())

        self._service = build("gmail", "v1", credentials=creds)
        return self._service

    def _fetch_email(self, service, msg_id: str) -> dict:
        msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
        headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
        body = self._extract_body(msg["payload"])
        if len(body) > self.max_body_chars:
            body = body[: self.max_body_chars] + "\n[… truncated]"
        return {
            "id": msg_id,
            "subject": headers.get("Subject", ""),
            "from": headers.get("From", ""),
            "date": headers.get("Date", ""),
            "snippet": msg.get("snippet", ""),
            "body": body,
        }

    @staticmethod
    def _extract_body(payload: dict) -> str:
        import base64

        if payload.get("body", {}).get("data"):
            data = payload["body"]["data"]
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
        for part in payload.get("parts", []):
            text = GmailTool._extract_body(part)
            if text:
                return text
        return ""
