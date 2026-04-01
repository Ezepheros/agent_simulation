"""Google Sheets tools — read/write spreadsheet data via the Sheets API v4.

Four tools are provided:
    - GetSpreadsheetInfoTool  → get_spreadsheet_info
    - ReadSheetTool           → read_sheet
    - AppendRowsTool          → append_rows
    - CreateSpreadsheetTool   → create_spreadsheet

Auth reuses the same OAuth2 credentials file as the Gmail tool
(GMAIL_CREDENTIALS_PATH) but stores a separate token at SHEETS_TOKEN_PATH.
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

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

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
            "SHEETS_TOKEN_PATH",
            "~/.config/small_agent/sheets_token.json",
        )
    ).expanduser()
    return creds, tok


def _build_service(credentials_path: Path, token_path: Path):
    """Build (and cache) a Google Sheets API v4 service client."""
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

    service = build("sheets", "v4", credentials=creds)
    _SERVICE_CACHE[cache_key] = service
    return service


# ---------------------------------------------------------------------------
# Tool 1 — get_spreadsheet_info
# ---------------------------------------------------------------------------

class GetSpreadsheetInfoTool(BaseTool):
    """Get metadata about a spreadsheet: title and list of sheet names/IDs."""

    def __init__(
        self,
        credentials_path: str | None = None,
        token_path: str | None = None,
    ) -> None:
        self._credentials_path, self._token_path = _resolve_auth(credentials_path, token_path)

    @property
    def name(self) -> str:
        return "get_spreadsheet_info"

    @property
    def description(self) -> str:
        return (
            "Get the title and list of sheet names/IDs inside a Google Spreadsheet. "
            "Use this to discover sheet names before reading or writing data."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "spreadsheet_id": {
                        "type": "string",
                        "description": "The spreadsheet ID from the Google Sheets URL.",
                    },
                },
                "required": ["spreadsheet_id"],
            },
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        t0 = time.monotonic()
        try:
            svc = _build_service(self._credentials_path, self._token_path)
            spreadsheet_id = arguments["spreadsheet_id"]
            result = svc.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            info = {
                "spreadsheet_id": spreadsheet_id,
                "title": result.get("properties", {}).get("title", ""),
                "spreadsheet_url": result.get("spreadsheetUrl", ""),
                "sheets": [
                    {
                        "sheet_id": s["properties"]["sheetId"],
                        "title": s["properties"]["title"],
                        "row_count": s["properties"].get("gridProperties", {}).get("rowCount"),
                        "column_count": s["properties"].get("gridProperties", {}).get("columnCount"),
                    }
                    for s in result.get("sheets", [])
                ],
            }
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output=json.dumps(info, ensure_ascii=False, indent=2),
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
# Tool 2 — read_sheet
# ---------------------------------------------------------------------------

class ReadSheetTool(BaseTool):
    """Read a range of cells from a Google Sheet."""

    def __init__(
        self,
        credentials_path: str | None = None,
        token_path: str | None = None,
    ) -> None:
        self._credentials_path, self._token_path = _resolve_auth(credentials_path, token_path)

    @property
    def name(self) -> str:
        return "read_sheet"

    @property
    def description(self) -> str:
        return (
            "Read a range of cells from a Google Sheet. "
            "Returns rows as a list of lists. "
            "Use A1 notation for the range, e.g. 'Sheet1!A1:D10' or just 'A1:D10' for the first sheet."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "spreadsheet_id": {
                        "type": "string",
                        "description": "The spreadsheet ID from the Google Sheets URL.",
                    },
                    "range": {
                        "type": "string",
                        "description": "A1 notation range, e.g. 'Sheet1!A1:D20'. Omit row bounds to read the whole column range.",
                    },
                },
                "required": ["spreadsheet_id", "range"],
            },
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        t0 = time.monotonic()
        try:
            svc = _build_service(self._credentials_path, self._token_path)
            spreadsheet_id = arguments["spreadsheet_id"]
            range_ = arguments["range"]
            result = (
                svc.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=range_)
                .execute()
            )
            values = result.get("values", [])
            output = {
                "range": result.get("range", range_),
                "row_count": len(values),
                "values": values,
            }
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output=json.dumps(output, ensure_ascii=False, indent=2),
                latency_ms=(time.monotonic() - t0) * 1000,
                metadata={"row_count": len(values)},
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
# Tool 3 — append_rows
# ---------------------------------------------------------------------------

class AppendRowsTool(BaseTool):
    """Append one or more rows to a Google Sheet."""

    def __init__(
        self,
        credentials_path: str | None = None,
        token_path: str | None = None,
    ) -> None:
        self._credentials_path, self._token_path = _resolve_auth(credentials_path, token_path)

    @property
    def name(self) -> str:
        return "append_rows"

    @property
    def description(self) -> str:
        return (
            "Append one or more rows to a Google Sheet. "
            "Rows are added after the last row that contains data. "
            "Each row is a list of cell values."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "spreadsheet_id": {
                        "type": "string",
                        "description": "The spreadsheet ID from the Google Sheets URL.",
                    },
                    "sheet_name": {
                        "type": "string",
                        "description": "Name of the sheet tab to append to, e.g. 'Sheet1'. Defaults to the first sheet.",
                    },
                    "rows": {
                        "type": "array",
                        "description": "List of rows to append. Each row is a list of cell values, e.g. [[\"Alice\", \"Work\", \"Summary...\"], [\"Bob\", \"Personal\", \"...\"]]. Values are strings or numbers.",
                        "items": {
                            "type": "array",
                            "items": {}
                        },
                    },
                },
                "required": ["spreadsheet_id", "rows"],
            },
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        t0 = time.monotonic()
        try:
            svc = _build_service(self._credentials_path, self._token_path)
            spreadsheet_id = arguments["spreadsheet_id"]
            sheet_name = arguments.get("sheet_name", "Sheet1")
            rows = arguments["rows"]
            range_ = f"{sheet_name}!A1"

            body = {"values": rows}
            result = (
                svc.spreadsheets()
                .values()
                .append(
                    spreadsheetId=spreadsheet_id,
                    range=range_,
                    valueInputOption="USER_ENTERED",
                    insertDataOption="INSERT_ROWS",
                    body=body,
                )
                .execute()
            )
            updates = result.get("updates", {})
            output = {
                "appended_range": updates.get("updatedRange", ""),
                "rows_appended": updates.get("updatedRows", len(rows)),
                "cells_updated": updates.get("updatedCells", 0),
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
# Tool 4 — create_spreadsheet
# ---------------------------------------------------------------------------

class CreateSpreadsheetTool(BaseTool):
    """Create a new Google Spreadsheet and optionally write a header row."""

    def __init__(
        self,
        credentials_path: str | None = None,
        token_path: str | None = None,
    ) -> None:
        self._credentials_path, self._token_path = _resolve_auth(credentials_path, token_path)

    @property
    def name(self) -> str:
        return "create_spreadsheet"

    @property
    def description(self) -> str:
        return (
            "Create a new Google Spreadsheet. "
            "Returns the spreadsheet ID and URL. "
            "Optionally provide a header row to write immediately after creation."
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
                        "description": "Title of the new spreadsheet.",
                    },
                    "sheet_name": {
                        "type": "string",
                        "description": "Name of the first sheet tab. Defaults to 'Sheet1'.",
                    },
                    "headers": {
                        "type": "array",
                        "description": "Optional header row to write in row 1, e.g. [\"Subject\", \"Sender\", \"Category\"].",
                        "items": {"type": "string"},
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
            sheet_name = arguments.get("sheet_name", "Sheet1")

            body: dict[str, Any] = {
                "properties": {"title": title},
                "sheets": [{"properties": {"title": sheet_name}}],
            }
            spreadsheet = svc.spreadsheets().create(body=body).execute()
            spreadsheet_id = spreadsheet["spreadsheetId"]
            spreadsheet_url = spreadsheet.get("spreadsheetUrl", "")

            if arguments.get("headers"):
                headers = arguments["headers"]
                svc.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=f"{sheet_name}!A1",
                    valueInputOption="USER_ENTERED",
                    body={"values": [headers]},
                ).execute()

            output = {
                "spreadsheet_id": spreadsheet_id,
                "spreadsheet_url": spreadsheet_url,
                "title": title,
                "sheet_name": sheet_name,
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
