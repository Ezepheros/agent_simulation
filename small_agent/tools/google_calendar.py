"""Google Calendar tools — read/write calendar events via the Calendar API v3.

Five tools are provided so the agent can pick exactly what it needs:
    - GetAllCalendarsTool        → get_all_calendars
    - SearchCalendarEventsTool   → search_calendar_events
    - CreateCalendarEventTool    → create_calendar_event
    - ModifyCalendarEventTool    → modify_calendar_event
    - RemoveCalendarEventTool    → remove_calendar_event

Auth reuses the same OAuth2 credentials file as the Gmail tool
(GMAIL_CREDENTIALS_PATH) but stores a separate token at CALENDAR_TOKEN_PATH
so the two scopes don't interfere.
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

_SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Module-level service cache so all tools within one run share a connection.
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
            "CALENDAR_TOKEN_PATH",
            "~/.config/small_agent/calendar_token.json",
        )
    ).expanduser()
    return creds, tok


def _build_service(credentials_path: Path, token_path: Path):
    """Build (and cache) a Google Calendar API v3 service client."""
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

    service = build("calendar", "v3", credentials=creds)
    _SERVICE_CACHE[cache_key] = service
    return service


def _format_event(event: dict) -> dict:
    """Extract the key fields from a raw Calendar API event dict."""
    start = event.get("start", {})
    end = event.get("end", {})
    return {
        "id": event.get("id", ""),
        "summary": event.get("summary", "(no title)"),
        "start": start.get("dateTime") or start.get("date", ""),
        "end": end.get("dateTime") or end.get("date", ""),
        "location": event.get("location", ""),
        "description": event.get("description", ""),
        "status": event.get("status", ""),
        "html_link": event.get("htmlLink", ""),
    }


# ---------------------------------------------------------------------------
# Tool 1 — get_all_calendars
# ---------------------------------------------------------------------------

class GetAllCalendarsTool(BaseTool):
    """List every calendar the authenticated user has access to."""

    def __init__(
        self,
        credentials_path: str | None = None,
        token_path: str | None = None,
    ) -> None:
        self._credentials_path, self._token_path = _resolve_auth(credentials_path, token_path)

    @property
    def name(self) -> str:
        return "get_all_calendars"

    @property
    def description(self) -> str:
        return (
            "List all Google Calendars the user has access to, including their IDs. "
            "Use this to find the correct calendar_id before searching or creating events."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        t0 = time.monotonic()
        try:
            svc = _build_service(self._credentials_path, self._token_path)
            result = svc.calendarList().list().execute()
            calendars = [
                {
                    "id": cal["id"],
                    "summary": cal.get("summary", ""),
                    "primary": cal.get("primary", False),
                    "access_role": cal.get("accessRole", ""),
                }
                for cal in result.get("items", [])
            ]
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output=json.dumps(calendars, ensure_ascii=False, indent=2),
                latency_ms=(time.monotonic() - t0) * 1000,
                metadata={"count": len(calendars)},
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
# Tool 2 — search_calendar_events
# ---------------------------------------------------------------------------

class SearchCalendarEventsTool(BaseTool):
    """Search for calendar events by keyword and/or date range."""

    def __init__(
        self,
        credentials_path: str | None = None,
        token_path: str | None = None,
        max_results: int = 10,
    ) -> None:
        self._credentials_path, self._token_path = _resolve_auth(credentials_path, token_path)
        self._max_results = max_results

    @property
    def name(self) -> str:
        return "search_calendar_events"

    @property
    def description(self) -> str:
        return (
            "Search for events on Google Calendar within an optional date range or by keyword. "
            "Returns event IDs, titles, start/end times, and descriptions. "
            "Always get the event ID here before calling modify_calendar_event or remove_calendar_event."
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
                        "description": "Free-text search (e.g. 'standup', 'interview with Alice').",
                    },
                    "time_min": {
                        "type": "string",
                        "description": "Start of search window in RFC3339, e.g. '2024-12-04T00:00:00Z'.",
                    },
                    "time_max": {
                        "type": "string",
                        "description": "End of search window in RFC3339, e.g. '2024-12-04T23:59:59Z'.",
                    },
                    "calendar_id": {
                        "type": "string",
                        "description": "Calendar to search. Defaults to 'primary'.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum events to return (default 10, max 50).",
                    },
                },
                "required": [],
            },
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        t0 = time.monotonic()
        try:
            svc = _build_service(self._credentials_path, self._token_path)
            calendar_id = arguments.get("calendar_id", "primary")
            max_results = min(int(arguments.get("max_results", self._max_results)), 50)
            kwargs: dict[str, Any] = dict(
                calendarId=calendar_id,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            if arguments.get("query"):
                kwargs["q"] = arguments["query"]
            if arguments.get("time_min"):
                kwargs["timeMin"] = arguments["time_min"]
            if arguments.get("time_max"):
                kwargs["timeMax"] = arguments["time_max"]

            result = svc.events().list(**kwargs).execute()
            events = [_format_event(e) for e in result.get("items", [])]
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output=json.dumps(events, ensure_ascii=False, indent=2),
                latency_ms=(time.monotonic() - t0) * 1000,
                metadata={"count": len(events)},
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
# Tool 3 — create_calendar_event
# ---------------------------------------------------------------------------

class CreateCalendarEventTool(BaseTool):
    """Create a new event on Google Calendar."""

    def __init__(
        self,
        credentials_path: str | None = None,
        token_path: str | None = None,
    ) -> None:
        self._credentials_path, self._token_path = _resolve_auth(credentials_path, token_path)

    @property
    def name(self) -> str:
        return "create_calendar_event"

    @property
    def description(self) -> str:
        return (
            "Create a new event on Google Calendar. "
            "Provide start and end times in RFC3339 format (e.g. '2024-12-05T09:00:00'). "
            "If no timezone is given, UTC is assumed."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Title of the event.",
                    },
                    "start_datetime": {
                        "type": "string",
                        "description": "Start time in RFC3339, e.g. '2024-12-05T09:00:00'.",
                    },
                    "end_datetime": {
                        "type": "string",
                        "description": "End time in RFC3339, e.g. '2024-12-05T09:30:00'.",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone, e.g. 'America/New_York'. Defaults to 'UTC'.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional notes/body for the event.",
                    },
                    "location": {
                        "type": "string",
                        "description": "Optional location string.",
                    },
                    "calendar_id": {
                        "type": "string",
                        "description": "Calendar to add the event to. Defaults to 'primary'.",
                    },
                },
                "required": ["summary", "start_datetime", "end_datetime"],
            },
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        t0 = time.monotonic()
        try:
            svc = _build_service(self._credentials_path, self._token_path)
            timezone = arguments.get("timezone", "UTC")
            calendar_id = arguments.get("calendar_id", "primary")
            body: dict[str, Any] = {
                "summary": arguments["summary"],
                "start": {"dateTime": arguments["start_datetime"], "timeZone": timezone},
                "end": {"dateTime": arguments["end_datetime"], "timeZone": timezone},
            }
            if arguments.get("description"):
                body["description"] = arguments["description"]
            if arguments.get("location"):
                body["location"] = arguments["location"]

            event = svc.events().insert(calendarId=calendar_id, body=body).execute()
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output=json.dumps(_format_event(event), ensure_ascii=False, indent=2),
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
# Tool 4 — modify_calendar_event
# ---------------------------------------------------------------------------

class ModifyCalendarEventTool(BaseTool):
    """Patch an existing calendar event — only the provided fields are changed."""

    def __init__(
        self,
        credentials_path: str | None = None,
        token_path: str | None = None,
    ) -> None:
        self._credentials_path, self._token_path = _resolve_auth(credentials_path, token_path)

    @property
    def name(self) -> str:
        return "modify_calendar_event"

    @property
    def description(self) -> str:
        return (
            "Modify an existing Google Calendar event. "
            "You must provide event_id (from search_calendar_events). "
            "Only the fields you supply are updated; everything else stays the same."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "ID of the event to modify (from search_calendar_events).",
                    },
                    "calendar_id": {
                        "type": "string",
                        "description": "Calendar containing the event. Defaults to 'primary'.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "New event title.",
                    },
                    "start_datetime": {
                        "type": "string",
                        "description": "New start time in RFC3339 format.",
                    },
                    "end_datetime": {
                        "type": "string",
                        "description": "New end time in RFC3339 format.",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone to use when updating times.",
                    },
                    "description": {
                        "type": "string",
                        "description": "New event description.",
                    },
                    "location": {
                        "type": "string",
                        "description": "New event location.",
                    },
                },
                "required": ["event_id"],
            },
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        t0 = time.monotonic()
        try:
            svc = _build_service(self._credentials_path, self._token_path)
            event_id = arguments["event_id"]
            calendar_id = arguments.get("calendar_id", "primary")

            # Fetch current event so we can preserve timezone if not overridden.
            existing = svc.events().get(calendarId=calendar_id, eventId=event_id).execute()
            patch: dict[str, Any] = {}

            if arguments.get("summary"):
                patch["summary"] = arguments["summary"]
            if arguments.get("description"):
                patch["description"] = arguments["description"]
            if arguments.get("location"):
                patch["location"] = arguments["location"]
            if arguments.get("start_datetime") or arguments.get("end_datetime"):
                tz = arguments.get("timezone") or existing.get("start", {}).get("timeZone", "UTC")
                if arguments.get("start_datetime"):
                    patch["start"] = {"dateTime": arguments["start_datetime"], "timeZone": tz}
                if arguments.get("end_datetime"):
                    patch["end"] = {"dateTime": arguments["end_datetime"], "timeZone": tz}

            updated = svc.events().patch(
                calendarId=calendar_id, eventId=event_id, body=patch
            ).execute()
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output=json.dumps(_format_event(updated), ensure_ascii=False, indent=2),
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
# Tool 5 — remove_calendar_event
# ---------------------------------------------------------------------------

class RemoveCalendarEventTool(BaseTool):
    """Permanently delete a calendar event."""

    def __init__(
        self,
        credentials_path: str | None = None,
        token_path: str | None = None,
    ) -> None:
        self._credentials_path, self._token_path = _resolve_auth(credentials_path, token_path)

    @property
    def name(self) -> str:
        return "remove_calendar_event"

    @property
    def description(self) -> str:
        return (
            "Permanently delete an event from Google Calendar. "
            "You must provide the event_id (from search_calendar_events). "
            "This action cannot be undone."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "ID of the event to delete (from search_calendar_events).",
                    },
                    "calendar_id": {
                        "type": "string",
                        "description": "Calendar containing the event. Defaults to 'primary'.",
                    },
                },
                "required": ["event_id"],
            },
        )

    def execute(self, arguments: dict) -> ToolResult:
        call_id = str(uuid.uuid4())
        t0 = time.monotonic()
        try:
            svc = _build_service(self._credentials_path, self._token_path)
            event_id = arguments["event_id"]
            calendar_id = arguments.get("calendar_id", "primary")
            svc.events().delete(calendarId=calendar_id, eventId=event_id).execute()
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                output=json.dumps({"deleted": True, "event_id": event_id}),
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
