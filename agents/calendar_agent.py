"""
Calendar Agent
Google Calendar integration with OAuth2.
Supports: create, search, delete events, conflict detection.
Falls back to mock mode if credentials not configured.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import GOOGLE_CREDENTIALS_PATH, GOOGLE_TOKEN_PATH, GOOGLE_SCOPES

# Google API imports with graceful fallback
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


class CalendarAgent:
    """
    Google Calendar agent with full OAuth2 authentication.

    Setup (one-time):
    1. Go to https://console.cloud.google.com
    2. Create project → Enable Calendar API
    3. Create OAuth2 Desktop credentials
    4. Download credentials.json → place at ~/.jarvis/credentials.json
    5. Run Jarvis — browser will open for auth on first use

    Falls back to mock mode automatically if credentials missing.
    """

    def __init__(self):
        self.service = None
        self.mock_events: List[Dict] = []
        self._mock_id_counter = 1
        self._init_service()

    def _init_service(self) -> None:
        if not GOOGLE_AVAILABLE:
            print("📅 CalendarAgent: Google API not installed — mock mode")
            return

        if not GOOGLE_CREDENTIALS_PATH.exists():
            print(f"📅 CalendarAgent: No credentials at {GOOGLE_CREDENTIALS_PATH} — mock mode")
            print("   To enable: download OAuth2 credentials.json from Google Cloud Console")
            return

        try:
            creds = None
            if GOOGLE_TOKEN_PATH.exists():
                creds = Credentials.from_authorized_user_file(
                    str(GOOGLE_TOKEN_PATH), GOOGLE_SCOPES
                )

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(GOOGLE_CREDENTIALS_PATH), GOOGLE_SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                    GOOGLE_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
                    GOOGLE_TOKEN_PATH.write_text(creds.to_json())

            self.service = build("calendar", "v3", credentials=creds)
            print("📅 CalendarAgent: Google Calendar connected ✅")
        except Exception as e:
            print(f"📅 CalendarAgent: Auth failed ({e}) — mock mode")

    @property
    def is_mock(self) -> bool:
        return self.service is None

    # ── Create event ───────────────────────────────────────────────────────

    async def create_event(
        self,
        title: str,
        start_time: str,
        end_time: str,
        attendees: Optional[List[str]] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a calendar event."""
        if self.is_mock:
            return self._mock_create(title, start_time, end_time, attendees, description, location)

        try:
            body = {
                "summary": title,
                "description": description,
                "location": location,
                "start": {"dateTime": start_time, "timeZone": "UTC"},
                "end":   {"dateTime": end_time,   "timeZone": "UTC"},
            }
            if attendees:
                body["attendees"] = [{"email": e} for e in attendees]

            event = self.service.events().insert(
                calendarId="primary",
                body=body,
                sendUpdates="all" if attendees else "none",
            ).execute()

            return {
                "success": True,
                "id": event["id"],
                "title": title,
                "start": start_time,
                "end": end_time,
                "link": event.get("htmlLink"),
                "attendees": attendees or [],
                "mock": False,
                "message": f"Created event '{title}' starting {start_time}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Search events ──────────────────────────────────────────────────────

    async def search_events(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        query: Optional[str] = None,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """Search for calendar events in a date range."""
        # Default to this week if no dates given
        now = datetime.utcnow()
        start_date = start_date or now.isoformat() + "Z"
        end_date   = end_date   or (now + timedelta(days=7)).isoformat() + "Z"

        if self.is_mock:
            return self._mock_search(start_date, end_date, query)

        try:
            result = self.service.events().list(
                calendarId="primary",
                timeMin=start_date,
                timeMax=end_date,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
                q=query,
            ).execute()

            events = result.get("items", [])
            items = [
                {
                    "id": e.get("id"),
                    "title": e.get("summary", "Untitled"),
                    "start": e.get("start", {}).get("dateTime", e.get("start", {}).get("date")),
                    "end":   e.get("end",   {}).get("dateTime", e.get("end",   {}).get("date")),
                    "location": e.get("location"),
                    "description": e.get("description"),
                    "attendees": [a.get("email") for a in e.get("attendees", [])],
                    "link": e.get("htmlLink"),
                }
                for e in events
            ]
            return {
                "success": True,
                "events": items,
                "count": len(items),
                "message": self._format_events(items),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Check conflicts ────────────────────────────────────────────────────

    async def check_conflicts(
        self, start_time: str, end_time: str
    ) -> Dict[str, Any]:
        """Check if a time slot has existing events."""
        result = await self.search_events(start_date=start_time, end_date=end_time)
        if not result.get("success"):
            return result
        conflicts = result.get("events", [])
        return {
            "success": True,
            "has_conflict": len(conflicts) > 0,
            "conflicts": conflicts,
            "message": (
                f"Conflict: {conflicts[0]['title']} already at that time."
                if conflicts else "Time slot is free."
            ),
        }

    # ── Delete event ───────────────────────────────────────────────────────

    async def delete_event(self, event_id: str) -> Dict[str, Any]:
        if self.is_mock:
            self.mock_events = [e for e in self.mock_events if e["id"] != event_id]
            return {"success": True, "message": f"Deleted event {event_id}"}
        try:
            self.service.events().delete(
                calendarId="primary", eventId=event_id
            ).execute()
            return {"success": True, "message": f"Deleted event {event_id}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Mock implementations ───────────────────────────────────────────────

    def _mock_create(self, title, start_time, end_time, attendees, description, location):
        event_id = f"mock_evt_{self._mock_id_counter}"
        self._mock_id_counter += 1
        event = {
            "id": event_id, "title": title,
            "start": start_time, "end": end_time,
            "attendees": attendees or [],
            "description": description, "location": location,
            "mock": True,
        }
        self.mock_events.append(event)
        attendee_str = f" with {', '.join(attendees)}" if attendees else ""
        return {
            "success": True, **event,
            "message": f"[Mock] Created '{title}'{attendee_str} at {start_time}",
        }

    def _mock_search(self, start_date, end_date, query):
        results = [
            e for e in self.mock_events
            if e.get("start", "") >= start_date[:10]
        ]
        if query:
            results = [e for e in results if query.lower() in e.get("title", "").lower()]
        return {
            "success": True,
            "events": results,
            "count": len(results),
            "message": self._format_events(results),
            "mock": True,
        }

    def _format_events(self, events: List[Dict]) -> str:
        if not events:
            return "No events found in that time range."
        lines = [f"Found {len(events)} event(s):"]
        for e in events:
            start = e.get("start", "")[:16].replace("T", " ")
            lines.append(f"  • {e['title']} — {start}")
            if e.get("location"):
                lines.append(f"    📍 {e['location']}")
        return "\n".join(lines)