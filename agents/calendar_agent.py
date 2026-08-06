"""
Calendar Agent
Google Calendar integration with OAuth2.
Supports: create, search, delete events, conflict detection.
Falls back to mock mode if credentials not configured.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import GOOGLE_CREDENTIALS_PATH, GOOGLE_TOKEN_PATH, GOOGLE_SCOPES
from config.settings import google_call

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



# Interactive OAuth opens a browser and blocks until a human finishes the
# consent screen. Both agents are built lazily, from a property first touched
# inside an async request handler — so running the flow there froze the entire
# event loop (every request, every WebSocket, the reminder scheduler) until
# someone clicked, and never returned at all on a headless deploy.
#
# It is now opt-in: the CLI and POST /google/reauth set this, the server does
# not. Without it a missing token means mock mode and a clear auth_error,
# which the adapters surface as DEGRADED rather than silently pretending.
import os as _os_oauth

def _rfc3339(dt: datetime) -> str:
    """RFC-3339 with a trailing Z, as timeMin/timeMax want."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _interactive_oauth_allowed() -> bool:
    return _os_oauth.getenv("JARVIS_INTERACTIVE_OAUTH", "").strip().lower() in (
        "1", "true", "yes", "on")


def _persist_token(path, creds) -> None:
    """Write the token with owner-only permissions.

    It carries a refresh token for the user's mail and calendar; the default
    umask leaves it world-readable (0644).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(creds.to_json())
    try:
        path.chmod(0o600)
    except OSError:
        pass  # non-POSIX filesystem — content is written either way


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
        # Captured so the UI / `/google/status` endpoint can show the user
        # exactly why auth failed instead of a generic "mock mode" label.
        self.auth_error: Optional[str] = None
        self._init_service()

    def _init_service(self) -> None:
        if not GOOGLE_AVAILABLE:
            self.auth_error = "google-api-python-client not installed"
            print(f"📅 CalendarAgent: {self.auth_error} — mock mode")
            return

        if not GOOGLE_CREDENTIALS_PATH.exists():
            self.auth_error = f"credentials.json not found at {GOOGLE_CREDENTIALS_PATH}"
            print(f"📅 CalendarAgent: {self.auth_error} — mock mode")
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
                    # Refresh the expired access token using the stored
                    # refresh_token. Save the refreshed credentials back
                    # to disk immediately so the next restart picks up
                    # the new access token rather than refreshing again.
                    try:
                        creds.refresh(Request())
                        print("📅 CalendarAgent: refreshed expired token")
                        _persist_token(GOOGLE_TOKEN_PATH, creds)
                    except Exception as refresh_exc:
                        # Refresh tokens for Google "Testing" mode apps
                        # expire after 7 days. Surface this so the user
                        # knows to run the OAuth flow again.
                        self.auth_error = (
                            f"token refresh failed ({type(refresh_exc).__name__}: "
                            f"{refresh_exc}). Delete token.json and restart to "
                            f"trigger fresh OAuth — or move the app to 'Production' "
                            f"in Google Cloud Console to stop refresh tokens "
                            f"expiring every 7 days."
                        )
                        print(f"📅 CalendarAgent: {self.auth_error}")
                        return
                elif not _interactive_oauth_allowed():
                    self.auth_error = (
                        "no valid token and interactive OAuth is disabled here. "
                        "Run the CLI (python3 main.py) or POST /google/reauth to "
                        "authorise; set JARVIS_INTERACTIVE_OAUTH=true to allow the "
                        "browser flow from this process."
                    )
                    print(f"📅 CalendarAgent: {self.auth_error}")
                    return
                else:
                    # No valid creds at all — interactive OAuth. Only reached
                    # when explicitly permitted, because it blocks on a human.
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(GOOGLE_CREDENTIALS_PATH), GOOGLE_SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                    _persist_token(GOOGLE_TOKEN_PATH, creds)
                    print("📅 CalendarAgent: completed fresh OAuth flow")

            self.service = build("calendar", "v3", credentials=creds)
            self.auth_error = None
            print("📅 CalendarAgent: Google Calendar connected ✅")
        except Exception as e:
            self.auth_error = f"{type(e).__name__}: {e}"
            print(f"📅 CalendarAgent: Auth failed — {self.auth_error}")
            print("   Tip: delete token.json and restart to re-auth.")

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
            # Detect timezone from the datetime string or use local
            import os
            local_tz = os.environ.get("TZ", "Europe/London")

            body = {
                "summary": title,
                "description": description,
                "location": location,
                "start": {"dateTime": start_time, "timeZone": local_tz},
                "end":   {"dateTime": end_time,   "timeZone": local_tz},
            }
            if attendees:
                body["attendees"] = [{"email": e} for e in attendees]

            event = await google_call(lambda: self.service.events().insert(
                calendarId="primary",
                body=body,
                sendUpdates="all" if attendees else "none",
            ).execute())

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
        # utcnow() is deprecated in 3.12+ and returns a NAIVE datetime.
        # Note the format helper: an aware datetime's isoformat() already ends
        # in "+00:00", so the old `+ "Z"` would now emit "...+00:00Z", which
        # the Calendar API rejects.
        now = datetime.now(timezone.utc)
        start_date = start_date or _rfc3339(now)
        end_date   = end_date   or _rfc3339(now + timedelta(days=7))

        if self.is_mock:
            return self._mock_search(start_date, end_date, query)

        try:
            result = await google_call(lambda: self.service.events().list(
                calendarId="primary",
                timeMin=start_date,
                timeMax=end_date,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
                q=query,
            ).execute())

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
            await google_call(lambda: self.service.events().delete(
                calendarId="primary", eventId=event_id
            ).execute())
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