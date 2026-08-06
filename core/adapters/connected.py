"""
Adapters — connected services (Gmail, Google Calendar, Spotify).

These are the tools with credentials, and they behave differently from the
rest in three ways the adapter has to be explicit about:

* **Mock mode is a real state, not a success.** ``GmailAgent`` and
  ``CalendarAgent`` fall back to a seeded mock inbox/calendar whenever OAuth
  fails, and return ``success: True`` while doing it. Nothing downstream can
  currently tell "your inbox" from "three fabricated emails" — the health
  dashboard shows green and the LLM answers confidently about mail that
  doesn't exist. Here mock results are marked ``degraded`` with the auth error
  attached, and ``health_check`` reports DEGRADED rather than OK.

* **Sending mail and creating events are destructive.** They're flagged, so
  the MCP layer and the confirmation gate can see it without a hardcoded list.

* **Auth failures must not be retried.** They come back as AUTH, which the
  circuit breaker treats as non-retryable — retrying an expired refresh token
  fourteen times just delays the error the user needs to see.

The OAuth *refresh* path itself is not fixed here: ``InstalledAppFlow.
run_local_server()`` blocks the event loop and waits indefinitely for a browser
(audit item 1.6). That's a change inside ``GmailAgent``/``CalendarAgent``, in
the Severity 1 pass.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.tool import (
    Action,
    BaseTool,
    HealthReport,
    HealthStatus,
    ToolAuthError,
    ToolInputError,
    ToolNotFoundError,
    ToolResult,
    ToolUpstreamError,
)

_EMAIL_RE = re.compile(r"^[^@\s,]+@[^@\s,]+\.[^@\s,]+$")
_ISO_HINT = "ISO-8601, e.g. 2026-07-28T14:00:00"


def _unwrap(payload: Dict[str, Any], *, what: str) -> Dict[str, Any]:
    if payload.get("success"):
        return payload
    err = str(payload.get("error") or f"{what} failed")
    low = err.lower()
    if any(k in low for k in ("credential", "token", "oauth", "unauthor",
                              "invalid_grant", "401", "403")):
        raise ToolAuthError(err)
    if "not found" in low or "notfound" in low or "404" in low:
        raise ToolNotFoundError(err)
    if "invalid" in low or "malformed" in low or "400" in low:
        raise ToolInputError(err)
    raise ToolUpstreamError(err)


def _validate_recipients(value: Any, field: str) -> List[str]:
    """Accept a string or list; reject anything that isn't a plausible address.

    Worth doing at the boundary: ``to`` reaches ``send_email`` straight from an
    LLM-generated plan, and a malformed address currently surfaces as an opaque
    Gmail API 400 after the draft has already been composed.
    """
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",")]
    elif isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value]
    else:
        raise ToolInputError(f"{field} must be a string or list of addresses")
    parts = [p for p in parts if p]
    if not parts:
        raise ToolInputError(f"{field} must contain at least one address")
    bad = [p for p in parts if not _EMAIL_RE.match(p)]
    if bad:
        raise ToolInputError(f"{field} contains invalid address(es): {', '.join(bad)}")
    return parts


class _ConnectedTool(BaseTool):
    """Shared mock-mode handling for the Google agents."""

    def __init__(self, agent: Any) -> None:
        self._a = agent
        super().__init__()

    @property
    def _is_mock(self) -> bool:
        return bool(getattr(self._a, "is_mock", False))

    @property
    def _auth_error(self) -> Optional[str]:
        return getattr(self._a, "auth_error", None)

    def _result(self, action: str, data: Dict[str, Any], message: str) -> ToolResult:
        """Tag anything produced in mock mode so callers can tell it apart."""
        if self._is_mock:
            return ToolResult.ok(
                self.name, action, data=data, message=message, degraded=True,
                meta={"mock": True, "auth_error": self._auth_error or "not authenticated"},
            )
        return ToolResult.ok(self.name, action, data=data, message=message)

    async def _check_health(self) -> HealthReport:
        if self._is_mock:
            return HealthReport(
                HealthStatus.DEGRADED, self.name,
                f"running on mock data — {self._auth_error or 'not authenticated'}",
            )
        try:
            ok, detail = await self._probe()
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name,
                                f"{type(exc).__name__}: {exc}")
        return HealthReport(HealthStatus.OK if ok else HealthStatus.ERROR,
                            self.name, detail)

    async def _probe(self) -> tuple[bool, str]:  # pragma: no cover - overridden
        return True, "no probe defined"


# ── Gmail ───────────────────────────────────────────────────────────────────


class GmailAdapter(_ConnectedTool):
    _name = "gmail"
    _description = "Read, search, send, reply to, draft and archive email."

    def _register_actions(self) -> None:
        email_id = {"type": "string", "minLength": 1, "description": "Gmail message id."}
        body = {"type": "string", "description": "Plain-text body."}

        self.add_action(Action(
            name="get_inbox", description="Recent messages. Defaults to unread only.",
            input_schema={"properties": {
                "max_results": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                "query": {"type": "string", "default": "is:unread",
                          "description": "Gmail search syntax, e.g. 'is:unread from:sarah'."},
            }},
            handler=self._inbox, timeout=30.0,
        ))
        self.add_action(Action(
            name="search_emails", description="Search the mailbox.",
            input_schema={"properties": {
                "query": {"type": "string", "minLength": 1},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
            }, "required": ["query"]},
            handler=self._search, timeout=30.0,
        ))
        self.add_action(Action(
            name="get_email_body", description="Full body of one message.",
            input_schema={"properties": {"email_id": email_id}, "required": ["email_id"]},
            handler=self._body, timeout=30.0,
        ))
        self.add_action(Action(
            name="get_thread", description="Every message in a thread.",
            input_schema={"properties": {"thread_id": {"type": "string", "minLength": 1}},
                          "required": ["thread_id"]},
            handler=self._thread, timeout=30.0,
        ))
        self.add_action(Action(
            name="send_email", description="Send an email immediately.",
            input_schema={"properties": {
                "to": {"description": "Address, comma-separated list, or array."},
                "subject": {"type": "string"},
                "body": body,
                "cc": {"description": "Optional cc recipients."},
            }, "required": ["to", "subject", "body"]},
            handler=self._send, timeout=45.0, read_only=False, destructive=True,
        ))
        self.add_action(Action(
            name="draft_email", description="Save a draft without sending.",
            input_schema={"properties": {
                "to": {"description": "Address, comma-separated list, or array."},
                "subject": {"type": "string"},
                "body": body,
            }, "required": ["to", "subject", "body"]},
            handler=self._draft, timeout=45.0, read_only=False,
        ))
        self.add_action(Action(
            name="mark_as_read", description="Mark one message read.",
            input_schema={"properties": {"email_id": email_id}, "required": ["email_id"]},
            handler=self._mark_read, timeout=20.0, read_only=False,
        ))
        self.add_action(Action(
            name="archive_email", description="Archive a message.",
            input_schema={"properties": {"email_id": email_id}, "required": ["email_id"]},
            handler=self._archive, timeout=20.0, read_only=False, destructive=True,
        ))

    async def _inbox(self, max_results: int = 5, query: str = "is:unread"):
        d = _unwrap(await self._a.get_inbox(max_results=max_results, query=query),
                    what="inbox read")
        emails = d.get("emails", [])
        return self._result("get_inbox", d, f"{len(emails)} message(s).")

    async def _search(self, query: str, max_results: int = 5):
        d = _unwrap(await self._a.search_emails(query=query, max_results=max_results),
                    what="email search")
        return self._result("search_emails", d,
                            f"{len(d.get('emails', []))} match(es) for {query!r}.")

    async def _body(self, email_id: str):
        d = _unwrap(await self._a.get_email_body(email_id), what="email body")
        return self._result("get_email_body", d, str(d.get("body", "")))

    async def _thread(self, thread_id: str):
        d = _unwrap(await self._a.get_thread(thread_id), what="thread read")
        return self._result("get_thread", d,
                            f"{len(d.get('messages', []))} message(s) in thread.")

    async def _send(self, to: Any, subject: str, body: str, cc: Any = None):
        recipients = _validate_recipients(to, "to")
        cc_list = _validate_recipients(cc, "cc") if cc else None
        if not subject.strip():
            raise ToolInputError("subject must not be empty")
        d = _unwrap(await self._a.send_email(
            to=",".join(recipients), subject=subject, body=body,
            cc=",".join(cc_list) if cc_list else None), what="send")
        return self._result("send_email", d,
                            d.get("message") or f"Sent to {', '.join(recipients)}.")

    async def _draft(self, to: Any, subject: str, body: str):
        recipients = _validate_recipients(to, "to")
        d = _unwrap(await self._a.draft_email(
            to=",".join(recipients), subject=subject, body=body), what="draft")
        return self._result("draft_email", d, d.get("message") or "Draft saved.")

    async def _mark_read(self, email_id: str):
        d = _unwrap(await self._a.mark_as_read(email_id), what="mark read")
        return self._result("mark_as_read", d, d.get("message") or "Marked as read.")

    async def _archive(self, email_id: str):
        d = _unwrap(await self._a.archive_email(email_id), what="archive")
        return self._result("archive_email", d, d.get("message") or "Archived.")

    async def _probe(self) -> tuple[bool, str]:
        d = await self._a.get_inbox(max_results=1, query="is:unread")
        if d.get("success"):
            return True, "gmail api responding"
        return False, str(d.get("error", "unknown"))


# ── Calendar ────────────────────────────────────────────────────────────────


class CalendarAdapter(_ConnectedTool):
    _name = "calendar"
    _description = "Read, create and delete Google Calendar events, and check for clashes."

    def _register_actions(self) -> None:
        when = {"type": "string", "description": _ISO_HINT}

        self.add_action(Action(
            name="search_events", description="Events in a date range, optionally filtered.",
            input_schema={"properties": {
                "start_date": {"type": "string", "description": "ISO date or datetime."},
                "end_date": {"type": "string", "description": "ISO date or datetime."},
                "query": {"type": "string", "description": "Free-text filter."},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
            }},
            handler=self._search, timeout=30.0,
        ))
        self.add_action(Action(
            name="create_event", description="Create a calendar event.",
            input_schema={"properties": {
                "title": {"type": "string", "minLength": 1},
                "start_time": when,
                "end_time": when,
                "attendees": {"description": "Address, comma-separated list, or array."},
                "description": {"type": "string"},
                "location": {"type": "string"},
            }, "required": ["title", "start_time", "end_time"]},
            handler=self._create, timeout=45.0, read_only=False, destructive=True,
        ))
        self.add_action(Action(
            name="check_conflicts", description="Whether anything already occupies a slot.",
            input_schema={"properties": {"start_time": when, "end_time": when},
                          "required": ["start_time", "end_time"]},
            handler=self._conflicts, timeout=30.0,
        ))
        self.add_action(Action(
            name="delete_event", description="Delete an event by id.",
            input_schema={"properties": {"event_id": {"type": "string", "minLength": 1}},
                          "required": ["event_id"]},
            handler=self._delete, timeout=30.0, read_only=False, destructive=True,
        ))

    async def _search(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                      query: Optional[str] = None, max_results: int = 10):
        d = _unwrap(await self._a.search_events(
            start_date=start_date, end_date=end_date, query=query,
            max_results=max_results), what="calendar search")
        events = d.get("events", [])
        return self._result("search_events", d, f"{len(events)} event(s).")

    async def _create(self, title: str, start_time: str, end_time: str,
                      attendees: Any = None, description: Optional[str] = None,
                      location: Optional[str] = None):
        if not start_time or not end_time:
            raise ToolInputError(f"start_time and end_time are required ({_ISO_HINT})")
        if start_time >= end_time:
            # Cheap lexical check — ISO-8601 sorts correctly as text. Catches
            # the common LLM slip of emitting the same value twice.
            raise ToolInputError(
                f"end_time ({end_time}) must be after start_time ({start_time})")
        guests = _validate_recipients(attendees, "attendees") if attendees else None
        d = _unwrap(await self._a.create_event(
            title=title, start_time=start_time, end_time=end_time,
            attendees=guests, description=description, location=location),
            what="event creation")
        return self._result("create_event", d,
                            d.get("message") or f"Created '{title}'.")

    async def _conflicts(self, start_time: str, end_time: str):
        d = _unwrap(await self._a.check_conflicts(start_time, end_time),
                    what="conflict check")
        clashes = d.get("conflicts", [])
        msg = (f"{len(clashes)} clash(es) in that slot." if clashes
               else "That slot is free.")
        return self._result("check_conflicts", d, msg)

    async def _delete(self, event_id: str):
        d = _unwrap(await self._a.delete_event(event_id), what="event deletion")
        return self._result("delete_event", d, d.get("message") or "Event deleted.")

    async def _probe(self) -> tuple[bool, str]:
        d = await self._a.search_events(max_results=1)
        if d.get("success"):
            return True, "calendar api responding"
        return False, str(d.get("error", "unknown"))


# ── Spotify ─────────────────────────────────────────────────────────────────


class SpotifyAdapter(BaseTool):
    _name = "spotify"
    _description = "Search and control Spotify playback: play, pause, skip, volume, queue."

    def __init__(self, tool: Any) -> None:
        self._t = tool
        super().__init__()

    def _register_actions(self) -> None:
        self.add_action(Action(
            name="get_now_playing", description="What's currently playing.",
            input_schema={"properties": {}}, handler=self._now, timeout=20.0,
        ))
        self.add_action(Action(
            name="search", description="Search tracks, albums, artists or playlists.",
            input_schema={"properties": {
                "query": {"type": "string", "minLength": 1},
                "search_type": {"type": "string",
                                "enum": ["track", "album", "artist", "playlist"],
                                "default": "track"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            }, "required": ["query"]},
            handler=self._search, timeout=25.0,
        ))
        self.add_action(Action(
            name="play",
            description="Play a track by name, by URI, or resume what's paused.",
            input_schema={"properties": {
                "query": {"type": "string", "description": "Song or artist name."},
                "uri": {"type": "string", "description": "Spotify URI, e.g. spotify:track:..."},
            }},
            handler=self._play, timeout=40.0, read_only=False,
        ))
        for name, desc in (("pause", "Pause playback."),
                           ("skip", "Skip to the next track."),
                           ("previous", "Go back to the previous track.")):
            self.add_action(Action(
                name=name, description=desc, input_schema={"properties": {}},
                handler=self._simple(name), timeout=20.0, read_only=False,
            ))
        self.add_action(Action(
            name="set_volume", description="Set playback volume 0–100.",
            input_schema={"properties": {"level": {"type": "integer", "minimum": 0,
                                                   "maximum": 100}},
                          "required": ["level"]},
            handler=self._volume, timeout=20.0, read_only=False,
        ))
        self.add_action(Action(
            name="get_devices", description="Available Spotify playback devices.",
            input_schema={"properties": {}}, handler=self._devices, timeout=20.0,
        ))

    async def _now(self):
        d = await self._t.get_now_playing()
        if not d.get("success") and d.get("error"):
            _unwrap(d, what="now playing")
        return d, self._t.format_now_playing(d)

    async def _search(self, query: str, search_type: str = "track", limit: int = 10):
        d = _unwrap(await self._t.search(query, search_type=search_type, limit=limit),
                    what="spotify search")
        return d, self._t.format_search(d)

    async def _play(self, query: Optional[str] = None, uri: Optional[str] = None):
        if uri:
            d = await self._t.play(uri)
        elif query:
            d = await self._t.play_by_name(query)
        else:
            d = await self._t.play()
        _unwrap(d, what="playback")
        return d, self._t.format_play_result(d)

    def _simple(self, method: str):
        async def _call():
            d = await getattr(self._t, method)()
            if isinstance(d, dict):
                _unwrap(d, what=f"spotify {method}")
                return d, d.get("message", f"{method.title()}d.")
            return {"ok": True}, f"{method.title()}d."
        _call.__name__ = f"_spotify_{method}"
        return _call

    async def _volume(self, level: int):
        d = await self._t.set_volume(level)
        if isinstance(d, dict):
            _unwrap(d, what="set volume")
        return d, f"Volume set to {level}%."

    async def _devices(self):
        d = _unwrap(await self._t.get_devices(), what="device list")
        devices = d.get("devices", [])
        return d, f"{len(devices)} device(s) available."

    async def _check_health(self) -> HealthReport:
        try:
            token = await self._t._get_token()
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name,
                                f"token refresh failed: {type(exc).__name__}: {exc}")
        if not token:
            return HealthReport(HealthStatus.UNAVAILABLE, self.name,
                                "not authenticated — run the Spotify OAuth flow")
        try:
            d = await self._t.get_devices()
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name, f"{type(exc).__name__}: {exc}")
        if not d.get("success"):
            return HealthReport(HealthStatus.ERROR, self.name, str(d.get("error")))
        if not d.get("devices"):
            # Authenticated and reachable, but nothing to play on. Distinct
            # from broken — worth its own state on the dashboard.
            return HealthReport(HealthStatus.DEGRADED, self.name,
                                "authenticated, but no active playback device")
        return HealthReport(HealthStatus.OK, self.name,
                            f"{len(d['devices'])} device(s) available")


__all__ = ["GmailAdapter", "CalendarAdapter", "SpotifyAdapter"]
