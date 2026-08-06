"""Tests for the connected-service adapters (Gmail, Calendar, Spotify).

The behaviour that matters most here is mock mode. Both Google agents fall back
to fabricated data when OAuth fails and report ``success: True`` while doing it,
so nothing downstream can tell real mail from seeded mail. These tests pin the
distinction.

No credentials, no network.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from core.adapters.connected import CalendarAdapter, GmailAdapter, SpotifyAdapter
from core.tool import ErrorType, HealthStatus


# ── Doubles ─────────────────────────────────────────────────────────────────


class FakeGmail:
    def __init__(self, *, mock: bool = False, auth_error: Optional[str] = None,
                 payload: Optional[Dict[str, Any]] = None):
        self.is_mock = mock
        self.auth_error = auth_error
        self.payload = payload
        self.sent: List[Dict[str, Any]] = []

    def _ok(self, extra):
        return self.payload if self.payload is not None else {"success": True, **extra}

    async def get_inbox(self, max_results=5, query="is:unread"):
        return self._ok({"emails": [{"id": "1", "subject": "Hi"}]})

    async def search_emails(self, query, max_results=5):
        return self._ok({"emails": []})

    async def get_email_body(self, email_id):
        return self._ok({"body": "the body"})

    async def get_thread(self, thread_id):
        return self._ok({"messages": [{"id": "1"}, {"id": "2"}]})

    async def send_email(self, to, subject, body, cc=None):
        self.sent.append({"to": to, "subject": subject, "cc": cc})
        return self._ok({"message": f"Sent to {to}"})

    async def draft_email(self, to, subject, body):
        return self._ok({"message": "Draft saved."})

    async def mark_as_read(self, email_id):
        return self._ok({"message": "Marked as read."})

    async def archive_email(self, email_id):
        return self._ok({"message": "Archived."})


class FakeCalendar:
    def __init__(self, *, mock: bool = False, auth_error: Optional[str] = None,
                 payload: Optional[Dict[str, Any]] = None):
        self.is_mock = mock
        self.auth_error = auth_error
        self.payload = payload
        self.created: List[Dict[str, Any]] = []

    def _ok(self, extra):
        return self.payload if self.payload is not None else {"success": True, **extra}

    async def search_events(self, start_date=None, end_date=None, query=None,
                            max_results=10):
        return self._ok({"events": [{"id": "e1", "summary": "Standup"}]})

    async def create_event(self, title, start_time, end_time, attendees=None,
                           description=None, location=None):
        self.created.append({"title": title, "attendees": attendees})
        return self._ok({"message": f"Created '{title}'", "event_id": "e9"})

    async def check_conflicts(self, start, end):
        return self._ok({"conflicts": []})

    async def delete_event(self, event_id):
        return self._ok({"message": f"Deleted event {event_id}"})


class FakeSpotify:
    def __init__(self, *, token="tok", devices=None, payload=None):
        self._token = token
        self._devices = devices if devices is not None else [{"id": "d1"}]
        self.payload = payload
        self.calls: List[str] = []

    def _ok(self, extra):
        return self.payload if self.payload is not None else {"success": True, **extra}

    async def _get_token(self):
        return self._token

    async def get_now_playing(self):
        return self._ok({"track": "Song", "artist": "Someone"})

    async def search(self, query, search_type="track", limit=10):
        return self._ok({"items": [{"name": "Song"}]})

    async def play(self, uri=None):
        self.calls.append(f"play:{uri}")
        return self._ok({"message": "Playing"})

    async def play_by_name(self, query):
        self.calls.append(f"play_by_name:{query}")
        return self._ok({"message": f"Playing {query}"})

    async def pause(self):
        self.calls.append("pause")
        return self._ok({"message": "Paused"})

    async def skip(self):
        self.calls.append("skip")
        return self._ok({})

    async def previous(self):
        self.calls.append("previous")
        return self._ok({})

    async def set_volume(self, level):
        self.calls.append(f"volume:{level}")
        return self._ok({})

    async def get_devices(self):
        return self._ok({"devices": self._devices})

    def format_now_playing(self, d):
        return f"{d.get('track')} — {d.get('artist')}"

    def format_search(self, d):
        return f"{len(d.get('items', []))} result(s)"

    def format_play_result(self, d):
        return d.get("message", "")


# ── Mock mode ───────────────────────────────────────────────────────────────


async def test_real_gmail_results_are_not_flagged_degraded():
    r = await GmailAdapter(FakeGmail()).execute("get_inbox")
    assert r.success is True
    assert r.degraded is False
    assert "mock" not in r.meta


async def test_mock_gmail_results_are_flagged_degraded_with_the_auth_error():
    """Mock mode currently returns success:True with fabricated mail and no
    signal. The LLM then answers confidently about email that doesn't exist."""
    agent = FakeGmail(mock=True, auth_error="token refresh failed (invalid_grant)")
    r = await GmailAdapter(agent).execute("get_inbox")
    assert r.success is True
    assert r.degraded is True
    assert r.meta["mock"] is True
    assert "invalid_grant" in r.meta["auth_error"]


async def test_mock_gmail_health_is_degraded_not_ok():
    h = await GmailAdapter(FakeGmail(mock=True, auth_error="no credentials")).health_check()
    assert h.status is HealthStatus.DEGRADED
    assert h.healthy is True          # usable, but not the real thing
    assert "mock" in h.detail


async def test_mock_calendar_is_flagged_too():
    r = await CalendarAdapter(FakeCalendar(mock=True)).execute("search_events")
    assert r.degraded is True
    assert r.meta["mock"] is True


async def test_live_gmail_health_probes_the_api():
    assert (await GmailAdapter(FakeGmail()).health_check()).status is HealthStatus.OK

    broken = FakeGmail(payload={"success": False, "error": "HTTP 500"})
    h = await GmailAdapter(broken).health_check()
    assert h.status is HealthStatus.ERROR
    assert "500" in h.detail


# ── Auth errors are non-retryable ───────────────────────────────────────────


@pytest.mark.parametrize("err", [
    "invalid_grant: Token has been expired or revoked",
    "HTTP 401 Unauthorized",
    "credentials.json not found",
    "oauth flow failed",
])
async def test_auth_failures_classify_as_auth_and_are_not_retried(err):
    a = GmailAdapter(FakeGmail(payload={"success": False, "error": err}))
    r = await a.execute("get_inbox")
    assert r.success is False
    assert r.error_type is ErrorType.AUTH
    assert r.retryable is False


async def test_transient_gmail_failure_is_retryable():
    a = GmailAdapter(FakeGmail(payload={"success": False, "error": "HTTP 503 backend error"}))
    r = await a.execute("get_inbox")
    assert r.error_type is ErrorType.UPSTREAM
    assert r.retryable is True


# ── Recipient validation ────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["notanemail", "a@b", "@example.com", "sarah@", ""])
async def test_send_rejects_malformed_addresses_before_hitting_the_api(bad):
    agent = FakeGmail()
    r = await GmailAdapter(agent).execute(
        "send_email", {"to": bad, "subject": "Hi", "body": "text"})
    assert r.success is False
    assert r.error_type is ErrorType.INPUT
    assert agent.sent == [], "a malformed address reached the Gmail API"


async def test_send_accepts_a_list_and_a_comma_separated_string():
    agent = FakeGmail()
    a = GmailAdapter(agent)

    assert (await a.execute("send_email", {
        "to": ["x@example.com", "y@example.com"],
        "subject": "S", "body": "B"})).success
    assert (await a.execute("send_email", {
        "to": "x@example.com, y@example.com", "subject": "S", "body": "B"})).success
    assert len(agent.sent) == 2
    assert agent.sent[0]["to"] == "x@example.com,y@example.com"


async def test_send_rejects_an_empty_subject():
    r = await GmailAdapter(FakeGmail()).execute(
        "send_email", {"to": "a@b.com", "subject": "   ", "body": "x"})
    assert r.error_type is ErrorType.INPUT


async def test_send_and_archive_are_marked_destructive():
    a = GmailAdapter(FakeGmail())
    assert a.actions["send_email"].destructive is True
    assert a.actions["archive_email"].destructive is True
    assert a.actions["draft_email"].destructive is False
    assert a.actions["get_inbox"].read_only is True


# ── Calendar ────────────────────────────────────────────────────────────────


async def test_create_event_rejects_end_before_start():
    """A common LLM slip is emitting the same timestamp twice."""
    agent = FakeCalendar()
    r = await CalendarAdapter(agent).execute("create_event", {
        "title": "Sync", "start_time": "2026-07-28T15:00:00",
        "end_time": "2026-07-28T14:00:00"})
    assert r.success is False
    assert r.error_type is ErrorType.INPUT
    assert agent.created == []


async def test_create_event_rejects_identical_times():
    r = await CalendarAdapter(FakeCalendar()).execute("create_event", {
        "title": "Sync", "start_time": "2026-07-28T14:00:00",
        "end_time": "2026-07-28T14:00:00"})
    assert r.error_type is ErrorType.INPUT


async def test_create_event_validates_attendees():
    agent = FakeCalendar()
    r = await CalendarAdapter(agent).execute("create_event", {
        "title": "Sync", "start_time": "2026-07-28T14:00:00",
        "end_time": "2026-07-28T15:00:00", "attendees": "not-an-email"})
    assert r.error_type is ErrorType.INPUT
    assert agent.created == []


async def test_create_event_happy_path():
    agent = FakeCalendar()
    r = await CalendarAdapter(agent).execute("create_event", {
        "title": "Sync", "start_time": "2026-07-28T14:00:00",
        "end_time": "2026-07-28T15:00:00",
        "attendees": ["a@example.com"]})
    assert r.success
    assert agent.created[0]["attendees"] == ["a@example.com"]


async def test_check_conflicts_message_reads_naturally():
    free = await CalendarAdapter(FakeCalendar()).execute("check_conflicts", {
        "start_time": "2026-07-28T14:00:00", "end_time": "2026-07-28T15:00:00"})
    assert free.message == "That slot is free."

    busy = FakeCalendar(payload={"success": True, "conflicts": [{"id": "e1"}]})
    clash = await CalendarAdapter(busy).execute("check_conflicts", {
        "start_time": "2026-07-28T14:00:00", "end_time": "2026-07-28T15:00:00"})
    assert "1 clash" in clash.message


async def test_calendar_writes_are_destructive():
    a = CalendarAdapter(FakeCalendar())
    assert a.actions["create_event"].destructive is True
    assert a.actions["delete_event"].destructive is True
    assert a.actions["search_events"].read_only is True


# ── Spotify ─────────────────────────────────────────────────────────────────


async def test_spotify_play_prefers_uri_over_query():
    t = FakeSpotify()
    await SpotifyAdapter(t).execute("play", {"uri": "spotify:track:x", "query": "song"})
    assert t.calls == ["play:spotify:track:x"]


async def test_spotify_play_by_name_when_only_a_query_given():
    t = FakeSpotify()
    await SpotifyAdapter(t).execute("play", {"query": "wonderwall"})
    assert t.calls == ["play_by_name:wonderwall"]


async def test_spotify_bare_play_resumes():
    t = FakeSpotify()
    await SpotifyAdapter(t).execute("play")
    assert t.calls == ["play:None"]


async def test_spotify_volume_bounds():
    a = SpotifyAdapter(FakeSpotify())
    assert (await a.execute("set_volume", {"level": 150})).error_type is ErrorType.INPUT
    assert (await a.execute("set_volume", {"level": 55})).success


async def test_spotify_search_type_is_constrained():
    a = SpotifyAdapter(FakeSpotify())
    bad = await a.execute("search", {"query": "x", "search_type": "podcast"})
    assert bad.error_type is ErrorType.INPUT
    assert (await a.execute("search", {"query": "x", "search_type": "album"})).success


async def test_spotify_health_unauthenticated_is_unavailable():
    h = await SpotifyAdapter(FakeSpotify(token=None)).health_check()
    assert h.status is HealthStatus.UNAVAILABLE
    assert "OAuth" in h.detail or "authenticated" in h.detail


async def test_spotify_health_no_device_is_degraded_not_broken():
    """Authenticated and reachable but nothing to play on — a distinct state
    that reads as 'broken' if you only have ok/error."""
    h = await SpotifyAdapter(FakeSpotify(devices=[])).health_check()
    assert h.status is HealthStatus.DEGRADED
    assert "no active playback device" in h.detail


async def test_spotify_health_ok_with_devices():
    h = await SpotifyAdapter(FakeSpotify()).health_check()
    assert h.status is HealthStatus.OK


async def test_spotify_token_refresh_explosion_is_contained():
    class Exploding(FakeSpotify):
        async def _get_token(self):
            raise RuntimeError("refresh endpoint 500")

    h = await SpotifyAdapter(Exploding()).health_check()
    assert h.status is HealthStatus.ERROR
    assert "refresh endpoint 500" in h.detail
