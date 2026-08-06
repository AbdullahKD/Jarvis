"""Fake Gmail and Calendar services.

``GmailAgent`` and ``CalendarAgent`` talk to whatever ``build()`` returned, via
chains like ``service.users().messages().list(...).execute()``. These fakes
mimic that chain against an in-memory mailbox and calendar, so the agents' real
code — MIME assembly, header parsing, body extraction, mock-mode branching —
runs unchanged with no credentials and no network.

That matters more than it sounds. The interesting bugs in these agents are in
the *parsing*: base64url payloads, multipart walking, HTML stripping, RFC-2822
address splitting. A mock that returns a tidy dict skips exactly the code
that's worth testing, so ``FakeGmailService`` stores messages in real Gmail API
shape — nested ``payload.parts``, base64url bodies, ``headers`` as a list of
name/value pairs — and the agent does the real work of pulling them apart.
"""

from __future__ import annotations

import base64
import email.utils
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _b64(text: str) -> str:
    """Gmail's base64url, padding stripped, as the API actually returns it."""
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode().rstrip("=")


class _Request:
    """Stands in for an HttpRequest: the agents call .execute() on it, often
    inside a lambda handed to google_call()."""

    def __init__(self, result: Any, *, error: Optional[Exception] = None):
        self._result = result
        self._error = error

    def execute(self, *args, **kwargs) -> Any:
        if self._error:
            raise self._error
        return self._result


# ── Gmail ───────────────────────────────────────────────────────────────────


@dataclass
class FakeMessage:
    id: str
    thread_id: str
    sender: str
    subject: str
    body: str
    to: str = "me@example.com"
    labels: List[str] = field(default_factory=lambda: ["INBOX", "UNREAD"])
    html: bool = False

    def to_api(self, fmt: str = "full") -> Dict[str, Any]:
        headers = [
            {"name": "From", "value": self.sender},
            {"name": "To", "value": self.to},
            {"name": "Subject", "value": self.subject},
            {"name": "Date", "value": email.utils.formatdate()},
            {"name": "Message-ID", "value": f"<{self.id}@example.com>"},
        ]
        if fmt == "metadata":
            return {"id": self.id, "threadId": self.thread_id,
                    "labelIds": list(self.labels),
                    "payload": {"headers": headers},
                    "snippet": self.body[:80]}

        # Real shape: multipart/alternative with the text part nested, which
        # is what forces the agent's _extract_body to actually walk parts.
        if self.html:
            payload = {
                "mimeType": "multipart/alternative", "headers": headers,
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": _b64(self.body)}},
                    {"mimeType": "text/html",
                     "body": {"data": _b64(f"<html><body><p>{self.body}</p></body></html>")}},
                ],
            }
        else:
            payload = {"mimeType": "text/plain", "headers": headers,
                       "body": {"data": _b64(self.body)}}

        return {"id": self.id, "threadId": self.thread_id,
                "labelIds": list(self.labels), "payload": payload,
                "snippet": self.body[:80]}


class FakeGmailService:
    """In-memory Gmail. Pass to GmailAgent as `.service`."""

    def __init__(self, messages: Optional[List[FakeMessage]] = None):
        # NOT `self.messages` — that name is the API chain method
        # (`service.users().messages()`), and an attribute would shadow it.
        self.store: Dict[str, FakeMessage] = {}
        for m in messages or _default_inbox():
            self.store[m.id] = m
        self.sent: List[Dict[str, Any]] = []
        self.drafts: List[Dict[str, Any]] = []
        self.modified: List[tuple[str, Dict[str, Any]]] = []
        self.raise_on: Dict[str, Exception] = {}   # method name -> exception
        self._ids = itertools.count(1000)

    # chain entry points
    def users(self):
        return self

    def messages(self):
        return _GmailMessages(self)

    def threads(self):
        return _GmailThreads(self)

    def drafts(self):
        return _GmailDrafts(self)

    def _err(self, name: str) -> Optional[Exception]:
        return self.raise_on.get(name)

    def next_id(self) -> str:
        return f"msg_{next(self._ids)}"


class _GmailMessages:
    def __init__(self, svc: FakeGmailService):
        self.svc = svc

    def list(self, userId="me", maxResults=10, q=None, labelIds=None, **kw):
        msgs = list(self.svc.store.values())
        if q:
            ql = q.lower()
            if "is:unread" in ql:
                msgs = [m for m in msgs if "UNREAD" in m.labels]
            # crude from: filter, enough to exercise the agent's own handling
            if "from:" in ql:
                who = ql.split("from:", 1)[1].split()[0]
                msgs = [m for m in msgs if who in m.sender.lower()]
        msgs = msgs[:maxResults]
        return _Request(
            {"messages": [{"id": m.id, "threadId": m.thread_id} for m in msgs],
             "resultSizeEstimate": len(msgs)},
            error=self.svc._err("list"))

    def get(self, userId="me", id=None, format="full", **kw):
        msg = self.svc.store.get(id)
        if msg is None:
            return _Request(None, error=_http_error(404, "Not Found"))
        return _Request(msg.to_api(format), error=self.svc._err("get"))

    def send(self, userId="me", body=None, **kw):
        mid = self.svc.next_id()
        self.svc.sent.append({"id": mid, **(body or {})})
        return _Request({"id": mid, "threadId": mid, "labelIds": ["SENT"]},
                        error=self.svc._err("send"))

    def modify(self, userId="me", id=None, body=None, **kw):
        self.svc.modified.append((id, body or {}))
        msg = self.svc.store.get(id)
        if msg:
            for label in (body or {}).get("removeLabelIds", []):
                if label in msg.labels:
                    msg.labels.remove(label)
            for label in (body or {}).get("addLabelIds", []):
                msg.labels.append(label)
        return _Request({"id": id}, error=self.svc._err("modify"))


class _GmailThreads:
    def __init__(self, svc: FakeGmailService):
        self.svc = svc

    def get(self, userId="me", id=None, format="full", **kw):
        msgs = [m for m in self.svc.store.values() if m.thread_id == id]
        if not msgs:
            return _Request(None, error=_http_error(404, "Not Found"))
        return _Request({"id": id, "messages": [m.to_api(format) for m in msgs]},
                        error=self.svc._err("thread_get"))


class _GmailDrafts:
    def __init__(self, svc: FakeGmailService):
        self.svc = svc

    def create(self, userId="me", body=None, **kw):
        did = f"draft_{len(self.svc.drafts) + 1}"
        self.svc.drafts.append({"id": did, **(body or {})})
        return _Request({"id": did, "message": {"id": self.svc.next_id()}},
                        error=self.svc._err("draft_create"))


def _default_inbox() -> List[FakeMessage]:
    return [
        FakeMessage("msg_1", "thr_1", "Sarah Khan <sarah@example.com>",
                    "Q3 numbers", "Can you send the Q3 figures before Friday?"),
        FakeMessage("msg_2", "thr_2", "no-reply@bank.example",
                    "Your statement is ready", "Your monthly statement is available.",
                    html=True),
        FakeMessage("msg_3", "thr_1", "Sarah Khan <sarah@example.com>",
                    "Re: Q3 numbers", "Thanks — got them.", labels=["INBOX"]),
    ]


def _http_error(status: int, reason: str) -> Exception:
    """A googleapiclient HttpError if the library is present, else a stand-in.

    Built lazily so these fakes stay importable in an environment without the
    Google client libraries installed.
    """
    try:
        from googleapiclient.errors import HttpError  # type: ignore

        class _Resp:
            def __init__(self, s): self.status, self.reason = s, reason
        return HttpError(_Resp(status), f'{{"error": "{reason}"}}'.encode())
    except ImportError:                               # pragma: no cover
        return RuntimeError(f"HTTP {status}: {reason}")


# ── Calendar ────────────────────────────────────────────────────────────────


@dataclass
class FakeEvent:
    id: str
    summary: str
    start: str
    end: str
    attendees: List[str] = field(default_factory=list)
    location: str = ""
    description: str = ""

    def to_api(self) -> Dict[str, Any]:
        return {
            "id": self.id, "summary": self.summary,
            "start": {"dateTime": self.start, "timeZone": "Europe/London"},
            "end": {"dateTime": self.end, "timeZone": "Europe/London"},
            "attendees": [{"email": a} for a in self.attendees],
            "location": self.location, "description": self.description,
            "htmlLink": f"https://calendar.google.com/event?eid={self.id}",
            "status": "confirmed",
        }


class FakeCalendarService:
    """In-memory Calendar. Pass to CalendarAgent as `.service`."""

    def __init__(self, events: Optional[List[FakeEvent]] = None):
        self.store: Dict[str, FakeEvent] = {e.id: e for e in (events or _default_events())}
        self.created: List[Dict[str, Any]] = []
        self.deleted: List[str] = []
        self.raise_on: Dict[str, Exception] = {}
        self._ids = itertools.count(100)

    def events(self):
        return _CalendarEvents(self)

    def calendarList(self):
        return _CalendarList(self)


class _CalendarEvents:
    def __init__(self, svc: FakeCalendarService):
        self.svc = svc

    def list(self, calendarId="primary", timeMin=None, timeMax=None, q=None,
             maxResults=10, singleEvents=True, orderBy=None, **kw):
        items = list(self.svc.store.values())
        # String comparison is valid for ISO-8601 and matches how the agent
        # builds these bounds.
        if timeMin:
            items = [e for e in items if e.end >= timeMin[:19]]
        if timeMax:
            items = [e for e in items if e.start <= timeMax[:19]]
        if q:
            items = [e for e in items if q.lower() in e.summary.lower()]
        items.sort(key=lambda e: e.start)
        return _Request({"items": [e.to_api() for e in items[:maxResults]]},
                        error=self.svc.raise_on.get("list"))

    def insert(self, calendarId="primary", body=None, sendUpdates=None, **kw):
        body = body or {}
        eid = f"evt_{next(self.svc._ids)}"
        ev = FakeEvent(
            id=eid, summary=body.get("summary", ""),
            start=(body.get("start") or {}).get("dateTime", ""),
            end=(body.get("end") or {}).get("dateTime", ""),
            attendees=[a["email"] for a in body.get("attendees", []) if "email" in a],
            location=body.get("location", "") or "",
            description=body.get("description", "") or "",
        )
        self.svc.store[eid] = ev
        self.svc.created.append(body)
        return _Request(ev.to_api(), error=self.svc.raise_on.get("insert"))

    def delete(self, calendarId="primary", eventId=None, **kw):
        err = self.svc.raise_on.get("delete")
        if err is None and eventId not in self.svc.store:
            err = _http_error(404, "Not Found")
        else:
            self.svc.store.pop(eventId, None)
            self.svc.deleted.append(eventId)
        return _Request("", error=err)

    def get(self, calendarId="primary", eventId=None, **kw):
        ev = self.svc.store.get(eventId)
        return _Request(ev.to_api() if ev else None,
                        error=None if ev else _http_error(404, "Not Found"))


class _CalendarList:
    def __init__(self, svc: FakeCalendarService):
        self.svc = svc

    def list(self, **kw):
        return _Request({"items": [{"id": "primary", "summary": "Abdullah",
                                    "primary": True}]})


def _default_events() -> List[FakeEvent]:
    return [
        FakeEvent("evt_1", "Standup", "2026-07-27T09:00:00", "2026-07-27T09:15:00"),
        FakeEvent("evt_2", "Dissertation supervision",
                  "2026-07-27T14:00:00", "2026-07-27T15:00:00",
                  attendees=["supervisor@example.ac.uk"]),
    ]


__all__ = [
    "FakeGmailService", "FakeCalendarService", "FakeMessage", "FakeEvent",
]
