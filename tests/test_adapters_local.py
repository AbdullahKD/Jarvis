"""Tests for the local-tool adapters.

Two things get real scrutiny here because both were live bugs:

* the three synchronous tools must not run on the event loop
* a confirmation must only ever execute the operation its own session staged
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from core.adapters.local import (
    PENDING_TTL_SECONDS,
    ContactsAdapter,
    DocumentAdapter,
    FileManagerAdapter,
    MacControlAdapter,
    RemindersAdapter,
    _PendingStore,
)
from core.tool import ErrorType, HealthStatus


# ── Doubles ─────────────────────────────────────────────────────────────────


class FakePendingOp:
    def __init__(self, operation: str, source: str, destination: Optional[str] = None):
        self.operation = operation
        self.source = Path(source)
        self.destination = Path(destination) if destination else None

    def summary(self):
        return f"⚠️  {self.operation} {self.source}"


class FakeFileManager:
    """Records the thread each call runs on, so we can prove the adapter took
    the synchronous work off the event loop."""

    def __init__(self):
        self.threads: List[int] = []
        self.executed: List[str] = []
        self.fail_prepare: Optional[Dict[str, Any]] = None

    def _tick(self):
        self.threads.append(threading.get_ident())

    def list_directory(self, path):
        self._tick()
        if path == "nope":
            return {"success": False, "error": "Path does not exist: nope"}
        return {"success": True, "path": path, "items": [{"name": "a.txt"}]}

    def search(self, query, location=None, content_search=False):
        self._tick()
        return {"success": True, "matches": [], "query": query,
                "content_search": content_search}

    def read_file(self, path):
        self._tick()
        if path.endswith(".bin"):
            return {"success": False, "error": "Cannot read binary file: x.bin"}
        return {"success": True, "content": "file body"}

    def get_info(self, path):
        self._tick()
        return {"success": True, "size": 12}

    def create_file(self, path, content=""):
        self._tick()
        return {"success": True, "message": f"Created {path}"}

    def create_folder(self, path):
        self._tick()
        return {"success": True, "message": f"Created folder {path}"}

    def prepare_delete(self, path):
        self._tick()
        return self.fail_prepare or FakePendingOp("delete", path)

    def prepare_move(self, src, dst):
        self._tick()
        return self.fail_prepare or FakePendingOp("move", src, dst)

    def prepare_rename(self, path, new_name):
        self._tick()
        return self.fail_prepare or FakePendingOp("rename", path, new_name)

    def execute_delete(self, op):
        self._tick()
        self.executed.append(f"delete:{op.source}")
        return {"success": True, "message": f"Deleted {op.source.name}"}

    def execute_move(self, op):
        self._tick()
        self.executed.append(f"move:{op.source}")
        return {"success": True, "message": "Moved"}

    def execute_rename(self, op):
        self._tick()
        self.executed.append(f"rename:{op.source}")
        return {"success": True, "message": "Renamed"}

    def format_listing(self, d):
        return f"{len(d.get('items', []))} items"

    def format_search(self, d):
        return f"{len(d.get('matches', []))} matches"

    def format_info(self, d):
        return f"{d.get('size')} bytes"


class FakeContacts:
    def __init__(self):
        self.store = {"sarah": {"name": "Sarah", "email": "sarah@example.com"}}

    def find(self, name):
        return self.store.get(name.lower().strip())

    def list_all(self):
        return list(self.store.values())

    def add(self, name, email, notes=""):
        self.store[name.lower()] = {"name": name, "email": email, "notes": notes}

    def delete(self, name):
        return self.store.pop(name.lower(), None) is not None

    def format_list(self):
        return f"{len(self.store)} contact(s)"


class FakeReminders:
    def __init__(self):
        self.items: Dict[str, Dict[str, Any]] = {}
        self.n = 0
        self.raise_on_list = False

    def add(self, title, body="", due_at=None, recurring_minutes=None,
            offset_minutes=None):
        self.n += 1
        rid = f"r{self.n}"
        self.items[rid] = {"id": rid, "title": title, "due_at": due_at or "soon"}
        return rid

    def list_pending(self):
        if self.raise_on_list:
            raise RuntimeError("database is locked")
        return list(self.items.values())

    def complete(self, rid):
        return self.items.pop(rid, None) is not None

    def delete(self, rid):
        return self.items.pop(rid, None) is not None

    def format_list(self, items):
        return f"{len(items)} pending"


class FakeDocument:
    def __init__(self, payload=None):
        self.payload = payload if payload is not None else {
            "success": True, "text": "extracted body", "pages": 3}

    async def extract(self, path):
        return self.payload

    def format_result(self, d, max_chars=20000):
        return str(d.get("text", ""))[:max_chars]


class FakeMac:
    def __init__(self, payload=None):
        self.payload = payload if payload is not None else {
            "success": True, "message": "done"}
        self.calls: List[tuple] = []

    def __getattr__(self, name):
        async def _call(**kw):
            self.calls.append((name, kw))
            return self.payload
        return _call


# ── FileManager: off-loop execution ─────────────────────────────────────────


async def test_sync_file_calls_run_off_the_event_loop():
    """FileManagerTool is fully synchronous and is currently called bare from
    async code — a Documents walk stalls every other request. The adapter must
    hand it to a worker thread."""
    t = FakeFileManager()
    a = FileManagerAdapter(t)
    loop_thread = threading.get_ident()

    await a.execute("list_directory", {"path": "desktop"})
    await a.execute("search", {"query": "invoice"})

    assert t.threads, "underlying tool was never called"
    assert all(tid != loop_thread for tid in t.threads), \
        "synchronous file work ran on the event loop thread"


async def test_event_loop_stays_responsive_during_a_slow_file_call():
    class SlowFM(FakeFileManager):
        def list_directory(self, path):
            time.sleep(0.3)
            return {"success": True, "items": []}

    a = FileManagerAdapter(SlowFM())
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        for _ in range(20):
            await asyncio.sleep(0.01)
            ticks += 1

    await asyncio.gather(a.execute("list_directory", {"path": "documents"}), heartbeat())
    assert ticks == 20, "event loop was blocked while the sync tool ran"


async def test_file_errors_are_classified():
    a = FileManagerAdapter(FakeFileManager())
    missing = await a.execute("list_directory", {"path": "nope"})
    assert missing.error_type is ErrorType.NOT_FOUND

    binary = await a.execute("read_file", {"path": "x.bin"})
    assert binary.error_type is ErrorType.INPUT


# ── FileManager: the confirmation gate ──────────────────────────────────────


async def test_prepare_then_confirm_executes_once():
    t = FakeFileManager()
    a = FileManagerAdapter(t)

    staged = await a.execute("prepare_delete",
                             {"path": "desktop/old.txt", "session_id": "sess-A"})
    assert staged.success
    token = staged.data["token"]
    assert t.executed == [], "prepare must not execute anything"

    done = await a.execute("confirm_operation", {"token": token, "session_id": "sess-A"})
    assert done.success
    assert t.executed == ["delete:desktop/old.txt"]

    # Token is single-use.
    again = await a.execute("confirm_operation", {"token": token, "session_id": "sess-A"})
    assert again.success is False
    assert again.error_type is ErrorType.NOT_FOUND
    assert t.executed == ["delete:desktop/old.txt"]


async def test_another_session_cannot_confirm_your_delete():
    """The live bug: _pending_file_op lives on one shared orchestrator, so a
    'yes' in tab B can execute a delete staged in tab A."""
    t = FakeFileManager()
    a = FileManagerAdapter(t)

    staged = await a.execute("prepare_delete",
                             {"path": "desktop/important.txt", "session_id": "sess-A"})
    token = staged.data["token"]

    hijack = await a.execute("confirm_operation",
                             {"token": token, "session_id": "sess-B"})
    assert hijack.success is False
    assert t.executed == [], "another session executed a delete it did not stage"

    # Still confirmable by its rightful owner.
    ok = await a.execute("confirm_operation", {"token": token, "session_id": "sess-A"})
    assert ok.success
    assert t.executed == ["delete:desktop/important.txt"]


async def test_two_sessions_can_stage_concurrently_without_clobbering():
    """One shared _pending_file_op slot means the second staging overwrites the
    first. Both must survive."""
    t = FakeFileManager()
    a = FileManagerAdapter(t)

    a_tok = (await a.execute("prepare_delete",
                             {"path": "desktop/a.txt", "session_id": "A"})).data["token"]
    b_tok = (await a.execute("prepare_delete",
                             {"path": "desktop/b.txt", "session_id": "B"})).data["token"]
    assert a_tok != b_tok

    await a.execute("confirm_operation", {"token": b_tok, "session_id": "B"})
    await a.execute("confirm_operation", {"token": a_tok, "session_id": "A"})
    assert sorted(t.executed) == ["delete:desktop/a.txt", "delete:desktop/b.txt"]


async def test_cancel_prevents_execution():
    t = FakeFileManager()
    a = FileManagerAdapter(t)
    token = (await a.execute("prepare_delete",
                             {"path": "desktop/x", "session_id": "S"})).data["token"]

    assert (await a.execute("cancel_operation",
                            {"token": token, "session_id": "S"})).success
    after = await a.execute("confirm_operation", {"token": token, "session_id": "S"})
    assert after.success is False
    assert t.executed == []


async def test_cancel_from_another_session_is_refused():
    a = FileManagerAdapter(FakeFileManager())
    token = (await a.execute("prepare_delete",
                             {"path": "desktop/x", "session_id": "S"})).data["token"]
    assert (await a.execute("cancel_operation",
                            {"token": token, "session_id": "OTHER"})).success is False


async def test_staged_operations_expire():
    store = _PendingStore()
    a = FileManagerAdapter(FakeFileManager(), pending=store)
    token = (await a.execute("prepare_delete",
                             {"path": "desktop/x", "session_id": "S"})).data["token"]

    # Age the entry past its TTL rather than sleeping five minutes.
    store._items[token].created_at -= (PENDING_TTL_SECONDS + 1)

    expired = await a.execute("confirm_operation", {"token": token, "session_id": "S"})
    assert expired.success is False
    assert "expired" in expired.error


async def test_prepare_failure_does_not_stage_anything():
    t = FakeFileManager()
    t.fail_prepare = {"success": False, "error": "Not found: ghost.txt"}
    a = FileManagerAdapter(t)

    r = await a.execute("prepare_delete", {"path": "ghost.txt", "session_id": "S"})
    assert r.success is False
    assert r.error_type is ErrorType.NOT_FOUND
    assert len(a.pending) == 0


async def test_prepare_requires_a_session_id():
    r = await FileManagerAdapter(FakeFileManager()).execute(
        "prepare_delete", {"path": "desktop/x"})
    assert r.success is False
    assert r.error_type is ErrorType.INPUT


def test_destructive_actions_are_flagged_for_mcp():
    a = FileManagerAdapter(FakeFileManager())
    assert a.actions["confirm_operation"].destructive is True
    # prepare_* stages only, so it is not itself destructive
    assert a.actions["prepare_delete"].destructive is False
    assert a.actions["list_directory"].read_only is True
    assert a.actions["create_file"].read_only is False


# ── Contacts ────────────────────────────────────────────────────────────────


async def test_contacts_find_and_missing():
    a = ContactsAdapter(FakeContacts())
    hit = await a.execute("find", {"name": "Sarah"})
    assert hit.success
    assert hit.data["email"] == "sarah@example.com"

    miss = await a.execute("find", {"name": "Nobody"})
    assert miss.error_type is ErrorType.NOT_FOUND


async def test_contacts_add_validates_email_shape():
    a = ContactsAdapter(FakeContacts())
    bad = await a.execute("add", {"name": "X", "email": "not-an-email"})
    assert bad.error_type is ErrorType.INPUT

    good = await a.execute("add", {"name": "Ali", "email": "ali@example.com"})
    assert good.success


async def test_contacts_delete_missing_is_not_found():
    a = ContactsAdapter(FakeContacts())
    assert (await a.execute("delete", {"name": "ghost"})).error_type is ErrorType.NOT_FOUND


# ── Reminders ───────────────────────────────────────────────────────────────


async def test_reminder_requires_a_time():
    """The raw store silently defaults, so 'remind me about the thing' quietly
    landed five minutes out. Make the caller say when."""
    r = await RemindersAdapter(FakeReminders()).execute("add", {"title": "Call mum"})
    assert r.success is False
    assert r.error_type is ErrorType.INPUT
    assert "offset_minutes" in r.error


async def test_reminder_add_and_list():
    t = FakeReminders()
    a = RemindersAdapter(t)
    added = await a.execute("add", {"title": "Standup", "offset_minutes": 30})
    assert added.success
    assert added.data["id"] == "r1"

    listed = await a.execute("list_pending")
    assert listed.message == "1 pending"


async def test_reminder_complete_missing_is_not_found():
    a = RemindersAdapter(FakeReminders())
    assert (await a.execute("complete", {"id": "nope"})).error_type is ErrorType.NOT_FOUND


async def test_reminder_health_reports_a_locked_database():
    t = FakeReminders()
    t.raise_on_list = True
    h = await RemindersAdapter(t).health_check()
    assert h.status is HealthStatus.ERROR
    assert "database is locked" in h.detail


async def test_reminder_sqlite_calls_run_off_loop():
    calls: List[int] = []

    class ThreadCheckingReminders(FakeReminders):
        def list_pending(self):
            calls.append(threading.get_ident())
            return []

    await RemindersAdapter(ThreadCheckingReminders()).execute("list_pending")
    assert calls and calls[0] != threading.get_ident()


# ── Documents ───────────────────────────────────────────────────────────────


async def test_document_extract_truncates():
    a = DocumentAdapter(FakeDocument({"success": True, "text": "x" * 5000}))
    r = await a.execute("extract", {"path": "/tmp/a.pdf", "max_chars": 100})
    assert r.success
    assert len(r.message) == 100


async def test_document_failure_is_typed():
    a = DocumentAdapter(FakeDocument({"success": False, "error": "File not found: /x"}))
    assert (await a.execute("extract", {"path": "/x"})).error_type is ErrorType.NOT_FOUND


# ── macOS ───────────────────────────────────────────────────────────────────


async def test_mac_actions_are_unavailable_off_darwin():
    """Running the server on Linux shouldn't surface mac failures as internal
    errors — UNAVAILABLE is a distinct, non-retryable state."""
    a = MacControlAdapter(FakeMac(), is_mac=False)
    r = await a.execute("get_volume")
    assert r.success is False
    assert r.error_type is ErrorType.UNAVAILABLE
    assert r.retryable is False


async def test_mac_health_off_darwin_is_unavailable_not_error():
    h = await MacControlAdapter(FakeMac(), is_mac=False).health_check()
    assert h.status is HealthStatus.UNAVAILABLE
    assert h.healthy is False


async def test_mac_passthrough_forwards_params():
    t = FakeMac()
    a = MacControlAdapter(t, is_mac=True)
    assert (await a.execute("set_volume", {"level": 40})).success
    assert t.calls[-1] == ("set_volume", {"level": 40})


async def test_mac_volume_bounds_enforced():
    a = MacControlAdapter(FakeMac(), is_mac=True)
    assert (await a.execute("set_volume", {"level": 400})).error_type is ErrorType.INPUT
    assert (await a.execute("set_brightness", {"level": 2.5})).error_type is ErrorType.INPUT


async def test_mac_destructive_actions_flagged():
    a = MacControlAdapter(FakeMac(), is_mac=True)
    assert a.actions["quit_app"].destructive is True
    assert a.actions["lock_screen"].destructive is True
    assert a.actions["empty_trash"].destructive is True
    assert a.actions["get_battery"].read_only is True


async def test_mac_failure_payload_is_surfaced():
    a = MacControlAdapter(FakeMac({"success": False, "error": "AppleScript error -1728"}),
                          is_mac=True)
    r = await a.execute("get_volume")
    assert r.success is False
    assert "1728" in r.error
