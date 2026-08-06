"""
Adapters — local tools (file manager, contacts, reminders, documents, macOS
system control).

Two problems specific to this group, both flagged in the Phase 1 audit:

1. **Three of these tools are fully synchronous** — ``FileManagerTool``,
   ``ContactBook`` and ``ReminderStore`` have no ``async`` methods at all, yet
   the orchestrator calls them straight from ``async def``. A directory walk
   over Documents, or a SQLite write, blocks the event loop and stalls every
   other request and WebSocket on the server. Every call here goes through
   ``asyncio.to_thread``.

2. **The confirmation gate is not session-scoped.** ``_pending_file_op`` lives
   on the single shared orchestrator, so a "confirm" typed in one browser tab
   can execute a delete staged in another. The prepare/confirm actions below
   issue a token bound to a ``session_id`` and expiring after five minutes, so
   a confirmation can only ever complete the operation it belongs to.

``MacControlTool`` is already async but makes 14 direct ``subprocess.run``
calls inside those async methods (audit item 1.8). The adapter can't fix that
from the outside — it's noted per-action and fixed in the tool itself as part
of the Severity 1 pass.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.tool import (
    Action,
    BaseTool,
    HealthReport,
    HealthStatus,
    ToolInputError,
    ToolNotFoundError,
    ToolResult,
    ToolUnavailableError,
    ToolUpstreamError,
)

# How long a staged destructive operation stays confirmable.
PENDING_TTL_SECONDS = 300

_SESSION = {
    "type": "string",
    "description": (
        "Opaque per-conversation id. Confirmations are only valid for the "
        "session that staged them."
    ),
}


def _unwrap(payload: Dict[str, Any], *, what: str) -> Dict[str, Any]:
    if payload.get("success"):
        return payload
    err = str(payload.get("error") or f"{what} failed")
    low = err.lower()
    if "not found" in low or "does not exist" in low:
        raise ToolNotFoundError(err)
    if "not allowed" in low or "permission denied" in low:
        raise ToolUnavailableError(err)
    if "already exists" in low or "use a different name" in low \
            or "not a directory" in low or "use list_directory" in low \
            or "cannot read binary" in low:
        raise ToolInputError(err)
    raise ToolUpstreamError(err)


# ── Pending destructive operations ──────────────────────────────────────────


@dataclass(slots=True)
class _Pending:
    token: str
    session_id: str
    op: Any
    created_at: float

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.created_at) > PENDING_TTL_SECONDS


class _PendingStore:
    """Session-scoped store for staged destructive operations.

    Deliberately *not* a single slot. The current design holds one
    ``_pending_file_op`` on the shared orchestrator, which is both a
    correctness bug (concurrent sessions clobber each other) and a safety bug
    (one session can confirm another's delete).
    """

    def __init__(self) -> None:
        self._items: Dict[str, _Pending] = {}

    def stage(self, session_id: str, op: Any) -> str:
        self._sweep()
        token = secrets.token_urlsafe(12)
        self._items[token] = _Pending(token, session_id, op, time.monotonic())
        return token

    def take(self, token: str, session_id: str) -> Any:
        self._sweep()
        item = self._items.get(token)
        if item is None:
            raise ToolNotFoundError(
                "no pending operation with that token — it may have expired "
                f"(operations are confirmable for {PENDING_TTL_SECONDS // 60} minutes)"
            )
        if not secrets.compare_digest(item.session_id, session_id):
            # Do not reveal what the other session staged.
            raise ToolInputError("that pending operation belongs to another session")
        del self._items[token]
        return item.op

    def discard(self, token: str, session_id: str) -> bool:
        item = self._items.get(token)
        if item is None or not secrets.compare_digest(item.session_id, session_id):
            return False
        del self._items[token]
        return True

    def _sweep(self) -> None:
        for tok in [t for t, i in self._items.items() if i.expired]:
            del self._items[tok]

    def __len__(self) -> int:
        self._sweep()
        return len(self._items)


# ── File manager ────────────────────────────────────────────────────────────


class FileManagerAdapter(BaseTool):
    _name = "files"
    _description = ("Browse, search, read and create files under Desktop, Documents "
                    "and Downloads. Destructive operations require confirmation.")

    def __init__(self, tool: Any, pending: Optional[_PendingStore] = None) -> None:
        self._t = tool
        # `pending or _PendingStore()` would be wrong: _PendingStore defines
        # __len__, so an empty injected store is falsy and would be silently
        # replaced by a fresh one.
        self.pending = pending if pending is not None else _PendingStore()
        super().__init__()

    def _register_actions(self) -> None:
        path = {"type": "string", "minLength": 1,
                "description": "Path or root alias (desktop / documents / downloads)."}

        self.add_action(Action(
            name="list_directory", description="List the contents of a directory.",
            input_schema={"properties": {"path": path}, "required": ["path"]},
            handler=self._list, timeout=20.0,
        ))
        self.add_action(Action(
            name="search",
            description="Find files by name, optionally searching their contents.",
            input_schema={"properties": {
                "query": {"type": "string", "minLength": 1},
                "location": {"type": "string", "description": "Root to search under."},
                "content_search": {"type": "boolean", "default": False,
                                   "description": "Also grep inside text files. Much slower."},
            }, "required": ["query"]},
            handler=self._search, timeout=60.0,
        ))
        self.add_action(Action(
            name="read_file", description="Read a text file's contents.",
            input_schema={"properties": {"path": path}, "required": ["path"]},
            handler=self._read, timeout=20.0,
        ))
        self.add_action(Action(
            name="get_info", description="Size, type and timestamps for a path.",
            input_schema={"properties": {"path": path}, "required": ["path"]},
            handler=self._info, timeout=10.0,
        ))
        self.add_action(Action(
            name="create_file", description="Create a new text file.",
            input_schema={"properties": {"path": path,
                                         "content": {"type": "string", "default": ""}},
                          "required": ["path"]},
            handler=self._create_file, timeout=15.0, read_only=False,
        ))
        self.add_action(Action(
            name="create_folder", description="Create a new folder.",
            input_schema={"properties": {"path": path}, "required": ["path"]},
            handler=self._create_folder, timeout=15.0, read_only=False,
        ))

        # ── Staged destructive operations ───────────────────────────────────
        for op_name, desc, extra in (
            ("delete", "Stage a file or folder for deletion.", {}),
            ("move", "Stage a move.", {"destination": {"type": "string", "minLength": 1}}),
            ("rename", "Stage a rename.", {"new_name": {"type": "string", "minLength": 1}}),
        ):
            props = {"path": path, "session_id": _SESSION, **extra}
            self.add_action(Action(
                name=f"prepare_{op_name}",
                description=f"{desc} Returns a token; nothing happens until confirm_operation.",
                input_schema={"properties": props,
                              "required": ["path", "session_id"] + list(extra)},
                handler=getattr(self, f"_prepare_{op_name}"),
                timeout=15.0, read_only=False, destructive=False,
            ))

        self.add_action(Action(
            name="confirm_operation",
            description="Execute a previously staged destructive operation.",
            input_schema={"properties": {
                "token": {"type": "string", "minLength": 1},
                "session_id": _SESSION,
            }, "required": ["token", "session_id"]},
            handler=self._confirm, timeout=60.0, read_only=False, destructive=True,
        ))
        self.add_action(Action(
            name="cancel_operation", description="Discard a staged operation.",
            input_schema={"properties": {
                "token": {"type": "string", "minLength": 1},
                "session_id": _SESSION,
            }, "required": ["token", "session_id"]},
            handler=self._cancel, timeout=5.0, read_only=False,
        ))

    # ── Read paths (all off-loop) ───────────────────────────────────────────

    async def _list(self, path: str):
        d = _unwrap(await asyncio.to_thread(self._t.list_directory, path),
                    what="directory listing")
        return d, self._t.format_listing(d)

    async def _search(self, query: str, location: Optional[str] = None,
                      content_search: bool = False):
        d = _unwrap(
            await asyncio.to_thread(self._t.search, query, location, content_search),
            what="file search")
        return d, self._t.format_search(d)

    async def _read(self, path: str):
        d = _unwrap(await asyncio.to_thread(self._t.read_file, path), what="file read")
        return d, d.get("content", "")

    async def _info(self, path: str):
        d = _unwrap(await asyncio.to_thread(self._t.get_info, path), what="file info")
        return d, self._t.format_info(d)

    async def _create_file(self, path: str, content: str = ""):
        d = _unwrap(await asyncio.to_thread(self._t.create_file, path, content),
                    what="file create")
        return d, d.get("message", "File created.")

    async def _create_folder(self, path: str):
        d = _unwrap(await asyncio.to_thread(self._t.create_folder, path),
                    what="folder create")
        return d, d.get("message", "Folder created.")

    # ── Staged operations ───────────────────────────────────────────────────

    async def _stage(self, session_id: str, op: Any, verb: str):
        # prepare_* returns either a PendingFileOp or the legacy error dict.
        if isinstance(op, dict):
            _unwrap(op, what=verb)
        token = self.pending.stage(session_id, op)
        summary = op.summary() if hasattr(op, "summary") else str(op)
        return (
            {"token": token, "operation": verb,
             "expires_in_seconds": PENDING_TTL_SECONDS, "summary": summary},
            summary,
        )

    async def _prepare_delete(self, path: str, session_id: str):
        op = await asyncio.to_thread(self._t.prepare_delete, path)
        return await self._stage(session_id, op, "delete")

    async def _prepare_move(self, path: str, destination: str, session_id: str):
        op = await asyncio.to_thread(self._t.prepare_move, path, destination)
        return await self._stage(session_id, op, "move")

    async def _prepare_rename(self, path: str, new_name: str, session_id: str):
        op = await asyncio.to_thread(self._t.prepare_rename, path, new_name)
        return await self._stage(session_id, op, "rename")

    async def _confirm(self, token: str, session_id: str):
        op = self.pending.take(token, session_id)
        runner = {
            "delete": self._t.execute_delete,
            "move": self._t.execute_move,
            "rename": self._t.execute_rename,
        }.get(getattr(op, "operation", ""))
        if runner is None:
            raise ToolInputError(f"unsupported staged operation: {getattr(op, 'operation', '?')}")
        d = _unwrap(await asyncio.to_thread(runner, op), what=op.operation)
        return d, d.get("message", f"{op.operation.title()} complete.")

    async def _cancel(self, token: str, session_id: str):
        ok = self.pending.discard(token, session_id)
        if not ok:
            raise ToolNotFoundError("no pending operation with that token for this session")
        return {"cancelled": True}, "Cancelled."

    async def _check_health(self) -> HealthReport:
        try:
            roots = await asyncio.to_thread(self._t.list_directory, "desktop")
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name, f"{type(exc).__name__}: {exc}")
        if roots.get("success"):
            return HealthReport(HealthStatus.OK, self.name,
                                f"{len(self.pending)} operation(s) awaiting confirmation")
        return HealthReport(HealthStatus.ERROR, self.name, str(roots.get("error")))


# ── Contacts ────────────────────────────────────────────────────────────────


class ContactsAdapter(BaseTool):
    _name = "contacts"
    _description = "Local address book mapping names to email addresses."

    def __init__(self, tool: Any) -> None:
        self._t = tool
        super().__init__()

    def _register_actions(self) -> None:
        self.add_action(Action(
            name="find", description="Look up a contact by name (partial matches allowed).",
            input_schema={"properties": {"name": {"type": "string", "minLength": 1}},
                          "required": ["name"]},
            handler=self._find, timeout=5.0,
        ))
        self.add_action(Action(
            name="list_all", description="Every saved contact.",
            input_schema={"properties": {}}, handler=self._list, timeout=5.0,
        ))
        self.add_action(Action(
            name="add", description="Add or update a contact.",
            input_schema={"properties": {
                "name": {"type": "string", "minLength": 1},
                "email": {"type": "string", "format": "email",
                          "pattern": r"^[^@\s]+@[^@\s]+\.[^@\s]+$"},
                "notes": {"type": "string", "default": ""},
            }, "required": ["name", "email"]},
            handler=self._add, timeout=5.0, read_only=False,
        ))
        self.add_action(Action(
            name="delete", description="Remove a contact by name.",
            input_schema={"properties": {"name": {"type": "string", "minLength": 1}},
                          "required": ["name"]},
            handler=self._delete, timeout=5.0, read_only=False, destructive=True,
        ))

    async def _find(self, name: str):
        c = await asyncio.to_thread(self._t.find, name)
        if not c:
            raise ToolNotFoundError(f"no contact matching {name!r}")
        return c, f"{c['name']} — {c['email']}"

    async def _list(self):
        items = await asyncio.to_thread(self._t.list_all)
        return {"contacts": items}, self._t.format_list()

    async def _add(self, name: str, email: str, notes: str = ""):
        await asyncio.to_thread(self._t.add, name, email, notes)
        return {"name": name, "email": email}, f"Saved {name} — {email}."

    async def _delete(self, name: str):
        ok = await asyncio.to_thread(self._t.delete, name)
        if not ok:
            raise ToolNotFoundError(f"no contact named {name!r}")
        return {"deleted": name}, f"Deleted {name}."

    async def _check_health(self) -> HealthReport:
        try:
            n = len(await asyncio.to_thread(self._t.list_all))
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name, f"{type(exc).__name__}: {exc}")
        return HealthReport(HealthStatus.OK, self.name, f"{n} contacts")


# ── Reminders ───────────────────────────────────────────────────────────────


class RemindersAdapter(BaseTool):
    _name = "reminders"
    _description = "One-shot and recurring reminders, persisted to SQLite."

    def __init__(self, tool: Any) -> None:
        self._t = tool
        super().__init__()

    def _register_actions(self) -> None:
        self.add_action(Action(
            name="add", description="Create a reminder. Give either due_at or offset_minutes.",
            input_schema={"properties": {
                "title": {"type": "string", "minLength": 1},
                "body": {"type": "string", "default": ""},
                "due_at": {"type": "string", "description": "ISO-8601 datetime."},
                "offset_minutes": {"type": "integer", "minimum": 1,
                                   "description": "Minutes from now."},
                "recurring_minutes": {"type": "integer", "minimum": 1,
                                      "description": "Repeat interval; omit for one-shot."},
            }, "required": ["title"]},
            handler=self._add, timeout=10.0, read_only=False,
        ))
        self.add_action(Action(
            name="list_pending", description="All incomplete reminders, soonest first.",
            input_schema={"properties": {}}, handler=self._list, timeout=10.0,
        ))
        self.add_action(Action(
            name="complete", description="Mark a reminder done.",
            input_schema={"properties": {"id": {"type": "string", "minLength": 1}},
                          "required": ["id"]},
            handler=self._complete, timeout=10.0, read_only=False,
        ))
        self.add_action(Action(
            name="delete", description="Delete a reminder.",
            input_schema={"properties": {"id": {"type": "string", "minLength": 1}},
                          "required": ["id"]},
            handler=self._delete, timeout=10.0, read_only=False, destructive=True,
        ))

    async def _add(self, title: str, body: str = "", due_at: Optional[str] = None,
                   offset_minutes: Optional[int] = None,
                   recurring_minutes: Optional[int] = None):
        if not due_at and offset_minutes is None:
            # The raw store defaults silently; being explicit stops "remind me
            # about the thing" quietly landing five minutes from now.
            raise ToolInputError("give either due_at (ISO-8601) or offset_minutes")
        rid = await asyncio.to_thread(
            self._t.add, title, body, due_at, recurring_minutes, offset_minutes)
        when = due_at or f"in {offset_minutes} minutes"
        return {"id": rid, "title": title, "due": when}, f"Reminder set: {title} — {when}."

    async def _list(self):
        items = await asyncio.to_thread(self._t.list_pending)
        return {"reminders": items}, self._t.format_list(items)

    async def _complete(self, id: str):  # noqa: A002 - matches the plan's param name
        ok = await asyncio.to_thread(self._t.complete, id)
        if not ok:
            raise ToolNotFoundError(f"no reminder with id {id!r}")
        return {"id": id, "completed": True}, "Reminder marked complete."

    async def _delete(self, id: str):  # noqa: A002
        ok = await asyncio.to_thread(self._t.delete, id)
        if not ok:
            raise ToolNotFoundError(f"no reminder with id {id!r}")
        return {"id": id, "deleted": True}, "Reminder deleted."

    async def _check_health(self) -> HealthReport:
        try:
            n = len(await asyncio.to_thread(self._t.list_pending))
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name,
                                f"sqlite unreachable: {type(exc).__name__}: {exc}")
        return HealthReport(HealthStatus.OK, self.name, f"{n} pending")


# ── Documents ───────────────────────────────────────────────────────────────


class DocumentAdapter(BaseTool):
    _name = "document"
    _description = "Extract text from PDF, DOCX, CSV and plain-text files."

    def __init__(self, tool: Any) -> None:
        self._t = tool
        super().__init__()

    def _register_actions(self) -> None:
        self.add_action(Action(
            name="extract", description="Pull the text out of a document.",
            input_schema={"properties": {
                "path": {"type": "string", "minLength": 1},
                "max_chars": {"type": "integer", "minimum": 100, "maximum": 200000,
                              "default": 20000},
            }, "required": ["path"]},
            handler=self._extract, timeout=120.0,
        ))

    async def _extract(self, path: str, max_chars: int = 20000):
        d = _unwrap(await self._t.extract(path), what="document extraction")
        return d, self._t.format_result(d, max_chars=max_chars)

    async def _check_health(self) -> HealthReport:
        missing = []
        for mod, why in (("pdfplumber", "PDF"), ("docx", "DOCX")):
            try:
                __import__(mod)
            except ImportError:
                missing.append(why)
        if missing:
            return HealthReport(HealthStatus.DEGRADED, self.name,
                                f"parsers unavailable: {', '.join(missing)}")
        return HealthReport(HealthStatus.OK, self.name, "pdf + docx parsers present")


# ── macOS system control ────────────────────────────────────────────────────


class MacControlAdapter(BaseTool):
    _name = "mac"
    _description = ("Control macOS: volume, brightness, apps, clipboard, notifications, "
                    "battery, Wi-Fi, dark mode, screenshots.")

    def __init__(self, tool: Any, *, is_mac: Optional[bool] = None) -> None:
        self._t = tool
        if is_mac is None:
            import sys
            is_mac = sys.platform == "darwin"
        self._is_mac = is_mac
        super().__init__()

    def _register_actions(self) -> None:
        app = {"type": "string", "minLength": 1, "description": "Application name."}

        simple = [
            ("get_volume", "Current output volume.", {}, True),
            ("get_brightness", "Current display brightness.", {}, True),
            ("get_battery", "Battery percentage and charging state.", {}, True),
            ("get_system_info", "CPU, memory and disk usage.", {}, True),
            ("get_wifi_info", "Wi-Fi network name and signal.", {}, True),
            ("get_dark_mode", "Whether dark mode is on.", {}, True),
            ("get_running_apps", "Applications currently running.", {}, True),
            ("get_clipboard", "Current clipboard contents.", {}, True),
            ("toggle_dark_mode", "Switch between light and dark mode.", {}, False),
            ("empty_trash", "Empty the Trash.", {}, False),
        ]
        for name, desc, props, read_only in simple:
            self.add_action(Action(
                name=name, description=desc,
                input_schema={"properties": props},
                handler=self._passthrough(name), timeout=20.0,
                read_only=read_only, destructive=(name == "empty_trash"),
            ))

        self.add_action(Action(
            name="set_volume", description="Set output volume 0–100.",
            input_schema={"properties": {"level": {"type": "integer", "minimum": 0,
                                                   "maximum": 100}},
                          "required": ["level"]},
            handler=self._passthrough("set_volume"), timeout=20.0, read_only=False,
        ))
        self.add_action(Action(
            name="set_brightness", description="Set display brightness 0.0–1.0.",
            input_schema={"properties": {"level": {"type": "number", "minimum": 0,
                                                   "maximum": 1}},
                          "required": ["level"]},
            handler=self._passthrough("set_brightness"), timeout=20.0, read_only=False,
        ))
        self.add_action(Action(
            name="open_app", description="Launch or focus an application.",
            input_schema={"properties": {"app_name": app,
                                         "new_window": {"type": "boolean", "default": False}},
                          "required": ["app_name"]},
            handler=self._passthrough("open_app"), timeout=25.0, read_only=False,
        ))
        self.add_action(Action(
            name="quit_app", description="Quit an application.",
            input_schema={"properties": {"app_name": app}, "required": ["app_name"]},
            handler=self._passthrough("quit_app"), timeout=20.0,
            read_only=False, destructive=True,
        ))
        self.add_action(Action(
            name="set_clipboard", description="Write text to the clipboard.",
            input_schema={"properties": {"text": {"type": "string"}}, "required": ["text"]},
            handler=self._passthrough("set_clipboard"), timeout=15.0, read_only=False,
        ))
        self.add_action(Action(
            name="send_notification", description="Show a macOS notification.",
            input_schema={"properties": {
                "message": {"type": "string", "minLength": 1},
                "title": {"type": "string", "default": "Jarvis"},
                "subtitle": {"type": "string", "default": ""},
            }, "required": ["message"]},
            handler=self._passthrough("send_notification"), timeout=15.0, read_only=False,
        ))
        self.add_action(Action(
            name="take_screenshot", description="Capture the screen to a file.",
            input_schema={"properties": {"save_path": {"type": "string"}}},
            handler=self._passthrough("take_screenshot"), timeout=30.0, read_only=False,
        ))
        self.add_action(Action(
            name="lock_screen", description="Lock the screen.",
            input_schema={"properties": {}},
            handler=self._passthrough("lock_screen"), timeout=15.0,
            read_only=False, destructive=True,
        ))

    def _passthrough(self, method: str):
        async def _call(**params):
            if not self._is_mac:
                raise ToolUnavailableError(
                    f"mac.{method} needs macOS; this process is not running on it"
                )
            fn = getattr(self._t, method, None)
            if fn is None:
                raise ToolUnavailableError(f"MacControlTool has no method {method!r}")
            result = await fn(**params)
            if isinstance(result, dict):
                _unwrap(result, what=f"mac.{method}")
                return result, result.get("message", "")
            return result, str(result or "")
        _call.__name__ = f"_mac_{method}"
        return _call

    async def _check_health(self) -> HealthReport:
        if not self._is_mac:
            return HealthReport(HealthStatus.UNAVAILABLE, self.name,
                                "not running on macOS — all actions disabled")
        try:
            d = await self._t.get_volume()
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name, f"{type(exc).__name__}: {exc}")
        if d.get("success"):
            return HealthReport(HealthStatus.OK, self.name, "applescript responding")
        return HealthReport(HealthStatus.ERROR, self.name,
                            f"applescript probe failed: {d.get('error')}")


__all__ = [
    "FileManagerAdapter", "ContactsAdapter", "RemindersAdapter",
    "DocumentAdapter", "MacControlAdapter", "_PendingStore",
    "PENDING_TTL_SECONDS",
]
