"""
Reminders Tool
SQLite-backed reminder store + asyncio scheduler + Mac notifications.

Uses the existing jarvis.db (SQLITE_PATH) with a new `reminders` table.
Mac notifications fire via osascript so no extra dependencies needed.
"""

from __future__ import annotations

import asyncio
import subprocess
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config.settings import SQLITE_PATH


# ── Store ──────────────────────────────────────────────────────────────────────

class ReminderStore:
    """
    CRUD operations for reminders backed by the Jarvis SQLite database.

    Schema:
        id                TEXT PRIMARY KEY
        title             TEXT NOT NULL
        body              TEXT
        due_at            TEXT   -- ISO-8601 datetime
        recurring_minutes INT    -- NULL = one-shot, else repeat every N minutes
        completed         INT    -- 0 = pending, 1 = done
        created_at        TEXT
    """

    def __init__(self):
        self._init_db()
        print("⏰ ReminderStore ready")

    def _init_db(self):
        with sqlite3.connect(SQLITE_PATH) as conn:
            # WAL allows concurrent readers + a writer without blocking the
            # async event loop. The scheduler polls this table once a minute
            # while user requests can also hit it.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id                TEXT PRIMARY KEY,
                    title             TEXT NOT NULL,
                    body              TEXT DEFAULT '',
                    due_at            TEXT NOT NULL,
                    recurring_minutes INTEGER,
                    completed         INTEGER DEFAULT 0,
                    created_at        TEXT
                )
            """)

    # ── Write ──────────────────────────────────────────────────────────────

    def add(
        self,
        title: str,
        body: str = "",
        due_at: Optional[str] = None,
        recurring_minutes: Optional[int] = None,
        offset_minutes: Optional[int] = None,
    ) -> str:
        """
        Add a reminder. Returns the new reminder ID.

        Args:
            title:              Short description shown in the notification
            body:               Optional longer detail
            due_at:             ISO datetime string; if None, uses offset_minutes from now
            recurring_minutes:  If set, reschedule this many minutes after firing
            offset_minutes:     Convenience: "in X minutes from now"
        """
        reminder_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat()

        if due_at is None:
            mins = offset_minutes or 5
            due_at = (datetime.now() + timedelta(minutes=mins)).isoformat()

        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.execute(
                "INSERT INTO reminders VALUES (?,?,?,?,?,0,?)",
                (reminder_id, title, body, due_at, recurring_minutes, created_at),
            )
        return reminder_id

    def complete(self, reminder_id: str) -> bool:
        with sqlite3.connect(SQLITE_PATH) as conn:
            cur = conn.execute(
                "UPDATE reminders SET completed=1 WHERE id=?", (reminder_id,)
            )
        return cur.rowcount > 0

    def delete(self, reminder_id: str) -> bool:
        with sqlite3.connect(SQLITE_PATH) as conn:
            cur = conn.execute("DELETE FROM reminders WHERE id=?", (reminder_id,))
        return cur.rowcount > 0

    # ── Read ───────────────────────────────────────────────────────────────

    def list_pending(self) -> List[Dict[str, Any]]:
        """Return all incomplete reminders ordered by due time."""
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM reminders WHERE completed=0 ORDER BY due_at ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def list_due(self) -> List[Dict[str, Any]]:
        """Return reminders whose due_at <= now and are not completed."""
        now = datetime.now().isoformat()
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM reminders WHERE completed=0 AND due_at <= ?",
                (now,),
            ).fetchall()
        return [dict(r) for r in rows]

    def reschedule(self, reminder_id: str, offset_minutes: int):
        """Move a recurring reminder forward by offset_minutes."""
        new_due = (datetime.now() + timedelta(minutes=offset_minutes)).isoformat()
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.execute(
                "UPDATE reminders SET due_at=? WHERE id=?",
                (new_due, reminder_id),
            )

    def format_list(self, reminders: List[Dict[str, Any]]) -> str:
        if not reminders:
            return "No pending reminders."
        lines = [f"You have {len(reminders)} pending reminder(s):"]
        for i, r in enumerate(reminders, 1):
            try:
                dt = datetime.fromisoformat(r["due_at"])
                due_str = dt.strftime("%-d %b at %-I:%M %p")
            except Exception:
                due_str = r["due_at"]
            rec = f"  (repeats every {r['recurring_minutes']}m)" if r.get("recurring_minutes") else ""
            lines.append(f"{i}. {r['title']} — due {due_str}{rec}")
        return "\n".join(lines)


# ── Scheduler ──────────────────────────────────────────────────────────────────

def _fire_notification(title: str, body: str):
    """Fire a macOS notification via osascript (no extra deps required)."""
    body_safe = body.replace('"', "'") or title
    title_safe = title.replace('"', "'")
    script = f'display notification "{body_safe}" with title "⏰ {title_safe}" sound name "Glass"'
    try:
        subprocess.run(["osascript", "-e", script], timeout=5, check=False)
    except Exception as e:
        print(f"⏰ Notification error: {e}")


class ReminderScheduler:
    """
    asyncio background task that checks every 60 seconds for due reminders
    and fires macOS notifications.
    """

    def __init__(self, store: ReminderStore):
        self.store = store
        self._task: Optional[asyncio.Task] = None

    def start(self):
        """Start the background scheduler loop."""
        self._task = asyncio.ensure_future(self._loop())
        print("⏰ ReminderScheduler running (checks every 60s)")

    async def _loop(self):
        while True:
            try:
                due = self.store.list_due()
                for r in due:
                    _fire_notification(r["title"], r.get("body", ""))
                    if r.get("recurring_minutes"):
                        self.store.reschedule(r["id"], r["recurring_minutes"])
                    else:
                        self.store.complete(r["id"])
                    print(f"⏰ Fired: {r['title']} (id={r['id']})")
            except Exception as e:
                print(f"⏰ Scheduler error: {e}")
            await asyncio.sleep(60)
