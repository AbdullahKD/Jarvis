"""The app's startup/shutdown contract.

Written when @app.on_event("startup") was replaced with a lifespan context
manager. The warning FastAPI printed was cosmetic; the bug behind it was not.
The old handler started a reminder scheduler and two background tasks and
stopped none of them, because on_event has no natural place to put teardown —
so every `--reload` restart left another scheduler polling the same SQLite
file, and Ctrl-C printed "Task was destroyed but it is pending".

These tests pin the behaviour that actually matters: shutdown leaves nothing
running.
"""

from __future__ import annotations

import asyncio
import warnings

import pytest

import server
from tools.reminders import ReminderScheduler


def test_app_uses_lifespan_not_on_event():
    """Starlette exposes registered on_event handlers as router lifecycle
    lists. Both empty means nothing slipped back in."""
    assert not server.app.router.on_startup
    assert not server.app.router.on_shutdown


def test_importing_server_emits_no_on_event_deprecation():
    """Re-import under an error filter. Already-imported modules are cached,
    so this checks the recorded warning rather than re-executing the module."""
    import importlib

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("server")
    assert module.app is not None
    assert not [w for w in caught
                if "on_event" in str(w.message)], "on_event is back"


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_cleanly():
    """The whole point: after the context exits, nothing is still running."""
    before = {t for t in asyncio.all_tasks()}
    async with server.lifespan(server.app):
        await asyncio.sleep(0)                     # let startup tasks schedule
    leftover = {t for t in asyncio.all_tasks()} - before - {asyncio.current_task()}
    assert not leftover, f"tasks outlived shutdown: {[t.get_coro() for t in leftover]}"


@pytest.mark.asyncio
async def test_lifespan_survives_a_failing_background_task():
    """LLM warmup calls Ollama, which is not running in tests. A background
    task that raises must not turn a clean exit into a traceback."""
    async with server.lifespan(server.app):
        await asyncio.sleep(0.05)
    # reaching here without an exception is the assertion


# ── the scheduler's half of the contract ────────────────────────────────────


@pytest.mark.asyncio
async def test_scheduler_stop_actually_awaits_the_task():
    """cancel() only *requests* cancellation. Returning without awaiting would
    race the loop's next tick, which is how the task used to survive."""
    class _Store:
        def list_due(self):
            return []

    sched = ReminderScheduler(_Store())
    sched.start()
    task = sched._task
    await sched.stop()
    assert task.done()
    assert sched._task is None


@pytest.mark.asyncio
async def test_scheduler_stop_is_safe_to_call_twice():
    """Shutdown paths get re-entered; a second stop must not raise."""
    class _Store:
        def list_due(self):
            return []

    sched = ReminderScheduler(_Store())
    sched.start()
    await sched.stop()
    await sched.stop()          # no-op, must not raise


@pytest.mark.asyncio
async def test_scheduler_stop_before_start_is_a_no_op():
    sched = ReminderScheduler(object())
    await sched.stop()
