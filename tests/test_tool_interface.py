"""Contract tests for core.tool — the Phase 2 interface every tool implements.

These pin the behaviours the rest of the system is allowed to rely on:
result shape, error typing, timeout enforcement, schema validation, and the
legacy-dict compatibility path that lets adapters be migrated incrementally.
"""

from __future__ import annotations

import asyncio

import pytest

from core.tool import (
    Action,
    BaseTool,
    ErrorType,
    HealthReport,
    HealthStatus,
    ToolResult,
    UnknownActionError,
)


# ── Result shape ────────────────────────────────────────────────────────────


async def test_successful_call_returns_toolresult(fake_tool):
    r = await fake_tool.execute("echo", {"text": "hello"})
    assert isinstance(r, ToolResult)
    assert r.success is True
    assert r.tool == "fake"
    assert r.action == "echo"
    assert r.data == {"echoed": "hello"}
    assert r.message == "You said: hello"
    assert r.error is None
    assert r.latency_ms > 0


async def test_tuple_handler_splits_data_and_message(fake_tool):
    r = await fake_tool.execute("tuple_shape")
    assert r.data == {"a": 1}
    assert r.message == "tuple message"


async def test_params_default_to_empty(fake_tool):
    r = await fake_tool.execute("tuple_shape", None)
    assert r.success


# ── Schema validation ───────────────────────────────────────────────────────


async def test_missing_required_param_is_input_error_not_crash(fake_tool):
    r = await fake_tool.execute("echo", {})
    assert r.success is False
    assert r.error_type is ErrorType.INPUT
    assert "text" in r.error
    # The handler must never have run.
    assert fake_tool.calls == []


async def test_wrong_type_is_input_error(fake_tool):
    r = await fake_tool.execute("echo", {"text": 42})
    assert r.success is False
    assert r.error_type is ErrorType.INPUT


async def test_extra_params_are_allowed_by_default(fake_tool):
    """The Planner LLM routinely emits stray keys. Schema validation must not
    reject them, but the handler will — as an INPUT error, not INTERNAL."""
    r = await fake_tool.execute("echo", {"text": "hi", "unexpected": True})
    assert r.success is False
    assert r.error_type is ErrorType.INPUT
    assert "parameter mismatch" in r.error


# ── Error typing ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind,expected", [
    ("auth", ErrorType.AUTH),
    ("input", ErrorType.INPUT),
    ("unavailable", ErrorType.UNAVAILABLE),
    ("not_found", ErrorType.NOT_FOUND),
    ("upstream", ErrorType.UPSTREAM),
])
async def test_typed_tool_errors_map_to_error_types(fake_tool, kind, expected):
    r = await fake_tool.execute("raises", {"kind": kind})
    assert r.success is False
    assert r.error_type is expected
    assert f"deliberate {kind} failure" in r.error


async def test_unexpected_exception_becomes_internal_with_traceback(fake_tool):
    r = await fake_tool.execute("boom")
    assert r.success is False
    assert r.error_type is ErrorType.INTERNAL
    assert "ZeroDivisionError" in r.error
    # This is the point of the whole exercise: the traceback survives instead
    # of being print()ed and dropped, so telemetry can store it.
    assert "traceback" in r.meta
    assert "ZeroDivisionError" in r.meta["traceback"]


async def test_retryable_classification():
    assert ErrorType.TIMEOUT.retryable
    assert ErrorType.UPSTREAM.retryable
    assert not ErrorType.INPUT.retryable
    assert not ErrorType.AUTH.retryable
    assert not ErrorType.INTERNAL.retryable


# ── Timeouts ────────────────────────────────────────────────────────────────


async def test_action_timeout_is_enforced(fake_tool):
    r = await fake_tool.execute("slow", {"seconds": 5})
    assert r.success is False
    assert r.error_type is ErrorType.TIMEOUT
    assert "timed out" in r.error
    # 0.2s budget — must not have actually waited 5s.
    assert r.latency_ms < 1000


async def test_per_call_timeout_overrides_action_default(fake_tool):
    r = await fake_tool.execute("slow", {"seconds": 0.05}, timeout=2.0)
    assert r.success is True


async def test_cancellation_is_not_swallowed(fake_tool):
    """Eating CancelledError here would make the server un-shutdown-able."""
    task = asyncio.create_task(fake_tool.execute("slow", {"seconds": 5}, timeout=30))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ── Legacy dict compatibility (migration path) ──────────────────────────────


async def test_legacy_success_dict_is_adapted(fake_tool):
    r = await fake_tool.execute("legacy_dict", {"ok": True})
    assert r.success is True
    assert r.data == {"n": 7}
    assert r.message == "legacy ok"


async def test_legacy_failure_dict_is_adapted_as_upstream(fake_tool):
    r = await fake_tool.execute("legacy_dict", {"ok": False})
    assert r.success is False
    assert r.error == "legacy failure"
    assert r.error_type is ErrorType.UPSTREAM


async def test_to_dict_emits_keys_the_orchestrator_already_reads(fake_tool):
    d = (await fake_tool.execute("echo", {"text": "x"})).to_dict()
    assert d["success"] is True
    assert d["result"] == {"echoed": "x"}
    assert d["message"] == "You said: x"
    assert "latency_ms" in d

    d2 = (await fake_tool.execute("raises", {"kind": "auth"})).to_dict()
    assert d2["success"] is False
    assert d2["error"]
    assert d2["error_type"] == "auth"


# ── Unknown actions ─────────────────────────────────────────────────────────


async def test_unknown_action_raises_rather_than_failing_softly(fake_tool):
    """A typo'd action is a programmer error. If it came back as a failed
    ToolResult it would look like a tool outage in the telemetry."""
    with pytest.raises(UnknownActionError) as exc:
        await fake_tool.execute("no_such_action")
    assert "no_such_action" in str(exc.value)
    assert "echo" in str(exc.value)


# ── Health ──────────────────────────────────────────────────────────────────


async def test_health_check_reports_ok(fake_tool):
    h = await fake_tool.health_check()
    assert h.status is HealthStatus.OK
    assert h.healthy is True
    assert h.tool == "fake"
    assert h.latency_ms > 0
    assert h.to_dict()["status"] == "ok"


async def test_health_check_survives_a_raising_probe():
    class Broken(BaseTool):
        _name = "broken"
        _description = "raises during health check"

        def _register_actions(self):
            self.add_action(Action(name="noop", description="",
                                   input_schema={}, handler=self._noop))

        async def _noop(self):
            return None

        async def _check_health(self):
            raise RuntimeError("probe exploded")

    h = await Broken().health_check()
    assert h.status is HealthStatus.ERROR
    assert "probe exploded" in h.detail


async def test_health_check_timeout_is_bounded():
    class Hanging(BaseTool):
        _name = "hanging"
        _description = "health check never returns"
        _health_timeout = 0.2

        def _register_actions(self):
            self.add_action(Action(name="noop", description="",
                                   input_schema={}, handler=self._noop))

        async def _noop(self):
            return None

        async def _check_health(self):
            await asyncio.sleep(60)

    h = await asyncio.wait_for(Hanging().health_check(), timeout=15)
    assert h.status is HealthStatus.ERROR
    assert "timed out" in h.detail


# ── MCP forward-compatibility (Phase 3 depends on this) ─────────────────────


def test_actions_export_as_mcp_declarations(fake_tool):
    decls = fake_tool.mcp_declarations()
    assert len(decls) == len(fake_tool.actions)
    by_name = {d["name"]: d for d in decls}
    assert "fake.echo" in by_name

    echo = by_name["fake.echo"]
    assert echo["inputSchema"]["type"] == "object"
    assert echo["inputSchema"]["required"] == ["text"]
    assert echo["annotations"]["readOnlyHint"] is True

    assert by_name["fake.wipe"]["annotations"]["destructiveHint"] is True
    assert by_name["fake.wipe"]["annotations"]["readOnlyHint"] is False
