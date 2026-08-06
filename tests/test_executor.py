"""Tests for core.executor — the DAG runner.

The first block is the audit's Severity 1.1: the old guard checked whether a
dependency was *present* in results, not whether it *succeeded*, which made the
BLOCKED branch unreachable and let subtasks run against failure payloads.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from core.executor import (
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    DagExecutor,
)
from core.registry import ToolRegistry
from core.tool import Action, BaseTool, ErrorType


@dataclass
class Task:
    id: str
    agent: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)


class ScriptedTool(BaseTool):
    """Succeeds or fails per action name; records call order and concurrency."""

    _name = "scripted"
    _description = "Test double with controllable outcomes."

    def __init__(self):
        self.calls: List[str] = []
        self.concurrent = 0
        self.peak_concurrent = 0
        self.delay = 0.0
        super().__init__()

    def _register_actions(self) -> None:
        for name, read_only in (("ok", True), ("boom", True),
                                ("write", False), ("wipe", False)):
            self.add_action(Action(
                name=name, description=f"{name} action",
                input_schema={"properties": {"tag": {"type": "string"}}},
                handler=self._make(name), read_only=read_only,
                destructive=(name == "wipe"), timeout=5.0,
            ))

    def _make(self, name):
        async def _call(tag: str = ""):
            self.calls.append(f"{name}:{tag}" if tag else name)
            self.concurrent += 1
            self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
            try:
                if self.delay:
                    await asyncio.sleep(self.delay)
                if name == "boom":
                    from core.tool import ToolUpstreamError
                    raise ToolUpstreamError("scripted failure")
                return {"value": f"{name}-result", "tag": tag}, f"{name} done"
            finally:
                self.concurrent -= 1
        _call.__name__ = f"_{name}"
        return _call


@pytest.fixture
def tool():
    return ScriptedTool()


@pytest.fixture
def reg(tool):
    r = ToolRegistry()
    r.register(tool)
    return r


@pytest.fixture
def ex(reg):
    return DagExecutor(reg)


# ── Severity 1.1: failed dependencies must block ────────────────────────────


async def test_dependent_subtask_does_not_run_when_its_dependency_failed(ex, tool):
    """The live bug: 'check my calendar, then email Sarah about the clash'
    sent the email even when the calendar step failed."""
    report = await ex.execute([
        Task("a", "scripted", "boom"),
        Task("b", "scripted", "ok", depends_on=["a"]),
    ])

    assert report.results["a"]["success"] is False
    assert report.results["b"]["success"] is False
    assert report.statuses["b"] == STATUS_BLOCKED
    assert "b" in report.blocked
    assert report.results["b"]["blocked_by"] == ["a"]
    assert "ok" not in tool.calls, "dependent subtask ran against a failed dependency"


async def test_blocking_cascades_through_a_chain(ex, tool):
    report = await ex.execute([
        Task("a", "scripted", "boom"),
        Task("b", "scripted", "ok", depends_on=["a"]),
        Task("c", "scripted", "ok", depends_on=["b"]),
    ])
    assert set(report.blocked) == {"b", "c"}
    assert tool.calls == ["boom"]


async def test_a_failure_only_blocks_its_own_dependents(ex, tool):
    report = await ex.execute([
        Task("a", "scripted", "boom"),
        Task("b", "scripted", "ok", depends_on=["a"]),
        Task("c", "scripted", "ok"),                 # independent
    ])
    assert report.statuses["c"] == STATUS_COMPLETED
    assert report.statuses["b"] == STATUS_BLOCKED


async def test_partial_dependency_failure_blocks(ex):
    """One good dependency does not license running on a bad one."""
    report = await ex.execute([
        Task("a", "scripted", "ok"),
        Task("b", "scripted", "boom"),
        Task("c", "scripted", "ok", depends_on=["a", "b"]),
    ])
    assert report.statuses["c"] == STATUS_BLOCKED
    assert report.results["c"]["blocked_by"] == ["b"]


async def test_all_dependencies_succeeding_runs_the_dependent(ex, tool):
    report = await ex.execute([
        Task("a", "scripted", "ok", {"tag": "a"}),
        Task("b", "scripted", "ok", {"tag": "b"}, depends_on=["a"]),
    ])
    assert report.all_ok
    assert tool.calls == ["ok:a", "ok:b"]


# ── Ordering ────────────────────────────────────────────────────────────────


async def test_dependency_order_is_respected(ex, tool):
    report = await ex.execute([
        Task("c", "scripted", "write", {"tag": "c"}, depends_on=["b"]),
        Task("a", "scripted", "write", {"tag": "a"}),
        Task("b", "scripted", "write", {"tag": "b"}, depends_on=["a"]),
    ])
    assert report.all_ok
    assert tool.calls == ["write:a", "write:b", "write:c"]


async def test_diamond_dependency(ex, tool):
    report = await ex.execute([
        Task("root", "scripted", "ok", {"tag": "root"}),
        Task("l", "scripted", "ok", {"tag": "l"}, depends_on=["root"]),
        Task("r", "scripted", "ok", {"tag": "r"}, depends_on=["root"]),
        Task("join", "scripted", "ok", {"tag": "join"}, depends_on=["l", "r"]),
    ])
    assert report.all_ok
    assert tool.calls[0] == "ok:root"
    assert tool.calls[-1] == "ok:join"


# ── Concurrency ─────────────────────────────────────────────────────────────


async def test_independent_reads_run_concurrently(reg, tool):
    """The old loop awaited each ready subtask in turn, so three independent
    API calls cost three round-trips."""
    tool.delay = 0.05
    ex = DagExecutor(reg, max_parallel=4)

    loop = asyncio.get_running_loop()
    start = loop.time()
    report = await ex.execute([Task(f"t{i}", "scripted", "ok") for i in range(4)])
    elapsed = loop.time() - start

    assert report.all_ok
    assert tool.peak_concurrent > 1, "reads ran sequentially"
    assert elapsed < 0.15, f"4 x 50ms reads took {elapsed:.3f}s — not parallel"


async def test_writes_run_one_at_a_time(reg, tool):
    """LLM plans don't reliably declare ordering between writes, so
    parallelising them could reorder side effects."""
    tool.delay = 0.02
    ex = DagExecutor(reg, max_parallel=4)
    await ex.execute([Task(f"w{i}", "scripted", "write") for i in range(4)])
    assert tool.peak_concurrent == 1, "writes ran concurrently"


async def test_destructive_actions_are_never_parallelised(reg, tool):
    tool.delay = 0.02
    ex = DagExecutor(reg, max_parallel=4)
    await ex.execute([Task(f"d{i}", "scripted", "wipe") for i in range(3)])
    assert tool.peak_concurrent == 1


async def test_parallelism_is_bounded(reg, tool):
    tool.delay = 0.05
    ex = DagExecutor(reg, max_parallel=2)
    await ex.execute([Task(f"t{i}", "scripted", "ok") for i in range(6)])
    assert tool.peak_concurrent <= 2


# ── Cycles and bad plans ────────────────────────────────────────────────────


async def test_cycle_is_reported_as_a_cycle(ex):
    report = await ex.execute([
        Task("a", "scripted", "ok", depends_on=["b"]),
        Task("b", "scripted", "ok", depends_on=["a"]),
    ])
    assert set(report.cyclic) == {"a", "b"}
    for r in report.results.values():
        assert r["success"] is False
        assert "circular" in r["error"]


async def test_self_dependency_is_a_cycle(ex):
    report = await ex.execute([Task("a", "scripted", "ok", depends_on=["a"])])
    assert report.cyclic == ["a"]


async def test_undefined_dependency_is_distinguished_from_a_cycle(ex, tool):
    """The old code labelled both 'Circular dependency'. They're different
    Planner bugs and need different fixes."""
    report = await ex.execute([Task("a", "scripted", "ok", depends_on=["ghost"])])
    assert report.cyclic == [], "an undefined dep was misreported as a cycle"
    assert report.unresolved_deps["a"] == ["ghost"]
    assert report.results["a"]["success"] is False
    assert "undefined subtask" in report.results["a"]["error"]


async def test_unknown_tool_fails_that_subtask_only(ex, tool):
    report = await ex.execute([
        Task("a", "teleporter", "engage"),
        Task("b", "scripted", "ok"),
    ])
    assert report.results["a"]["success"] is False
    assert report.results["a"]["error_type"] == ErrorType.NOT_FOUND.value
    assert report.statuses["b"] == STATUS_COMPLETED


async def test_unknown_action_fails_that_subtask_only(ex):
    report = await ex.execute([Task("a", "scripted", "nonsense")])
    assert report.results["a"]["success"] is False
    assert report.results["a"]["error_type"] == ErrorType.NOT_FOUND.value


async def test_empty_plan(ex):
    report = await ex.execute([])
    assert report.total == 0
    assert report.all_ok is False


# ── Dependency injection ────────────────────────────────────────────────────


async def test_inject_deps_hook_receives_prior_results(reg, tool):
    seen = []

    def inject(params, deps, completed):
        seen.append((dict(params), list(deps), dict(completed)))
        if deps:
            params["tag"] = completed[deps[0]]["result"]["value"]
        return params

    ex = DagExecutor(reg, inject_deps=inject)
    report = await ex.execute([
        Task("a", "scripted", "ok", {"tag": "first"}),
        Task("b", "scripted", "ok", depends_on=["a"]),
    ])
    assert report.all_ok
    assert tool.calls == ["ok:first", "ok:ok-result"]
    assert seen[1][1] == ["a"]


# ── Resilience ──────────────────────────────────────────────────────────────


async def test_a_registry_that_raises_does_not_abort_the_plan(reg, tool):
    class Exploding(ToolRegistry):
        async def execute(self, *a, **kw):
            raise RuntimeError("registry died")

    broken = Exploding()
    broken.register(tool)
    report = await DagExecutor(broken).execute([Task("a", "scripted", "ok")])
    assert report.results["a"]["success"] is False
    assert "registry died" in report.results["a"]["error"]


async def test_subtask_callback_is_invoked_per_result(reg):
    seen = []
    ex = DagExecutor(reg, on_subtask=lambda sid, res: seen.append((sid, res["success"])))
    await ex.execute([
        Task("a", "scripted", "ok"),
        Task("b", "scripted", "boom"),
        Task("c", "scripted", "ok", depends_on=["b"]),
    ])
    assert dict(seen) == {"a": True, "b": False, "c": False}


async def test_a_raising_callback_does_not_break_execution(reg):
    def bad(sid, res):
        raise RuntimeError("callback exploded")

    report = await DagExecutor(reg, on_subtask=bad).execute(
        [Task("a", "scripted", "ok")])
    assert report.all_ok


async def test_report_counts(ex):
    report = await ex.execute([
        Task("a", "scripted", "ok"),
        Task("b", "scripted", "ok"),
        Task("c", "scripted", "boom"),
    ])
    assert report.total == 3
    assert report.succeeded == 2
    assert report.all_ok is False


async def test_execution_terminates_on_a_large_chain(ex):
    """Guards the round bound — a long legitimate chain must not be mistaken
    for a cycle."""
    tasks = [Task("t0", "scripted", "ok")]
    tasks += [Task(f"t{i}", "scripted", "ok", depends_on=[f"t{i-1}"])
              for i in range(1, 30)]
    report = await ex.execute(tasks)
    assert report.all_ok
    assert report.total == 30
    assert report.cyclic == []
