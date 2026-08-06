"""
DAG execution.

Lifted out of ``JarvisOrchestrator`` so it can be tested without importing
ChromaDB, Ollama and fourteen tools. The orchestrator now owns *policy* (what
to plan, how to phrase the answer) and this owns *mechanism* (what runs, in
what order, what happens when something fails).

Three behaviour changes from the version this replaces:

1. **Failed dependencies actually block.** The old guard read::

       deps_done = all(d in completed for d in subtask.depends_on)
       if not deps_done:
           failed_deps = [d for d in subtask.depends_on
                          if d in completed and not completed[d].get("success")]
           if failed_deps: ...

   A failed dependency is still *in* ``completed``, so ``deps_done`` was True,
   the guard was skipped, and the BLOCKED branch was unreachable. Dependent
   subtasks ran against failure payloads — "check my calendar, then email
   Sarah about the clash" would send the email with an error dict where the
   event data should be.

2. **Independent reads run concurrently.** The old loop awaited each ready
   subtask in turn, so a "weather and news and scores" plan paid the sum of
   three network round-trips despite the DAG saying they were independent.
   Writes still run one at a time — see ``_partition`` for why.

3. **Cycles are reported, not guessed at.** The old code detected "nothing ran
   this round" and labelled everything remaining a circular dependency, which
   also caught plans referencing a subtask id that was never defined. Those are
   different bugs and now say so.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

from core.registry import ToolRegistry
from core.tool import ErrorType, ToolResult

logger = logging.getLogger("jarvis.executor")


class SubtaskLike(Protocol):
    """What the executor needs from a subtask. Deliberately structural, so
    ``config.models.Subtask`` works unchanged and tests need no imports."""

    id: str
    agent: str
    action: str
    params: Dict[str, Any]
    depends_on: List[str]


# Status strings the executor reports back. The orchestrator maps these onto
# its own TaskStatus enum — keeping the enum out of here is what lets this
# module be imported without config.models.
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked"


@dataclass(slots=True)
class ExecutionReport:
    """Outcome of running a plan."""

    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    statuses: Dict[str, str] = field(default_factory=dict)
    blocked: List[str] = field(default_factory=list)
    cyclic: List[str] = field(default_factory=list)
    unresolved_deps: Dict[str, List[str]] = field(default_factory=dict)
    rounds: int = 0

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results.values() if r.get("success"))

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def all_ok(self) -> bool:
        return self.total > 0 and self.succeeded == self.total


class DagExecutor:
    """Runs a subtask DAG against a ToolRegistry."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        inject_deps: Optional[Callable[[Dict[str, Any], List[str],
                                        Dict[str, Any]], Dict[str, Any]]] = None,
        max_parallel: int = 4,
        on_subtask: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.registry = registry
        self._inject = inject_deps or (lambda params, deps, completed: params)
        self.max_parallel = max(1, max_parallel)
        self._on_subtask = on_subtask

    # ── Public ──────────────────────────────────────────────────────────────

    async def execute(self, subtasks: Sequence[SubtaskLike]) -> ExecutionReport:
        report = ExecutionReport()
        pending: Dict[str, SubtaskLike] = {st.id: st for st in subtasks}

        if not pending:
            return report

        self._flag_unresolved(pending, report)

        # Bound the loop at the number of subtasks: every round retires at
        # least one (by running it or blocking it), or we detect a cycle and
        # stop. A blocking cascade retires one layer per round, so a chain of
        # n needs at most n rounds.
        for _ in range(len(pending) + 1):
            if not pending:
                break
            report.rounds += 1

            ready, blocked = self._classify(pending, report)

            for st_id in blocked:
                failed = [d for d in pending[st_id].depends_on
                          if d in report.results
                          and not report.results[d].get("success")]
                report.results[st_id] = {
                    "success": False,
                    "error": f"blocked by failed dependencies: {', '.join(failed)}",
                    "error_type": ErrorType.NOT_FOUND.value,
                    "blocked_by": failed,
                }
                report.statuses[st_id] = STATUS_BLOCKED
                report.blocked.append(st_id)
                self._emit(st_id, report.results[st_id])
                logger.info("subtask %s blocked by failed deps: %s", st_id, failed)
                del pending[st_id]

            if not ready:
                if not pending:
                    break
                # Blocking is progress: a cascade retires one layer per round,
                # and the next round can then classify its dependents. Only
                # declare a cycle when a whole round retired nothing.
                if blocked:
                    continue
                self._mark_cyclic(pending, report)
                break

            await self._run_round(ready, pending, report)

        return report

    # ── Internals ───────────────────────────────────────────────────────────

    def _flag_unresolved(self, pending: Dict[str, SubtaskLike],
                         report: ExecutionReport) -> None:
        """A plan referencing a subtask id that doesn't exist is a Planner bug,
        distinct from a cycle. The old code silently reported both as
        'Circular dependency'."""
        known = set(pending)
        for st_id, st in pending.items():
            missing = [d for d in (st.depends_on or []) if d not in known]
            if missing:
                report.unresolved_deps[st_id] = missing
                logger.warning("subtask %s depends on undefined subtask(s): %s",
                               st_id, missing)

    def _classify(self, pending: Dict[str, SubtaskLike],
                  report: ExecutionReport) -> tuple[List[str], List[str]]:
        """Split pending subtasks into runnable and blocked.

        The distinction the old code got wrong: a dependency being *present* in
        results is not the same as it having *succeeded*.
        """
        ready: List[str] = []
        blocked: List[str] = []
        for st_id, st in pending.items():
            deps = [d for d in (st.depends_on or [])
                    if d not in report.unresolved_deps.get(st_id, [])]
            if not all(d in report.results for d in deps):
                continue                       # still waiting
            if any(not report.results[d].get("success") for d in deps):
                blocked.append(st_id)
            else:
                ready.append(st_id)
        return ready, blocked

    def _mark_cyclic(self, pending: Dict[str, SubtaskLike],
                     report: ExecutionReport) -> None:
        for st_id in list(pending):
            report.results[st_id] = {
                "success": False,
                "error": "circular dependency — this subtask can never become runnable",
                "error_type": ErrorType.INPUT.value,
            }
            report.statuses[st_id] = STATUS_FAILED
            report.cyclic.append(st_id)
            self._emit(st_id, report.results[st_id])
        logger.warning("circular dependency among subtasks: %s", report.cyclic)

    def _partition(self, ready: List[str],
                   pending: Dict[str, SubtaskLike]) -> tuple[List[str], List[str]]:
        """Split a ready batch into read-only (safe to run concurrently) and
        everything else (run one at a time).

        Parallelising reads is free — three independent API calls should cost
        one round-trip, not three. Parallelising writes is not: two subtasks
        that both send email or both move files may depend on each other
        through the filesystem or an inbox without saying so in the DAG, and
        the plans come from an LLM that is not reliable about declaring that.
        Sequential writes preserve today's semantics exactly.
        """
        reads, writes = [], []
        for st_id in ready:
            st = pending[st_id]
            reads.append(st_id) if self._is_read_only(st) else writes.append(st_id)
        return reads, writes

    def _is_read_only(self, st: SubtaskLike) -> bool:
        try:
            res = self.registry.resolve(st.agent, st.action)
        except Exception:  # noqa: BLE001 - unknown tools fail in _dispatch_one
            return False
        spec = res.tool.actions.get(res.action)
        return bool(spec and spec.read_only and not spec.destructive)

    async def _run_round(self, ready: List[str], pending: Dict[str, SubtaskLike],
                         report: ExecutionReport) -> None:
        reads, writes = self._partition(ready, pending)

        if reads:
            sem = asyncio.Semaphore(self.max_parallel)

            async def _guarded(st_id: str):
                async with sem:
                    return st_id, await self._dispatch_one(pending[st_id], report)

            for st_id, result in await asyncio.gather(*(_guarded(i) for i in reads)):
                self._record(st_id, result, report)
                del pending[st_id]

        for st_id in writes:
            result = await self._dispatch_one(pending[st_id], report)
            self._record(st_id, result, report)
            del pending[st_id]

    async def _dispatch_one(self, st: SubtaskLike,
                            report: ExecutionReport) -> Dict[str, Any]:
        if st.id in report.unresolved_deps:
            missing = report.unresolved_deps[st.id]
            return {
                "success": False,
                "error": f"plan references undefined subtask(s): {', '.join(missing)}",
                "error_type": ErrorType.INPUT.value,
            }

        params = self._inject(dict(st.params or {}), list(st.depends_on or []),
                              report.results)
        try:
            result: ToolResult = await self.registry.execute(
                (st.agent or "").strip().lower(),
                (st.action or "").strip().lower(),
                params,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # The registry converts tool failures into results; reaching here
            # means the registry itself broke. Never let one subtask abort the
            # whole plan.
            logger.exception("registry raised while dispatching %s", st.id)
            return {"success": False, "error": f"{type(exc).__name__}: {exc}",
                    "error_type": ErrorType.INTERNAL.value}
        return result.to_dict()

    def _record(self, st_id: str, result: Dict[str, Any],
                report: ExecutionReport) -> None:
        report.results[st_id] = result
        report.statuses[st_id] = (STATUS_COMPLETED if result.get("success")
                                  else STATUS_FAILED)

        # Log rather than swallow. The dispatcher this replaced returned the
        # error string and discarded everything else, so no failure in the DAG
        # path was diagnosable after the fact — which is also why Phase 4's
        # telemetry would have had nothing truthful to record.
        if not result.get("success"):
            logger.warning("subtask %s failed: %s.%s -> [%s] %s",
                           st_id, result.get("tool"), result.get("action"),
                           result.get("error_type", "unknown"), result.get("error"))
            tb = (result.get("meta") or {}).get("traceback")
            if tb:
                logger.debug("subtask %s traceback:\n%s", st_id, tb)
        elif result.get("degraded"):
            logger.info("subtask %s succeeded on a fallback path: %s.%s",
                        st_id, result.get("tool"), result.get("action"))

        self._emit(st_id, result)

    def _emit(self, st_id: str, result: Dict[str, Any]) -> None:
        if self._on_subtask is None:
            return
        try:
            self._on_subtask(st_id, result)
        except Exception:  # noqa: BLE001
            logger.warning("subtask callback raised for %s", st_id, exc_info=True)


__all__ = ["DagExecutor", "ExecutionReport", "SubtaskLike",
           "STATUS_COMPLETED", "STATUS_FAILED", "STATUS_BLOCKED"]
