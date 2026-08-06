"""
Adapters — Forge and Sentinel.

Both existed only as web pages: the logic sat inside ``server.py`` behind a
route, so you could look at repo health or a security scan but you couldn't
*ask* about it. Registering them means "what have I not pushed?" and "have I
leaked anything?" resolve through the same planner and executor as everything
else.

Both are read-only. Nothing here stages, commits, pushes, rewrites history,
deletes a file or changes a permission — the fixes are returned as commands
for you to run, deliberately. A scanner that edits your repo to fix what it
found is a scanner you can't safely run on a hunch.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from core.tool import (
    Action,
    BaseTool,
    HealthReport,
    HealthStatus,
    ToolInputError,
    ToolNotFoundError,
    ToolResult,
)


# ── Forge ───────────────────────────────────────────────────────────────────


class ForgeAdapter(BaseTool):
    _name = "forge"
    _description = ("Repo health across your projects: uncommitted changes, "
                    "unpushed commits, and TODO/FIXME markers left in the code.")

    def __init__(self, tool: Any) -> None:
        self._t = tool
        super().__init__()

    def _register_actions(self) -> None:
        self.add_action(Action(
            name="status",
            description="Summary across every tracked project: what's uncommitted, unpushed and outstanding.",
            input_schema={"properties": {}},
            handler=self._status, timeout=120.0,
        ))
        self.add_action(Action(
            name="project",
            description="Detail for one project by name.",
            input_schema={"properties": {
                "name": {"type": "string", "minLength": 1,
                         "description": "Project folder name, e.g. 'Jarvis'."},
            }, "required": ["name"]},
            handler=self._project, timeout=90.0,
        ))
        self.add_action(Action(
            name="list_marks",
            description="TODO / FIXME / HACK markers left in the code.",
            input_schema={"properties": {
                "name": {"type": "string", "description": "Limit to one project."},
                "kind": {"type": "string", "enum": ["TODO", "FIXME", "HACK", "XXX", "BUG"],
                         "description": "Only this marker type."},
            }},
            handler=self._marks, timeout=120.0,
        ))
        self.add_action(Action(
            name="uncommitted",
            description="Which projects have work that only exists on this machine.",
            input_schema={"properties": {}},
            handler=self._uncommitted, timeout=120.0,
        ))

    async def _scan(self):
        # Walks every project and shells out to git repeatedly — firmly a
        # worker-thread job, not something to do on the event loop.
        return await asyncio.to_thread(self._t.scan)

    async def _status(self):
        projects = await self._scan()
        data = {"projects": [p.to_json() for p in projects],
                "rollup": self._t.rollup(projects)}
        return data, self._t.summarise(projects)

    async def _project(self, name: str):
        projects = await self._scan()
        match = next((p for p in projects if p.name.lower() == name.lower().strip()), None)
        if match is None:
            known = ", ".join(p.name for p in projects) or "none"
            raise ToolNotFoundError(f"no project named {name!r}. Tracked: {known}")
        return match.to_json(), match.summary_line()

    async def _marks(self, name: Optional[str] = None, kind: Optional[str] = None):
        projects = await self._scan()
        if name:
            projects = [p for p in projects if p.name.lower() == name.lower().strip()]
            if not projects:
                raise ToolNotFoundError(f"no project named {name!r}")
        out = []
        for p in projects:
            for m in p.todo_samples:
                if kind and m.kind != kind:
                    continue
                out.append({**m.to_json(), "project": p.name})
        if not out:
            return {"marks": []}, ("No " + (kind or "TODO/FIXME") + " markers found.")
        lines = [f"{m['kind']} — {m['project']}/{m['file']}:{m['line']} — {m['text']}"
                 for m in out[:15]]
        more = f"\n…and {len(out) - 15} more." if len(out) > 15 else ""
        return {"marks": out, "count": len(out)}, "\n".join(lines) + more

    async def _uncommitted(self):
        projects = await self._scan()
        dirty = [p for p in projects if p.dirty]
        if not dirty:
            return {"projects": []}, "Everything is committed."
        lines = [f"{p.name}: {p.dirty} uncommitted"
                 + (f", {p.ahead} unpushed" if p.ahead else "") for p in dirty]
        return ({"projects": [p.to_json() for p in dirty]},
                "\n".join(lines))

    async def _check_health(self) -> HealthReport:
        try:
            found = await asyncio.to_thread(self._t.find_projects)
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name, f"{type(exc).__name__}: {exc}")
        if not found:
            return HealthReport(HealthStatus.DEGRADED, self.name,
                                "no git projects found in the configured roots")
        return HealthReport(HealthStatus.OK, self.name, f"{len(found)} project(s) tracked")


# ── Sentinel ────────────────────────────────────────────────────────────────


class SentinelAdapter(BaseTool):
    _name = "sentinel"
    _description = ("Scan the project for exposed secrets, world-readable "
                    "credential files and secrets committed to git.")

    def __init__(self, tool: Any) -> None:
        self._t = tool
        super().__init__()

    def _register_actions(self) -> None:
        self.add_action(Action(
            name="scan",
            description="Scan the working tree for exposed secrets and risky permissions.",
            input_schema={"properties": {
                "severity": {"type": "string", "enum": ["high", "medium", "low"],
                             "description": "Only findings at this level."},
            }},
            handler=self._scan, timeout=120.0,
        ))
        self.add_action(Action(
            name="scan_history",
            description=("Look for credentials in past commits. A secret removed from "
                         "the working tree still lives in every commit that had it."),
            input_schema={"properties": {
                "max_commits": {"type": "integer", "minimum": 10, "maximum": 5000,
                                "default": 400},
            }},
            handler=self._history, timeout=180.0,
        ))
        self.add_action(Action(
            name="summary",
            description="One-line verdict: is anything exposed?",
            input_schema={"properties": {}},
            handler=self._summary, timeout=120.0,
        ))

    async def _scan(self, severity: Optional[str] = None):
        findings, summary = await asyncio.to_thread(self._t.scan)
        if severity:
            findings = [f for f in findings if f.severity == severity]
            summary = {severity: len(findings)}
        data = {"findings": [f.to_json() for f in findings], "summary": summary,
                "root": str(self._t.project_dir)}
        return data, self._t.summarise(findings, summary)

    async def _history(self, max_commits: int = 400):
        findings, summary = await asyncio.to_thread(self._t.scan_history, max_commits)
        data = {"findings": [f.to_json() for f in findings], "summary": summary,
                "root": str(self._t.project_dir), "scanned_commits": max_commits}
        if not findings:
            return data, (f"No credentials found in the last {max_commits} commits.")
        return data, self._t.summarise(findings, summary)

    async def _summary(self):
        findings, summary = await asyncio.to_thread(self._t.scan)
        # Degraded rather than failed when something IS exposed: the scan
        # worked. Failing here would make a working scanner look broken every
        # time it did its job.
        return ToolResult.ok(
            self.name, "summary",
            data={"summary": summary},
            message=self._t.summarise(findings, summary),
            degraded=bool(summary.get("high")),
        )

    async def _check_health(self) -> HealthReport:
        root = self._t.project_dir
        if not root.exists():
            return HealthReport(HealthStatus.ERROR, self.name,
                                f"project directory not found: {root}")
        return HealthReport(HealthStatus.OK, self.name, f"scanning {root}")


__all__ = ["ForgeAdapter", "SentinelAdapter"]
