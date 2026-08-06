"""
Forge — repo health across every project you're working on.

Lifted out of ``server.py`` so it can be tested and called as a tool. For each
git project it reports what's uncommitted, what's unpushed, the recent commits,
and the TODO/FIXME/HACK markers left in the code.

Everything shells out to git in read-only mode: ``status --porcelain``,
``rev-list``, ``log``. Nothing stages, commits, pushes or checks out.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

TODO_RX = re.compile(r"\b(TODO|FIXME|HACK|XXX|BUG)\b")
MARK_KINDS = ("TODO", "FIXME", "HACK", "XXX", "BUG")

SKIP_DIRS = {".git", "venv", "__pycache__", "node_modules", ".cache",
             "site-packages", "dist", "build", ".next", "Pods", "Packages",
             "vendor", "target", "DerivedData", ".pytest_cache", ".mypy_cache"}
SKIP_PATTERNS = ("venv", "virtualenv", "site-packages", ".egg-info")
SKIP_SUFFIXES = (".bak", ".old", ".orig", ".icloud")

CODE_EXT = {".py", ".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte", ".go",
            ".rs", ".rb", ".java", ".kt", ".swift", ".c", ".h", ".cpp", ".cs",
            ".php", ".sh", ".sql", ".html", ".css", ".scss"}

MAX_FILE_BYTES = 800_000
MAX_SAMPLES = 12
MAX_PROJECTS = 12


def skip_dir(name: str) -> bool:
    if name in SKIP_DIRS or name.startswith("."):
        return True
    low = name.lower()
    return any(p in low for p in SKIP_PATTERNS) or low.endswith(SKIP_SUFFIXES)


def git(cwd: Path, *args: str, timeout: int = 8) -> str:
    try:
        r = subprocess.run(["git", "-C", str(cwd), *args],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


@dataclass
class Mark:
    kind: str
    file: str
    line: int
    text: str

    def to_json(self) -> Dict[str, Any]:
        return {"kind": self.kind, "file": self.file, "line": self.line, "text": self.text}


@dataclass
class Project:
    name: str
    path: str
    is_git: bool = False
    branch: str = ""
    dirty: int = 0
    ahead: int = 0
    behind: int = 0
    files: int = 0
    todos: Dict[str, int] = field(default_factory=dict)
    todo_samples: List[Mark] = field(default_factory=list)
    commits: List[Dict[str, str]] = field(default_factory=list)
    last_commit: Optional[Dict[str, str]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name, "path": self.path, "is_git": self.is_git,
            "branch": self.branch, "dirty": self.dirty,
            "ahead": self.ahead, "behind": self.behind, "files": self.files,
            "todos": self.todos,
            "todo_samples": [m.to_json() for m in self.todo_samples],
            "commits": self.commits, "last_commit": self.last_commit,
        }

    def summary_line(self) -> str:
        bits = [f"{self.name} ({self.branch or 'no branch'})"]
        bits.append(f"{self.dirty} uncommitted" if self.dirty else "clean")
        if self.ahead:
            bits.append(f"{self.ahead} unpushed")
        if self.behind:
            bits.append(f"{self.behind} behind")
        total = self.todos.get("total", 0)
        if total:
            bits.append(f"{total} code marks")
        return " · ".join(bits)


class ForgeTool:
    """Scans configured roots for git projects and reports their state."""

    def __init__(self, scan_roots: Sequence[str], always_include: Optional[Path] = None):
        self.scan_roots = [str(r) for r in scan_roots]
        self.always_include = Path(always_include) if always_include else None

    # ── Discovery ───────────────────────────────────────────────────────────

    def find_projects(self, limit: int = MAX_PROJECTS) -> List[Path]:
        found: Dict[str, Path] = {}
        if self.always_include and self.always_include.exists():
            found[str(self.always_include)] = self.always_include
        for root in self.scan_roots:
            rp = Path(root)
            if not rp.exists():
                continue
            try:
                for child in sorted(rp.iterdir()):
                    if child.is_dir() and (child / ".git").exists():
                        found[str(child)] = child
            except Exception:  # noqa: BLE001
                continue
        return list(found.values())[:limit]

    # ── Per-project ─────────────────────────────────────────────────────────

    def count_marks(self, proj: Path) -> tuple[Dict[str, int], List[Mark], int]:
        counts = {k: 0 for k in MARK_KINDS}
        samples: List[Mark] = []
        files = 0
        for dirpath, dirnames, filenames in os.walk(proj):
            dirnames[:] = [d for d in dirnames if not skip_dir(d)]
            for fn in filenames:
                fp = Path(dirpath) / fn
                if fp.suffix.lower() not in CODE_EXT:
                    continue
                files += 1
                try:
                    if fp.stat().st_size > MAX_FILE_BYTES:
                        continue
                    text = fp.read_text(errors="ignore")
                except Exception:  # noqa: BLE001
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    m = TODO_RX.search(line)
                    if not m or len(line) >= 300:
                        continue
                    counts[m.group(1)] = counts.get(m.group(1), 0) + 1
                    if len(samples) < MAX_SAMPLES:
                        try:
                            rel = str(fp.relative_to(proj))
                        except ValueError:
                            rel = fp.name
                        samples.append(Mark(m.group(1), rel, i, line.strip()[:120]))
        counts["total"] = sum(v for k, v in counts.items() if k != "total")
        return counts, samples, files

    def project_info(self, proj: Path) -> Project:
        p = Project(name=proj.name, path=str(proj), is_git=(proj / ".git").exists())
        if p.is_git:
            p.branch = git(proj, "rev-parse", "--abbrev-ref", "HEAD") or "?"
            dirty = git(proj, "status", "--porcelain")
            p.dirty = len([l for l in dirty.splitlines() if l.strip()])
            ab = git(proj, "rev-list", "--left-right", "--count", "@{u}...HEAD")
            if ab and "\t" in ab:
                behind, ahead = ab.split("\t")[:2]
                p.behind = int(behind or 0)
                p.ahead = int(ahead or 0)
            for line in git(proj, "log", "-6", "--pretty=%h\x1f%s\x1f%cr").splitlines():
                if "\x1f" in line:
                    h, s, rel = line.split("\x1f")
                    p.commits.append({"hash": h, "msg": s[:80], "rel": rel})
            if p.commits:
                p.last_commit = p.commits[0]
        p.todos, p.todo_samples, p.files = self.count_marks(proj)
        return p

    # ── Aggregate ───────────────────────────────────────────────────────────

    def scan(self) -> List[Project]:
        projects = [self.project_info(p) for p in self.find_projects()]
        # Jarvis first, then whatever has the most uncommitted work, then name.
        home = self.always_include.name if self.always_include else None
        projects.sort(key=lambda p: (p.name != home, -p.dirty, p.name))
        return projects

    def rollup(self, projects: Sequence[Project]) -> Dict[str, int]:
        return {
            "projects": len(projects),
            "dirty": sum(p.dirty for p in projects),
            "unpushed": sum(p.ahead for p in projects),
            "behind": sum(p.behind for p in projects),
            "marks": sum(p.todos.get("total", 0) for p in projects),
            "clean_repos": sum(1 for p in projects if not p.dirty),
        }

    def summarise(self, projects: Sequence[Project]) -> str:
        """One-line answer for the tool path."""
        if not projects:
            return "No git projects found in the configured roots."
        r = self.rollup(projects)
        if not r["dirty"] and not r["unpushed"]:
            return (f"All {r['projects']} projects are committed and pushed. "
                    f"{r['marks']} code marks outstanding.")
        parts = [f"{r['projects']} projects"]
        if r["dirty"]:
            parts.append(f"{r['dirty']} uncommitted changes")
        if r["unpushed"]:
            parts.append(f"{r['unpushed']} unpushed commits")
        if r["marks"]:
            parts.append(f"{r['marks']} code marks")
        busiest = max(projects, key=lambda p: p.dirty)
        tail = (f" Most work sits in {busiest.name} ({busiest.dirty} uncommitted)."
                if busiest.dirty else "")
        return ", ".join(parts) + "." + tail


__all__ = ["ForgeTool", "Project", "Mark", "TODO_RX", "MARK_KINDS",
           "skip_dir", "git", "CODE_EXT"]
