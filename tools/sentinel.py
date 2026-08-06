"""
Sentinel — local secret and permission scanner.

Lifted out of ``server.py`` so it can be tested, and so the orchestrator can
call it as a tool rather than the capability existing only as a web page.

Four checks against the working tree:

1. Sensitive files present, and whether their POSIX mode lets other users read
   them.
2. A regex content scan for credential patterns, with placeholder filtering.
3. Secret files that git is *tracking*.
4. ``.env`` present but not in ``.gitignore``.

And one against history:

5. **Secrets in past commits.** This is the one that matters most and the one
   the working-tree scan structurally cannot see. ``git rm --cached`` — which
   check 3 tells you to run — removes a file from tracking *going forward* and
   leaves every historical copy intact. So a repo can look clean while an old
   commit still contains a live key, and pushing it anywhere public exposes it.
   ``scan_history`` walks the diffs and says so.

Everything here is read-only. Nothing writes, deletes or rewrites history.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

MAX_FILE_BYTES = 1_500_000

SKIP_DIRS = {".git", "venv", "__pycache__", "node_modules", ".cache",
             "site-packages", "logs", "data", "dist", "build", ".next",
             ".pytest_cache", ".mypy_cache"}
SKIP_PATTERNS = ("venv", "virtualenv", "site-packages", ".egg-info")
SKIP_SUFFIXES = (".bak", ".old", ".orig", ".icloud")

TEXT_EXT = {".py", ".js", ".ts", ".json", ".env", ".sh", ".yaml", ".yml",
            ".toml", ".ini", ".cfg", ".txt", ".md", ".html", ".conf", ""}

SECRET_FILENAMES = {"credentials.json", "token.json", ".env", "service_account.json",
                    "id_rsa", "id_ed25519", "client_secret.json"}

# (name, severity, regex). Most-specific first — the loop breaks on first hit,
# so a GitHub token must not be caught by the generic assignment rule.
PATTERNS: List[tuple] = [
    ("Private key block", "high", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("AWS access key", "high", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GitHub token", "high", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("Anthropic key", "high", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI key", "high", re.compile(r"sk-[A-Za-z0-9_\-]{20,}")),
    ("Google API key", "high", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("Slack token", "high", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("Google OAuth secret", "high", re.compile(r"GOCSPX-[A-Za-z0-9_\-]{10,}")),
    ("Stripe key", "high", re.compile(r"[sr]k_(?:live|test)_[A-Za-z0-9]{20,}")),
    ("Generic secret assignment", "medium",
     re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd|client[_-]?secret)"
                r"\s*[:=]\s*['\"][^'\"]{8,}['\"]")),
]

PLACEHOLDER = re.compile(
    r"(?i)(your[_-]?|example|placeholder|xxxx|<[^>]+>|changeme|\.\.\.|"
    r"dummy|sample|fake|test[_-]?(?:value|secret|key|token)|redacted|"
    r"hunter2|correct[_-]?horse|s3cr3t|foo|bar|baz|lorem)")

# A constant inside the test suite is a fixture. Reporting it at full severity
# trains you to ignore the list, which is how the real one gets missed.
TEST_PATH = re.compile(
    r"(?:^|/)(tests?|spec|fixtures?|__tests__|conftest\.py|"
    r"test_[^/]+\.py|[^/]+_test\.py|[^/]+\.test\.[jt]s)(?:/|$)")

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def is_test_path(rel: str) -> bool:
    return bool(TEST_PATH.search(str(rel).replace("\\", "/")))


def skip_dir(name: str) -> bool:
    if name in SKIP_DIRS or name.startswith("."):
        return True
    low = name.lower()
    return any(p in low for p in SKIP_PATTERNS) or low.endswith(SKIP_SUFFIXES)


def redact(s: str) -> str:
    """Never echo a live credential back, not even into a local web page."""
    s = (s or "").strip()
    if len(s) <= 8:
        return s[:2] + "***"
    return s[:4] + "***" + s[-2:]


@dataclass
class Finding:
    severity: str
    type: str
    file: str
    line: int = 0
    detail: str = ""
    commit: str = ""

    def to_json(self) -> Dict[str, Any]:
        d = {"severity": self.severity, "type": self.type, "file": self.file,
             "line": self.line, "detail": self.detail}
        if self.commit:
            d["commit"] = self.commit
        return d


def _git(cwd: Path, *args: str, timeout: int = 20) -> str:
    try:
        r = subprocess.run(["git", "-C", str(cwd), *args],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def match_line(line: str, rel: str) -> Optional[Finding]:
    """First credential pattern matching a line, or None."""
    if len(line) > 500:
        return None
    for name, sev, rx in PATTERNS:
        m = rx.search(line)
        if not m:
            continue
        if PLACEHOLDER.search(m.group(0)):
            return None
        in_test = is_test_path(rel)
        return Finding(
            severity="low" if in_test else sev,
            type=name + (" (test fixture)" if in_test else ""),
            file=rel, line=0,
            detail=f"{name} found: {redact(m.group(0))}" + (
                " — inside the test suite, so almost certainly a fixture "
                "rather than a real credential." if in_test else ""),
        )
    return None


class SentinelTool:
    """Read-only scanner over a project directory."""

    def __init__(self, project_dir: Path | str):
        self.project_dir = Path(project_dir)

    # ── Working tree ────────────────────────────────────────────────────────

    def scan(self) -> tuple[List[Finding], Dict[str, int]]:
        findings: List[Finding] = []

        for dirpath, dirnames, filenames in os.walk(self.project_dir):
            dirnames[:] = [d for d in dirnames if not skip_dir(d)]
            for fn in filenames:
                fp = Path(dirpath) / fn
                try:
                    rel = str(fp.relative_to(self.project_dir))
                except ValueError:
                    continue

                if fn in SECRET_FILENAMES or fn.endswith(".pem"):
                    findings.extend(self._check_permissions(fp, fn, rel))

                if fp.suffix.lower() not in TEXT_EXT:
                    continue
                try:
                    if fp.stat().st_size > MAX_FILE_BYTES:
                        continue
                    text = fp.read_text(errors="ignore")
                except Exception:  # noqa: BLE001
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    hit = match_line(line, rel)
                    if hit:
                        hit.line = i
                        findings.append(hit)

        findings.extend(self._check_git_tracking())
        findings.extend(self._check_gitignore())
        return self._finalise(findings)

    def _check_permissions(self, fp: Path, fn: str, rel: str) -> List[Finding]:
        try:
            mode = fp.stat().st_mode
        except Exception:  # noqa: BLE001
            return []
        if not (mode & (stat.S_IRGRP | stat.S_IROTH)):
            return []
        return [Finding("medium", "World/group-readable secret", rel, 0,
                        f"{fn} is readable by other users (mode "
                        f"{oct(mode & 0o777)}). Run: chmod 600 {rel}")]

    def _check_git_tracking(self) -> List[Finding]:
        out = _git(self.project_dir, "ls-files")
        tracked = {Path(p).name for p in out.splitlines()}
        return [
            Finding("high", "Secret committed to git", n, 0,
                    f"{n} is tracked by git — it may be in your commit history. "
                    f"Untrack with: git rm --cached {n}, then add it to .gitignore.")
            for n in sorted(SECRET_FILENAMES & tracked)
        ]

    def _check_gitignore(self) -> List[Finding]:
        env = self.project_dir / ".env"
        if not env.exists():
            return []
        gi = self.project_dir / ".gitignore"
        try:
            if gi.exists() and any(l.strip() in (".env", "*.env", ".env*")
                                   for l in gi.read_text().splitlines()):
                return []
        except Exception:  # noqa: BLE001
            pass
        return [Finding("medium", ".env not in .gitignore", ".gitignore", 0,
                        "A .env file exists but isn't listed in .gitignore — "
                        "add '.env' to avoid committing it.")]

    # ── History ─────────────────────────────────────────────────────────────

    def scan_history(self, max_commits: int = 400) -> tuple[List[Finding], Dict[str, int]]:
        """Look for credentials in past commits.

        The working-tree scan cannot see these by construction, and the fix it
        recommends for a tracked secret — ``git rm --cached`` — does nothing
        about them: the file stays in every commit that already had it. A repo
        can therefore look clean while an old commit still holds a live key.

        Reports the *first* commit that introduced each distinct secret, since
        that is what determines how far back a history rewrite has to go.
        """
        if not (self.project_dir / ".git").exists():
            return [], {"high": 0, "medium": 0, "low": 0}

        # -G greps the diff content; --oneline keeps the payload small. Doing
        # this per-pattern is far cheaper than diffing the whole history once
        # and scanning it in Python.
        findings: List[Finding] = []
        seen: Set[tuple] = set()

        for name, sev, rx in PATTERNS:
            if name.startswith("Generic"):
                # Too noisy over history — every refactor of a config file
                # matches. The specific patterns are what actually matter.
                continue
            out = _git(self.project_dir, "log", "--all", f"-n{max_commits}",
                       "--format=%H%x1f%cr%x1f%s", "-G", rx.pattern,
                       "--name-only", "--diff-filter=AM", timeout=45)
            if not out.strip():
                continue

            commit = rel_time = subject = ""
            for line in out.splitlines():
                if "\x1f" in line:
                    parts = line.split("\x1f")
                    commit, rel_time, subject = (parts + ["", "", ""])[:3]
                    continue
                path = line.strip()
                if not path or is_test_path(path):
                    continue
                key = (name, path)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(Finding(
                    severity="high",
                    type=f"{name} in git history",
                    file=path, line=0, commit=commit[:8],
                    detail=(
                        f"A {name.lower()} appears in {path} in commit "
                        f"{commit[:8]} ({rel_time}: {subject[:60]}). Removing "
                        f"the file now does NOT remove it from history — "
                        f"rotate the credential, then rewrite history with "
                        f"git filter-repo --path {path} --invert-paths"),
                ))
        return self._finalise(findings)

    # ── Shared ──────────────────────────────────────────────────────────────

    def _finalise(self, findings: List[Finding]) -> tuple[List[Finding], Dict[str, int]]:
        seen, uniq = set(), []
        for f in findings:
            key = (f.type, f.file, f.line, f.commit)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(f)
        uniq.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 3), f.file, f.line))
        summary = {"high": 0, "medium": 0, "low": 0}
        for f in uniq:
            summary[f.severity] = summary.get(f.severity, 0) + 1
        return uniq, summary

    def summarise(self, findings: Sequence[Finding], summary: Dict[str, int]) -> str:
        """One-line answer, for when this is called as a tool rather than read."""
        total = sum(summary.values())
        if not total:
            return "No exposed secrets, loose permissions or committed credentials."
        bits = [f"{summary[k]} {k}" for k in ("high", "medium", "low") if summary.get(k)]
        lead = f"{total} finding{'s' if total != 1 else ''} ({', '.join(bits)})."
        worst = [f for f in findings if f.severity == "high"][:3]
        if worst:
            lead += " Most serious: " + "; ".join(
                f"{f.type} in {f.file}" + (f" @ {f.commit}" if f.commit else "")
                for f in worst) + "."
        return lead


__all__ = ["SentinelTool", "Finding", "PATTERNS", "PLACEHOLDER",
           "is_test_path", "skip_dir", "redact", "match_line", "SECRET_FILENAMES"]
