"""Tests for Forge and Sentinel, and their adapters.

Real git repositories are created in tmp_path — a fake for `git log` would
test the fake, and the parsing of git's output is exactly where these break.
The one thing not exercised for real is a live credential: the fixtures use
syntactically valid but fictional keys.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from core.adapters.devtools import ForgeAdapter, SentinelAdapter
from core.tool import ErrorType, HealthStatus
from tools.forge import ForgeTool, skip_dir as forge_skip
from tools.sentinel import (
    PLACEHOLDER,
    SentinelTool,
    is_test_path,
    match_line,
    redact,
    skip_dir as sent_skip,
)

# Fictional but structurally valid: detection keys off shape, so a value like
# "xxx" wouldn't exercise the patterns at all.
#
# Note what these must NOT contain. The first draft used AWS's own documented
# form, AKIA...EXAMPLE — and the scanner correctly rejected it, because
# "example" is in the placeholder list. The fixture was wrong, not the filter.
FAKE_AWS = "AKIA" + "Q7ZB3MPLDNVWXYZ2"
FAKE_GH = "ghp_" + "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8"
FAKE_GOOGLE_OAUTH = "GOCSPX-" + "9fJqL2mNp4RtVwXz"


def has_git():
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


requires_git = pytest.mark.skipif(not has_git(), reason="git not available")


def init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(["git", "-C", str(path), *a],
                                    capture_output=True, text=True, timeout=10)
    run("init", "-q", "-b", "main")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "Test")
    run("config", "commit.gpgsign", "false")
    return path


def commit_all(path: Path, msg: str):
    subprocess.run(["git", "-C", str(path), "add", "-A"], capture_output=True, timeout=10)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", msg],
                   capture_output=True, timeout=10)


# ── Sentinel: pattern matching ──────────────────────────────────────────────


@pytest.mark.parametrize("line,expect", [
    (f'aws_key = "{FAKE_AWS}"', "AWS access key"),
    (f'token: {FAKE_GH}', "GitHub token"),
    (f'client_secret = "{FAKE_GOOGLE_OAUTH}"', "Google OAuth secret"),
    ("-----BEGIN RSA PRIVATE KEY-----", "Private key block"),
    ('password = "s0me-real-looking-value"', "Generic secret assignment"),
])
def test_credential_patterns_match(line, expect):
    hit = match_line(line, "app/config.py")
    assert hit is not None, f"missed: {line}"
    assert hit.type == expect


@pytest.mark.parametrize("line", [
    'api_key = "your-api-key-here"',
    'secret = "changeme"',
    'token = "<YOUR_TOKEN>"',
    'password = "example-password"',
    'PASSWORD = "hunter2-correct-horse"',
    'key = "dummy-value-1234"',
])
def test_placeholders_are_not_reported(line):
    """A scanner that flags the README's example config is a scanner nobody
    reads."""
    assert match_line(line, "app/config.py") is None


def test_a_specific_pattern_wins_over_the_generic_one():
    """Ordering matters: the loop stops at the first match, so a GitHub token
    must not be reported as a vague 'generic secret assignment'."""
    hit = match_line(f'github_token = "{FAKE_GH}"', "app.py")
    assert hit.type == "GitHub token"
    assert hit.severity == "high"


def test_secrets_are_redacted_in_output():
    hit = match_line(f'aws = "{FAKE_AWS}"', "app.py")
    assert FAKE_AWS not in hit.detail, "the scanner echoed the credential back"
    assert "***" in hit.detail


def test_redact_handles_short_values():
    assert redact("abc") == "ab***"
    assert "***" in redact("a-much-longer-secret-value")


def test_test_fixtures_are_downgraded_not_hidden():
    """Reporting a fixture at full severity trains you to ignore the list; not
    reporting it at all hides a real leak that happens to live in tests/."""
    hit = match_line(f'aws = "{FAKE_AWS}"', "tests/test_thing.py")
    assert hit.severity == "low"
    assert "test fixture" in hit.type


@pytest.mark.parametrize("path,is_test", [
    ("tests/test_ws_guard.py", True), ("conftest.py", True),
    ("src/app.test.ts", True), ("spec/models_spec.py", True),
    ("core/tool.py", False), ("server.py", False), ("app/testing_utils.py", False),
])
def test_test_path_detection(path, is_test):
    assert is_test_path(path) is is_test


def test_long_lines_are_skipped():
    """Minified bundles produce nothing but false positives."""
    assert match_line('key = "' + "x" * 600 + '"', "bundle.min.js") is None


# ── Sentinel: working tree ──────────────────────────────────────────────────


@pytest.fixture
def project(tmp_path):
    p = tmp_path / "proj"
    (p / "app").mkdir(parents=True)
    (p / "app" / "config.py").write_text(f'AWS_KEY = "{FAKE_AWS}"\n')
    (p / "app" / "clean.py").write_text("def hello():\n    return 1\n")
    (p / "README.md").write_text('Set api_key = "your-api-key-here" in .env\n')
    (p / "tests").mkdir()
    (p / "tests" / "test_auth.py").write_text(f'TOKEN = "{FAKE_GH}"\n')
    return p


def test_scan_finds_a_real_secret(project):
    findings, summary = SentinelTool(project).scan()
    types = {f.type for f in findings}
    assert "AWS access key" in types
    assert summary["high"] >= 1


def test_scan_ignores_the_readme_placeholder(project):
    findings, _ = SentinelTool(project).scan()
    assert not any(f.file == "README.md" for f in findings)


def test_scan_downgrades_the_test_fixture(project):
    findings, _ = SentinelTool(project).scan()
    fixture = [f for f in findings if f.file.startswith("tests/")]
    assert fixture and all(f.severity == "low" for f in fixture)


def test_scan_reports_line_numbers(project):
    (project / "app" / "multi.py").write_text(
        "import os\n\n\n" + f'KEY = "{FAKE_AWS}"\n')
    findings, _ = SentinelTool(project).scan()
    hit = next(f for f in findings if f.file.endswith("multi.py"))
    assert hit.line == 4


def test_world_readable_secret_file_is_flagged(project):
    env = project / ".env"
    env.write_text("SOMETHING=1\n")
    env.chmod(0o644)
    findings, _ = SentinelTool(project).scan()
    perm = [f for f in findings if f.type == "World/group-readable secret"]
    assert perm, "a 0644 .env was not flagged"
    assert "chmod 600" in perm[0].detail


def test_owner_only_secret_file_is_not_flagged(project):
    env = project / ".env"
    env.write_text("SOMETHING=1\n")
    env.chmod(0o600)
    findings, _ = SentinelTool(project).scan()
    assert not [f for f in findings if f.type == "World/group-readable secret"]


def test_env_without_gitignore_is_flagged(project):
    (project / ".env").write_text("X=1\n")
    (project / ".env").chmod(0o600)
    findings, _ = SentinelTool(project).scan()
    assert any(f.type == ".env not in .gitignore" for f in findings)


def test_env_listed_in_gitignore_is_not_flagged(project):
    (project / ".env").write_text("X=1\n")
    (project / ".env").chmod(0o600)
    (project / ".gitignore").write_text("__pycache__\n.env\n")
    findings, _ = SentinelTool(project).scan()
    assert not any(f.type == ".env not in .gitignore" for f in findings)


def test_vendored_directories_are_not_scanned(project):
    vend = project / "node_modules" / "pkg"
    vend.mkdir(parents=True)
    (vend / "index.js").write_text(f'var k = "{FAKE_AWS}";\n')
    findings, _ = SentinelTool(project).scan()
    assert not any("node_modules" in f.file for f in findings)


def test_findings_are_sorted_worst_first(project):
    (project / ".env").write_text("X=1\n")
    (project / ".env").chmod(0o644)
    findings, _ = SentinelTool(project).scan()
    sevs = [f.severity for f in findings]
    assert sevs == sorted(sevs, key=lambda s: {"high":0,"medium":1,"low":2}[s])


def test_clean_project_reports_nothing(tmp_path):
    p = tmp_path / "clean"
    p.mkdir()
    (p / "main.py").write_text("print('hi')\n")
    findings, summary = SentinelTool(p).scan()
    assert findings == []
    assert sum(summary.values()) == 0


# ── Sentinel: git history — the check the working tree can't do ─────────────


@requires_git
def test_secret_in_a_past_commit_is_found_after_deletion(tmp_path):
    """The headline case. `git rm --cached` removes a file from tracking going
    forward and leaves every historical copy — so the working tree looks clean
    while an old commit still holds a live key."""
    repo = init_repo(tmp_path / "repo")
    leak = repo / "settings.py"
    leak.write_text(f'AWS_KEY = "{FAKE_AWS}"\n')
    commit_all(repo, "add settings")

    leak.write_text("AWS_KEY = os.environ['AWS_KEY']\n")
    commit_all(repo, "move key to env")

    tool = SentinelTool(repo)
    tree_findings, _ = tool.scan()
    assert not any(f.type == "AWS access key" for f in tree_findings), \
        "working tree should now be clean"

    hist_findings, summary = tool.scan_history()
    assert any("AWS access key" in f.type for f in hist_findings), \
        "history scan missed a secret that is still in the repo"
    assert summary["high"] >= 1


@requires_git
def test_history_finding_names_the_commit_and_the_fix(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "conf.py").write_text(f'GH = "{FAKE_GH}"\n')
    commit_all(repo, "oops")

    findings, _ = SentinelTool(repo).scan_history()
    assert findings
    f = findings[0]
    assert f.commit and len(f.commit) == 8
    assert "rotate the credential" in f.detail
    assert "filter-repo" in f.detail


@requires_git
def test_clean_history_reports_nothing(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "main.py").write_text("print('hello')\n")
    commit_all(repo, "init")
    findings, summary = SentinelTool(repo).scan_history()
    assert findings == []
    assert sum(summary.values()) == 0


@requires_git
def test_history_ignores_test_fixtures(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_keys.py").write_text(f'AWS = "{FAKE_AWS}"\n')
    commit_all(repo, "add tests")
    findings, _ = SentinelTool(repo).scan_history()
    assert findings == [], "a fixture in tests/ was reported as a history leak"


def test_history_scan_on_a_non_git_directory_is_not_an_error(tmp_path):
    p = tmp_path / "plain"
    p.mkdir()
    findings, summary = SentinelTool(p).scan_history()
    assert findings == [] and sum(summary.values()) == 0


# ── Forge ───────────────────────────────────────────────────────────────────


@pytest.fixture
def workspace(tmp_path):
    # Skips inside the fixture rather than via a mark: pytest refuses marks on
    # fixtures, and every test using this one needs git anyway.
    if not has_git():
        pytest.skip("git not available")
    root = tmp_path / "ws"
    root.mkdir()

    a = init_repo(root / "alpha")
    (a / "main.py").write_text("# TODO: handle empty input\nprint(1)\n")
    (a / "util.py").write_text("# FIXME: this leaks a handle\n# HACK: sleep to avoid a race\n")
    commit_all(a, "first commit")
    (a / "dirty.py").write_text("x = 1\n")            # uncommitted

    b = init_repo(root / "beta")
    (b / "app.js").write_text("// TODO: debounce\n")
    commit_all(b, "init")

    (root / "notarepo").mkdir()
    (root / "notarepo" / "x.py").write_text("# TODO: ignored\n")
    return root


@requires_git
def test_finds_only_git_projects(workspace):
    tool = ForgeTool(scan_roots=[str(workspace)])
    names = {p.name for p in tool.find_projects()}
    assert names == {"alpha", "beta"}


@requires_git
def test_reports_uncommitted_and_branch(workspace):
    tool = ForgeTool(scan_roots=[str(workspace)])
    alpha = next(p for p in tool.scan() if p.name == "alpha")
    assert alpha.is_git
    assert alpha.branch == "main"
    assert alpha.dirty == 1
    beta = next(p for p in tool.scan() if p.name == "beta")
    assert beta.dirty == 0


@requires_git
def test_counts_code_marks_by_kind(workspace):
    tool = ForgeTool(scan_roots=[str(workspace)])
    alpha = next(p for p in tool.scan() if p.name == "alpha")
    assert alpha.todos["TODO"] == 1
    assert alpha.todos["FIXME"] == 1
    assert alpha.todos["HACK"] == 1
    assert alpha.todos["total"] == 3


@requires_git
def test_mark_samples_carry_file_and_line(workspace):
    tool = ForgeTool(scan_roots=[str(workspace)])
    alpha = next(p for p in tool.scan() if p.name == "alpha")
    todo = next(m for m in alpha.todo_samples if m.kind == "TODO")
    assert todo.file == "main.py"
    assert todo.line == 1
    assert "handle empty input" in todo.text


@requires_git
def test_recent_commits_are_parsed(workspace):
    tool = ForgeTool(scan_roots=[str(workspace)])
    alpha = next(p for p in tool.scan() if p.name == "alpha")
    assert alpha.commits
    assert alpha.commits[0]["msg"] == "first commit"
    assert alpha.last_commit["hash"]


@requires_git
def test_rollup_totals(workspace):
    tool = ForgeTool(scan_roots=[str(workspace)])
    projects = tool.scan()
    r = tool.rollup(projects)
    assert r["projects"] == 2
    assert r["dirty"] == 1
    assert r["marks"] == 4
    assert r["clean_repos"] == 1


@requires_git
def test_summary_reads_as_a_sentence(workspace):
    tool = ForgeTool(scan_roots=[str(workspace)])
    text = tool.summarise(tool.scan())
    assert "2 projects" in text
    assert "alpha" in text


def test_forge_summary_with_no_projects(tmp_path):
    assert "No git projects" in ForgeTool(scan_roots=[str(tmp_path)]).summarise([])


@pytest.mark.parametrize("name", ["node_modules", "venv", ".git", "Pods",
                                  "Packages", "vendor", "build.bak"])
def test_forge_skips_vendored_directories(name):
    assert forge_skip(name) is True


@pytest.mark.parametrize("name", ["src", "app", "core", "Assets"])
def test_forge_keeps_source_directories(name):
    assert forge_skip(name) is False


# ── Adapters ────────────────────────────────────────────────────────────────


@requires_git
async def test_forge_status_through_the_adapter(workspace):
    a = ForgeAdapter(ForgeTool(scan_roots=[str(workspace)]))
    r = await a.execute("status")
    assert r.success, r.error
    assert r.data["rollup"]["projects"] == 2
    assert "2 projects" in r.message


@requires_git
async def test_forge_project_lookup_is_case_insensitive(workspace):
    a = ForgeAdapter(ForgeTool(scan_roots=[str(workspace)]))
    assert (await a.execute("project", {"name": "ALPHA"})).success


@requires_git
async def test_forge_unknown_project_lists_what_exists(workspace):
    a = ForgeAdapter(ForgeTool(scan_roots=[str(workspace)]))
    r = await a.execute("project", {"name": "gamma"})
    assert r.success is False
    assert r.error_type is ErrorType.NOT_FOUND
    assert "alpha" in r.error


@requires_git
async def test_forge_marks_filter_by_kind(workspace):
    a = ForgeAdapter(ForgeTool(scan_roots=[str(workspace)]))
    r = await a.execute("list_marks", {"kind": "FIXME"})
    assert r.success
    assert all(m["kind"] == "FIXME" for m in r.data["marks"])


async def test_forge_rejects_an_unknown_mark_kind(tmp_path):
    a = ForgeAdapter(ForgeTool(scan_roots=[str(tmp_path)]))
    r = await a.execute("list_marks", {"kind": "NONSENSE"})
    assert r.error_type is ErrorType.INPUT


@requires_git
async def test_forge_uncommitted_lists_only_dirty_repos(workspace):
    a = ForgeAdapter(ForgeTool(scan_roots=[str(workspace)]))
    r = await a.execute("uncommitted")
    assert r.success
    assert [p["name"] for p in r.data["projects"]] == ["alpha"]


async def test_forge_health_degrades_with_no_projects(tmp_path):
    h = await ForgeAdapter(ForgeTool(scan_roots=[str(tmp_path)])).health_check()
    assert h.status is HealthStatus.DEGRADED


async def test_sentinel_scan_through_the_adapter(project):
    r = await SentinelAdapter(SentinelTool(project)).execute("scan")
    assert r.success, r.error
    assert r.data["summary"]["high"] >= 1
    assert "finding" in r.message


async def test_sentinel_severity_filter(project):
    r = await SentinelAdapter(SentinelTool(project)).execute("scan", {"severity": "low"})
    assert r.success
    assert all(f["severity"] == "low" for f in r.data["findings"])


async def test_sentinel_summary_is_degraded_when_something_is_exposed(project):
    """The scan working and finding something is not the scan failing — but it
    isn't a clean bill of health either."""
    r = await SentinelAdapter(SentinelTool(project)).execute("summary")
    assert r.success is True
    assert r.degraded is True


async def test_sentinel_summary_is_clean_on_a_clean_project(tmp_path):
    p = tmp_path / "clean"; p.mkdir()
    (p / "a.py").write_text("print(1)\n")
    r = await SentinelAdapter(SentinelTool(p)).execute("summary")
    assert r.success and r.degraded is False
    assert "No exposed secrets" in r.message


@requires_git
async def test_sentinel_history_through_the_adapter(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "c.py").write_text(f'K = "{FAKE_AWS}"\n')
    commit_all(repo, "leak")
    r = await SentinelAdapter(SentinelTool(repo)).execute("scan_history")
    assert r.success
    assert r.data["findings"]


async def test_sentinel_history_bounds_max_commits(tmp_path):
    a = SentinelAdapter(SentinelTool(tmp_path))
    assert (await a.execute("scan_history", {"max_commits": 99999})).error_type is ErrorType.INPUT


async def test_every_devtool_action_is_read_only(project, tmp_path):
    """Neither tool may be able to change anything: they run on a hunch, and a
    scanner that edits your repo is one you can't safely run that way."""
    for adapter in (SentinelAdapter(SentinelTool(project)),
                    ForgeAdapter(ForgeTool(scan_roots=[str(tmp_path)]))):
        for name, action in adapter.actions.items():
            assert action.read_only is True, f"{adapter.name}.{name} is not read-only"
            assert action.destructive is False, f"{adapter.name}.{name} is destructive"
