"""Every Python file in the repo must compile.

This exists because a mechanical rewrite of tools/mac_control.py shipped an
`await` inside a `lambda` — legal to *parse*, illegal to *compile*. The check
that let it through used ast.parse(), which builds the tree without running the
compiler's syntax checks. `await` outside an async function, `return` outside a
function, `yield` in a comprehension and duplicate parameter names all parse
cleanly and fail at compile.

The repo has 47 modules and most can't be imported in a test (they need
ChromaDB, Ollama, macOS, Google credentials). Compiling them needs none of
that, and catches the whole class of error that import-time smoke testing would
otherwise be the only guard against.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", "node_modules",
    "site-packages", "dist-packages", "_to_delete", "build", "dist",
}

# Substring/suffix matches, for directories an exact-name list misses. The repo
# currently contains `venv.icloud.bak` — 15,268 vendored .py files. An exact
# "venv" match doesn't catch that name, and collecting it would turn this file
# into 15,000 tests over third-party code we don't control.
SKIP_PATTERNS = ("venv", "virtualenv", ".egg-info")
SKIP_SUFFIXES = (".bak", ".old", ".orig")


def _skipped(path: Path) -> bool:
    for part in path.parts[:-1]:                       # directories only
        if part in SKIP_DIRS:
            return True
        low = part.lower()
        if any(pat in low for pat in SKIP_PATTERNS):
            return True
        if low.endswith(SKIP_SUFFIXES):
            return True
    return False


def _python_files():
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if not _skipped(path):
            yield path


ALL_FILES = list(_python_files())


def test_repo_has_python_files():
    """Guard the guard: a bad glob would make every test below vacuous."""
    assert len(ALL_FILES) >= 5, f"only found {len(ALL_FILES)} files — check SKIP_DIRS"


def test_vendored_directories_are_not_collected():
    """A vendored tree slipping through turns this file into thousands of
    tests over code we don't own — and one broken vendored file would fail
    the build for no useful reason."""
    assert len(ALL_FILES) < 500, (
        f"collected {len(ALL_FILES)} files — a vendored directory is leaking "
        f"through the skip list"
    )
    for path in ALL_FILES:
        parts = " ".join(path.parts).lower()
        assert "venv" not in parts and "site-packages" not in parts, path


@pytest.mark.parametrize(
    "path", ALL_FILES, ids=[str(p.relative_to(REPO_ROOT)) for p in ALL_FILES])
def test_file_compiles(path: Path):
    """Builtin compile(), not py_compile — no .pyc is written anywhere."""
    source = path.read_text(encoding="utf-8")
    try:
        compile(source, str(path), "exec", dont_inherit=True)
    except SyntaxError as exc:
        pytest.fail(
            f"{path.relative_to(REPO_ROOT)}:{exc.lineno} does not compile: "
            f"{exc.msg}\n    {(exc.text or '').strip()}")


def test_compile_catches_what_ast_parse_misses():
    """The exact gap that let the bug through: `await` inside a lambda parses
    cleanly and fails to compile."""
    import ast
    bad = "async def g():\n    return (lambda: (await f()))\n"
    ast.parse(bad)                                   # no error — this is the gap
    with pytest.raises(SyntaxError):
        compile(bad, "<canary>", "exec", dont_inherit=True)


# ── Import, not just compile ────────────────────────────────────────────────
#
# Compiling server.py is not enough. A module-level `_Path(...)` referring to an
# alias imported 600 lines further down compiles perfectly and raises NameError
# the moment the file is executed — which shipped, and took the server down on
# startup with nothing in the suite to catch it. Module-level statements only
# run on import, so importing is the only thing that exercises them.


def test_server_imports():
    """server.py has ~2700 lines of module-level statements. Compiling proves
    they parse; only importing proves they *run*."""
    import importlib

    module = importlib.import_module("server")
    assert module.app is not None


def test_server_module_level_names_resolve_in_order():
    """The specific failure: an alias used above the line that defines it.

    Guarded by a canary rather than by re-reading server.py, because the real
    protection is test_server_imports above — this documents the shape of the
    bug so the next person understands why that import test exists."""
    src = "X = _Alias('/tmp')\nfrom pathlib import Path as _Alias\n"
    compile(src, "<canary>", "exec", dont_inherit=True)   # compiles fine
    with pytest.raises(NameError):
        exec(src, {})                                     # and dies on execution
