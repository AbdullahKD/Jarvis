"""Shared fixtures.

Nothing here touches Ollama, Google, the network or macOS.

The environment variables are set at *import* time, before anything from the
application is imported, because ``config.settings`` reads them at module scope
and computes DATA_DIR / SQLITE_PATH / CHROMA_DIR once. Setting them in a
fixture would be too late — the paths would already point at the real
~/Desktop/Jarvis/data, and a test run would write into the live reminder and
evaluation databases.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# ── Redirect every on-disk store into a throwaway directory ─────────────────
# Must happen before any `config.settings` import. mkdtemp rather than pytest's
# tmp_path because this runs at collection time, before fixtures exist.
_TEST_HOME = Path(tempfile.mkdtemp(prefix="jarvis-tests-"))
os.environ.update(
    JARVIS_DATA_DIR=str(_TEST_HOME),
    JARVIS_LOGS_DIR=str(_TEST_HOME / "logs"),
    JARVIS_NOTES_DIR=str(_TEST_HOME / "notes"),
    CHROMA_DIR=str(_TEST_HOME / "chroma"),
    JARVIS_SQLITE_PATH=str(_TEST_HOME / "jarvis.db"),
    # Never open a browser for OAuth from a test, whatever else happens.
    JARVIS_INTERACTIVE_OAUTH="false",
)

import asyncio
from typing import Any, Dict

import pytest

from core.tool import Action, BaseTool, HealthReport, HealthStatus, ToolError, ToolResult


class FakeTool(BaseTool):
    """A tool with one of each interesting handler shape, so the base class's
    normalisation and error mapping can be tested without any real service."""

    _name = "fake"
    _description = "Test double covering every handler return shape."

    def __init__(self, *, health: HealthStatus = HealthStatus.OK) -> None:
        self._health = health
        self.calls: list[tuple[str, Dict[str, Any]]] = []
        super().__init__()

    def _register_actions(self) -> None:
        self.add_action(Action(
            name="echo", description="Return what it was given.",
            input_schema={"properties": {"text": {"type": "string"}},
                          "required": ["text"]},
            handler=self._echo,
        ))
        self.add_action(Action(
            name="tuple_shape", description="Handler returning (data, message).",
            input_schema={"properties": {}}, handler=self._tuple_shape,
        ))
        self.add_action(Action(
            name="legacy_dict", description="Handler returning the old dict shape.",
            input_schema={"properties": {"ok": {"type": "boolean"}}},
            handler=self._legacy_dict,
        ))
        self.add_action(Action(
            name="raises", description="Handler raising a typed ToolError.",
            input_schema={"properties": {"kind": {"type": "string"}}},
            handler=self._raises,
        ))
        self.add_action(Action(
            name="boom", description="Handler raising an untyped exception.",
            input_schema={"properties": {}}, handler=self._boom,
        ))
        self.add_action(Action(
            name="slow", description="Handler that sleeps.",
            input_schema={"properties": {"seconds": {"type": "number"}}},
            handler=self._slow, timeout=0.2,
        ))
        self.add_action(Action(
            name="wipe", description="Destructive no-op.",
            input_schema={"properties": {}}, handler=self._wipe,
            destructive=True, read_only=False,
        ))

    async def _echo(self, text: str):
        self.calls.append(("echo", {"text": text}))
        return {"echoed": text}, f"You said: {text}"

    async def _tuple_shape(self):
        return {"a": 1}, "tuple message"

    async def _legacy_dict(self, ok: bool = True):
        if ok:
            return {"success": True, "result": {"n": 7}, "message": "legacy ok"}
        return {"success": False, "error": "legacy failure"}

    async def _raises(self, kind: str = "upstream"):
        from core.tool import (ToolAuthError, ToolInputError, ToolNotFoundError,
                               ToolUnavailableError, ToolUpstreamError)
        mapping = {
            "auth": ToolAuthError, "input": ToolInputError,
            "unavailable": ToolUnavailableError, "not_found": ToolNotFoundError,
            "upstream": ToolUpstreamError,
        }
        raise mapping[kind](f"deliberate {kind} failure")

    async def _boom(self):
        return 1 / 0

    async def _slow(self, seconds: float = 1.0):
        await asyncio.sleep(seconds)
        return {"slept": seconds}, "done"

    async def _wipe(self):
        return {"wiped": True}, "wiped"

    async def _check_health(self) -> HealthReport:
        return HealthReport(status=self._health, tool=self.name, detail="test double")


@pytest.fixture
def fake_tool() -> FakeTool:
    return FakeTool()


@pytest.fixture
def registry(fake_tool):
    from core.registry import ToolRegistry
    r = ToolRegistry()
    r.register(fake_tool, aliases=("fakey", "double"),
               action_aliases={"say": "echo", "repeat": "echo"})
    return r


# ═══════════════════════════════════════════════════════════════════════════
# Harness: the real orchestrator, with only its boundaries faked
# ═══════════════════════════════════════════════════════════════════════════

from tests.fakes import (                       # noqa: E402  (after env setup)
    FakeCalendarService,
    FakeGmailService,
    FakeOllamaClient,
    FakeShell,
    install_network_block,
)


@pytest.fixture
def test_home() -> Path:
    """The throwaway data directory this run is using."""
    return _TEST_HOME


@pytest.fixture(autouse=True)
def no_network():
    """Fail loudly on any outbound socket.

    A backstop, not the fake itself. Its job is to turn "this test silently hit
    the real ESPN API and passed" into an immediate, named failure. Loopback
    stays open so ChromaDB and SQLite are unaffected.
    """
    undo = install_network_block(allow_local=True)
    try:
        yield
    finally:
        undo()


@pytest.fixture
def fake_llm() -> FakeOllamaClient:
    return FakeOllamaClient()


@pytest.fixture
def fake_shell() -> FakeShell:
    return FakeShell()


@pytest.fixture
def fake_gmail_service() -> FakeGmailService:
    return FakeGmailService()


@pytest.fixture
def fake_calendar_service() -> FakeCalendarService:
    return FakeCalendarService()


@pytest.fixture
def pretend_macos(monkeypatch):
    """Make the macOS-only code paths reachable off a Mac.

    MacControlTool checks ``platform_guard.is_mac()`` twice — once at module
    level and once in __init__, where it replaces every coroutine method with a
    "macOS only" stub. Without this, the mac tool is inert everywhere except a
    developer's own Mac, so its AppleScript output parsing — the part with the
    regexes, and therefore the bugs — would only ever be exercised there.

    Patched in both namespaces because mac_control imports the name directly.
    """
    import tools.mac_control as mac_mod
    import tools.platform_guard as guard

    monkeypatch.setattr(guard, "is_mac", lambda: True)
    monkeypatch.setattr(mac_mod, "is_mac", lambda: True)
    yield


@pytest.fixture
def jarvis(monkeypatch, tmp_path, fake_llm, fake_shell, pretend_macos,
           fake_gmail_service, fake_calendar_service):
    """A real JarvisOrchestrator with every external boundary faked.

    This is the fixture the integration tests want. Everything inside the
    process is the production code path — the real router, planner, critic,
    executor, registry, adapters and tools. Only four things are replaced:

    * the LLM client (deterministic, records prompts)
    * the Gmail and Calendar service objects (in-memory, real API shapes)
    * subprocess.run inside mac_control (canned macOS output)
    * the on-disk stores (temp dir, via the env vars set at import)

    Attached for convenience: ``jarvis.fake_llm``, ``.fake_shell``,
    ``.fake_gmail``, ``.fake_calendar``.

    Skips rather than fails if the heavy dependencies aren't installed, so the
    fast unit tests still run in a bare environment.
    """
    pytest.importorskip("chromadb", reason="orchestrator needs ChromaDB")
    pytest.importorskip("aiohttp", reason="orchestrator needs aiohttp")

    import orchestrator as orch_mod
    import tools.contacts as contacts_mod
    import tools.mac_control as mac_mod

    # Contacts persist to ~/.jarvis/contacts.json — redirect before construction.
    monkeypatch.setattr(contacts_mod, "CONTACTS_PATH", tmp_path / "contacts.json")
    # mac_control shells out; _run_off_loop and _run_applescript both go through
    # this one name, so patching it covers every path.
    monkeypatch.setattr(mac_mod.subprocess, "run", fake_shell.run)
    monkeypatch.setattr(orch_mod, "OllamaClient", lambda *a, **k: fake_llm)

    # build_registry() is called inside __init__ and auto-detects the platform.
    # Force it, so the mac adapter is live rather than reporting UNAVAILABLE.
    import core.bootstrap as bootstrap_mod
    real_build = bootstrap_mod.build_registry
    monkeypatch.setattr(
        orch_mod, "build_registry",
        lambda o, **kw: real_build(o, **{**kw, "is_mac": True}))

    j = orch_mod.JarvisOrchestrator()

    # Google agents are lazy properties backed by _gmail/_calendar. Pre-seed
    # them with agents wired to the fake services, so nothing authenticates.
    from agents.calendar_agent import CalendarAgent
    from agents.gmail_agent import GmailAgent

    gmail = GmailAgent.__new__(GmailAgent)
    gmail.service = fake_gmail_service
    gmail.auth_error = None
    _seed_agent_defaults(gmail)

    cal = CalendarAgent.__new__(CalendarAgent)
    cal.service = fake_calendar_service
    cal.auth_error = None
    _seed_agent_defaults(cal)

    j._gmail = gmail
    j._calendar = cal

    j.fake_llm = fake_llm
    j.fake_shell = fake_shell
    j.fake_gmail = fake_gmail_service
    j.fake_calendar = fake_calendar_service
    return j


def _seed_agent_defaults(agent: Any) -> None:
    """Fill in attributes the real __init__ would have set.

    __new__ skips __init__ deliberately: running it would attempt OAuth. Any
    attribute the agent's methods read must be set here, and a missing one
    shows up immediately as an AttributeError in the test rather than as
    mysterious behaviour later.
    """
    for name, value in (("_mock_inbox", []), ("_mock_events", []),
                        ("user_email", "me@example.com")):
        if not hasattr(agent, name):
            setattr(agent, name, value)
