"""Tests for core.bootstrap — the wiring that makes every tool reachable.

The headline case is the audit finding: six tools existed but had no branch in
``orchestrator._dispatch``, so the Planner could never call them. A test that
enumerates the orchestrator's tool attributes and asserts each is registered
turns that from "remember to add an elif" into a failing build.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from core.bootstrap import ACTION_ALIASES, TOOL_ALIASES, build_registry
from core.registry import ToolRegistry
from core.tool import HealthStatus

from tests.test_adapters_connected import FakeCalendar, FakeGmail, FakeSpotify
from tests.test_adapters_information import (
    FakeMarkets,
    FakeNews,
    FakePrayer,
    FakeSearch,
    FakeSports,
    FakeWeather,
)
from tests.test_adapters_local import (
    FakeContacts,
    FakeDocument,
    FakeFileManager,
    FakeMac,
    FakeReminders,
)


class FakeMemoryItem:
    def __init__(self, content, degraded=False):
        self.content = content
        self.id = "mem_1"
        self.metadata = {"degraded": degraded}


class FakeMemory:
    def __init__(self, degraded=False, count=7):
        self.degraded = degraded
        self._count = count

    async def retrieve(self, query, k=5, **kw):
        return [FakeMemoryItem("you live in High Wycombe")]

    async def recall_facts(self, query, k=5):
        return [FakeMemoryItem("a stored fact")]

    async def store(self, content, **kw):
        return FakeMemoryItem(content, degraded=self.degraded)

    def get_count(self):
        return self._count


class FakeLLM:
    async def health_check(self):
        return True


class FakeSummariser:
    llm = FakeLLM()

    async def summarise(self, text, max_words=150):
        return " ".join(text.split()[:max_words])


class FakeFinEx:
    async def chat(self, question, company="Bestway Cement", **kw):
        return {"answer": f"{company}: answer to {question}", "sql": "SELECT 1"}


class FakeForge:
    def scan(self): return []
    def find_projects(self, limit=12): return []
    def rollup(self, projects): return {"projects": 0, "dirty": 0, "unpushed": 0,
                                        "behind": 0, "marks": 0, "clean_repos": 0}
    def summarise(self, projects): return "No git projects found."


class FakeSentinel:
    project_dir = __import__("pathlib").Path("/tmp")
    def scan(self): return [], {"high": 0, "medium": 0, "low": 0}
    def scan_history(self, max_commits=400): return [], {"high": 0, "medium": 0, "low": 0}
    def summarise(self, findings, summary): return "Nothing exposed."


class FakeOrchestrator:
    """Mirrors the attribute names JarvisOrchestrator.__init__ actually sets."""

    def __init__(self):
        self.memory = FakeMemory()
        self.summariser = FakeSummariser()
        self._finex_built = 0
        self.weather = FakeWeather()
        self.websearch = FakeSearch()
        self.news = FakeNews()
        self.sports = FakeSports()
        self.markets = FakeMarkets()
        self.prayer = FakePrayer()
        self.files = FakeFileManager()
        self.contacts = FakeContacts()
        self.reminders = FakeReminders()
        self.document = FakeDocument()
        self.mac = FakeMac()
        self.spotify = FakeSpotify()
        self.forge = FakeForge()
        self.sentinel = FakeSentinel()
        self._gmail_built = 0
        self._calendar_built = 0

    def _resolve_temporal(self, phrase):
        return None if phrase == "gibberish" else f"2026-07-28T09:00:00 ({phrase})"

    @property
    def finex(self):
        self._finex_built += 1
        return FakeFinEx()

    # Lazy properties, as on the real orchestrator — touching them constructs
    # the agent and fires an OAuth refresh.
    @property
    def gmail(self):
        self._gmail_built += 1
        return FakeGmail()

    @property
    def calendar(self):
        self._calendar_built += 1
        return FakeCalendar()


@pytest.fixture
def orch():
    return FakeOrchestrator()


@pytest.fixture
def reg(orch):
    return build_registry(orch, is_mac=True)


# ── Coverage ────────────────────────────────────────────────────────────────


# The 14 user-facing tools...
TOOLS = {
    "weather", "websearch", "news", "sports", "markets", "prayer",
    "files", "contacts", "reminders", "document", "mac",
    "spotify", "gmail", "calendar",
}
# ...plus the orchestrator's own faculties, registered rather than
# special-cased so _dispatch stays a single lookup.
FACULTIES = {"memory", "summariser", "finex", "internal", "forge", "sentinel"}
EXPECTED = TOOLS | FACULTIES


def test_all_fourteen_tools_are_registered(reg):
    assert TOOLS <= set(reg.names)
    assert len(TOOLS) == 14


def test_faculties_are_registered_too(reg):
    assert FACULTIES <= set(reg.names)
    assert set(reg.names) == EXPECTED


def test_registry_covers_every_orchestrator_tool(orch, reg):
    """Guards the exact regression class from the audit: a tool that exists on
    the orchestrator but is unreachable because nobody wired it up."""
    tool_attrs = [
        a for a in dir(orch)
        if not a.startswith("_") and a not in ("gmail", "calendar", "finex")
    ]
    unregistered = [a for a in tool_attrs if a not in reg]
    assert unregistered == [], (
        f"orchestrator holds tools nothing can reach: {unregistered}"
    )


def test_previously_unreachable_tools_are_now_reachable(reg):
    """sports, markets, prayer, files, contacts had no dispatcher branch."""
    for name in ("sports", "markets", "prayer", "files", "contacts"):
        assert name in reg
        assert reg.get(name).actions


async def test_every_registered_action_resolves(reg):
    for tool in reg:
        for action in tool.actions:
            res = reg.resolve(tool.name, action)
            assert res.action == action


# ── Aliases ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("alias,expected", [
    ("email", "gmail"), ("mail", "gmail"), ("inbox", "gmail"),
    ("cal", "calendar"), ("schedule", "calendar"),
    ("web", "websearch"), ("search", "websearch"),
    ("system", "mac"), ("file", "files"), ("music", "spotify"),
])
def test_tool_aliases_resolve(reg, alias, expected):
    assert reg.get(alias).name == expected


@pytest.mark.parametrize("tool,alias,expected", [
    ("calendar", "get_events", "search_events"),
    ("calendar", "add_event", "create_event"),
    ("gmail", "read_emails", "get_inbox"),
    ("gmail", "send", "send_email"),
    ("weather", "get_weather", "get_current"),
    ("sports", "get_table", "get_standings"),
    ("markets", "get_quote", "get_price"),
    ("spotify", "next", "skip"),
    ("reminders", "add_reminder", "add"),
    ("files", "ls", "list_directory"),
])
def test_action_aliases_resolve(reg, tool, alias, expected):
    assert reg.resolve(tool, alias).action == expected


def test_no_alias_table_entry_points_at_a_missing_tool(reg):
    for tool_name in TOOL_ALIASES:
        assert tool_name in EXPECTED, f"alias table references unknown tool {tool_name!r}"
    for tool_name in ACTION_ALIASES:
        assert tool_name in EXPECTED, f"action alias table references unknown tool {tool_name!r}"


def test_every_action_alias_points_at_a_real_action(reg):
    """Registration already rejects this, but assert it explicitly — the old
    dispatcher hid the same mistake until a user hit that branch."""
    for tool_name, mapping in ACTION_ALIASES.items():
        actions = reg.get(tool_name).actions
        for alias, target in mapping.items():
            assert target in actions, f"{tool_name}.{alias} -> {target!r} does not exist"


# ── Lazy Google construction ────────────────────────────────────────────────


def test_registration_does_not_authenticate_google(orch):
    """Building GmailAgent/CalendarAgent fires an OAuth refresh. It must not
    happen just because the registry was constructed at server startup."""
    build_registry(orch, is_mac=True)
    assert orch._gmail_built == 0
    assert orch._calendar_built == 0


def test_google_actions_are_declared_before_authentication(orch):
    reg = build_registry(orch, is_mac=True)
    assert "get_inbox" in reg.get("gmail").actions
    assert "create_event" in reg.get("calendar").actions
    assert orch._gmail_built == 0, "declaring actions triggered authentication"


async def test_google_agent_is_built_on_first_execute(orch):
    reg = build_registry(orch, is_mac=True)
    assert orch._gmail_built == 0
    r = await reg.execute("gmail", "get_inbox")
    assert r.success
    assert orch._gmail_built == 1

    # Built once, then reused.
    await reg.execute("gmail", "get_inbox")
    assert orch._gmail_built == 1


async def test_lazy_wrapper_exports_mcp_declarations_without_authenticating(orch):
    reg = build_registry(orch, is_mac=True)
    decls = reg.mcp_declarations()
    names = {d["name"] for d in decls}
    assert "gmail.send_email" in names
    assert "calendar.create_event" in names
    assert orch._gmail_built == 0


# ── Shared instances ────────────────────────────────────────────────────────


def test_adapters_wrap_the_orchestrators_own_instances(orch, reg):
    """Two sets of tool objects would mean two Spotify tokens and two SQLite
    handles while the old dispatcher and the registry run side by side."""
    assert reg.get("spotify")._t is orch.spotify
    assert reg.get("files")._t is orch.files
    assert reg.get("reminders")._t is orch.reminders


def test_file_pending_store_is_shared_when_injected(orch):
    from core.adapters.local import _PendingStore
    store = _PendingStore()
    reg = build_registry(orch, pending_store=store, is_mac=True)
    assert reg.get("files").pending is store


# ── Health + introspection ──────────────────────────────────────────────────


async def test_health_check_returns_a_report_per_tool(reg):
    reports = await reg.health_check()
    assert set(reports) == EXPECTED
    for name, rep in reports.items():
        assert rep.status in set(HealthStatus), f"{name} returned a bogus status"


async def test_mac_reports_unavailable_off_darwin(orch):
    reg = build_registry(orch, is_mac=False)
    reports = await reg.health_check()
    assert reports["mac"].status is HealthStatus.UNAVAILABLE


def test_planner_catalogue_lists_every_tool_and_marks_destructive(reg):
    cat = reg.planner_catalogue()
    for name in EXPECTED:
        assert name in cat, f"{name} missing from the Planner catalogue"
    assert "[destructive]" in cat
    assert "send_email" in cat
    assert "get_scores" in cat


def test_describe_is_json_serialisable(reg):
    import json
    json.dumps(reg.describe())          # must not raise


def test_mcp_declarations_are_unique_and_well_formed(reg):
    decls = reg.mcp_declarations()
    names = [d["name"] for d in decls]
    assert len(names) == len(set(names)), "duplicate MCP tool names"
    for d in decls:
        assert d["inputSchema"]["type"] == "object"
        assert d["description"]
        assert "readOnlyHint" in d["annotations"]


def test_a_broken_adapter_does_not_take_out_the_registry(orch, caplog):
    """One malformed tool must not leave the assistant toolless."""
    class Exploding:
        name = "weather"          # collides deliberately

        @property
        def actions(self):
            raise RuntimeError("boom")

    reg = ToolRegistry()
    built = build_registry(orch, registry=reg, is_mac=True)
    assert len(built) == len(EXPECTED)


# ── Faculties ───────────────────────────────────────────────────────────────


async def test_memory_retrieve_through_the_registry(reg):
    r = await reg.execute("memory", "retrieve_context", {"query": "where do I live"})
    assert r.success
    assert "High Wycombe" in r.message


async def test_memory_store_flags_a_degraded_embedding(orch):
    """A hash-fallback embedding is deterministic noise — the memory is stored
    but can never be found by similarity. Silently poisoning the store was the
    old behaviour."""
    orch.memory = FakeMemory(degraded=True)
    reg = build_registry(orch, is_mac=True)
    r = await reg.execute("memory", "store_fact", {"content": "something"})
    assert r.success is True
    assert r.degraded is True
    assert r.meta["embedding"] == "hash-fallback"


async def test_memory_store_is_not_degraded_normally(reg):
    r = await reg.execute("memory", "store_fact", {"content": "x"})
    assert r.success and r.degraded is False


async def test_summariser_through_the_registry(reg):
    r = await reg.execute("summariser", "summarise",
                          {"text": "word " * 500, "max_words": 10})
    assert r.success
    assert len(r.message.split()) == 10


async def test_internal_resolve_temporal(reg):
    r = await reg.execute("internal", "resolve_temporal", {"phrase": "next Tuesday"})
    assert r.success
    assert "next Tuesday" in r.message


async def test_internal_unresolvable_phrase_is_not_found(reg):
    from core.tool import ErrorType
    r = await reg.execute("internal", "resolve_temporal", {"phrase": "gibberish"})
    assert r.success is False
    assert r.error_type is ErrorType.NOT_FOUND


async def test_finex_stays_lazy_until_called(orch):
    reg = build_registry(orch, is_mac=True)
    assert orch._finex_built == 0
    r = await reg.execute("finex", "chat", {"question": "what was revenue"})
    assert r.success
    assert orch._finex_built == 1


async def test_finex_uses_the_default_company(reg):
    r = await reg.execute("finex", "chat", {"question": "revenue?"})
    assert "Bestway Cement" in r.message


async def test_faculty_aliases_resolve(reg):
    assert reg.get("recall").name == "memory"
    assert reg.resolve("memory", "get_context").action == "retrieve_context"
    assert reg.resolve("summariser", "summarize").action == "summarise"
    assert reg.resolve("finex", "ask").action == "chat"
    assert reg.resolve("internal", "validate").action == "validate_output"
