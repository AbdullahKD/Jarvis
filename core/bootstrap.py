"""
Wire every Jarvis tool into a registry.

This is the one place that knows which adapters exist and what the Planner is
allowed to call them. Two consequences worth stating:

* **Reachability becomes structural.** In the current dispatcher, sports,
  markets, prayer times, file manager, contacts and the brain indexer have no
  branch at all — the Planner can emit a subtask for them and it comes back
  "Unknown agent/action". Registering a tool here is what makes it callable,
  and ``test_registry_covers_every_orchestrator_tool`` fails if one is missed.

* **The alias table is visible.** Today the Planner's naming inconsistency
  (``email``/``gmail``/``mail``, ``get_events``/``search_events``) is absorbed
  by hand-written ``elif action in (...)`` clauses spread through 240 lines.
  Collected here, they can be checked at registration and printed into the
  Planner prompt from the same source.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from core.adapters.cognitive import (
    FinExAdapter,
    InternalAdapter,
    MemoryAdapter,
    SummariserAdapter,
)
from core.adapters.connected import CalendarAdapter, GmailAdapter, SpotifyAdapter
from core.adapters.devtools import ForgeAdapter, SentinelAdapter
from core.adapters.information import (
    MarketsAdapter,
    NewsAdapter,
    PrayerTimesAdapter,
    SportsAdapter,
    WeatherAdapter,
    WebSearchAdapter,
)
from core.adapters.local import (
    ContactsAdapter,
    DocumentAdapter,
    FileManagerAdapter,
    MacControlAdapter,
    RemindersAdapter,
    _PendingStore,
)
from core.registry import ToolRegistry

logger = logging.getLogger("jarvis.bootstrap")


# Aliases the Planner LLM is known to emit. Keeping the mapping declarative
# means adding a synonym is a one-line change instead of another elif.
TOOL_ALIASES: dict[str, tuple[str, ...]] = {
    "gmail":     ("email", "mail", "inbox"),
    "calendar":  ("cal", "events", "schedule"),
    "websearch": ("web", "search", "web_search", "browse"),
    "mac":       ("system", "mac_control", "maccontrol"),
    "files":     ("file", "file_manager", "filemanager", "filesystem"),
    "markets":   ("market", "stocks", "finance_prices"),
    "sports":    ("sport", "scores"),
    "prayer":    ("prayer_times", "prayertimes", "salah"),
    "reminders": ("reminder",),
    "contacts":  ("contact", "addressbook", "address_book"),
    "document":  ("documents", "doc"),
    "news":      ("headlines",),
    "weather":   ("forecast",),
    "spotify":   ("music", "player"),
    "memory":    ("recall", "remember"),
    "summariser":("summarise", "summarizer", "summary"),
    "finex":     ("finance", "financials"),
    "forge":     ("repos", "projects", "git"),
    "sentinel":  ("security", "secrets", "audit"),
}

# Action synonyms, per tool. Same reasoning.
ACTION_ALIASES: dict[str, dict[str, str]] = {
    "calendar": {
        "get_events": "search_events",
        "list_events": "search_events",
        "read_calendar": "search_events",
        "add_event": "create_event",
        "schedule_event": "create_event",
        "cancel_event": "delete_event",
    },
    "gmail": {
        "read_emails": "get_inbox",
        "check_email": "get_inbox",
        "list_emails": "get_inbox",
        "read_email": "get_email_body",
        "search": "search_emails",
        "send": "send_email",
        "reply": "draft_email",
    },
    "weather": {
        "get_weather": "get_current",
        "current": "get_current",
        "forecast": "get_forecast",
    },
    "sports": {
        "get_fixtures": "get_scores",
        "get_results": "get_scores",
        "get_table": "get_standings",
        "team": "search_team",
    },
    "markets": {
        "get_stock": "get_price",
        "get_quote": "get_price",
        "get_markets": "get_all",
    },
    "news": {"get_news": "get_headlines", "headlines": "get_headlines"},
    "spotify": {
        "play_track": "play",
        "play_by_name": "play",
        "next": "skip",
        "search_tracks": "search",
        "now_playing": "get_now_playing",
    },
    "files": {
        "list": "list_directory",
        "ls": "list_directory",
        "read": "read_file",
        "find": "search",
    },
    "reminders": {"add_reminder": "add", "list_reminders": "list_pending",
                  "complete_reminder": "complete", "delete_reminder": "delete"},
    "contacts": {"lookup": "find", "list_contacts": "list_all"},
    "document": {"extract_text": "extract", "read_document": "extract"},
    "prayer": {"get_prayer_times": "get_times", "next_prayer": "get_next_prayer"},
    "memory": {"retrieve": "retrieve_context", "get_context": "retrieve_context",
               "remember": "store_fact", "store": "store_fact"},
    "summariser": {"summarize": "summarise", "condense": "summarise"},
    "finex": {"ask": "chat", "query": "chat", "analyse": "chat"},
    "internal": {"resolve_date": "resolve_temporal", "validate": "validate_output"},
    "forge": {"get_status": "status", "repos": "status", "todos": "list_marks",
              "list_todos": "list_marks", "dirty": "uncommitted",
              "unpushed": "uncommitted"},
    "sentinel": {"check": "scan", "audit": "scan", "secrets": "scan",
                 "history": "scan_history", "git_history": "scan_history"},
}


def build_registry(orchestrator: Any, *,
                   registry: Optional[ToolRegistry] = None,
                   pending_store: Optional[_PendingStore] = None,
                   is_mac: Optional[bool] = None) -> ToolRegistry:
    """Register every tool held by ``orchestrator``.

    Takes the orchestrator rather than constructing tools itself, so the
    registry wraps the *same* instances the existing code paths use. During
    migration both the old dispatcher and the registry are live at once; two
    sets of tool objects would mean two Spotify tokens and two SQLite handles.

    Google agents are reached through the orchestrator's lazy properties, so
    registration does not itself trigger an OAuth refresh.
    """
    reg = registry if registry is not None else ToolRegistry()
    store = pending_store if pending_store is not None else _PendingStore()

    def _add(adapter: Any) -> None:
        try:
            reg.register(adapter,
                         aliases=TOOL_ALIASES.get(adapter.name, ()),
                         action_aliases=ACTION_ALIASES.get(adapter.name))
        except Exception as exc:  # noqa: BLE001
            # One malformed adapter must not leave the whole assistant
            # toolless. Log loudly; health_check will show it missing.
            logger.error("failed to register %s: %s", adapter.name, exc)

    # ── Information ─────────────────────────────────────────────────────────
    _add(WeatherAdapter(orchestrator.weather))
    _add(WebSearchAdapter(orchestrator.websearch))
    _add(NewsAdapter(orchestrator.news))
    _add(SportsAdapter(orchestrator.sports))
    _add(MarketsAdapter(orchestrator.markets))
    _add(PrayerTimesAdapter(orchestrator.prayer))

    # ── Local ───────────────────────────────────────────────────────────────
    _add(FileManagerAdapter(orchestrator.files, pending=store))
    _add(ContactsAdapter(orchestrator.contacts))
    _add(RemindersAdapter(orchestrator.reminders))
    _add(DocumentAdapter(orchestrator.document))
    _add(MacControlAdapter(orchestrator.mac, is_mac=is_mac))

    # ── Connected ───────────────────────────────────────────────────────────
    _add(SpotifyAdapter(orchestrator.spotify))
    _add(_LazyGoogle(GmailAdapter, lambda: orchestrator.gmail, "gmail",
                     "Read, search, send, reply to, draft and archive email."))
    _add(_LazyGoogle(CalendarAdapter, lambda: orchestrator.calendar, "calendar",
                     "Read, create and delete Google Calendar events, and check for clashes."))

    # ── Developer tools ─────────────────────────────────────────────────────
    # Both were web-page-only: the logic lived behind a route in server.py, so
    # you could look at repo health but not ask about it.
    _add(ForgeAdapter(orchestrator.forge))
    _add(SentinelAdapter(orchestrator.sentinel))

    # ── The orchestrator's own faculties ────────────────────────────────────
    # Registered rather than special-cased, so _dispatch stays a single lookup
    # even for plans that mention memory or summarisation.
    _add(MemoryAdapter(orchestrator.memory))
    _add(SummariserAdapter(orchestrator.summariser))
    _add(InternalAdapter(orchestrator._resolve_temporal))
    # FinEx stays lazy — importing it pulls psycopg2 and ChromaDB.
    _add(FinExAdapter(lambda: orchestrator.finex))

    logger.info("registry ready — %d tools, %d actions",
                len(reg), sum(len(t.actions) for t in reg))
    return reg


class _LazyGoogle:
    """Defer building a Google adapter until something actually calls it.

    ``orchestrator.gmail`` is a property that constructs ``GmailAgent()`` on
    first access, which fires an OAuth token refresh over the network. Touching
    it during registration would put that refresh on the server-startup path —
    exactly what the lazy properties were introduced to avoid.

    Implements the ``Tool`` surface by forwarding to the real adapter, built on
    first use. Actions are declared eagerly from the adapter class so the
    registry, the Planner catalogue and the MCP declarations are all complete
    before any authentication happens.
    """

    def __init__(self, adapter_cls: type, get_agent: Any, name: str, description: str):
        self._adapter_cls = adapter_cls
        self._get_agent = get_agent
        self._name = name
        self._description = description
        self._real: Any = None
        # Declaring actions needs an adapter instance, but building one only
        # needs an object with `is_mock`/`auth_error` — not a live connection.
        self._template = adapter_cls(_UnauthenticatedStub())

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def actions(self):
        return self._template.actions

    def _resolve(self):
        if self._real is None:
            self._real = self._adapter_cls(self._get_agent())
        return self._real

    async def execute(self, action, params=None, *, timeout=None):
        return await self._resolve().execute(action, params, timeout=timeout)

    async def health_check(self):
        return await self._resolve().health_check()

    def mcp_declarations(self):
        return self._template.mcp_declarations()

    def __repr__(self) -> str:  # pragma: no cover
        state = "built" if self._real is not None else "lazy"
        return f"<_LazyGoogle {self._name!r} {state}>"


class _UnauthenticatedStub:
    """Stands in for a Google agent purely so an adapter can be constructed for
    schema declaration. Every call raises — it must never serve real traffic."""

    is_mock = True
    auth_error = "not yet authenticated"

    def __getattr__(self, item):
        async def _never(*_a, **_kw):
            raise RuntimeError(
                f"_UnauthenticatedStub.{item} called — the lazy adapter was not resolved"
            )
        return _never


__all__ = ["build_registry", "TOOL_ALIASES", "ACTION_ALIASES"]
