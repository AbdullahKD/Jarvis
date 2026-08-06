#!/usr/bin/env python3
"""
Jarvis end-to-end health check — every agent and every tool.
=============================================================

A single, non-destructive smoke test that answers one question:
"Is every agent and tool actually working right now?"

Two tiers per component:

  • STRUCTURAL — import + construct the component and confirm its public
    interface exists. Pure, deterministic, no network. Always runs.
  • LIVE       — a real, READ-ONLY probe (weather for London, AAPL price,
    a router classification via Ollama, a Gmail inbox peek, …). Runs by
    default; disable with --offline. Anything whose backing service or key
    is missing is reported as SKIP, never FAIL.

Destructive actions are NEVER performed — no email is sent, no calendar
event is created/deleted, no volume/brightness is changed, no file outside
a temp dir is written. The only writes are a self-deleting temp .txt
(DocumentTool) and an add→delete round-trip on the reminders SQLite store.

Usage:
    python -m tests.test_jarvis_health            # structural + live
    python tests/test_jarvis_health.py --offline  # structural only
    python tests/test_jarvis_health.py --verbose  # show tracebacks
    python tests/test_jarvis_health.py --strict    # live failures fail the run
    python tests/test_jarvis_health.py --json out.json

Exit code: 0 if all STRUCTURAL checks pass (and, with --strict, all LIVE
checks too); non-zero otherwise — so it slots into CI.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_ICON = {PASS: "✅", FAIL: "❌", SKIP: "⏭️ "}

LIVE_TIMEOUT = 20.0   # seconds — hard cap per live probe so one hang can't stall CI


# ── Result bookkeeping ──────────────────────────────────────────────────────


@dataclass
class CheckResult:
    section: str
    name: str
    tier: str            # "structural" | "live"
    status: str          # PASS | FAIL | SKIP
    detail: str = ""
    ms: float = 0.0


@dataclass
class Runner:
    offline: bool = False
    verbose: bool = False
    strict: bool = False
    results: List[CheckResult] = field(default_factory=list)
    loop: asyncio.AbstractEventLoop = field(default=None, repr=False)

    # gating flags, filled in by detect()
    ollama: bool = False
    google: bool = False
    mac: bool = False

    def detect(self) -> None:
        """Probe the environment once so live checks can SKIP cleanly."""
        try:
            from tools.platform_guard import is_mac
            self.mac = bool(is_mac())
        except Exception:
            self.mac = sys.platform == "darwin"
        # Google creds present?
        self.google = (ROOT / "token.json").exists() or (ROOT / "credentials.json").exists()
        # Ollama reachable? (only worth checking when not offline)
        if not self.offline:
            self.ollama = self._probe_ollama()

    def _probe_ollama(self) -> bool:
        try:
            from config.llm_client import OllamaClient
            llm = OllamaClient()

            async def _ping() -> bool:
                out = await asyncio.wait_for(
                    llm.chat([{"role": "user", "content": "say ok"}], max_tokens=5),
                    timeout=10.0,
                )
                return bool(out)

            return bool(self.loop.run_until_complete(_ping()))
        except Exception:
            return False

    # ── core recorder ────────────────────────────────────────────────────
    def run(
        self,
        section: str,
        name: str,
        tier: str,
        fn: Callable[[], Any],
        *,
        requires: Optional[str] = None,
    ) -> CheckResult:
        # Gating
        if tier == "live" and self.offline:
            return self._add(section, name, tier, SKIP, "offline mode")
        if tier == "live" and requires == "ollama" and not self.ollama:
            return self._add(section, name, tier, SKIP, "Ollama not reachable")
        if tier == "live" and requires == "mac" and not self.mac:
            return self._add(section, name, tier, SKIP, "not macOS")
        if tier == "live" and requires == "google" and not self.google:
            return self._add(section, name, tier, SKIP, "no Google credentials")

        t0 = time.perf_counter()
        try:
            res = fn()
            if inspect.iscoroutine(res):
                res = self.loop.run_until_complete(asyncio.wait_for(res, LIVE_TIMEOUT))
            ok, detail = self._interpret(res)
            status = PASS if ok else FAIL
        except _SkipProbe as exc:
            return self._add(section, name, tier, SKIP, str(exc),
                             (time.perf_counter() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001
            status = FAIL
            detail = f"{type(exc).__name__}: {exc}"
            if self.verbose:
                traceback.print_exc()
        ms = (time.perf_counter() - t0) * 1000
        return self._add(section, name, tier, status, detail, ms)

    @staticmethod
    def _interpret(res: Any) -> tuple[bool, str]:
        """Normalise a probe return value into (ok, detail)."""
        if res is None:
            return True, ""
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], bool):
            return res[0], str(res[1])
        if isinstance(res, bool):
            return res, ""
        if isinstance(res, dict):
            if "success" in res:
                ok = bool(res["success"])
                detail = res.get("error", "") if not ok else _short(res)
                return ok, detail
            return True, _short(res)
        if isinstance(res, (list, str)):
            return True, _short(res)
        return True, _short(res)

    def _add(self, section, name, tier, status, detail="", ms=0.0) -> CheckResult:
        r = CheckResult(section, name, tier, status, _trim(detail), ms)
        self.results.append(r)
        print(f"  {_ICON[status]} [{tier:<10}] {name:<28} {_trim(detail, 70)}")
        return r

    # ── reporting ──────────────────────────────────────────────────────────
    def summary(self) -> int:
        n = lambda s, t=None: sum(  # noqa: E731
            1 for r in self.results if r.status == s and (t is None or r.tier == t)
        )
        print("\n" + "=" * 78)
        print(" SUMMARY")
        print("=" * 78)
        print(f"  Structural : {n(PASS,'structural')} pass, "
              f"{n(FAIL,'structural')} fail, {n(SKIP,'structural')} skip")
        print(f"  Live       : {n(PASS,'live')} pass, "
              f"{n(FAIL,'live')} fail, {n(SKIP,'live')} skip")
        print(f"  Environment: Ollama={'up' if self.ollama else 'down'}  "
              f"Google={'yes' if self.google else 'no'}  "
              f"macOS={'yes' if self.mac else 'no'}  "
              f"mode={'offline' if self.offline else 'live'}")

        struct_fail = n(FAIL, "structural")
        live_fail = n(FAIL, "live")
        if struct_fail:
            print(f"\n  ❌ {struct_fail} STRUCTURAL failure(s) — these are real bugs.")
        if live_fail:
            tag = "❌" if self.strict else "⚠️ "
            print(f"  {tag} {live_fail} LIVE failure(s) — usually an external "
                  f"service/key issue (see details above).")
        if not struct_fail and not live_fail:
            print("\n  ✅ All checks green.")

        exit_code = 1 if struct_fail or (self.strict and live_fail) else 0
        print(f"\n  exit code: {exit_code}")
        return exit_code


class _SkipProbe(Exception):
    """Raise inside a probe to report SKIP with a reason."""


def _is_mock(agent: Any) -> bool:
    """is_mock is a @property on the Google agents; tolerate either shape."""
    m = getattr(agent, "is_mock", False)
    try:
        return bool(m() if callable(m) else m)
    except Exception:
        return True


def _short(obj: Any) -> str:
    try:
        if isinstance(obj, dict):
            keys = [k for k in obj if k != "success"][:4]
            return "{" + ", ".join(f"{k}={_trim(str(obj[k]),18)}" for k in keys) + "}"
        if isinstance(obj, (list, tuple)):
            return f"{len(obj)} item(s)"
        return _trim(str(obj), 60)
    except Exception:
        return ""


def _trim(s: Any, n: int = 90) -> str:
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


# ── Probes ──────────────────────────────────────────────────────────────────
#
# Each register_* function adds its structural + live checks to the runner.
# Components are constructed lazily inside try/except so one bad constructor
# never aborts the whole run.


def register_subsystems(r: Runner, llm) -> None:
    sec = "SUBSYSTEMS"

    def voice_config():
        from voice.config import load_config
        cfg = load_config()
        assert cfg.elevenlabs_fast_model and cfg.elevenlabs_model
        return True, f"fast={cfg.elevenlabs_fast_model}, model={cfg.elevenlabs_model}"
    r.run(sec, "voice.config", "structural", voice_config)

    def reminders_db():
        from tools.reminders import ReminderStore
        store = ReminderStore()
        rid = store.add(title="__healthcheck__", offset_minutes=1)
        ok = store.delete(rid)
        return ok, "add→delete round-trip on SQLite"
    r.run(sec, "reminders SQLite", "structural", reminders_db)

    def chroma_memory():
        from memory.memory_agent import MemoryAgent
        mem = MemoryAgent(llm)
        c = mem.get_count()
        return True, f"chroma collection reachable, {c} memories"
    r.run(sec, "memory / ChromaDB", "structural", chroma_memory)


def register_agents(r: Runner, llm) -> None:
    sec = "AGENTS"
    shared = {}  # carry a live TaskPlan to critic/evaluator if planner succeeds

    # Router
    def router_struct():
        from agents.router import RouterAgent
        shared["router"] = RouterAgent(llm)
        assert callable(shared["router"].route)
        return True, "constructed"
    r.run(sec, "RouterAgent", "structural", router_struct)

    async def router_live():
        d = await shared["router"].route("what's the weather in London today?")
        agent = getattr(d, "primary_agent", None)
        return True, f"routed → {getattr(agent,'value',agent)}"
    if "router" in shared:
        r.run(sec, "RouterAgent.route", "live", router_live, requires="ollama")

    # Planner
    def planner_struct():
        from agents.planner import PlannerAgent
        shared["planner"] = PlannerAgent(llm)
        return True, "constructed"
    r.run(sec, "PlannerAgent", "structural", planner_struct)

    async def planner_live():
        plan = await shared["planner"].plan("what is the weather in London")
        shared["plan"] = plan
        n = len(getattr(plan, "subtasks", []) or [])
        return True, f"plan with {n} subtask(s), intent={getattr(plan,'intent','?')}"
    if "planner" in shared:
        r.run(sec, "PlannerAgent.plan", "live", planner_live, requires="ollama")

    # Critic
    def critic_struct():
        from agents.critic import CriticAgent
        shared["critic"] = CriticAgent(llm)
        return True, "constructed"
    r.run(sec, "CriticAgent", "structural", critic_struct)

    async def critic_live():
        if "plan" not in shared:
            raise _SkipProbe("no live plan available (planner live skipped)")
        verdict = await shared["critic"].review_plan(shared["plan"])
        return True, f"verdict approved={getattr(verdict,'approved','?')}"
    if "critic" in shared:
        r.run(sec, "CriticAgent.review_plan", "live", critic_live, requires="ollama")

    # Evaluator (pure, no network)
    def evaluator_struct():
        from agents.evaluator import EvaluatorAgent
        ev = EvaluatorAgent()
        assert callable(ev.evaluate)
        return True, "constructed"
    r.run(sec, "EvaluatorAgent", "structural", evaluator_struct)

    # Summariser
    def summariser_struct():
        from agents.summariser import SummariserAgent
        shared["summariser"] = SummariserAgent(llm)
        return True, "constructed"
    r.run(sec, "SummariserAgent", "structural", summariser_struct)

    async def summariser_live():
        text = ("Jarvis is a multi-agent assistant. " * 40)
        out = await shared["summariser"].summarise(text, max_words=20)
        return bool(out and out.strip()), f"{len(out.split())} words out"
    if "summariser" in shared:
        r.run(sec, "SummariserAgent.summarise", "live", summariser_live, requires="ollama")

    # Memory agent
    def memory_struct():
        from memory.memory_agent import MemoryAgent
        shared["memory"] = MemoryAgent(llm)
        return True, "constructed"
    r.run(sec, "MemoryAgent", "structural", memory_struct)

    async def memory_live():
        mems = await shared["memory"].retrieve("test query")
        return True, f"retrieve() → {len(mems)} memories (read-only)"
    if "memory" in shared:
        r.run(sec, "MemoryAgent.retrieve", "live", memory_live, requires="ollama")

    # Calendar (Google) — read-only
    def calendar_struct():
        from agents.calendar_agent import CalendarAgent
        cal = CalendarAgent()
        shared["calendar"] = cal
        return True, f"constructed (mock={_is_mock(cal)})"
    r.run(sec, "CalendarAgent", "structural", calendar_struct)

    async def calendar_live():
        cal = shared.get("calendar")
        if cal is None:
            raise _SkipProbe("constructor failed")
        if _is_mock(cal):
            raise _SkipProbe("mock mode (no live Google auth)")
        data = await cal.search_events(query="", max_results=1)
        return data
    if "calendar" in shared:
        r.run(sec, "CalendarAgent.search_events", "live", calendar_live, requires="google")

    # Gmail (Google) — read-only
    def gmail_struct():
        from agents.gmail_agent import GmailAgent
        gm = GmailAgent()
        shared["gmail"] = gm
        return True, f"constructed (mock={_is_mock(gm)})"
    r.run(sec, "GmailAgent", "structural", gmail_struct)

    async def gmail_live():
        gm = shared.get("gmail")
        if gm is None:
            raise _SkipProbe("constructor failed")
        if _is_mock(gm):
            raise _SkipProbe("mock mode (no live Google auth)")
        data = await gm.get_inbox(max_results=1)
        return data
    if "gmail" in shared:
        r.run(sec, "GmailAgent.get_inbox", "live", gmail_live, requires="google")

    # FinEx — needs Postgres + Chroma + Ollama
    def finex_struct():
        from agents.finex_agent import FinExAgent
        shared["finex"] = FinExAgent()
        return True, "constructed"
    r.run(sec, "FinExAgent", "structural", finex_struct)

    async def finex_live():
        fx = shared.get("finex")
        if fx is None:
            raise _SkipProbe("constructor failed")
        data = await fx.list_companies()
        return data
    if "finex" in shared:
        r.run(sec, "FinExAgent.list_companies", "live", finex_live)


def register_tools(r: Runner, llm) -> None:
    sec = "TOOLS"

    # Helper: structural-construct then optional live probe
    def _struct(name, factory, store, key):
        def fn():
            store[key] = factory()
            return True, "constructed"
        r.run(sec, name, "structural", fn)

    store: dict = {}

    # Weather
    _struct("WeatherTool", lambda: __import__("tools.weather", fromlist=["WeatherTool"]).WeatherTool(), store, "weather")
    async def weather_live():
        return await store["weather"].get_current_for_location("London")
    if "weather" in store:
        r.run(sec, "WeatherTool.get_current", "live", weather_live)

    # Web search
    _struct("WebSearchTool", lambda: __import__("tools.web_search", fromlist=["WebSearchTool"]).WebSearchTool(), store, "web")
    async def web_live():
        return await store["web"].search("python programming language", max_results=3)
    if "web" in store:
        r.run(sec, "WebSearchTool.search", "live", web_live)

    # News
    _struct("NewsTool", lambda: __import__("tools.news", fromlist=["NewsTool"]).NewsTool(), store, "news")
    async def news_live():
        return await store["news"].get_headlines(max_items=2)
    if "news" in store:
        r.run(sec, "NewsTool.get_headlines", "live", news_live)

    # Markets
    _struct("MarketsTool", lambda: __import__("tools.markets", fromlist=["MarketsTool"]).MarketsTool(), store, "markets")
    async def markets_live():
        return await store["markets"].get_price("AAPL")
    if "markets" in store:
        r.run(sec, "MarketsTool.get_price", "live", markets_live)

    # Sports
    _struct("SportsTool", lambda: __import__("tools.sports", fromlist=["SportsTool"]).SportsTool(), store, "sports")
    async def sports_live():
        return await store["sports"].get_scores("premier_league", limit=3)
    if "sports" in store:
        r.run(sec, "SportsTool.get_scores", "live", sports_live)

    # Prayer times
    _struct("PrayerTimesTool", lambda: __import__("tools.prayer_times", fromlist=["PrayerTimesTool"]).PrayerTimesTool(), store, "prayer")
    async def prayer_live():
        return await store["prayer"].get_times()
    if "prayer" in store:
        r.run(sec, "PrayerTimesTool.get_times", "live", prayer_live)

    # Spotify
    _struct("SpotifyTool", lambda: __import__("tools.spotify", fromlist=["SpotifyTool"]).SpotifyTool(llm=llm), store, "spotify")
    async def spotify_live():
        try:
            data = await store["spotify"].search("daft punk", limit=1) if "limit" in inspect.signature(store["spotify"].search).parameters else await store["spotify"].search("daft punk")
        except TypeError:
            data = await store["spotify"].search("daft punk")
        if isinstance(data, dict) and not data.get("success") and "auth" in str(data.get("error", "")).lower():
            raise _SkipProbe("Spotify not authorised")
        return data
    if "spotify" in store:
        r.run(sec, "SpotifyTool.search", "live", spotify_live)

    # macOS control (read-only)
    _struct("MacControlTool", lambda: __import__("tools.mac_control", fromlist=["MacControlTool"]).MacControlTool(), store, "mac")
    async def mac_live():
        return await store["mac"].get_volume()
    if "mac" in store:
        r.run(sec, "MacControlTool.get_volume", "live", mac_live, requires="mac")

    # Document extraction (temp file — self-deleting)
    _struct("DocumentTool", lambda: __import__("tools.document", fromlist=["DocumentTool"]).DocumentTool(), store, "doc")
    async def doc_live():
        import tempfile
        tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        tmp.write("Jarvis health check document.\nLine two.\n")
        tmp.close()
        try:
            return await store["doc"].extract(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
    if "doc" in store:
        r.run(sec, "DocumentTool.extract", "live", doc_live)

    # File manager (read-only on the project root)
    _struct("FileManagerTool", lambda: __import__("tools.file_manager", fromlist=["FileManagerTool"]).FileManagerTool(), store, "files")
    def files_live():
        return store["files"].list_directory(str(ROOT))
    if "files" in store:
        r.run(sec, "FileManagerTool.list_directory", "live", files_live)

    # Contacts (local JSON, pure)
    _struct("ContactBook", lambda: __import__("tools.contacts", fromlist=["ContactBook"]).ContactBook(), store, "contacts")
    def contacts_live():
        return store["contacts"].list_all()
    if "contacts" in store:
        r.run(sec, "ContactBook.list_all", "live", contacts_live)

    # Briefing (pure logic)
    _struct("BriefingHandler", lambda: __import__("tools.briefing", fromlist=["BriefingHandler"]).BriefingHandler(), store, "brief")
    def brief_live():
        b = store["brief"]
        intents = b.detect_intents("what's the weather and the news today")
        return bool(intents is not None), f"detect_intents → {intents}"
    if "brief" in store:
        r.run(sec, "BriefingHandler.detect_intents", "live", brief_live)

    # Email composer (needs Ollama)
    _struct("EmailComposer", lambda: __import__("tools.email_composer", fromlist=["EmailComposer"]).EmailComposer(llm), store, "composer")

    # Query parser (needs Ollama)
    _struct("QueryParser", lambda: __import__("tools.query_parser", fromlist=["QueryParser"]).QueryParser(llm), store, "parser")
    async def parser_live():
        return await store["parser"].classify("what time is it in Tokyo")
    if "parser" in store:
        r.run(sec, "QueryParser.classify", "live", parser_live, requires="ollama")


# ── main ──────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Jarvis agent + tool health check")
    ap.add_argument("--offline", action="store_true", help="structural checks only (no network/Ollama)")
    ap.add_argument("--verbose", action="store_true", help="print tracebacks on failure")
    ap.add_argument("--strict", action="store_true", help="live failures also fail the run")
    ap.add_argument("--json", metavar="PATH", help="write results to a JSON file")
    args = ap.parse_args(argv)

    print("=" * 78)
    print(" JARVIS HEALTH CHECK — agents + tools")
    print("=" * 78)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    r = Runner(offline=args.offline, verbose=args.verbose, strict=args.strict, loop=loop)

    print("\n[detecting environment …]")
    r.detect()

    try:
        from config.llm_client import OllamaClient
        llm = OllamaClient()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Could not construct the shared LLM client: {exc}")
        return 2

    print("\n── SUBSYSTEMS ──")
    register_subsystems(r, llm)
    print("\n── AGENTS ──")
    register_agents(r, llm)
    print("\n── TOOLS ──")
    register_tools(r, llm)

    code = r.summary()

    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(x) for x in r.results], indent=2), encoding="utf-8"
        )
        print(f"  results written → {args.json}")

    try:
        loop.close()
    except Exception:
        pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
