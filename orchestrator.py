"""
Jarvis Orchestrator
The central coordinator. Receives a user request and drives the full
pipeline: Router → Memory → Planner → Critic → Executor → Evaluator.

This is what makes Jarvis a proper Multi-Agent System.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.critic import CriticAgent
from agents.evaluator import EvaluatorAgent
from agents.planner import PlannerAgent
from agents.router import RouterAgent
from agents.summariser import SummariserAgent
from tools.contacts import ContactBook
from tools.email_composer import EmailComposer, EmailDraft
from agents.calendar_agent import CalendarAgent
from agents.gmail_agent import GmailAgent
from config.llm_client import OllamaClient
from config.models import (
    AgentRole,
    JarvisResponse,
    MemoryType,
    Subtask,
    TaskPlan,
    TaskStatus,
)
from config.settings import OLLAMA_CHAT_MODEL, OLLAMA_ROUTER_MODEL
from memory.memory_agent import MemoryAgent
from tools.document import DocumentTool
from tools.sports import SportsTool
from tools.markets import MarketsTool
from tools.prayer_times import PrayerTimesTool
from tools.briefing import BriefingHandler
from tools.mac_control import MacControlTool
from tools.news import NewsTool
from tools.spotify import SpotifyTool
from tools.weather import WeatherTool
from tools.web_search import WebSearchTool
from tools.file_manager import FileManagerTool, PendingFileOp
from tools.reminders import ReminderStore
from tools.forge import ForgeTool
from tools.sentinel import SentinelTool
from core.bootstrap import build_registry
from core.executor import (
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    DagExecutor,
)

# Planner replan attempts. Each replan adds a planner+critic round-trip.
# Historically pinned to 0 for demo latency on the old CPU-bound machine;
# now env-configurable — on the M5 (4x faster prefill) one replan attempt is
# affordable, so set MAX_REPLAN_ATTEMPTS=1 in .env to re-enable the full
# critic→replan loop for evaluation runs. Default stays 0 (demo behaviour).
import os as _os_replan
MAX_REPLAN_ATTEMPTS = int(_os_replan.getenv("MAX_REPLAN_ATTEMPTS", "0"))

logger = logging.getLogger("jarvis.orchestrator")

# Voice replies are TTS'd via ElevenLabs. Each extra token = ~0.3 s of speaking.
# 100 tokens was too tight — it truncated answers mid-sentence (a definition or
# a 2-3 item answer would get cut off), which reads as "incomplete". 220 lets
# replies finish their thought (~3-4 short sentences) while staying snappy:
# sentence-streaming TTS means speech still starts on the FIRST sentence, so
# perceived latency is unchanged even though the full answer is longer.
# Text-mode replies stay on the original 180 cap for the web UI.
VOICE_MAX_TOKENS = 220


# ── One-paragraph enforcement ──────────────────────────────────────────────────
# The system prompt asks for a single paragraph, but small models like
# llama3.2:latest will sometimes ignore it and emit multi-paragraph or
# bulleted responses anyway. This post-processor enforces the rule
# deterministically before we ever ship the response to the UI.

_BULLET_LINE_RE = re.compile(r'^\s*(?:[\*\-•]|\d+\.)\s+', re.M)


def _enforce_single_paragraph(text: str) -> str:
    """
    Collapse a model response to exactly one paragraph of plain prose.

    - Strips markdown bold/italic markers (**, *, _) used as emphasis.
    - Removes bullet markers (*, -, •, 1.) at the start of lines.
    - Drops level-1/2/3 markdown headers (lines starting with #).
    - Joins multiple paragraphs into one with single spaces.
    - Trims trailing follow-up offers ("Would you like to know more?" etc).
    """
    if not text:
        return text
    cleaned = text.strip()
    # Strip markdown headers
    cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.M)
    # Remove bullet/numbered prefixes from line starts
    cleaned = _BULLET_LINE_RE.sub('', cleaned)
    # Strip bold/italic emphasis markers but keep the words
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', cleaned)
    cleaned = cleaned.replace('*', '')
    # Collapse any sequence of whitespace (including newlines) to single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Strip common follow-up offers at the end
    cleaned = re.sub(
        r'\s*(?:Would you like (?:to know |me to )?[^.?!]*[\.\?!]|'
        r'Let me know if you[^.?!]*[\.\?!]|'
        r'Feel free to ask[^.?!]*[\.\?!])\s*$',
        '',
        cleaned,
        flags=re.I,
    ).strip()
    return cleaned


class JarvisOrchestrator:
    """
    Coordinates all Jarvis agents to execute user requests.

    Pipeline per request:
    1.  Router    — classify intent, decide primary agent
    2.  Memory    — retrieve relevant context
    3.  Planner   — decompose into subtask DAG
    4.  Critic    — review plan quality, trigger replan if needed
    5.  Executor  — run subtasks respecting dependency order
    6.  Critic    — review results
    7.  Evaluator — score and persist benchmark data
    8.  Memory    — store episodic memory of this interaction
    """

    def __init__(self, model: str = OLLAMA_CHAT_MODEL):
        self.model = model

        # Shared LLM client (all agents can use same instance)
        self.llm = OllamaClient(model=model)

        # Core agents
        self.router    = RouterAgent(self.llm)
        self.memory    = MemoryAgent(self.llm)
        self.planner   = PlannerAgent(self.llm)
        self.critic    = CriticAgent(self.llm)
        self.evaluator = EvaluatorAgent()
        self.summariser = SummariserAgent(self.llm)
        # Google APIs lazy-built — see properties below. The OAuth refresh in
        # CalendarAgent / GmailAgent constructors used to block server startup
        # whenever the saved token had expired, so we defer instantiation
        # until something actually needs them.
        self._calendar: Optional[CalendarAgent] = None
        self._gmail: Optional[GmailAgent] = None
        self.contacts  = ContactBook()
        self.composer  = EmailComposer(self.llm)
        # FinEx sub-agent — lazy. Importing it pulls psycopg2, ChromaDB, etc.
        # so we hold off until the first finance query lands.
        self._finex = None
        # Pending confirmation states
        self._pending_email: EmailDraft | None = None
        self._pending_meeting: dict | None = None
        self._pending_file_op: PendingFileOp | None = None
        # Last inbox shown to the user, so "read/reply/archive email N" can
        # resolve the number N to a real Gmail message. Populated whenever the
        # inbox is read; refreshed on demand if empty when an N-command lands.
        self._last_inbox: List[Dict] = []

        # Tool instances
        self.weather    = WeatherTool()
        self.websearch  = WebSearchTool()
        self.news       = NewsTool()
        self.mac        = MacControlTool()
        self.spotify    = SpotifyTool(llm=self.llm)
        self.document   = DocumentTool()
        self.sports     = SportsTool()
        self.markets    = MarketsTool()
        self.prayer     = PrayerTimesTool()
        self.briefing   = BriefingHandler()
        self.files      = FileManagerTool()
        self.reminders  = ReminderStore()
        # Read-only developer tools. Same instances the /forge and /sentinel
        # pages use, so a chat answer and the dashboard can never disagree.
        from pathlib import Path as _P
        _project = _P(__file__).parent
        self.forge      = ForgeTool(
            scan_roots=[str(_P.home() / "Desktop"), str(_P.home() / "Documents")],
            always_include=_project)
        self.sentinel   = SentinelTool(_project)

        # ── User profile (personalisation) ──────────────────────────────────
        # Loaded once and injected into the shared system prompt so EVERY chat
        # call (handle, handle_stream, websearch answers) is grounded in who
        # Jarvis is assisting. Editable at data/profile.json.
        try:
            from config.profile import load_profile
            self.profile = load_profile()
            self.llm.JARVIS_SYSTEM_PROMPT = (
                self.llm.JARVIS_SYSTEM_PROMPT + "\n\n" + self.profile.summary()
            )
            print(f"👤 Profile loaded — assisting {self.profile.preferred_name} "
                  f"({self.profile.tone} tone)")
        except Exception as exc:
            self.profile = None
            print(f"⚠️  Profile not loaded ({exc}); using base persona")

        # ── Tool registry ───────────────────────────────────────────────────
        # Every tool, behind one interface. The DAG executor resolves through
        # this instead of a 240-line if/elif chain, which is also what makes
        # the six tools that had no branch (sports, markets, prayer, files,
        # contacts) reachable from a plan for the first time.
        #
        # Built last, so every tool attribute above already exists. Google
        # agents are registered lazily and are NOT authenticated here.
        self.tools = build_registry(self)

        print(f"\n🤖 Jarvis Orchestrator ready — model: {model}")
        print(f"   Agents: Router, Memory, Planner, Critic, Evaluator, Summariser, Calendar(lazy), Gmail(lazy), FinEx(lazy)")
        print(f"   Tools:  Weather, WebSearch, News, Mac, Spotify, Document, FileManager")
        print(f"   Registry: {len(self.tools)} tools, "
              f"{sum(len(t.actions) for t in self.tools)} actions\n")

    # ── Lazy Google + FinEx accessors ──────────────────────────────────────
    # Building CalendarAgent / GmailAgent fires a Google OAuth token refresh
    # over the network. If the saved token has expired (which happens every
    # 7 days while the OAuth app is in "Testing" mode), that refresh used to
    # block the orchestrator constructor — and therefore server startup.
    # Lazy-loading defers that risk to the first user request that actually
    # needs Calendar or Gmail.
    @property
    def calendar(self) -> "CalendarAgent":
        if self._calendar is None:
            self._calendar = CalendarAgent()
        return self._calendar

    @calendar.setter
    def calendar(self, value: "CalendarAgent") -> None:
        # /google/reauth in server.py re-assigns to refresh credentials.
        self._calendar = value

    @property
    def gmail(self) -> "GmailAgent":
        if self._gmail is None:
            self._gmail = GmailAgent()
        return self._gmail

    @gmail.setter
    def gmail(self, value: "GmailAgent") -> None:
        self._gmail = value

    @property
    def finex(self):
        """Lazy-built FinEx sub-agent. Imported on first use so missing
        psycopg2 / chroma deps don't break Jarvis startup for non-finance demos."""
        if self._finex is None:
            from agents.finex_agent import FinExAgent
            self._finex = FinExAgent()
        return self._finex

    @finex.setter
    def finex(self, value) -> None:
        self._finex = value

    # ── Shared pending-state intercept ─────────────────────────────────────
    #
    # When the orchestrator is in the middle of a multi-turn flow (waiting
    # for an email address, a meeting duration, a file-op confirmation,
    # etc.) the user's next message MUST be interpreted as a continuation
    # of that flow — not routed through the LLM router, which will
    # confidently misclassify bare email addresses as "news", short
    # numeric strings as "weather", and so on.
    #
    # This helper catches those continuations and short-circuits the
    # router. Returns a JarvisResponse if handled, None if not.

    async def _try_pending_state_intercept(self, user_request: str):
        import re as _repe
        req = user_request.strip()
        req_lower = req.lower()

        # ── Pending email: user is replying with the missing email ────────
        if self._pending_email and getattr(self._pending_email, "needs_email", False):
            # Strip a markdown-link wrapper Slack/Outlook sometimes leave
            # behind (e.g. "[mail@x.com](mailto:mail@x.com)").
            cleaned = _repe.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", req)
            email_match = _repe.search(r"[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}", cleaned)
            if email_match:
                self._pending_email.recipient_email = email_match.group(0).strip()
                self._pending_email.needs_email = False
                # Auto-save the contact so we remember them next time
                if (self._pending_email.recipient_name
                        and self._pending_email.recipient_email):
                    self.contacts.add(
                        self._pending_email.recipient_name,
                        self._pending_email.recipient_email,
                    )
                msg = self.composer.format_draft_for_confirmation(self._pending_email)
                return JarvisResponse(success=True, message=msg, latency_ms=0.0)

            # User said "add NAME EMAIL" inline
            add_m = _repe.search(
                r"add\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+([\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,})",
                req, _repe.IGNORECASE,
            )
            if add_m:
                name = add_m.group(1).strip().title()
                email = add_m.group(2).strip()
                self.contacts.add(name, email)
                self._pending_email.recipient_email = email
                self._pending_email.recipient_name = name
                self._pending_email.needs_email = False
                msg = (
                    f"Contact saved: {name} → {email}\n\n"
                    + self.composer.format_draft_for_confirmation(self._pending_email)
                )
                return JarvisResponse(success=True, message=msg, latency_ms=0.0)

            # Anything else while waiting for an email — gently re-prompt
            # rather than letting the router run wild.
            if req_lower in ("cancel", "no", "abort", "never mind", "nevermind"):
                name = self._pending_email.recipient_name or "the recipient"
                self._pending_email = None
                return JarvisResponse(
                    success=True,
                    message=f"Email to {name} cancelled.",
                    latency_ms=0.0,
                )
            return JarvisResponse(
                success=True,
                message=(
                    f"I still need an email address for "
                    f"'{self._pending_email.recipient_name or 'the recipient'}'. "
                    "Reply with just the address (e.g. name@example.com), or "
                    "say 'cancel' to drop the email."
                ),
                latency_ms=0.0,
            )

        # ── Pending email draft: user is confirming / cancelling / editing ─
        # Catches just the unambiguous responses; richer phrasing like
        # "make it shorter" still falls through to the existing
        # _try_shortcut edit handler.
        if self._pending_email and not getattr(self._pending_email, "needs_email", False):
            confirm_words = ("yes", "yes send it", "send it", "send", "confirm",
                             "go ahead", "yeah", "yep", "yup", "ok", "okay")
            cancel_words = ("no", "cancel", "don't send", "do not send",
                            "abort", "scrap it", "delete it")
            if req_lower in confirm_words:
                # CRITICAL: actually send the email here. Earlier versions
                # cleared _pending_email and returned None hoping the
                # downstream _try_shortcut confirm handler would catch
                # "yes" — but the router was classifying "yes" as Tier 2
                # general chat, which skips _try_shortcut entirely. The
                # LLM would then hallucinate a fake "I'll draft and send
                # the email..." response without anything ever being sent.
                draft = self._pending_email
                self._pending_email = None
                result = await self._send_pending_draft(draft)
                msg = result.get(
                    "message",
                    f"Email sent to {draft.recipient_email}." if result.get("success")
                    else f"Could not send: {result.get('error', 'unknown error')}",
                )
                # Auto-save the contact on successful send
                if result.get("success") and draft.recipient_email and draft.recipient_name:
                    self.contacts.add(draft.recipient_name, draft.recipient_email)
                return JarvisResponse(
                    success=result.get("success", False),
                    message=msg,
                    latency_ms=0.0,
                )
            if req_lower in cancel_words:
                self._pending_email = None
                return JarvisResponse(
                    success=True,
                    message="Email cancelled. No email was sent.",
                    latency_ms=0.0,
                )

        # ── Pending meeting waiting for duration ──────────────────────────
        if self._pending_meeting and not self._pending_meeting.get("needs_new_time"):
            # Catch bare numeric durations and worded ones — the existing
            # _try_shortcut handler already does this parsing; we just
            # need to make sure these strings don't get routed.
            if (_repe.search(r"\b\d+\s*(?:m|min|mins|minute|minutes|hour|hours|hr|hrs|h)\b", req_lower)
                    or req_lower in ("half hour", "quarter hour", "an hour",
                                        "one hour", "two hours", "yes", "confirm",
                                        "book it")):
                return None  # let _try_shortcut handle parsing
            if req_lower in ("cancel", "no", "abort", "never mind"):
                title = self._pending_meeting.get("title", "Meeting")
                self._pending_meeting = None
                return JarvisResponse(
                    success=True,
                    message=f"'{title}' booking cancelled.",
                    latency_ms=0.0,
                )

        # ── Pending file op waiting for confirm/cancel ────────────────────
        if self._pending_file_op:
            if req_lower in ("confirm", "yes", "do it", "go ahead",
                             "proceed", "ok", "okay"):
                return None  # let _try_shortcut handle the actual execute
            if req_lower in ("cancel", "no", "stop", "abort",
                             "nevermind", "never mind"):
                self._pending_file_op = None
                return JarvisResponse(
                    success=True,
                    message="Cancelled.",
                    latency_ms=0.0,
                )

        return None  # no pending state caught the input — proceed normally

    # ── Shared follow-up / elaborate handling ──────────────────────────────

    # Phrases that, when said on their own (or as a short standalone request),
    # mean "expand on the previous answer" rather than "answer this as a new
    # query". The router doesn't know about conversation state, so without
    # this intercept "elaborate" would be routed as a web search for the word
    # "elaborate" — useless.
    _FOLLOWUP_TRIGGERS = [
        "elaborate", "tell me more", "more detail", "more details",
        "explain more", "expand", "expand on that", "go into more detail",
        "go on", "continue", "and?", "more", "what else", "say more",
        "explain in detail", "go into detail", "give me more",
        "give me more detail", "give me more details",
        "in more detail", "in detail", "give more detail",
    ]

    def _detect_followup(self, user_request: str) -> bool:
        """True if the user's message is a bare 'elaborate'-style follow-up."""
        stripped = user_request.strip().lower().rstrip("?.!")
        if stripped in self._FOLLOWUP_TRIGGERS:
            return True
        # Allow short variations like "can you elaborate on that"
        if len(user_request.split()) <= 6:
            if any(stripped.startswith(t) for t in self._FOLLOWUP_TRIGGERS):
                return True
            if any(t in stripped for t in ("elaborate", "more detail", "more details", "in detail")):
                return True
        return False

    def _extract_prev_exchange(self, history: List[Dict]) -> tuple:
        """
        Walk history backwards looking for the most recent complete
        user→assistant exchange. Returns (prev_user_msg, prev_assistant_msg)
        or (None, None) if no eligible exchange exists.
        """
        prev_user_msg = None
        prev_assistant_msg = None
        for turn in reversed(history):
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant" and prev_assistant_msg is None:
                prev_assistant_msg = content
            elif role == "user" and prev_assistant_msg is not None:
                candidate = content.lower().rstrip("?.!")
                if (candidate not in self._FOLLOWUP_TRIGGERS
                        and len(candidate.split()) > 2):
                    prev_user_msg = content
                    break
        return prev_user_msg, prev_assistant_msg

    def _build_elaborate_messages(
        self, prev_user_msg: str, prev_assistant_msg: str,
    ) -> List[Dict[str, str]]:
        """
        Build the message list for an elaborate request. Crucially this
        provides an EXPLICIT system message — that suppresses the default
        JARVIS_SYSTEM_PROMPT, whose strict "one paragraph, never list
        capabilities" rules fight the elaborate request and cause small
        models to defensively dump their capability list instead of
        actually elaborating.
        """
        elaborate_system = (
            "You are answering a follow-up request for more detail. "
            "The user has already received a brief answer and is now "
            "asking you to expand on it. "
            "Produce a thorough, well-organised response of 200-350 words. "
            "You may use multiple paragraphs and numbered points (1. 2. 3.) "
            "where helpful. "
            "Do NOT use markdown asterisks (*), bold/italic markers, or "
            "hash symbols (#). "
            "Do NOT introduce yourself, mention your name, or list your "
            "capabilities — the user already knows who you are. "
            "Do NOT mention or cite sources or URLs. "
            "Stay strictly on the topic of the previous question — go "
            "deeper on background, key facts, context, and significance "
            "of that specific topic only. Do not pivot to other topics."
        )
        elaborate_user = (
            f"Previous question from the user:\n"
            f"{prev_user_msg}\n\n"
            f"Brief answer you gave earlier:\n"
            f"{(prev_assistant_msg or '')[:2000]}\n\n"
            f"Now elaborate. Expand on the topic above with more depth, "
            f"background, and detail. Do not repeat the brief answer "
            f"verbatim — add information the brief answer left out."
        )
        return [
            {"role": "system", "content": elaborate_system},
            {"role": "user",   "content": elaborate_user},
        ]

    # ── Main entry point ───────────────────────────────────────────────────

    async def handle(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
        _routing=None,  # pre-computed RouterDecision — skips router step when provided
        conversation_history: Optional[List[Dict]] = None,
        voice_mode: bool = False,
    ) -> JarvisResponse:
        """
        Handle a user request end-to-end.

        Args:
            user_request:   Natural language request from user
            context:        Optional dict with datetime, timezone, user_id
            model_override: Force a specific model (for benchmarking)

        Returns:
            JarvisResponse with message, evaluation, and task data
        """
        start_time = time.time()
        model = model_override or self.model

        print(f"\n{'='*60}")
        print(f"📨 Request: {user_request}")
        print(f"   Model: {model}")
        print(f"{'='*60}")

        ctx = context or {
            "current_datetime": datetime.now().isoformat(),
            "timezone": "Europe/London",
            "user_id": "user_001",
        }

        def _t(label: str, t0: float):
            elapsed = time.time() - t0
            print(f"[JARVIS] ⏱  {label}: {elapsed:.2f}s")
            return elapsed

        try:
            # ── Step -1: Pending-state intercept (must run before router) ─
            # If we're waiting for an email address / meeting duration /
            # file-op confirmation, the user's next message is a CONTINUATION
            # of that flow. The router would misclassify (e.g. a bare email
            # address gets routed as "news" with confidence 0.90 by the 1b
            # router model). Catch those early.
            if _routing is None:
                pending = await self._try_pending_state_intercept(user_request)
                if pending is not None:
                    pending.latency_ms = (time.time() - start_time) * 1000
                    return pending

            # ── Step 0: Elaborate / follow-up intercept ────────────────────
            # MUST run before the router. The router would otherwise treat
            # "elaborate" / "give me more detail" as a fresh standalone
            # query and route it (web search, planner, etc.), which is
            # wrong — these phrases only make sense relative to the prior
            # assistant turn. This intercept resolves them by re-asking
            # the prior question with an "expand" system prompt.
            #
            # Skip when _routing is provided (means handle_stream already
            # decided this is NOT a follow-up and is delegating to us for
            # the Tier 3 full pipeline).
            if _routing is None and self._detect_followup(user_request):
                history = conversation_history or []
                if history:
                    prev_user_msg, prev_assistant_msg = self._extract_prev_exchange(history)
                    if prev_user_msg:
                        t0 = time.time()
                        msgs = self._build_elaborate_messages(prev_user_msg, prev_assistant_msg)
                        # 900 tokens ≈ 640 words. The old 450 cap existed so
                        # the blocking HTTP path fit inside the 60s LLM
                        # timeout on the CPU-only 8GB machine; the M5 at
                        # ~40 tok/s clears 900 tokens in ~25s worst case.
                        try:
                            full_text = await self.llm.chat(msgs, model=model, max_tokens=900)
                        except Exception as exc:
                            # Surface a useful error instead of the bare
                            # "I encountered an error:" — the user has been
                            # staring at a thinking indicator for ~minute.
                            err_msg = str(exc) or type(exc).__name__
                            total_ms = (time.time() - start_time) * 1000
                            print(f"[JARVIS] ⚠️ Elaborate failed: {err_msg}")
                            return JarvisResponse(
                                success=False,
                                message=(
                                    "The detailed answer took too long to generate. "
                                    "This usually means the model is cold-loading "
                                    "or another query is in flight. Please try again."
                                ),
                                error=err_msg,
                                latency_ms=total_ms,
                            )
                        _t("elaborate", t0)
                        total_ms = (time.time() - start_time) * 1000
                        print(f"[JARVIS] ⚡ Elaborate complete — total: {total_ms/1000:.2f}s")
                        return JarvisResponse(
                            success=True,
                            message=full_text.strip(),
                            latency_ms=total_ms,
                        )
                # No history or no eligible prior turn — fall back to asking
                return JarvisResponse(
                    success=True,
                    message="What would you like me to elaborate on?",
                    latency_ms=(time.time() - start_time) * 1000,
                )

            # ── Memory command intercept (remember / forget / recall) ──────
            if _routing is None:
                mem_resp = await self._try_memory_command(user_request)
                if mem_resp is not None:
                    mem_resp.latency_ms = (time.time() - start_time) * 1000
                    return mem_resp

            # ── Morning Brief intercept (BEFORE the router) ────────────────
            # Must run before the router because the LLM router classifies
            # "morning brief" as Tier-2 general_chat, which then hallucinates
            # a fake brief. Catching it here bypasses the router entirely.
            if _routing is None and self.briefing.is_morning_briefing(user_request):
                t0 = time.time()
                msg = await self._morning_briefing(voice_mode=voice_mode)
                total_ms = (time.time() - start_time) * 1000
                _t("morning_brief", t0)
                print(f"[JARVIS] ⚡ Morning brief ({'voice' if voice_mode else 'text'}) — {total_ms/1000:.2f}s")
                return JarvisResponse(
                    success=True,
                    message=msg,
                    latency_ms=total_ms,
                )

            # ── Multi-ACTION intercept (BEFORE the router) ─────────────────
            # "dim the screen, play Despacito and remind me at 5" is three
            # separate commands. The router only picks ONE agent, so without
            # this intercept only the first action would run. We split the
            # request into atomic commands and execute each through the normal
            # routing+shortcut path, then aggregate the replies.
            if _routing is None and self.briefing.looks_multi_action(user_request):
                t0 = time.time()
                multi = await self._handle_multi_action(
                    user_request,
                    voice_mode=voice_mode,
                    conversation_history=conversation_history,
                )
                if multi is not None:
                    _t("multi_action", t0)
                    total_ms = (time.time() - start_time) * 1000
                    print(f"[JARVIS] ⚡ Multi-action — {total_ms/1000:.2f}s")
                    return JarvisResponse(success=True, message=multi, latency_ms=total_ms)

            # ── Step 1: Route ──────────────────────────────────────────────
            if _routing is not None:
                routing = _routing
                print(f"🔀 Router skipped — reusing pre-computed routing ({routing.primary_agent.value}, tier {routing.tier})")
            else:
                t0 = time.time()
                routing = await self.router.route(user_request)
                _t("router", t0)
            tier = routing.tier

            # ── Tier 1: tool-only — skip memory, run shortcut, return fast ─
            if tier == 1:
                t0 = time.time()
                shortcut = await self._try_shortcut(routing.primary_agent, user_request, voice_mode=voice_mode, conversation_history=conversation_history)
                _t("tool_shortcut", t0)
                if shortcut is not None:
                    print(f"[JARVIS] ⚡ Tier 1 complete — total: {time.time()-start_time:.2f}s")
                    return shortcut

            # ── Step 2: Memory retrieval (Tier 2 + 3 only) ────────────────
            t0 = time.time()
            memories = await self.memory.retrieve(user_request)
            _t("memory_retrieve", t0)
            print(f"🧠 Retrieved {len(memories)} relevant memories")

            # ── Tier 1 fallback: shortcut missed, treat as Tier 2 ─────────
            # NOTE: we used to call _try_shortcut a SECOND time here with
            # identical arguments — nothing it reads changes between the two
            # calls, so the retry could only ever return None again and just
            # re-paid the full keyword-scan cost on every misrouted Tier-1
            # request. Escalate straight to Tier 2 instead.
            if tier == 1:
                tier = 2  # escalate

            # ── Tier 2: single LLM call, skip Planner/Critic/Evaluator ────
            if tier == 2:
                t0 = time.time()
                context_str = ""
                agent = routing.primary_agent

                # ── FinEx fast-path ────────────────────────────────────────
                # FinEx has its own deterministic Python router + level-specific
                # handlers; we don't want the generic Jarvis LLM to also fire.
                # Send the question straight through.
                if agent == AgentRole.FINEX:
                    fx = await self.finex.chat(question=user_request)
                    answer = fx.get("answer", "") or "No answer from FinEx."
                    _t("finex_chat", t0)
                    total_ms = (time.time() - start_time) * 1000
                    print(f"[JARVIS] ⚡ Tier 2 FinEx ({fx.get('level_label','?')}) — {total_ms/1000:.2f}s")
                    asyncio.ensure_future(self.memory.store_task_result(
                        user_request, "finex_query", True, answer[:100]
                    ))
                    return JarvisResponse(
                        success=True,
                        message=_enforce_single_paragraph(answer),
                        latency_ms=total_ms,
                    )

                if agent == AgentRole.WEBSEARCH:
                    data = await self.websearch.search(user_request)
                    context_str = self.websearch.format_results(data)
                    _t("websearch_tool", t0)
                elif agent == AgentRole.NEWS:
                    data = await self.news.get_headlines(query=user_request, max_items=5)
                    context_str = self.news.format_headlines(data)
                    _t("news_tool", t0)

                # Voice mode tightens the spoken format: 1-2 short sentences max,
                # no markdown of any kind (TTS reads it verbatim).
                _voice_suffix = (
                    "\n\nVOICE MODE — your reply will be read aloud:\n"
                    "- Maximum 2 short sentences (≤ 30 words total).\n"
                    "- Speak conversationally — no markdown, no bullet points, no lists.\n"
                    "- No URLs, no citations, no follow-up offers."
                ) if voice_mode else ""

                if context_str and agent == AgentRole.WEBSEARCH:
                    user_content = (
                        f"Answer this question: {user_request}\n\n"
                        f"Source material:\n{context_str}\n\n"
                        f"STRICT OUTPUT FORMAT:\n"
                        f"- Output EXACTLY ONE paragraph, 3-5 sentences, plain prose.\n"
                        f"- Absolutely no bullet points, no numbered lists, no headers, no markdown.\n"
                        f"- No blank lines anywhere — keep the entire answer on a single paragraph.\n"
                        f"- Never introduce yourself or mention your name.\n"
                        f"- Never cite, mention, or list sources, URLs, or websites — just answer directly.\n"
                        f"- No follow-up offers ('Would you like to know more?' etc.)."
                        f"{_voice_suffix}"
                    )
                else:
                    user_content = (
                        f"Context:\n{context_str}\n\nUser: {user_request}{_voice_suffix}"
                        if context_str else f"{user_request}{_voice_suffix}"
                    )
                history = conversation_history or []
                # Voice turns trim history harder — every extra token costs both
                # TTFT (re-eval) and (downstream) TTS speaking time.
                _hist_cap = 4 if voice_mode else 8
                recent_history = history[-_hist_cap:] if len(history) > _hist_cap else history

                msgs = recent_history + [{"role": "user", "content": user_content}]

                t0 = time.time()
                # 180 token cap = ~130 words = ~5 medium sentences. Hard upper
                # bound that prevents the model from running away even when it
                # ignores the prompt's "one paragraph" instruction.
                # Voice mode clamps tighter (VOICE_MAX_TOKENS) because
                # ElevenLabs is going to read the whole thing aloud and we don't
                # want the demo dominated by Jarvis speaking.
                _tier2_tokens = VOICE_MAX_TOKENS if voice_mode else 180
                llm_response = await self.llm.chat(msgs, max_tokens=_tier2_tokens)
                _t("llm_single_call", t0)

                # Deterministic single-paragraph enforcement — guarantees the
                # response is one paragraph of plain prose regardless of what
                # the model emitted.
                llm_response = _enforce_single_paragraph(llm_response)

                total_ms = (time.time() - start_time) * 1000
                print(f"[JARVIS] ⚡ Tier 2 complete — total: {total_ms/1000:.2f}s")
                asyncio.ensure_future(self.memory.store_task_result(
                    user_request=user_request, intent=routing.primary_agent.value,
                    success=True, summary=llm_response[:100]
                ))
                return JarvisResponse(
                    success=True, message=llm_response,
                    latency_ms=total_ms,
                )

            # ── Tier 3: full pipeline ──────────────────────────────────────
            # Short-circuit for deterministic tools even in Tier 3
            t0 = time.time()
            shortcut = await self._try_shortcut(routing.primary_agent, user_request)
            _t("shortcut_check", t0)
            if shortcut is not None:
                return shortcut

            # ── Step 3: Plan ───────────────────────────────────────────────
            t0 = time.time()
            plan = await self.planner.plan(
                user_request, ctx, memories, model_override=model
            )
            _t("planner", t0)

            # ── Step 4: Critic ─────────────────────────────────────────────
            # Tightened for the demo: only invoke the critic for low-confidence
            # plans with more than 3 subtasks, OR for the research agent
            # specifically (which benefits from a sanity check). Most demo
            # prompts have confidence > 0.9 and ≤ 2 subtasks so they skip
            # the critic entirely, saving an LLM round-trip.
            _needs_critic = (
                (routing.confidence < 0.70 and len(plan.subtasks) > 3)
                or routing.primary_agent.value == "research"
            )
            planning_score = 0.8

            if _needs_critic:
                t0 = time.time()
                plan_verdict = await self.critic.review_plan(plan)
                planning_score = plan_verdict.score
                _t("critic_plan", t0)
                replan_attempts = 0
                while plan_verdict.replan_needed and replan_attempts < MAX_REPLAN_ATTEMPTS:
                    replan_attempts += 1
                    print(f"🔄 Replanning (attempt {replan_attempts})...")
                    feedback_ctx = {
                        **ctx,
                        "critic_feedback": "; ".join(plan_verdict.issues),
                        "critic_suggestions": "; ".join(plan_verdict.suggestions),
                    }
                    t0 = time.time()
                    plan = await self.planner.plan(
                        user_request, feedback_ctx, memories, model_override=model
                    )
                    plan.replan_count = replan_attempts
                    plan_verdict = await self.critic.review_plan(plan)
                    planning_score = max(planning_score, plan_verdict.score)
                    _t(f"replan_{replan_attempts}", t0)
            else:
                print(f"⚡ Critic skipped — high confidence ({routing.confidence:.2f})")

            # ── Step 5: Execute ────────────────────────────────────────────
            t0 = time.time()
            results = await self._execute_dag(plan, routing.primary_agent)
            _t("execute_dag", t0)

            # ── Step 6: Critic result review ───────────────────────────────
            # Disabled for the demo. The verdict was only ever logged — it
            # didn't gate any re-execution path — and burned another
            # ~1-2 s of LLM time. Re-enable for the dissertation
            # evaluation runs only.
            # if _needs_critic:
            #     t0 = time.time()
            #     result_verdict = await self.critic.review_result(plan, results)
            #     _t("critic_result", t0)

            # ── Step 7: Evaluate ───────────────────────────────────────────
            t0 = time.time()
            # evaluate_async persists off the event loop; evaluate() writes to
            # SQLite inline and this is the async request path.
            evaluation = await self.evaluator.evaluate_async(
                plan, results, start_time, planning_score=planning_score
            )
            _t("evaluator", t0)

            # ── Step 8: Store episodic memory ──────────────────────────────
            asyncio.ensure_future(self.memory.store_task_result(
                user_request=user_request,
                intent=plan.intent,
                success=evaluation.success,
                summary=evaluation.feedback,
            ))

            # ── Build response ─────────────────────────────────────────────
            t0 = time.time()
            message = self._build_response_message(
                user_request, plan, results, routing.primary_agent
            )
            _t("build_response", t0)

            total_ms = (time.time() - start_time) * 1000
            print(f"[JARVIS] ✅ Tier 3 complete — total: {total_ms/1000:.2f}s")

            return JarvisResponse(
                success=evaluation.success,
                message=message,
                task_plan=plan.to_dict(),
                evaluation=evaluation.to_dict(),
                latency_ms=evaluation.latency_ms,
            )

        except Exception as exc:
            latency_ms = (time.time() - start_time) * 1000
            print(f"❌ Orchestrator error: {exc}")
            import traceback
            traceback.print_exc()
            # Fall back to a useful explanation rather than the bare
            # "I encountered an error:" when the exception has no message
            # (common for timeouts on small models).
            err_text = str(exc) or type(exc).__name__
            if "Timeout" in err_text or "timeout" in err_text:
                user_msg = (
                    "The model took too long to respond. This usually means "
                    "the LLM is cold-loading. Try again — the second attempt "
                    "should be much faster."
                )
            else:
                user_msg = f"I encountered an error: {err_text}"
            return JarvisResponse(
                success=False,
                message=user_msg,
                error=err_text,
                latency_ms=latency_ms,
            )

    # ── Streaming entry point ─────────────────────────────────────────────

    async def handle_stream(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        voice_mode: bool = False,
    ):
        """
        Async generator version of handle().
        Yields dicts: {"type":"thinking"}, {"type":"chunk","text":"..."}, {"type":"response",...}

        Tier 1 → single {"type":"response"} (instant, no LLM)
        Tier 2 → {"type":"thinking"} then streamed {"type":"chunk"} tokens then {"type":"response"}
        Tier 3 → {"type":"thinking"} then full pipeline result as {"type":"response"}
        """
        start_time = time.time()
        model = model_override or self.model
        ctx = context or {
            "current_datetime": datetime.now().isoformat(),
            "timezone": "Europe/London",
            "user_id": "user_001",
        }

        def _t(label: str, t0: float):
            elapsed = time.time() - t0
            print(f"[JARVIS] ⏱  {label}: {elapsed:.2f}s")

        try:
            # ── Pending-state intercept ────────────────────────────────────
            # MUST run before the router. When _pending_email needs an
            # address, "user@gmail.com" is a continuation, not a query.
            # The 1b router model will otherwise classify a bare email
            # address as "news" with high confidence and the address
            # gets piped into the news headlines tool. Same risk for
            # meeting durations ("45 minutes") and file-op confirms.
            pending = await self._try_pending_state_intercept(user_request)
            if pending is not None:
                pending.latency_ms = (time.time() - start_time) * 1000
                yield {
                    "type": "response",
                    "message": pending.message,
                    "success": pending.success,
                    "latency_ms": pending.latency_ms,
                }
                return

            # ── Early intercept: pure follow-up with no new topic ─────────
            # "elaborate", "tell me more", "give me more detail" etc. must
            # be resolved against conversation history — never routed as a
            # standalone query. The detection + prev-exchange walk +
            # elaborate message construction all live on shared helpers
            # so the WebSocket streaming path and the /chat HTTP path stay
            # in lockstep (they were drifting before — only this path had
            # the fix, so /chat would hallucinate capability lists).
            if self._detect_followup(user_request):
                _history = conversation_history or []
                if not _history:
                    yield {
                        "type": "response",
                        "message": "What would you like me to elaborate on?",
                        "success": True,
                        "latency_ms": (time.time() - start_time) * 1000,
                    }
                    return

                prev_user_msg, prev_assistant_msg = self._extract_prev_exchange(_history)
                if not prev_user_msg:
                    yield {
                        "type": "response",
                        "message": "What would you like me to elaborate on?",
                        "success": True,
                        "latency_ms": (time.time() - start_time) * 1000,
                    }
                    return

                yield {"type": "thinking"}
                msgs = self._build_elaborate_messages(prev_user_msg, prev_assistant_msg)
                full_text = ""
                _stream_err = None
                # chat_stream auto-injects JARVIS_SYSTEM_PROMPT only when
                # no system message is present — our explicit system
                # message above suppresses that injection.
                try:
                    async for chunk in self.llm.chat_stream(msgs, model=model, max_tokens=700):
                        full_text += chunk
                        yield {"type": "chunk", "text": chunk}
                except Exception as _exc:  # noqa: BLE001
                    _stream_err = _exc
                    print(f"[JARVIS] ⚠️ Elaborate stream interrupted: {type(_exc).__name__}: {_exc}")
                # Keep whatever already streamed — a long answer shouldn't be
                # thrown away just because the tail hit the idle timeout.
                if full_text.strip():
                    yield {
                        "type": "response",
                        "message": full_text + ("" if _stream_err is None else "\n\n[Response was cut off early — ask me to continue.]"),
                        "success": True,
                        "latency_ms": (time.time() - start_time) * 1000,
                    }
                    return
                if _stream_err is not None:
                    raise _stream_err  # nothing streamed → let the handler report it
                yield {
                    "type": "response",
                    "message": full_text,
                    "success": True,
                    "latency_ms": (time.time() - start_time) * 1000,
                }
                return

            # ── Memory command intercept (remember / forget / recall) ──────
            mem_resp = await self._try_memory_command(user_request)
            if mem_resp is not None:
                yield {
                    "type": "response",
                    "message": mem_resp.message,
                    "success": mem_resp.success,
                    "latency_ms": (time.time() - start_time) * 1000,
                }
                return

            # ── Morning Brief intercept (BEFORE the router) ────────────────
            # Same intercept as in handle() — must run before the LLM router
            # classifies "morning brief" as Tier-2 general_chat.
            if self.briefing.is_morning_briefing(user_request):
                yield {"type": "thinking"}
                t0 = time.time()
                msg = await self._morning_briefing(voice_mode=voice_mode)
                total_ms = (time.time() - start_time) * 1000
                _t("morning_brief", t0)
                print(f"[JARVIS] ⚡ Stream morning brief ({'voice' if voice_mode else 'text'}) — {total_ms/1000:.2f}s")
                # Emit the whole brief as one chunk so the streaming voice
                # brain can sentence-split it for TTS in voice mode.
                yield {"type": "chunk", "text": msg}
                yield {
                    "type": "response",
                    "message": msg,
                    "success": True,
                    "latency_ms": total_ms,
                }
                return

            # ── Multi-ACTION intercept (BEFORE the router) ────────────────
            # The WebSocket path previously routed every request to a single
            # agent, so a compound command like "turn brightness to 10, play
            # Despacito and remind me to call mum at 5" only ever ran the
            # first action. Split it into atomic commands and run each one.
            if self.briefing.looks_multi_action(user_request):
                yield {"type": "thinking"}
                t0 = time.time()
                multi = await self._handle_multi_action(
                    user_request,
                    voice_mode=voice_mode,
                    conversation_history=conversation_history,
                )
                _t("multi_action", t0)
                if multi is not None:
                    total_ms = (time.time() - start_time) * 1000
                    print(f"[JARVIS] ⚡ Stream multi-action — {total_ms/1000:.2f}s")
                    yield {"type": "chunk", "text": multi}
                    yield {
                        "type": "response",
                        "message": multi,
                        "success": True,
                        "latency_ms": total_ms,
                    }
                    return

            # ── Router (always needed — 1b model, fast) ───────────────────
            t0 = time.time()
            routing = await self.router.route(user_request)
            _t("router", t0)
            tier = routing.tier

            # ── Tier 1: instant tool response ─────────────────────────────
            if tier == 1:
                t0 = time.time()
                shortcut = await self._try_shortcut(routing.primary_agent, user_request, voice_mode=voice_mode, conversation_history=conversation_history)
                _t("tool_shortcut", t0)
                if shortcut is not None:
                    print(f"[JARVIS] ⚡ Stream Tier 1 — {(time.time()-start_time):.2f}s")
                    yield {
                        "type": "response",
                        "message": shortcut.message,
                        "success": shortcut.success,
                        "latency_ms": (time.time() - start_time) * 1000,
                    }
                    return
                tier = 2  # escalate if no shortcut matched

            # ── Tier 2: stream LLM response token by token ────────────────
            if tier == 2:
                yield {"type": "thinking"}

                context_str = ""
                agent = routing.primary_agent

                # ── FinEx fast-path (no Jarvis LLM, no streaming — FinEx
                # has its own engine and returns a final answer directly).
                if agent == AgentRole.FINEX:
                    t0 = time.time()
                    fx = await self.finex.chat(question=user_request)
                    answer = fx.get("answer", "") or "No answer from FinEx."
                    _t("finex_chat", t0)
                    total_ms = (time.time() - start_time) * 1000
                    print(f"[JARVIS] ⚡ Stream Tier 2 FinEx — {total_ms/1000:.2f}s")
                    cleaned = _enforce_single_paragraph(answer)
                    # Emit the whole answer in one chunk so the streaming voice
                    # brain (server.py) treats it as the sentence buffer.
                    yield {"type": "chunk", "text": cleaned}
                    asyncio.ensure_future(self.memory.store_task_result(
                        user_request, "finex_query", True, cleaned[:100]
                    ))
                    yield {
                        "type": "response",
                        "message": cleaned,
                        "success": True,
                        "latency_ms": total_ms,
                    }
                    return

                if agent == AgentRole.WEBSEARCH:
                    t0 = time.time()
                    data = await self.websearch.search(user_request)
                    context_str = self.websearch.format_results(data)
                    _t("websearch_tool", t0)
                elif agent == AgentRole.NEWS:
                    t0 = time.time()
                    data = await self.news.get_headlines(query=user_request, max_items=5)
                    context_str = self.news.format_headlines(data)
                    _t("news_tool", t0)

                _voice_suffix = (
                    "\n\nVOICE MODE — your reply will be read aloud:\n"
                    "- Maximum 2 short sentences (≤ 30 words total).\n"
                    "- Speak conversationally — no markdown, no bullet points, no lists.\n"
                    "- No URLs, no citations, no follow-up offers."
                ) if voice_mode else ""

                if context_str and agent == AgentRole.WEBSEARCH:
                    user_content = (
                        f"Answer this question: {user_request}\n\n"
                        f"Source material:\n{context_str}\n\n"
                        f"STRICT OUTPUT FORMAT:\n"
                        f"- Output EXACTLY ONE paragraph, 3-5 sentences, plain prose.\n"
                        f"- Absolutely no bullet points, no numbered lists, no headers, no markdown.\n"
                        f"- No blank lines anywhere — keep the entire answer on a single paragraph.\n"
                        f"- Never introduce yourself or mention your name.\n"
                        f"- Never cite, mention, or list sources, URLs, or websites — just answer directly.\n"
                        f"- No follow-up offers ('Would you like to know more?' etc.)."
                        f"{_voice_suffix}"
                    )
                else:
                    user_content = (
                        f"Context:\n{context_str}\n\nUser: {user_request}{_voice_suffix}"
                        if context_str else f"{user_request}{_voice_suffix}"
                    )

                # Build message list: inject recent conversation history for context.
                # Voice mode trims harder so TTFT stays low.
                history = conversation_history or []
                _hist_cap = 4 if voice_mode else 8
                recent_history = history[-_hist_cap:] if len(history) > _hist_cap else history

                # Personalisation: pull any remembered facts relevant to this
                # query and prepend them (with the base persona) as a system
                # message. Skipped in voice mode to protect time-to-first-token.
                _sys_msgs = []
                if not voice_mode:
                    recall = await self._recall_block(user_request)
                    if recall:
                        _sys_msgs = [{
                            "role": "system",
                            "content": (self.llm.JARVIS_SYSTEM_PROMPT
                                        + "\n\nThings you remember about the user:\n"
                                        + recall),
                        }]
                msgs = _sys_msgs + recent_history + [{"role": "user", "content": user_content}]

                full_text = ""
                t0 = time.time()
                # 180 token cap matches the non-streaming Tier 2 path — keeps
                # the model honest about the one-paragraph rule. Voice mode
                # clamps tighter so TTS doesn't run away.
                _tier2_tokens = VOICE_MAX_TOKENS if voice_mode else 180
                _stream_err = None
                try:
                    async for chunk in self.llm.chat_stream(msgs, model=model, max_tokens=_tier2_tokens):
                        full_text += chunk
                        yield {"type": "chunk", "text": chunk}
                except Exception as _exc:  # noqa: BLE001
                    _stream_err = _exc
                    print(f"[JARVIS] ⚠️ Tier 2 stream interrupted: {type(_exc).__name__}: {_exc}")
                _t("llm_stream", t0)
                # If the stream died before producing anything, surface the
                # error; otherwise keep the partial text (cleaned below).
                if not full_text.strip() and _stream_err is not None:
                    raise _stream_err

                # Enforce single-paragraph format on the final aggregated text.
                # We stream chunks for perceived latency, then send a final
                # cleaned message that overrides what the UI buffered.
                cleaned = _enforce_single_paragraph(full_text)

                total_ms = (time.time() - start_time) * 1000
                print(f"[JARVIS] ⚡ Stream Tier 2 — {total_ms/1000:.2f}s")
                asyncio.ensure_future(self.memory.store_task_result(
                    user_request, routing.primary_agent.value, True, cleaned[:100]
                ))
                yield {
                    "type": "response",
                    "message": cleaned,
                    "success": True,
                    "latency_ms": total_ms,
                }
                return

            # ── Tier 3: full pipeline, show thinking indicator ─────────────
            yield {"type": "thinking"}

            # Pass pre-computed routing to avoid double-routing (saves ~2-3s).
            # voice_mode flows through so Tier 3 also clamps to short replies.
            response = await self.handle(
                user_request,
                context=ctx,
                model_override=model_override,
                _routing=routing,
                conversation_history=conversation_history,
                voice_mode=voice_mode,
            )
            yield {
                "type": "response",
                "message": response.message,
                "success": response.success,
                "latency_ms": response.latency_ms,
            }

        except Exception as exc:
            print(f"❌ Stream error: {exc}")
            import traceback; traceback.print_exc()
            yield {
                "type": "response",
                "message": f"I encountered an error: {exc}",
                "success": False,
                "latency_ms": (time.time() - start_time) * 1000,
            }

    # ── DAG Execution ──────────────────────────────────────────────────────

    async def _execute_dag(
        self,
        plan: TaskPlan,
        primary_agent: AgentRole,
    ) -> Dict[str, Any]:
        """Execute the plan's subtasks in dependency order.

        Delegates to ``core.executor.DagExecutor``; this method now only
        translates between the executor's plain-string statuses and the
        orchestrator's TaskStatus enum, and keeps the per-subtask timestamps
        the evaluator reads.

        Two fixes ride along with the move:

        * A subtask whose dependency FAILED is now genuinely blocked. The old
          guard tested whether a dependency was present in the results dict,
          not whether it had succeeded — and a failed dependency is present —
          so the BLOCKED branch was unreachable and dependents ran against
          failure payloads.

        * Independent read-only subtasks run concurrently instead of one after
          another. Writes stay sequential; LLM-generated plans don't reliably
          declare ordering between side effects.
        """
        by_id = {st.id: st for st in plan.subtasks}
        started: Dict[str, datetime] = {}

        for st in plan.subtasks:
            st.status = TaskStatus.IN_PROGRESS
            started[st.id] = datetime.now()

        def _on_subtask(st_id: str, result: Dict[str, Any]) -> None:
            st = by_id.get(st_id)
            if st is None:
                return
            st.started_at = started.get(st_id)
            st.completed_at = datetime.now()
            st.result = result
            icon = "✅" if result.get("success") else "❌"
            print(f"   {icon} [{st_id}] {st.agent}.{st.action}")

        executor = DagExecutor(
            self.tools,
            inject_deps=self._inject_deps,
            on_subtask=_on_subtask,
        )
        report = await executor.execute(plan.subtasks)

        status_map = {
            STATUS_COMPLETED: TaskStatus.COMPLETED,
            STATUS_BLOCKED: TaskStatus.BLOCKED,
        }
        for st_id, status in report.statuses.items():
            st = by_id.get(st_id)
            if st is not None:
                st.status = status_map.get(status, TaskStatus.FAILED)

        if report.blocked:
            logger.warning("plan %s: %d subtask(s) blocked by failed dependencies: %s",
                           getattr(plan, "intent", "?"), len(report.blocked),
                           report.blocked)
        if report.cyclic:
            logger.error("plan %s: circular dependency among %s",
                         getattr(plan, "intent", "?"), report.cyclic)
        if report.unresolved_deps:
            logger.error("plan %s: subtasks depend on undefined ids: %s",
                         getattr(plan, "intent", "?"), report.unresolved_deps)

        return report.results

    # ── Dependency injection ───────────────────────────────────────────────

    def _inject_deps(
        self,
        params: Dict[str, Any],
        depends_on: List[str],
        completed: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Replace {subtask_id.result.field} templates with actual values."""
        if not depends_on:
            return params

        enriched = params.copy()
        for key, value in enriched.items():
            if not isinstance(value, str) or "{" not in value:
                continue
            for dep_id in depends_on:
                if dep_id not in completed:
                    continue
                dep_result = completed[dep_id].get("result", {})
                if isinstance(dep_result, dict):
                    for field, field_val in dep_result.items():
                        placeholder = f"{{{dep_id}.result.{field}}}"
                        if placeholder in value:
                            enriched[key] = value.replace(placeholder, str(field_val))
        return enriched

    # ── Duration extraction ────────────────────────────────────────────────

    def _extract_duration_minutes(self, text: str) -> Optional[int]:
        """
        Pull an explicit duration out of a natural language meeting request.

        Returns minutes (int) when a duration is found, or None when the user
        didn't specify one (caller should then ask). Examples that match:
          "30 minute meeting"       → 30
          "1 hour call"             → 60
          "1.5 hour sync"           → 90
          "half hour"               → 30
          "quarter hour"            → 15
          "an hour"                 → 60
          "two hours"               → 120
          "45 mins"                 → 45
        """
        import re as _redur
        t = text.lower()

        # Numeric: "30 minutes", "45 mins", "1 hour", "2 hrs", "1.5 hour"
        m = _redur.search(
            r'\b(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m)\b',
            t,
        )
        if m:
            value = float(m.group(1))
            unit = m.group(2)
            if unit.startswith(("hour", "hr", "h")):
                return int(round(value * 60))
            return int(round(value))

        # Worded numbers + "hour(s)"
        worded = {
            "an hour": 60, "one hour": 60, "a hour": 60,
            "two hours": 120, "three hours": 180, "four hours": 240,
            "half hour": 30, "half an hour": 30, "half-hour": 30,
            "quarter hour": 15, "quarter of an hour": 15, "quarter-hour": 15,
        }
        for phrase, mins in worded.items():
            if phrase in t:
                return mins

        return None

    # ── Temporal resolution ────────────────────────────────────────────────

    def _resolve_temporal(self, phrase: str) -> Dict[str, str]:
        """Convert natural language time phrases to ISO datetime with timezone."""
        import re
        from datetime import timedelta, timezone
        from zoneinfo import ZoneInfo

        # Use local timezone
        try:
            local_tz = ZoneInfo("Europe/London")
        except Exception:
            local_tz = timezone.utc

        now = datetime.now(local_tz)
        target = now
        p = phrase.lower()

        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
        }

        # Date resolution
        if "tomorrow" in p:
            target = now + timedelta(days=1)
        elif "next week" in p:
            target = now + timedelta(days=7)
        elif "today" in p or "tonight" in p:
            target = now
        else:
            for day_name, day_num in day_map.items():
                if day_name in p:
                    days_ahead = (day_num - now.weekday()) % 7 or 7
                    target = now + timedelta(days=days_ahead)
                    break

        # Time resolution — find the LAST time mention in the phrase
        # to avoid picking up times from earlier context
        time_matches = list(re.finditer(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", p))
        if not time_matches:
            # Try without am/pm but only if a clear time word is nearby
            time_matches = list(re.finditer(r"at\s+(\d{1,2})(?::(\d{2}))?(?!\s*(?:am|pm))", p))
            if time_matches:
                # Reformat match groups
                m = time_matches[-1]
                h = int(m.group(1))
                mins = int(m.group(2) or 0)
                # Default: if hour < 8, assume pm (e.g. "at 2" = 2pm)
                if h < 8:
                    h += 12
                target = target.replace(hour=h, minute=mins, second=0, microsecond=0)
            else:
                # No time found — default to 9am
                target = target.replace(hour=9, minute=0, second=0, microsecond=0)
        else:
            m = time_matches[-1]
            h = int(m.group(1))
            mins = int(m.group(2) or 0)
            period = m.group(3)
            if period == "pm" and h != 12:
                h += 12
            elif period == "am" and h == 12:
                h = 0
            target = target.replace(hour=h, minute=mins, second=0, microsecond=0)

        end = target + timedelta(hours=1)

        return {
            "datetime": target.isoformat(),
            "date": target.date().isoformat(),
            "time": target.strftime("%H:%M"),
            "end_datetime": end.isoformat(),
            "timezone": str(local_tz),
        }

    # ── Response building ─────────────────────────────────────────────────

    def _build_response_message(
        self,
        request: str,
        plan: TaskPlan,
        results: Dict[str, Any],
        primary_agent: AgentRole,
    ) -> str:
        """Build a natural language response from execution results.

        IMPORTANT: failures must NOT be silently dropped. Previously this only
        collected messages from successful steps, so a failed action (e.g. a
        brightness change that didn't go through) would either vanish or be
        replaced with a vague "Completed 0/1 steps" — or worse, a sibling
        success message made it look like everything worked. We now surface
        failure messages/errors first so the user is always told the truth.
        """
        success_msgs = []
        failure_msgs = []
        for result in results.values():
            if result.get("success"):
                if result.get("message"):
                    success_msgs.append(result["message"])
            else:
                # Prefer an explicit message, fall back to the error text.
                fmsg = result.get("message") or result.get("error")
                if fmsg:
                    failure_msgs.append(fmsg)

        # Lead with failures so a real failure is never buried under a
        # success confirmation, then add any genuine successes.
        if failure_msgs:
            return " ".join(failure_msgs + success_msgs)
        if success_msgs:
            return " ".join(success_msgs)

        # Fallback: summarise what happened
        successes = sum(1 for r in results.values() if r.get("success"))
        total = len(results)
        if successes == total and total > 0:
            return f"Done — completed {successes}/{total} steps."
        return (
            f"I couldn't fully complete that — only {successes}/{total} steps "
            f"succeeded for: {plan.intent.replace('_', ' ')}."
        )


    # ── Shortcut handler ──────────────────────────────────────────────────

    async def _send_pending_draft(self, draft: "EmailDraft") -> Dict[str, Any]:
        """
        Send a confirmed pending email draft.

        Reply drafts go through GmailAgent.reply_to_email so the message stays
        in-thread (threadId + In-Reply-To/References headers); everything else
        is a fresh send_email. Centralising this means both confirmation paths
        (the pending-state intercept and the shortcut handler) behave the same.
        """
        if getattr(draft, "is_reply", False) and getattr(draft, "original_email", None):
            return await self.gmail.reply_to_email(draft.original_email, draft.body)
        return await self.gmail.send_email(
            to=draft.recipient_email,
            subject=draft.subject,
            body=draft.body,
        )

    async def _try_shortcut(
        self,
        primary_agent: AgentRole,
        user_request: str,
        *,
        voice_mode: bool = False,
        conversation_history: Optional[List[Dict]] = None,
    ):
        """
        Bypass LLM planning for deterministic single-tool intents.
        Returns a JarvisResponse if handled, None otherwise.

        voice_mode flag is passed through to shortcut handlers that have
        a voice-friendly alternative output shape (e.g. file listings,
        morning brief). conversation_history is used by the save-conversation
        file shortcut.
        """
        import time as _time
        import re
        start = _time.time()

        # Also catch weather requests the router misclassified
        req_lower = user_request.lower()

        # ── Battery (must be before weather — "temperature" could collide) ──
        if any(kw in req_lower for kw in ["battery", "battery level", "how much battery"]):
            import time as _tbat2
            _sbat2 = _tbat2.time()
            result = await self.mac.get_battery()
            if result.get("success"):
                pct = result.get("battery_pct", "unknown")
                msg = f"Battery is at {pct}%."
            else:
                msg = "Could not read battery level."
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tbat2.time()-_sbat2)*1000)

        # ── System diagnostic — real subprocess checks, no LLM ──────────────
        # Without this, "run a system diagnostic / check / health check" falls
        # through to Tier-2 LLM chat which fabricates plausible-looking numbers
        # (e.g. invents "92% disk usage", "56°C") from training data instead of
        # measuring the actual machine. We collect the same metrics the
        # /hardware endpoint exposes — battery, disk, memory, CPU, thermal —
        # and format them as a real diagnostic report.
        _diag_phrases = (
            "system diagnostic", "system check", "system health",
            "run diagnostic", "run a diagnostic", "run a quick diagnostic",
            "run a system", "diagnose my mac", "diagnose my system",
            "health check", "system status", "hardware check",
            "how is my mac", "how's my mac", "check my mac",
        )
        if any(p in req_lower for p in _diag_phrases):
            import asyncio as _adiag
            import subprocess as _sp, re as _re_diag, time as _tdiag
            _sdiag = _tdiag.time()

            async def _shell(cmd: str) -> str:
                try:
                    proc = await _adiag.create_subprocess_shell(
                        cmd,
                        stdout=_sp.PIPE,
                        stderr=_sp.DEVNULL,
                    )
                    stdout, _ = await _adiag.wait_for(proc.communicate(), timeout=4)
                    return (stdout or b"").decode(errors="ignore").strip()
                except Exception:
                    return ""

            # All probes run in parallel — total wall-clock ~1-1.5s on M-series
            (bat_pct, bat_state, disk_raw, mem_total_raw, vm_raw,
             cpu_raw, uptime_raw, sw_ver) = await _adiag.gather(
                _shell("pmset -g batt | grep -Eo '[0-9]+%' | head -1"),
                _shell("pmset -g batt | grep -Eo 'AC Power|Battery Power' | head -1"),
                _shell("df -h / | tail -1 | awk '{print $2, $3, $4, $5}'"),
                _shell("sysctl -n hw.memsize"),
                _shell("vm_stat | head -20"),
                _shell("top -l 1 -n 0 | awk '/CPU usage/ {print $3, $5}'"),
                _shell("uptime"),
                _shell("sw_vers -productVersion"),
            )

            # Parse battery
            battery_str = "—"
            if bat_pct:
                bs = bat_state or ""
                battery_str = f"{bat_pct.strip()} ({'plugged in' if 'AC' in bs else 'on battery'})"

            # Parse disk
            disk_str = "—"
            parts = disk_raw.split() if disk_raw else []
            if len(parts) >= 4:
                # total, used, available, pct
                disk_str = f"{parts[1]} used of {parts[0]} ({parts[3]} full, {parts[2]} free)"

            # Parse memory
            mem_str = "—"
            try:
                total_b = int(mem_total_raw) if mem_total_raw else 0
                total_gb = total_b / (1024 ** 3)
                page_size = 4096
                pm = _re_diag.search(r"page size of (\d+)", vm_raw)
                if pm:
                    page_size = int(pm.group(1))
                def _vm(field):
                    m = _re_diag.search(rf"{field}:\s+(\d+)\.", vm_raw)
                    return int(m.group(1)) if m else 0
                used_pages = _vm("Pages active") + _vm("Pages wired down") + _vm("Pages occupied by compressor")
                used_gb = used_pages * page_size / (1024 ** 3)
                if total_gb:
                    pct = round(used_gb / total_gb * 100, 1)
                    mem_str = f"{used_gb:.1f} GB used of {total_gb:.1f} GB ({pct}%)"
            except Exception:
                pass

            # Parse CPU
            cpu_str = "—"
            try:
                nums = _re_diag.findall(r"([\d.]+)%", cpu_raw or "")
                if nums:
                    total = round(sum(float(n) for n in nums), 1)
                    cpu_str = f"{total}% active"
            except Exception:
                pass

            # Parse uptime
            uptime_str = "—"
            if uptime_raw:
                m = _re_diag.search(r"up\s+(.+?),\s+\d+\s+user", uptime_raw)
                if m:
                    uptime_str = m.group(1).strip()

            os_str = f"macOS {sw_ver}" if sw_ver else "macOS"

            # Mini health verdict — green/amber/red per metric
            def _verdict_disk():
                if len(parts) >= 4 and parts[3].endswith("%"):
                    try:
                        pct = int(parts[3].rstrip("%"))
                        if pct >= 90: return "critical"
                        if pct >= 80: return "watch"
                    except ValueError:
                        pass
                return "ok"

            def _verdict_mem():
                try:
                    pct_match = _re_diag.search(r"\(([\d.]+)%\)", mem_str)
                    if pct_match:
                        pct = float(pct_match.group(1))
                        if pct >= 90: return "critical"
                        if pct >= 75: return "watch"
                except Exception:
                    pass
                return "ok"

            disk_verdict = _verdict_disk()
            mem_verdict = _verdict_mem()
            overall = "All systems nominal."
            if "critical" in (disk_verdict, mem_verdict):
                overall = "One or more metrics need attention."
            elif "watch" in (disk_verdict, mem_verdict):
                overall = "Everything's within range, a couple of values are worth watching."

            lines = [
                f"**System diagnostic** — {os_str}",
                "",
                f"**Battery**   {battery_str}",
                f"**Disk**      {disk_str}",
                f"**Memory**    {mem_str}",
                f"**CPU**       {cpu_str}",
                f"**Uptime**    {uptime_str}",
                "",
                f"**Verdict**   {overall}",
            ]
            msg = "\n".join(lines)
            print(f"Jarvis shortcut: diagnostic — {(_tdiag.time()-_sdiag)*1000:.0f}ms")
            return JarvisResponse(
                success=True,
                message=msg,
                latency_ms=(_tdiag.time() - _sdiag) * 1000,
            )

        weather_keywords = ["weather", "temperature", "forecast", "humid", "rain", "sunny", "cloudy", "wind speed"]
        is_weather_request = (
            primary_agent == AgentRole.WEATHER or
            any(kw in req_lower for kw in weather_keywords)
        )

        if is_weather_request:
            req = req_lower
            is_forecast = any(w in req for w in ["forecast", "this week", "next week", "7 day", "seven day", "weekly"])

            # Detect if user specified a location other than the default
            location = self._extract_location(user_request)

            if location:
                if is_forecast:
                    data = await self.weather.get_forecast_for_location(location)
                else:
                    data = await self.weather.get_current_for_location(location)
            else:
                if is_forecast:
                    data = await self.weather.get_forecast()
                else:
                    data = await self.weather.get_current()

            msg = self.weather.format_forecast(data) if is_forecast else self.weather.format_current(data)
            asyncio.ensure_future(self.memory.store_task_result(
                user_request, "get_weather", data.get("success", False), msg[:100]
            ))
            print(f"Jarvis shortcut: weather — {(_time.time()-start)*1000:.0f}ms")
            return JarvisResponse(
                success=data.get("success", False),
                message=msg,
                latency_ms=(_time.time() - start) * 1000,
            )

        # News shortcut — smart category/source/topic detection
        news_keywords = [
            "news", "headlines", "latest news", "top stories", "breaking",
            "what's happening", "whats happening", "current events",
            "sports news", "tech news", "business news", "world news",
            "science news", "ai news", "uk news", "football news",
        ]
        if any(kw in req_lower for kw in news_keywords):
            import time as _t
            _s = _t.time()
            detailed = any(w in req_lower for w in ["detailed", "detail", "more info", "tell me more"])
            data = await self.news.get_headlines(
                query=user_request,
                max_items=6,
            )
            msg = self.news.format_headlines(data, detailed=detailed)
            asyncio.ensure_future(self.memory.store_task_result(user_request, "get_news", True, msg[:100]))
            print(f"Jarvis shortcut: news — {(_t.time()-_s)*1000:.0f}ms")
            return JarvisResponse(success=True, message=msg, latency_ms=(_t.time()-_s)*1000)

        # ── Early Spotify intercept — catches "play X" BEFORE sports routing ──
        _early_play = (
            req_lower.startswith("play ") and
            not any(kw in req_lower for kw in ["play premier", "play match", "play game",
                                                "play the game", "play the match"])
        )
        if _early_play:
            import time as _tep
            _sep = _tep.time()
            _query_ep = re.sub(r'^play\s+', '', user_request, flags=re.IGNORECASE).strip()
            _query_ep = re.sub(r'\s+on spotify$', '', _query_ep, flags=re.IGNORECASE).strip()
            if _query_ep and _query_ep.lower() not in ("music", "something", "anything", "spotify"):
                result = await self.spotify.play_by_name(_query_ep)
                msg = self.spotify.format_play_result(result)
            else:
                result = await self.spotify.play()
                msg = "Resumed playback." if result.get("success") else result.get("error", "Could not resume.")
            return JarvisResponse(success=result.get("success", False), message=msg,
                                  latency_ms=(_tep.time()-_sep)*1000)

        # ── Sports shortcuts ──────────────────────────────────────────────────
        sports_keywords = [
            "scores", "results", "fixtures", "standings", "table",
            "premier league", "champions league", "la liga", "serie a",
            "bundesliga", "nfl", "nba", "nhl", "mlb", "f1", "formula 1",
            "football scores", "football results", "basketball scores",
            "sports results", "match results", "game scores", "football score",
            "who won", "vs", "final score", "match score",
        ]
        # ── Calendar pre-check — must come before sports to avoid "call" collision ──
        _has_meeting = any(w in req_lower for w in ["meeting", "appointment", "event", "session"])
        _has_schedule = any(w in req_lower for w in [
            "schedule", "book", "create", "add", "set", "arrange",
            "block", "put", "plan", "organise", "organize", "new"
        ])
        _explicit_cal = any(kw in req_lower for kw in ["add to calendar", "calendar event", "add event"])
        if (_has_meeting and _has_schedule) or _explicit_cal:
            # Jump straight to calendar section below
            pass
        else:
            sports_context = any(kw in req_lower for kw in sports_keywords)
        from config.settings import FAVOURITE_TEAMS
        team_sports = [
            # Premier League
            "arsenal", "chelsea", "liverpool", "manchester", "spurs",
            "tottenham", "city", "united", "west ham", "newcastle",
            "aston villa", "brighton", "everton", "wolves", "fulham",
            "brentford", "crystal palace", "bournemouth", "ipswich",
            "leicester", "southampton",
            # European
            "real madrid", "barcelona", "atletico", "juventus", "inter",
            "ac milan", "napoli", "bayern", "dortmund", "psg",
            # NBA
            "lakers", "celtics", "warriors", "bulls", "heat", "nets",
            "knicks", "clippers", "suns", "bucks", "nuggets", "76ers",
            # NFL
            "patriots", "chiefs", "cowboys", "packers", "eagles",
            # Cricket
            "pakistan", "india", "england cricket", "australia cricket",
            "west indies", "south africa cricket", "new zealand cricket",
        ] + [t.lower() for t in FAVOURITE_TEAMS]
        team_mentioned = any(t in req_lower for t in team_sports)

        sports_context = any(kw in req_lower for kw in sports_keywords)
        # Exclude clear music/Spotify requests from sports routing
        _is_music_req = (
            req_lower.startswith("play ") or
            any(kw in req_lower for kw in ["play song", "play track", "play music", "by michael", "by drake",
                                            "by the weeknd", "by kanye", "by taylor", "by eminem",
                                            "on spotify", "spotify", "pause music", "skip track"])
        )
        if (sports_context or team_mentioned) and not ((_has_meeting and _has_schedule) or _explicit_cal) and not _is_music_req:
            import time as _tsp
            _ssp = _tsp.time()

            # Team → league mapping (used when detect_league returns nothing)
            _team_league_map = {
                # La Liga
                "real madrid": "la_liga", "barcelona": "la_liga", "atletico": "la_liga",
                "sevilla": "la_liga", "villarreal": "la_liga", "real sociedad": "la_liga",
                # Serie A
                "juventus": "serie_a", "inter": "serie_a", "ac milan": "serie_a",
                "napoli": "serie_a", "roma": "serie_a", "lazio": "serie_a",
                # Bundesliga
                "bayern": "bundesliga", "dortmund": "bundesliga", "leverkusen": "bundesliga",
                "leipzig": "bundesliga",
                # Ligue 1
                "psg": "ligue_1", "paris saint": "ligue_1", "marseille": "ligue_1",
                # NBA
                "lakers": "nba", "celtics": "nba", "warriors": "nba", "bulls": "nba",
                "heat": "nba", "nets": "nba", "knicks": "nba", "clippers": "nba",
                "suns": "nba", "bucks": "nba", "nuggets": "nba", "76ers": "nba",
                # Cricket
                "pakistan": "cricket", "india": "cricket", "west indies": "cricket",
                # Premier League teams stay as default
            }

            # Detect league from request, then from team name
            league_key = self.sports.detect_league(user_request)
            if not league_key:
                for team_kw, mapped_league in _team_league_map.items():
                    if team_kw in req_lower:
                        league_key = mapped_league
                        break

            # If team mentioned, search for that team
            if team_mentioned and not any(kw in req_lower for kw in ["table", "standings"]):
                if not league_key:
                    league_key = "premier_league"
                data = await self.sports.search_team(user_request, league_key)
                if data.get("success") and data.get("games"):
                    msg = self.sports.format_scores(data)
                else:
                    # Fallback to full league scores
                    data = await self.sports.get_scores(league_key or "premier_league")
                    msg = self.sports.format_scores(data)
            elif any(kw in req_lower for kw in ["table", "standings", "top of", "who is top", "who leads"]):
                if not league_key:
                    league_key = "premier_league"
                data = await self.sports.get_standings(league_key)
                msg = self.sports.format_standings(data)
            else:
                if not league_key:
                    league_key = "premier_league"
                data = await self.sports.get_scores(league_key)
                msg = self.sports.format_scores(data)

            asyncio.ensure_future(self.memory.store_task_result(user_request, "sports", True, msg[:100]))
            return JarvisResponse(success=True, message=msg, latency_ms=(_tsp.time()-_ssp)*1000)

        # News digest — all categories
        if any(kw in req_lower for kw in ["morning briefing", "daily briefing", "news digest", "all news", "full news"]):
            import time as _td
            _sd = _td.time()
            digest = await self.news.get_all_categories(max_per_category=2)
            lines = ["Your news digest:"]
            for cat, items in digest.get("digest", {}).items():
                if items:
                    lines.append(f"**{cat.title()}**")
                    for item in items:
                        lines.append(f"  • {item['title']}")
                    lines.append("")
            msg = chr(10).join(lines)
            return JarvisResponse(success=True, message=msg, latency_ms=(_td.time()-_sd)*1000)

        # List available news sources
        if any(kw in req_lower for kw in ["news sources", "available news", "what news sources"]):
            return JarvisResponse(success=True, message=self.news.list_sources())

        if primary_agent == AgentRole.NEWS:
            data = await self.news.get_headlines(max_items=5)
            msg = self.news.format_headlines(data)
            asyncio.ensure_future(self.memory.store_task_result(
                user_request, "get_news", True, msg[:100]
            ))
            return JarvisResponse(
                success=True,
                message=msg,
                latency_ms=(_time.time() - start) * 1000,
            )

        # Email shortcut — read inbox
        email_read_keywords = ["check my email", "read my email", "my inbox", "any emails", "new emails", "unread"]
        if any(kw in req_lower for kw in email_read_keywords):
            import time as _t2
            _s2 = _t2.time()
            # Use the SAME source/order as the sidebar (is:inbox, newest first)
            # so the numbered list here and a follow-up "reply to email N" both
            # match what the user sees in the inbox panel.
            result = await self.gmail.get_inbox(max_results=8, query="is:inbox")
            # Cache the indexed list so a follow-up "reply to email 2" resolves.
            self._last_inbox = result.get("emails", []) or []
            msg = result.get("message", "Could not read inbox.")
            asyncio.ensure_future(self.memory.store_task_result(user_request, "read_email", True, msg[:100]))
            return JarvisResponse(success=True, message=msg, latency_ms=(_t2.time()-_s2)*1000)

        # ── Email shortcut — read / reply / archive a NUMBERED inbox email ──
        # Handles "read email 1", "reply to email 2 with a heart emoji",
        # "archive email 3". Without this the request falls through to the LLM,
        # which hallucinates a draft (and later a fake "Email sent.") because no
        # real pending draft is ever created. The number maps into the last
        # inbox the user saw; if the cache is empty we fetch it on demand.
        import re as _re_emn
        _emn = _re_emn.search(
            r'\b(read|reply|archive)\b(?:\s+to)?\s+(?:email|mail|message)\s*#?\s*(\d+)',
            req_lower,
        )
        if _emn:
            import time as _temn
            _semn = _temn.time()
            action_kind = _emn.group(1)
            idx = int(_emn.group(2))

            # Resolve the number against a FRESH is:inbox / newest-first list —
            # the same source and order as the sidebar — so "email N" is always
            # the message the user sees at position N. We deliberately do NOT
            # trust self._last_inbox here: other commands ("summarize inbox")
            # may have cached an is:unread list with a different order, which
            # previously caused replies to target the wrong email. Fall back to
            # the cache only if the live fetch fails (offline / mock).
            _inb = await self.gmail.get_inbox(max_results=8, query="is:inbox")
            _fresh = _inb.get("emails", []) or []
            if _fresh:
                self._last_inbox = _fresh

            if not self._last_inbox:
                return JarvisResponse(
                    success=False,
                    message="I couldn't read your inbox to find that email. Try 'check my inbox' first.",
                    latency_ms=(_temn.time()-_semn)*1000,
                )
            if idx < 1 or idx > len(self._last_inbox):
                return JarvisResponse(
                    success=False,
                    message=(
                        f"I only see {len(self._last_inbox)} email"
                        f"{'s' if len(self._last_inbox) != 1 else ''} in your inbox. "
                        f"Pick a number between 1 and {len(self._last_inbox)}."
                    ),
                    latency_ms=(_temn.time()-_semn)*1000,
                )

            target = self._last_inbox[idx - 1]

            # ── READ ───────────────────────────────────────────────────────
            if action_kind == "read":
                body_res = await self.gmail.get_email_body(target.get("id", ""))
                if not body_res.get("success"):
                    return JarvisResponse(
                        success=False,
                        message=f"Could not read email {idx}: {body_res.get('error', 'unknown error')}",
                        latency_ms=(_temn.time()-_semn)*1000,
                    )
                sender = body_res.get("from", target.get("from", "Unknown"))
                subj = body_res.get("subject", target.get("subject", "(no subject)"))
                body_text = (body_res.get("body") or "").strip() or "(empty body)"
                msg = f"From: {sender}\nSubject: {subj}\n\n{body_text}"
                asyncio.ensure_future(self.memory.store_task_result(user_request, "read_email", True, subj[:100]))
                return JarvisResponse(success=True, message=msg, latency_ms=(_temn.time()-_semn)*1000)

            # ── ARCHIVE ──────────────────────────────────────────────────────
            if action_kind == "archive":
                ar = await self.gmail.archive_email(target.get("id", ""))
                ok = ar.get("success", False)
                msg = ar.get("message") or ("Email archived." if ok else
                      f"Could not archive: {ar.get('error', 'unknown error')}")
                if ok:
                    # Drop it from the cached list so indices stay meaningful.
                    self._last_inbox.pop(idx - 1)
                asyncio.ensure_future(self.memory.store_task_result(user_request, "archive_email", ok, msg[:100]))
                return JarvisResponse(success=ok, message=msg, latency_ms=(_temn.time()-_semn)*1000)

            # ── REPLY ────────────────────────────────────────────────────────
            # Extract the reply instruction = text after "email N".
            instruction = _re_emn.sub(
                r'^.*?\b(?:email|mail|message)\s*#?\s*\d+\b[\s,:-]*', '', user_request, count=1
            ).strip()
            if instruction.lower().startswith(("with ", "saying ", "say ", "that ")):
                instruction = instruction.split(" ", 1)[1].strip() if " " in instruction else instruction
            if not instruction:
                instruction = "a brief, polite acknowledgement"

            reply_body = await self.composer.compose_reply(target, instruction)

            parsed = self.gmail._parse_address(target.get("from", ""))
            r_name = parsed.get("name") or parsed.get("email") or "them"
            r_email = parsed.get("email", "")
            subj = target.get("subject", "")
            if not subj.lower().startswith("re:"):
                subj = f"Re: {subj}"

            draft = EmailDraft(
                recipient_name=r_name,
                recipient_email=r_email,
                subject=subj,
                body=reply_body,
                tone="reply",
                intent="reply",
                contact_found=bool(r_email),
                needs_email=False,
                is_reply=True,
                original_email=target,
            )
            self._pending_email = draft
            msg = self.composer.format_draft_for_confirmation(draft)
            return JarvisResponse(success=True, message=msg, latency_ms=(_temn.time()-_semn)*1000)

        # Email shortcut — confirmation check first
        confirm_keywords = ["yes send it", "yes", "send it", "confirm", "go ahead", "yeah send"]
        if self._pending_email and any(kw in req_lower for kw in confirm_keywords):
            import time as _tc
            _sc = _tc.time()
            draft = self._pending_email
            self._pending_email = None
            result = await self._send_pending_draft(draft)
            msg = result.get("message", f"Email sent to {draft.recipient_email}")
            # Auto-save contact after successful send
            if result.get("success") and draft.recipient_email and draft.recipient_name:
                self.contacts.add(draft.recipient_name, draft.recipient_email)
            asyncio.ensure_future(self.memory.store_task_result(user_request, "send_email", result.get("success", False), msg[:100]))
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tc.time()-_sc)*1000)

        # Cancel pending email
        cancel_keywords = ["no", "cancel", "don't send", "do not send", "abort"]
        if self._pending_email and any(kw in req_lower for kw in cancel_keywords):
            self._pending_email = None
            return JarvisResponse(success=True, message="Email cancelled. No email was sent.")

        # ── Email address reply — MUST be first, before everything ─────────────
        # ── Email address reply — MUST be first, before everything ─────────────
        import re as _reemail
        # Strip markdown link format if present e.g. [email](mailto:email)
        _ur_clean = _reemail.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", user_request.strip())
        _email_m = _reemail.search(r"[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}", _ur_clean)
        _is_email_reply = _email_m and self._pending_email and getattr(self._pending_email, "needs_email", False)
        if _is_email_reply:
            import time as _teer
            _seer = _teer.time()
            self._pending_email.recipient_email = _email_m.group(0)
            self._pending_email.needs_email = False
            # Auto-save this new contact so we remember them next time
            if self._pending_email.recipient_name and self._pending_email.recipient_email:
                self.contacts.add(self._pending_email.recipient_name, self._pending_email.recipient_email)
            msg = self.composer.format_draft_for_confirmation(self._pending_email)
            return JarvisResponse(success=True, message=msg, latency_ms=(_teer.time()-_seer)*1000)

        # ── Edit pending email ─────────────────────────────────────────────────
        _edit_kws = ["edit", "change", "modify", "rewrite", "update", "tell her",
                     "tell him", "add that", "also say", "mention", "make it"]
        if self._pending_email and not getattr(self._pending_email, "needs_email", False) and any(kw in req_lower for kw in _edit_kws):
            import time as _tedit
            _sedit = _tedit.time()
            ep = "Edit this email body based on the instruction.\n\nCurrent body:\n" + self._pending_email.body + "\n\nInstruction: " + user_request + "\n\nReturn ONLY the updated body."
            try:
                nb = await self.llm.chat([{"role": "user", "content": ep}])
                self._pending_email.body = nb.strip()
                msg = "Updated! " + self.composer.format_draft_for_confirmation(self._pending_email)
            except Exception as e:
                msg = "Could not edit: " + str(e)
            return JarvisResponse(success=True, message=msg, latency_ms=(_tedit.time()-_sedit)*1000)


        # ── Multi-query — handle compound requests before the full briefing ─────
        # e.g. "what's the weather and latest news and premier league scores?"
        intents = self.briefing.detect_intents(user_request)
        if len(intents) >= 2 and not self.briefing.is_morning_briefing(user_request):
            import time as _tmq
            _smq = _tmq.time()
            multi_msg = await self._handle_multi_query(user_request, intents)
            if multi_msg:
                return JarvisResponse(success=True, message=multi_msg, latency_ms=(_tmq.time()-_smq)*1000)

        # ── Morning Briefing ─────────────────────────────────────────────────
        # NOTE: this intercept is now done at the TOP of handle() and
        # handle_stream() — before the router runs — because the LLM router
        # was classifying "morning brief" as Tier-2 chat and skipping
        # _try_shortcut entirely. Kept as a comment so the routing audit
        # trail is clear; the logic itself lives in
        # _try_morning_brief_intercept() and _morning_briefing().

        # ── File Manager — confirmation flow ──────────────────────────────────
        import time as _tfile
        _sfile = _tfile.time()

        # Handle pending file operation confirmation
        if self._pending_file_op:
            _low = req_lower.strip()
            if _low in ("confirm", "yes", "do it", "go ahead", "proceed", "ok"):
                op = self._pending_file_op
                self._pending_file_op = None
                if op.operation == "delete":
                    result = self.files.execute_delete(op)
                elif op.operation == "move":
                    result = self.files.execute_move(op)
                elif op.operation == "rename":
                    result = self.files.execute_rename(op)
                else:
                    result = {"success": False, "error": "Unknown operation"}
                msg = result.get("message", "Done.") if result.get("success") else f"Failed: {result.get('error')}"
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tfile.time()-_sfile)*1000)
            elif _low in ("cancel", "no", "stop", "abort", "nevermind", "never mind"):
                self._pending_file_op = None
                return JarvisResponse(success=True, message="Cancelled.",
                                      latency_ms=(_tfile.time()-_sfile)*1000)

        # ── Open a file by extension ────────────────────────────────────────
        # "open README.md", "show me the contents of foo.txt", "read notes.md"
        # → route through file_manager.read_file (NOT the mac open-app path).
        # The mac open-app shortcut above already skips when an extension is
        # detected; this is the matching positive intercept that actually
        # does the read.
        import re as _refile_open
        # IMPORTANT: alternation order matters because the lazy `.*?`
        # before `\.` will accept the first matching extension. List
        # LONGER extensions BEFORE shorter ones that share a prefix,
        # otherwise "package.json" gets captured as "package.js" and
        # "notes.markdown" as "notes.mark".
        _open_file_match = _refile_open.search(
            r'(?:open|read|show(?:\s+me)?(?:\s+the)?(?:\s+contents?\s+of)?|'
            r'display|cat|preview|peek\s+at|look\s+at)\s+'
            r'(?:the\s+|my\s+|file\s+)?'
            r'["\']?([\w\-][\w\-\.\s]*?\.'
            r'(?:markdown|md|'
            r'json|jsx|js|tsx|ts|'
            r'yaml|yml|'
            r'html|htm|css|'
            r'pptx|ppt|xlsx|xls|docx|doc|'
            r'txt|csv|toml|cfg|ini|env|log|xml|rst|sql|graphql|'
            r'py|sh|zsh|java|cpp|c|h|go|rb|php|swift|kt|r))'
            r'["\']?\b',
            user_request, _refile_open.IGNORECASE,
        )
        if _open_file_match:
            import time as _tfo
            _sfo = _tfo.time()
            target_file = _open_file_match.group(1).strip()
            data = self.files.read_file(target_file)
            if data.get("success"):
                content = data.get("content", "")
                trunc = "\n\n[File truncated at 50KB]" if data.get("truncated") else ""
                if voice_mode:
                    # Read aloud the first few lines, not the whole thing
                    snippet = content[:500].strip()
                    msg = (
                        f"{data['name']} — {data['lines']} lines, "
                        f"{self.files._fmt_size(data['size'])}. "
                        f"It starts with: {snippet[:300]}…"
                        if len(content) > 300
                        else f"{data['name']}: {content}"
                    )
                else:
                    msg = f"**{data['display_path']}**\n\n{content}{trunc}"
            else:
                msg = data.get("error", f"Could not open {target_file}.")
                # Suggest alternative — maybe they meant an app with a dot
                if "extension" in data.get("error", "") or "not allowed" in data.get("error", "").lower():
                    msg += " (Try giving the full path, or check Desktop/Documents/Downloads.)"
            return JarvisResponse(
                success=data.get("success", False),
                message=msg,
                latency_ms=(_tfo.time() - _sfo) * 1000,
            )

        # Detect file intent keywords. Order doesn't matter (we just check
        # any-match) but each new phrase should be reflected in the parser
        # below or it'll fall through to "no action".
        _file_kw = [
            "folder", "directory", "file", "files", "desktop", "documents", "downloads",
            "create folder", "make folder", "new folder", "create file", "new file",
            "create a note", "make a note", "new note",
            "delete file", "delete folder", "remove file", "remove folder",
            "rename", "move to", "find file", "search file", "list files",
            "show files", "browse", "what's on my desktop", "whats on my desktop",
            "what's on my documents", "whats on my documents",
            "show me what's on", "show me whats on", "show me my files",
            "what files", "what do i have on my", "list my files",
            # NEW — content-bearing creation
            "with content", "with text", "with body", "saying ", "containing ",
            "that says", "that contains",
            # NEW — save chat / transcript
            "save this conversation", "save the conversation", "save our conversation",
            "save this chat", "save the chat", "save our chat",
            "save the discussion", "save the transcript",
            "export the conversation", "export this chat",
            "dump the conversation", "dump the chat",
            # NEW — extension intent searches
            "all my pdfs", "find pdfs", "find all pdfs", "list pdfs",
            "my images", "my photos", "my screenshots",
            "my spreadsheets", "my videos", "my notes",
        ]
        _is_file_request = any(kw in req_lower for kw in _file_kw)

        if _is_file_request:
            parsed = self.files.parse_request(user_request)
            action = parsed.get("action")
            loc = parsed.get("location", "desktop")
            path = parsed.get("path")
            name = parsed.get("name")
            dest = parsed.get("destination")

            # ── List directory ──────────────────────────────────────────────
            if action == "list" or (not action and any(w in req_lower for w in ["list", "show", "browse", "what's on", "whats on"])):
                target = loc  # location key ('desktop','documents','downloads','all')
                # Optional modifiers parsed from natural language
                include_hidden = any(p in req_lower for p in (
                    "include hidden", "with hidden", "show hidden",
                    "all files including hidden", "hidden files",
                    "dotfiles",
                ))
                sort_by = "name"
                reverse = False
                if any(p in req_lower for p in (
                    "by date", "by modified", "newest first", "most recent",
                    "recently modified",
                )):
                    sort_by, reverse = "modified", True
                elif any(p in req_lower for p in (
                    "oldest first", "by oldest",
                )):
                    sort_by, reverse = "modified", False
                elif any(p in req_lower for p in (
                    "by size", "biggest first", "largest first",
                )):
                    sort_by, reverse = "size", True
                elif any(p in req_lower for p in (
                    "smallest first",
                )):
                    sort_by, reverse = "size", False

                data = self.files.list_directory(
                    target,
                    include_hidden=include_hidden,
                    sort_by=sort_by,
                    reverse=reverse,
                )
                # Voice mode gets the prose summary; text mode gets the rich list.
                if voice_mode:
                    msg = self.files.format_listing_voice(data)
                else:
                    msg = self.files.format_listing(data)
                return JarvisResponse(success=data.get("success", False), message=msg,
                                      latency_ms=(_tfile.time()-_sfile)*1000)

            # ── Search ──────────────────────────────────────────────────────
            elif action == "search" and path:
                data = self.files.search(path, location=loc)
                msg = self.files.format_search(data)
                return JarvisResponse(success=True, message=msg,
                                      latency_ms=(_tfile.time()-_sfile)*1000)

            # ── Read file ───────────────────────────────────────────────────
            elif action == "read" and path:
                data = self.files.read_file(path)
                if data.get("success"):
                    content = data.get("content", "")
                    trunc = "\n\n[File truncated at 50KB]" if data.get("truncated") else ""
                    # Let LLM summarise if large
                    if len(content) > 2000:
                        prompt = (
                            f"The user wants to read this file: {data['name']}\n\n"
                            f"Content:\n{content[:4000]}\n\n"
                            f"Give a brief summary of what this file contains, then show the first ~30 lines."
                        )
                        summary = await self.llm.chat([{"role": "user", "content": prompt}])
                        msg = summary.strip() + trunc
                    else:
                        msg = f"{data['display_path']}:\n\n{content}{trunc}"
                else:
                    msg = data.get("error", "Could not read file.")
                return JarvisResponse(success=data.get("success", False), message=msg,
                                      latency_ms=(_tfile.time()-_sfile)*1000)

            # ── Create folder ───────────────────────────────────────────────
            elif action == "create_folder" and name:
                # Build full path using detected location
                from pathlib import Path as _P
                from tools.file_manager import ALLOWED_ROOTS as _ROOTS
                target_root = _ROOTS.get(loc, _ROOTS["desktop"])
                full_path = str(target_root / name)
                data = self.files.create_folder(full_path)
                msg = data.get("message", data.get("error", "Could not create folder."))
                return JarvisResponse(success=data.get("success", False), message=msg,
                                      latency_ms=(_tfile.time()-_sfile)*1000)

            # ── Create file (with optional content body) ────────────────────
            elif action == "create_file" and name:
                from tools.file_manager import ALLOWED_ROOTS as _ROOTS
                target_root = _ROOTS.get(loc, _ROOTS["desktop"])
                # Default extension if user gave only a stem like "notes"
                _name = name
                if "." not in _name:
                    _name = f"{_name}.txt"
                full_path = str(target_root / _name)
                content = parsed.get("content") or ""
                data = self.files.create_file(full_path, content=content)
                if data.get("success") and content:
                    # Nice user feedback noting the content was written
                    data["message"] = (
                        f"Created {data['display_path']} "
                        f"({len(content)} character{'s' if len(content) != 1 else ''})."
                    )
                msg = data.get("message", data.get("error", "Could not create file."))
                return JarvisResponse(success=data.get("success", False), message=msg,
                                      latency_ms=(_tfile.time()-_sfile)*1000)

            # ── Save conversation to file ──────────────────────────────────
            elif action == "save_conversation" and name:
                from tools.file_manager import ALLOWED_ROOTS as _ROOTS
                target_root = _ROOTS.get(loc, _ROOTS["desktop"])
                _name = name if "." in name else f"{name}.md"
                full_path = str(target_root / _name)

                # Render the conversation history as Markdown
                history = conversation_history or []
                if not history:
                    return JarvisResponse(
                        success=False,
                        message="There's nothing to save yet — no conversation history in this session.",
                        latency_ms=(_tfile.time()-_sfile)*1000,
                    )
                from datetime import datetime as _dt
                lines_md = [
                    f"# Conversation transcript",
                    f"Saved: {_dt.now().strftime('%A, %d %B %Y at %H:%M')}",
                    "",
                ]
                for turn in history:
                    role = turn.get("role", "?")
                    content = turn.get("content", "")
                    if role == "user":
                        lines_md.append(f"## You")
                    elif role == "assistant":
                        lines_md.append(f"## Jarvis")
                    else:
                        lines_md.append(f"## {role}")
                    lines_md.append(content.strip())
                    lines_md.append("")
                body = "\n".join(lines_md)
                data = self.files.create_file(full_path, content=body)
                msg = (
                    f"Conversation saved to {data['display_path']} "
                    f"({len(history)} message{'s' if len(history) != 1 else ''})."
                    if data.get("success")
                    else data.get("error", "Could not save conversation.")
                )
                return JarvisResponse(
                    success=data.get("success", False),
                    message=msg,
                    latency_ms=(_tfile.time()-_sfile)*1000,
                )

            # ── Delete — requires approval ───────────────────────────────────
            elif action == "delete" and path:
                from tools.file_manager import ALLOWED_ROOTS as _ROOTS
                # Try direct resolve first, then search by name in detected location
                op = self.files.prepare_delete(path)
                if isinstance(op, dict):
                    # Not found directly — search by name in detected root
                    search_root = _ROOTS.get(loc, None)
                    found_path = None
                    search_roots = [search_root] if search_root else list(_ROOTS.values())
                    for root in search_roots:
                        for candidate in root.rglob("*"):
                            if candidate.name.lower() == path.lower() or \
                               candidate.name.lower().replace(" ", "") == path.lower().replace(" ", ""):
                                found_path = candidate
                                break
                        if found_path:
                            break
                    if found_path:
                        op = self.files.prepare_delete(str(found_path))
                    else:
                        return JarvisResponse(success=False,
                            message=f"Could not find \"{path}\" on your {loc}. Try: find {path}",
                            latency_ms=(_tfile.time()-_sfile)*1000)
                if isinstance(op, dict):
                    return JarvisResponse(success=False, message=op.get("error", "Could not prepare delete."),
                                          latency_ms=(_tfile.time()-_sfile)*1000)
                self._pending_file_op = op
                return JarvisResponse(success=True, message=op.summary(),
                                      latency_ms=(_tfile.time()-_sfile)*1000)

            # ── Rename — requires approval ───────────────────────────────────
            elif action == "rename" and path and name:
                op = self.files.prepare_rename(path, name)
                if isinstance(op, dict):
                    return JarvisResponse(success=False, message=op.get("error", "Could not prepare rename."),
                                          latency_ms=(_tfile.time()-_sfile)*1000)
                self._pending_file_op = op
                return JarvisResponse(success=True, message=op.summary(),
                                      latency_ms=(_tfile.time()-_sfile)*1000)

            # ── Move — requires approval ─────────────────────────────────────
            elif action == "move" and path and dest:
                op = self.files.prepare_move(path, dest)
                if isinstance(op, dict):
                    return JarvisResponse(success=False, message=op.get("error", "Could not prepare move."),
                                          latency_ms=(_tfile.time()-_sfile)*1000)
                self._pending_file_op = op
                return JarvisResponse(success=True, message=op.summary(),
                                      latency_ms=(_tfile.time()-_sfile)*1000)

        # Pending email — user replied with just an email address
        import re as _re2
        email_only_match = _re2.search(r'^\s*[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}\s*$', user_request)
        if self._pending_email and self._pending_email.needs_email and email_only_match:
            import time as _ter
            _ser = _ter.time()
            email_addr = email_only_match.group(0).strip()
            self._pending_email.recipient_email = email_addr
            self._pending_email.needs_email = False
            msg = self.composer.format_draft_for_confirmation(self._pending_email)
            return JarvisResponse(success=True, message=msg, latency_ms=(_ter.time()-_ser)*1000)

        # Email shortcut — send email (Level 2-4 pipeline)
        send_keywords = ["send an email", "send email", "email to", "send a message to", "write an email", "draft an email"]
        if any(kw in req_lower for kw in send_keywords):
            import time as _ts
            _ss = _ts.time()

            # Pass the raw request straight through. The composer's _BODY_PROMPT
            # already enforces identity rules (Jarvis writing on Abdullah's
            # behalf) and a strict "stay on topic, do not fabricate" guard.
            # The previous wrapper said "Write in first person as Abdullah",
            # which CONTRADICTED the composer prompt and caused the model to
            # hallucinate filler content trying to reconcile the two voices.
            _jarvis_context = (
                "Stay strictly on the topic stated below. Do NOT invent or "
                "add any facts, plans, projects, meetings, or details that "
                "are not explicitly in the request.\n\n"
                f"Email request: {user_request}"
            )
            draft = await self.composer.compose(_jarvis_context, self.contacts)

            # Level 3: Contact not found — ask for email
            if draft.needs_email:
                self._pending_email = draft
                msg = (
                    f"I don't have an email address for '{draft.recipient_name}' in your contacts.\n"
                    f"What is their email address? (or say 'add [name] [email]' to save them)"
                )
                return JarvisResponse(success=True, message=msg, latency_ms=(_ts.time()-_ss)*1000)

            # No recipient at all
            if not draft.recipient_email:
                return JarvisResponse(success=False, message="I need an email address to send to. Who would you like to email?", latency_ms=(_ts.time()-_ss)*1000)

            # Level 4: Show draft for confirmation
            self._pending_email = draft
            msg = self.composer.format_draft_for_confirmation(draft)
            return JarvisResponse(success=True, message=msg, latency_ms=(_ts.time()-_ss)*1000)

        # Add contact shortcut
        add_contact_match = re.search(r'add\s+(\w+)\s+([\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,})', req_lower)
        if add_contact_match:
            name = add_contact_match.group(1).capitalize()
            email = add_contact_match.group(2)
            self.contacts.add(name, email)
            # If we have a pending email waiting for this contact
            if self._pending_email and self._pending_email.needs_email:
                self._pending_email.recipient_email = email
                self._pending_email.needs_email = False
                msg = self.composer.format_draft_for_confirmation(self._pending_email)
                return JarvisResponse(success=True, message=f"Contact saved! {msg}")
            return JarvisResponse(success=True, message=f"Contact saved: {name} → {email}")

        # List contacts shortcut
        if any(kw in req_lower for kw in ["my contacts", "list contacts", "show contacts"]):
            return JarvisResponse(success=True, message=self.contacts.format_list())

        # ── Spotify shortcuts ──────────────────────────────────────────────────
        import time as _tsp2
        _ssp2 = _tsp2.time()

        _spotify_kw = [
            "play", "pause", "skip", "next song", "previous song", "last song",
            "spotify", "music", "song", "track", "artist", "playlist",
            "volume up", "volume down", "shuffle", "repeat",
            "what's playing", "whats playing", "now playing", "currently playing",
            "queue", "add to queue",
        ]
        _is_spotify = any(kw in req_lower for kw in _spotify_kw)

        # Avoid collision with mac volume/open app shortcuts
        _is_mac_vol = bool(re.search(r'(?:set\s+)?(?:volume|vol)\s+(?:to\s+)?\d+', req_lower))
        _is_open    = req_lower.startswith("open ") or req_lower.startswith("launch ")

        if _is_spotify and not _is_mac_vol and not _is_open:

            # Now playing
            if any(kw in req_lower for kw in ["what's playing", "whats playing", "now playing", "currently playing", "what song"]):
                data = await self.spotify.get_now_playing()
                msg = self.spotify.format_now_playing(data)
                return JarvisResponse(success=data.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # Pause
            if any(kw in req_lower for kw in ["pause", "stop music", "stop playing", "stop spotify"]):
                result = await self.spotify.pause()
                msg = "Paused." if result.get("success") else result.get("error", "Could not pause.")
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # Skip / next
            if any(kw in req_lower for kw in ["skip", "next song", "next track", "next please"]):
                result = await self.spotify.skip()
                msg = "Skipped to next track." if result.get("success") else result.get("error", "Could not skip.")
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # Previous
            if any(kw in req_lower for kw in ["previous", "last song", "go back", "prev track"]):
                result = await self.spotify.previous()
                msg = "Going back to previous track." if result.get("success") else result.get("error", "Could not go back.")
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # Shuffle
            if "shuffle" in req_lower:
                state = "off" not in req_lower
                result = await self.spotify.shuffle(state)
                msg = f"Shuffle {'on' if state else 'off'}." if result.get("success") else result.get("error", "Could not set shuffle.")
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # Repeat
            if "repeat" in req_lower:
                if "off" in req_lower:
                    mode = "off"
                elif "track" in req_lower or "song" in req_lower:
                    mode = "track"
                else:
                    mode = "context"
                result = await self.spotify.repeat(mode)
                msg = f"Repeat set to {mode}." if result.get("success") else result.get("error", "Could not set repeat.")
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # Spotify volume
            _spvol = re.search(r'(?:spotify\s+)?volume\s+(?:to\s+)?(\d+)', req_lower)
            if _spvol:
                level = int(_spvol.group(1))
                result = await self.spotify.set_volume(level)
                msg = f"Spotify volume set to {level}%." if result.get("success") else result.get("error", "Could not set volume.")
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # My playlists
            if any(kw in req_lower for kw in ["my playlists", "show playlists", "list playlists"]):
                data = await self.spotify.get_playlists()
                if data.get("success"):
                    plists = data.get("playlists", [])
                    if plists:
                        lines = ["Your Spotify playlists:\n"]
                        for i, p in enumerate(plists, 1):
                            lines.append(f"{i}. {p['name']} ({p['tracks']} tracks)")
                        msg = "\n".join(lines)
                    else:
                        msg = "No playlists found."
                else:
                    msg = data.get("error", "Could not get playlists.")
                return JarvisResponse(success=data.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # Play / resume — with or without a song name
            _play_match = re.search(
                r'play\s+(?:some\s+)?(?:me\s+)?(?:the\s+)?(?:song\s+|track\s+|artist\s+|playlist\s+)?'
                r'["\']?(.+?)["\']?\s*(?:on spotify|by .+)?$',
                user_request, re.IGNORECASE
            )
            if "play" in req_lower:
                if _play_match:
                    query = _play_match.group(1).strip()
                    # Strip trailing "on spotify"
                    query = re.sub(r'\s+on spotify$', '', query, flags=re.IGNORECASE).strip()
                    if query and query.lower() not in ("spotify", "music", "something", "anything"):
                        result = await self.spotify.play_by_name(query)
                        msg = self.spotify.format_play_result(result)
                    else:
                        # Resume
                        result = await self.spotify.play()
                        msg = "Resumed playback." if result.get("success") else result.get("error", "Could not resume.")
                else:
                    # Just "play" or "resume"
                    result = await self.spotify.play()
                    msg = "Resumed playback." if result.get("success") else result.get("error", "Could not resume.")
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

        # Calendar shortcut — check schedule
        cal_read_keywords = ["what's on my calendar", "my schedule", "my meetings", "what do i have", "events today", "events this week"]
        if any(kw in req_lower for kw in cal_read_keywords):
            import time as _t3
            _s3 = _t3.time()
            result = await self.calendar.search_events()
            msg = result.get("message", "No events found.")
            asyncio.ensure_future(self.memory.store_task_result(user_request, "check_calendar", True, msg[:100]))
            return JarvisResponse(success=True, message=msg, latency_ms=(_t3.time()-_s3)*1000)

        # ── Mac Control shortcuts ──────────────────────────────────────────
        import re as _rem

        # Open app — guarded so we don't fire on "open the file README.md"
        # (that's routed to file_manager via the file-extension check below).
        _open_trigger = (
            req_lower.startswith("open ") or req_lower.startswith("launch ")
            or req_lower.startswith("start ") or req_lower.startswith("can you open ")
            or req_lower.startswith("could you open ") or req_lower.startswith("please open ")
        )
        # If the user is referring to a specific file ("open README.md",
        # "show me the contents of foo.txt"), DON'T treat this as app-open.
        # File extensions present in the request → file_manager handles it.
        # Alternation order matters — longer extensions before shorter
        # prefixes (json/jsx/tsx must come before js/ts).
        _file_ext_re = _rem.compile(
            r'\b[\w\-]+\.'
            r'(?:markdown|md|'
            r'json|jsx|js|tsx|ts|'
            r'yaml|yml|'
            r'pptx|ppt|xlsx|xls|docx|doc|'
            r'html|htm|css|'
            r'pdf|txt|csv|toml|cfg|ini|env|log|xml|rst|sh|zsh|py)\b',
            _rem.IGNORECASE,
        )
        _has_file_extension = bool(_file_ext_re.search(user_request))

        if _open_trigger and not _has_file_extension:
            import time as _to
            _so = _to.time()
            # Known app aliases — longest first so multi-word matches win.
            # Add liberally; if a real app exists with the alias name macOS
            # resolves it via Launch Services regardless.
            app_aliases = {
                # Browsers
                "google chrome": "Google Chrome", "chrome": "Google Chrome",
                "safari": "Safari", "firefox": "Firefox", "brave": "Brave Browser",
                "arc": "Arc", "edge": "Microsoft Edge", "opera": "Opera",
                # Code editors / IDEs
                "visual studio code": "Visual Studio Code", "vs code": "Visual Studio Code",
                "vscode": "Visual Studio Code", "code editor": "Visual Studio Code",
                "cursor": "Cursor", "windsurf": "Windsurf",
                "xcode": "Xcode", "intellij": "IntelliJ IDEA",
                "pycharm": "PyCharm", "webstorm": "WebStorm",
                "sublime text": "Sublime Text", "sublime": "Sublime Text",
                "android studio": "Android Studio",
                # Terminals
                "iterm": "iTerm", "iterm2": "iTerm",
                "warp": "Warp", "terminal": "Terminal", "ghostty": "Ghostty",
                # Productivity
                "notion": "Notion", "obsidian": "Obsidian", "bear": "Bear",
                "things": "Things3", "things 3": "Things3",
                "todoist": "Todoist", "fantastical": "Fantastical",
                "raycast": "Raycast", "alfred": "Alfred",
                "1password": "1Password", "bitwarden": "Bitwarden",
                # Communication
                "slack": "Slack", "discord": "Discord",
                "microsoft teams": "Microsoft Teams", "teams": "Microsoft Teams",
                "zoom": "zoom.us", "webex": "Webex",
                "whatsapp": "WhatsApp", "telegram": "Telegram",
                "signal": "Signal",
                # Apple apps
                "mail": "Mail", "calendar": "Calendar", "notes": "Notes",
                "reminders": "Reminders", "messages": "Messages",
                "facetime": "FaceTime", "photos": "Photos", "maps": "Maps",
                "music": "Music", "tv": "TV", "podcasts": "Podcasts",
                "finder": "Finder", "calculator": "Calculator",
                "preview": "Preview", "freeform": "Freeform",
                "system preferences": "System Preferences",
                "system settings": "System Settings",
                "activity monitor": "Activity Monitor",
                # Office
                "word": "Microsoft Word", "microsoft word": "Microsoft Word",
                "excel": "Microsoft Excel", "microsoft excel": "Microsoft Excel",
                "powerpoint": "Microsoft PowerPoint",
                "microsoft powerpoint": "Microsoft PowerPoint",
                "outlook": "Microsoft Outlook", "onenote": "Microsoft OneNote",
                # Creative
                "figma": "Figma", "sketch": "Sketch", "framer": "Framer",
                "photoshop": "Adobe Photoshop", "illustrator": "Adobe Illustrator",
                "indesign": "Adobe InDesign", "premiere": "Adobe Premiere Pro",
                "after effects": "Adobe After Effects",
                "blender": "Blender", "final cut": "Final Cut Pro",
                "logic": "Logic Pro", "garageband": "GarageBand",
                # Media
                "spotify": "Spotify", "vlc": "VLC", "iina": "IINA",
                "infuse": "Infuse",
                # Dev tools
                "postman": "Postman", "insomnia": "Insomnia",
                "tableplus": "TablePlus", "dbeaver": "DBeaver",
                "docker": "Docker", "docker desktop": "Docker",
                "github desktop": "GitHub Desktop",
                "sourcetree": "Sourcetree", "fork": "Fork",
                # Cloud / sync
                "dropbox": "Dropbox", "google drive": "Google Drive",
                "onedrive": "OneDrive",
                # Bookkeeping the original list
                "vs": "Visual Studio Code",   # ambiguous but VS Code is most common
                "code": "Visual Studio Code",
            }
            # Sort aliases by length descending so "visual studio code" wins
            # over "code" when both could match.
            sorted_aliases = sorted(app_aliases.items(), key=lambda kv: -len(kv[0]))

            app = None
            for alias, real_name in sorted_aliases:
                if alias in req_lower:
                    app = real_name
                    break
            # Fallback: extract everything after open/launch/start, up to a
            # natural stopping word (in/on/with/from/to/please). Allows
            # multi-word app names we haven't aliased explicitly.
            if not app:
                open_match = _rem.search(
                    r'(?:open|launch|start)\s+'
                    r'(?:the\s+|app\s+|application\s+)?'
                    r'([\w][\w\s\.\-]*?)'
                    r'(?:\s+(?:in|on|with|from|to|please|now|app|application)\b|\s*$)',
                    user_request, _rem.IGNORECASE,
                )
                if open_match:
                    raw_app = open_match.group(1).strip()
                    # Title-case for consistency with Launch Services. Don't
                    # title-case if it's clearly a known camel/lowercase id.
                    if not any(c.isupper() for c in raw_app):
                        raw_app = " ".join(w.capitalize() for w in raw_app.split())
                    app = raw_app
            if app:
                new_window = any(kw in req_lower for kw in ["new window", "new tab", "open new", "new session"])

                # VSCode new window — Cmd+Shift+N
                if app == "Visual Studio Code" and new_window:
                    script = (
                        "tell application \"Visual Studio Code\" to activate\n"
                        "delay 0.5\n"
                        "tell application \"System Events\"\n"
                        "    keystroke \"n\" using {command down, shift down}\n"
                        "end tell"
                    )
                    await self.mac._async_script(script)
                    return JarvisResponse(success=True, message="Opening new VSCode window.", latency_ms=(_to.time()-_so)*1000)

                result = await self.mac.open_app(app, new_window=new_window)
                action = "Opening new window in" if new_window else "Opening"
                msg = f"{action} {app}." if result.get("success") else f"Could not open {app}: {result.get('error')}"
                return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_to.time()-_so)*1000)

        # Set volume
        vol_match = _rem.search(r'(?:set\s+)?(?:volume|vol)\s+(?:to\s+)?(\d+)', req_lower)
        if vol_match:
            import time as _tv
            _sv = _tv.time()
            level = int(vol_match.group(1))
            result = await self.mac.set_volume(level)
            msg = f"Volume set to {level}." if result.get("success") else f"Could not set volume: {result.get('error')}"
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tv.time()-_sv)*1000)

        # Mute / unmute
        if any(kw in req_lower for kw in ["mute", "silence", "quiet"]) and "volume" not in req_lower:
            import time as _tm
            _sm = _tm.time()
            result = await self.mac.mute()
            return JarvisResponse(success=result.get("success", False), message="Muted.", latency_ms=(_tm.time()-_sm)*1000)

        if any(kw in req_lower for kw in ["unmute", "unsilence", "turn sound on"]):
            import time as _tum
            _sum = _tum.time()
            result = await self.mac.unmute()
            return JarvisResponse(success=result.get("success", False), message="Unmuted.", latency_ms=(_tum.time()-_sum)*1000)

        # Set brightness
        bright_match = _rem.search(r'(?:set\s+)?brightness\s+(?:to\s+)?(\d+)', req_lower)
        if bright_match:
            import time as _tb
            _sb = _tb.time()
            level = min(100, int(bright_match.group(1))) / 100.0
            result = await self.mac.set_brightness(level)
            msg = f"Brightness set to {int(level*100)}%." if result.get("success") else f"Could not set brightness: {result.get('error')}"
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tb.time()-_sb)*1000)

        # Battery
        if any(kw in req_lower for kw in ["battery", "battery level", "how much battery"]):
            import time as _tbat
            _sbat = _tbat.time()
            result = await self.mac.get_battery()
            if result.get("success"):
                pct = result.get("battery_pct", "unknown")
                msg = f"Battery is at {pct}%."
            else:
                msg = "Could not read battery level."
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tbat.time()-_sbat)*1000)

        # Lock screen — broaden trigger phrasing
        _lock_phrases = (
            "lock screen", "lock my screen", "lock the screen",
            "lock my mac", "lock the mac", "lock my computer",
            "lock the computer", "lock my laptop", "lock the laptop",
            "lock it", "lock me out",
        )
        if any(kw in req_lower for kw in _lock_phrases):
            import time as _tl
            _sl = _tl.time()
            result = await self.mac.lock_screen()
            msg = result.get("message") or ("Screen locked." if result.get("success") else result.get("error", "Could not lock the screen."))
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tl.time()-_sl)*1000)

        # Get clipboard
        if any(kw in req_lower for kw in ["clipboard", "what did i copy", "whats in my clipboard"]):
            import time as _tcb
            _scb = _tcb.time()
            result = await self.mac.get_clipboard()
            text = result.get("text", "")
            msg = f"Clipboard contains: {text[:200]}" if text else "Clipboard is empty."
            return JarvisResponse(success=True, message=msg, latency_ms=(_tcb.time()-_scb)*1000)

        # Send Mac notification — broaden phrasing.
        # Catches:
        #   "send me a notification: X"
        #   "notify me: X" / "notify me that X"
        #   "ping me with X" / "ping me: X"
        #   "show notification X" / "post notification X"
        #   "tell me X" (but only if a colon or "that" anchors the body —
        #     otherwise it's too greedy and would intercept normal Q&A)
        notif_match = _rem.search(
            r'(?:send\s+(?:me\s+)?(?:a\s+)?notification|'
            r'show\s+(?:a\s+|me\s+a\s+)?notification|'
            r'post\s+(?:a\s+)?notification|'
            r'give\s+me\s+(?:a\s+)?notification|'
            r'notify\s+me|ping\s+me)'
            r'(?:\s+(?:with|that|saying|to\s+say))?\s*[:\-]?\s*(.+)$',
            user_request, _rem.IGNORECASE,
        )
        if notif_match:
            import time as _tn
            _sn = _tn.time()
            message = notif_match.group(1).strip().strip('"').strip("'")
            # Allow "with title 'X'" subtitle override
            title = "Jarvis"
            title_match = _rem.search(r'(?:title|titled)\s+["\']([^"\']+)["\']', message, _rem.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()
                message = (message[:title_match.start()] + message[title_match.end():]).strip()
            if not message:
                return JarvisResponse(
                    success=False,
                    message='I need the text for the notification. Try: "notify me: take a break".',
                    latency_ms=(_tn.time()-_sn)*1000,
                )
            result = await self.mac.send_notification(message, title=title)
            msg = (
                f"Notification sent: \"{message}\"."
                if result.get("success")
                else f"Could not send notification: {result.get('error','unknown error')}"
            )
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tn.time()-_sn)*1000)

        # Quit / close app — tightened regex.
        # OLD: captured `[a-zA-Z][a-zA-Z0-9\s]+` which greedily consumed
        # trailing words ("quit chrome now" → "chrome now" → fail).
        # NEW: stop at a natural boundary (in/on/for/now/please/!.,?) so
        # we only get the app token(s) themselves.
        quit_match = _rem.search(
            r'(?:quit|close|kill|exit|terminate|force\s+quit)\s+'
            # Strip ALL combinations of "the"/"app"/"application" prefixes.
            # Stacking with * so "the app discord" / "the application X" all work.
            r'(?:(?:the|app|application)\s+)*'
            r'([\w][\w\s\.\-]*?)'
            r'(?:\s+(?:in|on|for|now|please|app|application)\b|[\.!?,]|\s*$)',
            user_request, _rem.IGNORECASE,
        )
        # Don't fire quit on file-manager intents like "close this file"
        # or "exit the document".
        _is_close_file = any(w in req_lower for w in (
            "close this file", "close the file", "close the document",
            "close that document", "exit the document",
        ))
        if quit_match and not _is_close_file:
            import time as _tq
            _sq = _tq.time()
            # Reuse the full open-app alias map for quit so any app you
            # can open by alias you can also quit by alias.
            app_aliases_q = {
                "chrome": "Google Chrome", "google chrome": "Google Chrome",
                "safari": "Safari", "firefox": "Firefox", "brave": "Brave Browser",
                "arc": "Arc", "edge": "Microsoft Edge",
                "spotify": "Spotify", "vlc": "VLC", "iina": "IINA",
                "vscode": "Visual Studio Code", "vs code": "Visual Studio Code",
                "code": "Visual Studio Code", "cursor": "Cursor",
                "xcode": "Xcode", "warp": "Warp", "iterm": "iTerm",
                "terminal": "Terminal", "ghostty": "Ghostty",
                "slack": "Slack", "discord": "Discord",
                "teams": "Microsoft Teams", "microsoft teams": "Microsoft Teams",
                "zoom": "zoom.us", "whatsapp": "WhatsApp", "telegram": "Telegram",
                "notion": "Notion", "obsidian": "Obsidian", "raycast": "Raycast",
                "figma": "Figma", "sketch": "Sketch",
                "postman": "Postman", "insomnia": "Insomnia",
                "tableplus": "TablePlus", "docker": "Docker",
                "github desktop": "GitHub Desktop",
                "mail": "Mail", "calendar": "Calendar", "notes": "Notes",
                "messages": "Messages", "facetime": "FaceTime",
                "photos": "Photos", "maps": "Maps", "music": "Music",
                "finder": "Finder", "calculator": "Calculator",
                "preview": "Preview",
                "word": "Microsoft Word", "excel": "Microsoft Excel",
                "powerpoint": "Microsoft PowerPoint", "outlook": "Microsoft Outlook",
                "1password": "1Password",
            }
            raw_app = quit_match.group(1).strip().lower()
            app_q = app_aliases_q.get(raw_app, quit_match.group(1).strip().title())
            result = await self.mac.quit_app(app_q)
            msg = f"Closed {app_q}." if result.get("success") else f"Could not close {app_q}: {result.get('error')}"
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tq.time()-_sq)*1000)

        # ── Information tool shortcuts ─────────────────────────────────────

        # ── Search / Research — router-driven, no keyword list needed ──────
        # If the router said websearch or research, we search.
        # Also catch common question patterns the router might miss.
        question_starters = (
            'what', 'who', 'when', 'where', 'why', 'how', 'which',
            'tell', 'explain', 'describe', 'define', 'search', 'find',
            'look', 'google', 'research', 'investigate', 'analyse',
            'analyze', 'give me', 'show me', 'can you find',
        )

        already_handled = any(kw in req_lower for kw in [
            'weather', 'battery', 'wifi', 'volume', 'brightness',
            'screenshot', 'dark mode', 'trash', 'news', 'headlines',
            'schedule', 'remind', 'open ', 'quit ', 'close ',
            'sleep the mac', 'lock screen', 'check my email',
            'send email', 'send an email', 'my calendar', 'my emails',
            'mute', 'unmute', 'clipboard', 'whats on my calendar',
        ])

        # Use primary_agent passed in to detect search intent
        primary_agent_val = primary_agent.value if primary_agent else ''

        is_search = (
            primary_agent_val in ('websearch', 'research') or
            (req_lower.split()[0] in question_starters if req_lower.split() else False)
        ) and not already_handled

        is_research = (
            primary_agent_val == 'research' or
            any(kw in req_lower for kw in [
                'research', 'deep dive', 'detailed', 'comprehensive',
                'everything about', 'investigate', 'analyse', 'analyze',
                'in depth', 'give me a full', 'full overview',
            ])
        ) and not already_handled

        if is_search or is_research:
            import time as _tws
            _sws = _tws.time()

            # Detect query complexity for adaptive length
            _elaborate_triggers = ["elaborate", "explain in detail", "tell me more",
                                   "more detail", "go into detail", "expand on",
                                   "comprehensive", "deep dive", "in depth", "full overview",
                                   "everything about", "give me a full"]
            _req_low = user_request.lower()
            is_elaborate = any(t in _req_low for t in _elaborate_triggers)

            # ── Resolve follow-up queries using conversation history ────────
            # If the message is a bare follow-up ("elaborate", "tell me more", etc.)
            # with no new topic, pull the previous user query as the actual topic.
            _history = conversation_history or []
            _is_pure_followup = (
                is_elaborate and
                len(user_request.split()) <= 5 and
                _history
            )
            if _is_pure_followup:
                # Find the last user message that was a real query (not a follow-up itself)
                prev_query = None
                for turn in reversed(_history):
                    if turn.get("role") == "user":
                        candidate = turn["content"]
                        if not any(t in candidate.lower() for t in _elaborate_triggers):
                            prev_query = candidate
                            break
                if prev_query:
                    user_request = prev_query  # expand on this topic
                    is_research = True  # force detailed response

            # Use the search tool query parser to clean the query
            query = self.websearch.parse_query(user_request)
            # Also strip research/investigate triggers for cleaner queries
            for trigger in ['research ', 'investigate ', 'analyse ', 'analyze ']:
                if query.lower().startswith(trigger):
                    query = query[len(trigger):].strip()
                    break
            if not query or len(query) < 2:
                query = user_request

            msg = "Could not find information about: " + query

            if is_elaborate or is_research:
                length_instruction = "- Give a thorough response of 200-350 words. Cover background, key facts, current state, and significance. Use numbered lists where appropriate."
            else:
                # Default for ALL web queries: one concise paragraph.
                # Only elaborate when the user explicitly asks for more detail.
                length_instruction = (
                    "- Answer in a single concise paragraph (3-5 sentences maximum). "
                    "Be direct and factual. Do NOT use bullet points, numbered lists, or multiple paragraphs. "
                    "If the user wants more detail they will ask you to elaborate."
                )

            if is_research:
                # Run all 3 searches IN PARALLEL instead of sequentially
                search_queries = [query, query + " explained", query + " overview"]
                search_results = await asyncio.gather(
                    *[self.websearch.search(q, max_results=3) for q in search_queries],
                    return_exceptions=True
                )
                all_results = []
                for data in search_results:
                    if isinstance(data, Exception): continue
                    if data.get("success") and data.get("results"):
                        for r in data.get("results", []):
                            if r not in all_results:
                                all_results.append(r)
                if len(all_results) < 3:
                    wiki_data = await self.websearch._wiki(query, 3)
                    if wiki_data.get("success"):
                        for r in wiki_data.get("results", []):
                            if r not in all_results:
                                all_results.append(r)
                if all_results:
                    snips = [r.get("snippet", "")[:300] for r in all_results[:8]]
                    combined2 = " ".join(snips)
                    rp = (
                        f"Answer this question: {query}\n\n"
                        f"Source material:\n{combined2}\n\n"
                        f"Instructions:\n"
                        f"{length_instruction}\n"
                        f"- Organise logically. Use numbered points (1. 2. 3.) for lists only if elaborating.\n"
                        f"- Write in clear prose. NO markdown asterisks (*) or hash (#).\n"
                        f"- Never introduce yourself or mention your name.\n"
                        f"- Never cite, mention, or list sources, URLs, or websites — just answer directly.\n"
                        f"- Do NOT add 'Would you like to know more?' or similar follow-up offers."
                    )
                    recent_hist = _history[-6:] if len(_history) > 6 else _history
                    report = await self.llm.chat(recent_hist + [{"role": "user", "content": rp}])
                    msg = report.strip()
                else:
                    msg = "Could not find enough information about: " + query
            else:
                # Single search
                data = await self.websearch.search(query, max_results=5)
                if data.get("success") and data.get("results"):
                    snips2 = [r.get("snippet", "")[:400] for r in data["results"][:5]]
                    ct = " ".join(snips2)
                    ap = (
                        f"Answer this question: {query}\n\n"
                        f"Source material:\n{ct}\n\n"
                        f"Instructions:\n"
                        f"{length_instruction}\n"
                        f"- Write in natural prose. NO markdown asterisks (*) or hash (#) symbols.\n"
                        f"- Never introduce yourself or mention your name.\n"
                        f"- Never cite, mention, or list sources, URLs, or websites — just answer directly.\n"
                        f"- Do NOT add 'Would you like to know more?' or similar follow-up offers."
                    )
                    recent_hist = _history[-6:] if len(_history) > 6 else _history
                    summary = await self.llm.chat(recent_hist + [{"role": "user", "content": ap}])
                    msg = summary.strip()
                else:
                    msg = "Could not find results for: " + query

            asyncio.ensure_future(self.memory.store_task_result(user_request, "web_search", True, msg[:100]))
            return JarvisResponse(success=True, message=msg, latency_ms=(_tws.time()-_sws)*1000)

        # Volume up/down
        vol_up = any(kw in req_lower for kw in ["volume up", "turn up", "louder", "increase volume", "raise volume"])
        vol_down = any(kw in req_lower for kw in ["volume down", "turn down", "quieter", "decrease volume", "lower volume"])
        if vol_up or vol_down:
            import time as _tvd
            _svd = _tvd.time()
            amount_match = _rem.search(r'(\d+)', req_lower)
            amount = int(amount_match.group(1)) if amount_match else 10
            result = await self.mac.adjust_volume("up" if vol_up else "down", amount)
            direction = "up" if vol_up else "down"
            msg = f"Volume turned {direction} to {result.get('volume', '?')}%."
            return JarvisResponse(success=True, message=msg, latency_ms=(_tvd.time()-_svd)*1000)

        # Screenshot
        if any(kw in req_lower for kw in ["screenshot", "take a screenshot", "capture screen", "screen capture"]):
            import time as _tss
            _sss = _tss.time()
            result = await self.mac.take_screenshot()
            return JarvisResponse(success=result.get("success", False), message=result.get("message", "Screenshot taken."), latency_ms=(_tss.time()-_sss)*1000)

        # Dark mode toggle
        if any(kw in req_lower for kw in ["dark mode", "light mode", "toggle dark", "toggle light", "switch to dark", "switch to light"]):
            import time as _tdm
            _sdm = _tdm.time()
            if "off" in req_lower or "light mode" in req_lower or "switch to light" in req_lower:
                # Force light mode
                result = await self.mac.get_dark_mode()
                if result.get("dark_mode"):
                    result = await self.mac.toggle_dark_mode()
                msg = "Switched to light mode."
            elif "on" in req_lower or "dark mode" in req_lower or "switch to dark" in req_lower:
                result = await self.mac.get_dark_mode()
                if not result.get("dark_mode"):
                    result = await self.mac.toggle_dark_mode()
                msg = "Switched to dark mode."
            else:
                result = await self.mac.toggle_dark_mode()
                msg = "Toggled dark/light mode."
            return JarvisResponse(success=True, message=msg, latency_ms=(_tdm.time()-_sdm)*1000)

        # System info
        if any(kw in req_lower for kw in ["system info", "disk space", "storage", "cpu usage", "ram usage", "memory usage", "how much storage"]):
            import time as _tsi
            _ssi = _tsi.time()
            result = await self.mac.get_system_info()
            msg = result.get("message", "Could not get system info.")
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tsi.time()-_ssi)*1000)

        # WiFi info
        if any(kw in req_lower for kw in ["wifi", "wi-fi", "network", "internet connection", "what network", "connected to"]):
            import time as _twifi
            _swifi = _twifi.time()
            result = await self.mac.get_wifi_info()
            msg = result.get("message", "Could not get WiFi info.")
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_twifi.time()-_swifi)*1000)

        # Empty trash
        if any(kw in req_lower for kw in ["empty trash", "clear trash", "delete trash"]):
            import time as _ttr
            _str = _ttr.time()
            result = await self.mac.empty_trash()
            msg = "Trash emptied." if result.get("success") else f"Could not empty trash: {result.get('error')}"
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_ttr.time()-_str)*1000)

        # Sleep Mac — narrow trigger to avoid "I'm sleepy", "go to sleep
        # mode", "asleep at the wheel" false-positives.
        # Require an explicit machine reference, OR a clear command verb.
        _sleep_command_phrases = (
            "sleep the mac", "sleep my mac", "sleep my computer",
            "sleep the computer", "sleep my laptop", "sleep the laptop",
            "put the mac to sleep", "put my mac to sleep",
            "put the computer to sleep", "put my computer to sleep",
            "send my mac to sleep", "send the mac to sleep",
            "shut the screen", "screen off",
        )
        _is_sleep_command = (
            any(p in req_lower for p in _sleep_command_phrases)
            and "reminder" not in req_lower
        )
        if _is_sleep_command:
            import time as _tslp
            _sslp = _tslp.time()
            result = await self.mac.sleep()
            msg = result.get("message") or ("Putting your Mac to sleep." if result.get("success") else result.get("error", "Could not put the Mac to sleep."))
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tslp.time()-_sslp)*1000)

        # ── Reminder shortcuts ────────────────────────────────────────────────
        # Must come BEFORE the calendar shortcuts because "remind me about
        # the 3pm meeting" contains "meeting" and would otherwise trigger
        # the calendar booking flow.
        import re as _rerem
        # List reminders
        if any(kw in req_lower for kw in [
            "my reminders", "list reminders", "list my reminders",
            "what reminders", "show reminders", "show my reminders",
            "pending reminders", "any reminders",
        ]):
            import time as _trml
            _srml = _trml.time()
            pending = self.reminders.list_pending()
            msg = self.reminders.format_list(pending)
            return JarvisResponse(success=True, message=msg, latency_ms=(_trml.time()-_srml)*1000)

        # Create reminder. "remind me to X" / "set a reminder to X" / "alert me to X"
        _remind_match = _rerem.search(
            r'\b(?:remind me to|remind me|set a reminder to|set a reminder for|'
            r'alert me to|alert me)\s+(.+)',
            user_request,
            _rerem.IGNORECASE,
        )
        if _remind_match:
            import time as _trsa
            _srsa = _trsa.time()
            tail = _remind_match.group(1).strip()

            # Two timing styles to handle:
            #   - "in N minutes/hours" → offset from now
            #   - "at 3pm", "tomorrow at 9", "on Monday at 10am" → absolute
            offset_minutes = None
            offset_match = _rerem.search(
                r'\bin\s+(\d+)\s*(minute|min|m|hour|hr|h)s?\b',
                tail, _rerem.IGNORECASE,
            )
            if offset_match:
                val = int(offset_match.group(1))
                unit = offset_match.group(2).lower()
                offset_minutes = val * 60 if unit.startswith(('h',)) else val
                # Title is everything before " in N ..."
                title = tail[:offset_match.start()].strip(" ,.;:")
                due_at = None
            else:
                # Try absolute time resolution
                temporal = self._resolve_temporal(tail)
                due_at_iso = temporal.get("datetime", "")
                if due_at_iso:
                    due_at = due_at_iso
                    # Strip the time/date words out of the title
                    title = tail
                    for _noise in [
                        r'\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b',
                        r'\btomorrow\b', r'\btonight\b', r'\btoday\b',
                        r'\bnext week\b',
                        r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                    ]:
                        title = _rerem.sub(_noise, '', title, flags=_rerem.IGNORECASE)
                    title = _rerem.sub(r'\s+', ' ', title).strip(" ,.;:")
                else:
                    # No time given — default to 5 minutes from now
                    offset_minutes = 5
                    title = tail.strip(" ,.;:")
                    due_at = None

            # Clean leading "to" / "that" from "remind me to call John"
            title = _rerem.sub(r'^(?:to|that)\s+', '', title, flags=_rerem.IGNORECASE)
            title = title.strip(" ,.;:") or "Reminder"

            rid = self.reminders.add(
                title=title,
                due_at=due_at,
                offset_minutes=offset_minutes,
            )

            # Build a human-readable confirmation
            if offset_minutes is not None:
                when = f"in {offset_minutes} minute{'s' if offset_minutes != 1 else ''}"
            else:
                try:
                    from datetime import datetime as _dt
                    when = _dt.fromisoformat(due_at).strftime("%-d %b at %-I:%M %p")
                except Exception:
                    when = due_at or "soon"
            msg = f"Reminder set — '{title}' {when}."
            asyncio.ensure_future(self.memory.store_task_result(
                user_request, "set_reminder", True, msg[:100]
            ))
            return JarvisResponse(success=True, message=msg, latency_ms=(_trsa.time()-_srsa)*1000)

        # ── Calendar shortcuts ────────────────────────────────────────────────
        import re as _recal

        # Pending meeting — user gave a new time after conflict
        if self._pending_meeting and self._pending_meeting.get("needs_new_time"):
            time_match = _recal.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', req_lower)
            if not time_match:
                time_match = _recal.search(r'at\s+(\d{1,2})(?::(\d{2}))?', req_lower)
            if time_match or any(kw in req_lower for kw in ["tomorrow", "monday","tuesday","wednesday","thursday","friday","saturday","sunday"]):
                import time as _tnt
                _snt = _tnt.time()

                # Preserve the original date if user only gave a new time
                original_start = self._pending_meeting.get("start_time", "")
                has_date_word = any(kw in req_lower for kw in [
                    "tomorrow", "today", "monday","tuesday","wednesday",
                    "thursday","friday","saturday","sunday","next week"
                ])

                if has_date_word:
                    # User gave a full new date+time — resolve normally
                    new_temporal = self._resolve_temporal(user_request)
                    new_start = new_temporal.get("datetime", "")
                else:
                    # User only gave a new time — keep original date, just change time
                    from datetime import datetime as _dtfix, timezone
                    from zoneinfo import ZoneInfo
                    local_tz = ZoneInfo("Europe/London")
                    orig_dt = _dtfix.fromisoformat(original_start)
                    # Extract new time from user request
                    new_temporal = self._resolve_temporal(user_request)
                    new_time_str = new_temporal.get("time", "")
                    if new_time_str:
                        h, m = map(int, new_time_str.split(":"))
                        new_dt = orig_dt.replace(hour=h, minute=m, second=0, microsecond=0)
                        new_start = new_dt.isoformat()
                    else:
                        new_start = new_temporal.get("datetime", "")
                if new_start:
                    from datetime import timedelta, datetime as _dtnt
                    duration_mins = self._pending_meeting.get("duration_mins", 60)
                    new_end = (_dtnt.fromisoformat(new_start) + timedelta(minutes=duration_mins)).isoformat()

                    # Check conflicts again
                    conflict2 = await self.calendar.check_conflicts(new_start, new_end)
                    if conflict2.get("has_conflict"):
                        ct = conflict2["conflicts"][0].get("title", "another event")
                        return JarvisResponse(success=False,
                            message=f"That time also conflicts with '{ct}'. What other time works?",
                            latency_ms=(_tnt.time()-_snt)*1000)

                    # Book it
                    result = await self.calendar.create_event(
                        title=self._pending_meeting["title"],
                        start_time=new_start,
                        end_time=new_end,
                        attendees=self._pending_meeting.get("attendees") or None,
                    )
                    self._pending_meeting = None
                    if result.get("success"):
                        start_fmt = new_start[:16].replace("T", " at ")
                        dur_str = f"{duration_mins} minutes" if duration_mins != 60 else "1 hour"
                        msg = f"Done! Rescheduled to {start_fmt} for {dur_str}."
                        if result.get("link"):
                            msg += " View: " + result["link"]
                    else:
                        msg = f"Could not create event: {result.get('error')}"
                    return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tnt.time()-_snt)*1000)

        # Schedule/create event — pending duration confirmation
        if self._pending_meeting and any(kw in req_lower for kw in
            ["minute", "hour", "min", "hr", "30", "45", "60", "90", "15", "yes", "confirm", "book it", "1 hour", "2 hour"]):
            import time as _tconf
            _sconf = _tconf.time()
            meeting = self._pending_meeting

            # Extract duration from reply — handle all common formats
            duration_mins = 60  # default 1 hour
            dur_match = _recal.search(r'(\d+)\s*(?:hours?|hrs?|minutes?|mins?|m)', req_lower)
            if dur_match:
                val = int(dur_match.group(1))
                is_hours = any(u in req_lower[dur_match.start():dur_match.end()+2] for u in ["hour", "hr"])
                duration_mins = val * 60 if is_hours else val
            elif "half" in req_lower:
                duration_mins = 30
            elif "quarter" in req_lower:
                duration_mins = 15
            elif "one hour" in req_lower or "an hour" in req_lower:
                duration_mins = 60
            elif "two hour" in req_lower:
                duration_mins = 120
            # Clamp to sensible range
            duration_mins = max(15, min(480, duration_mins))

            from datetime import timedelta
            from datetime import datetime as _dt
            start_time = meeting["start_time"]
            start_dt = _dt.fromisoformat(start_time)
            end_dt = start_dt + timedelta(minutes=duration_mins)
            end_time = end_dt.isoformat()

            # Check conflicts with actual duration
            conflict = await self.calendar.check_conflicts(start_time, end_time)
            if conflict.get("has_conflict"):
                conflict_title = conflict["conflicts"][0].get("title", "another event")
                # Keep pending meeting alive so user can pick new time
                self._pending_meeting["duration_mins"] = duration_mins
                self._pending_meeting["needs_new_time"] = True
                return JarvisResponse(
                    success=False,
                    message=f"Conflict detected — '{conflict_title}' is already at that time. What time would you like instead?",
                    latency_ms=(_tconf.time()-_sconf)*1000
                )

            # Create the event
            result = await self.calendar.create_event(
                title=meeting["title"],
                start_time=start_time,
                end_time=end_time,
                attendees=meeting.get("attendees") or None,
            )
            self._pending_meeting = None

            if result.get("success"):
                attendee_str = f" with {', '.join(meeting['attendees'])}" if meeting.get("attendees") else ""
                start_fmt = start_time[:16].replace("T", " at ")
                dur_str = f"{duration_mins} minutes" if duration_mins != 60 else "1 hour"
                msg = f"✅ '{meeting['title']}' scheduled{attendee_str} on {start_fmt} for {dur_str}."
                if result.get("link"):
                    msg += ' View: ' + result['link']
            else:
                msg = f"Could not create event: {result.get('error', 'unknown error')}"

            asyncio.ensure_future(self.memory.store_task_result(user_request, "schedule_meeting", result.get("success", False), msg[:100]))
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tconf.time()-_sconf)*1000)

        # Cancel pending meeting
        if self._pending_meeting and any(kw in req_lower for kw in ["cancel", "no", "don't book", "abort", "never mind"]):
            self._pending_meeting = None
            return JarvisResponse(success=True, message="Meeting cancelled. Nothing was added to your calendar.")

        # Schedule/create event shortcut
        # Use broader schedule detection — check individual trigger words + meeting context
        has_meeting_word = any(w in req_lower for w in ["meeting", "call", "appointment", "event", "session"])
        has_action_word = any(w in req_lower for w in [
            "schedule", "book", "create", "add", "set", "arrange", "block",
            "put", "plan", "organise", "organize", "new"
        ])
        explicit_schedule = any(kw in req_lower for kw in [
            "add to calendar", "calendar event", "add event",
        ])

        if (has_meeting_word and has_action_word) or explicit_schedule:
            import time as _tcal
            _scal = _tcal.time()

            # Extract title — look for "called X", "named X", "titled X", "call it X"
            # Extract title using clean patterns
            title = None
            req_lower_t = user_request.lower()

            # Pattern: call it X / called X / named X / titled X
            import re as _ret
            m1 = _ret.search(r"call(?:ed)?\s+it\s+(.+?)(?:\s+(?:for|on|at)\s+\d|\s*$)", user_request, _ret.IGNORECASE)
            m2 = _ret.search(r"(?:called|named|titled)\s+(.+?)(?:\s+(?:at|on)\s+\d|\s+(?:today|tomorrow|next|for)\s|\s*$)", user_request, _ret.IGNORECASE)
            m3 = _ret.search(r"about\s+([\w\s]+?)(?:\s+(?:at|on|for|today|tomorrow)|$)", user_request, _ret.IGNORECASE)

            if m1:
                title = m1.group(1).strip()
            elif m2:
                title = m2.group(1).strip()
            elif m3:
                title = m3.group(1).strip().title()
            else:
                title = "Meeting"

            # Strip time/date/duration noise from title
            noise = ["today","tomorrow","tonight","monday","tuesday","wednesday",
                     "thursday","friday","saturday","sunday","next week",
                     "9pm","8pm","7pm","6pm","5pm","4pm","3pm","2pm","1pm",
                     "12pm","11am","10am","9am","8am","am","pm",
                     # Duration phrases — would otherwise leak into the title
                     # for requests like "schedule a 30 minute meeting…"
                     "minute","minutes","mins","min","hour","hours","hrs","hr",
                     "half","quarter","an hour","one hour","two hours"]
            for n in noise:
                title = _ret.sub(r"\b" + n + r"\b", "", title, flags=_ret.IGNORECASE).strip()
            # Strip any standalone digit groups left behind by duration removal
            # (e.g. "30 meeting" → "meeting"), then collapse whitespace.
            title = _ret.sub(r"\b\d+\b", "", title)
            title = _ret.sub(r"\s+", " ", title).strip(". ").title() or "Meeting"

            # Extract attendees — "with John and Sarah", "with john@email.com"
            attendees = []
            attendee_match = _recal.search(
                r'\bwith\s+([\w\s,]+?)(?:\s+(?:on|at|for|today|tomorrow|next|about|at\s+\d)|$)',
                user_request, _recal.IGNORECASE
            )
            if attendee_match:
                names = attendee_match.group(1).strip()
                for name in _recal.split(r'[,\s]+(?:and\s+)?', names):
                    name = name.strip()
                    if name:
                        contact = self.contacts.find(name)
                        if contact:
                            attendees.append(contact["email"])

            # Resolve time
            temporal = self._resolve_temporal(user_request)
            start_time = temporal.get("datetime", "")

            if not start_time:
                return JarvisResponse(success=False, message="I couldn't figure out when to schedule the meeting. Could you specify a date and time?", latency_ms=(_tcal.time()-_scal)*1000)

            # Try to extract duration from the original request so we can
            # one-shot the booking when the user already said how long.
            # Examples we want to catch:
            #   "schedule a 30 minute meeting..."
            #   "book a 1 hour call..."
            #   "set up a half-hour sync..."
            initial_duration_mins = self._extract_duration_minutes(user_request)

            # Store pending meeting state — needed whether we ask for duration
            # or skip straight to booking.
            self._pending_meeting = {
                "title": title,
                "start_time": start_time,
                "attendees": attendees,
            }

            start_fmt = start_time[:16].replace("T", " at ")
            attendee_str = f" with {', '.join(attendees)}" if attendees else ""

            if initial_duration_mins is None:
                # No duration given — fall back to the two-step flow.
                msg = (
                    f"I will schedule {title!r}{attendee_str} on {start_fmt}. "
                    "How long should the meeting be? (e.g. 30 minutes, 1 hour, 45 minutes)"
                )
                return JarvisResponse(success=True, message=msg, latency_ms=(_tcal.time()-_scal)*1000)

            # ── One-shot booking path ───────────────────────────────────────
            # User gave us everything (title, time, duration). Check for
            # conflicts and create the event in a single turn.
            from datetime import timedelta as _td, datetime as _dt
            duration_mins = max(15, min(480, initial_duration_mins))
            start_dt = _dt.fromisoformat(start_time)
            end_dt = start_dt + _td(minutes=duration_mins)
            end_time = end_dt.isoformat()

            conflict = await self.calendar.check_conflicts(start_time, end_time)
            if conflict.get("has_conflict"):
                conflict_title = conflict["conflicts"][0].get("title", "another event")
                # Keep the pending meeting so the next turn can pick a new time
                self._pending_meeting["duration_mins"] = duration_mins
                self._pending_meeting["needs_new_time"] = True
                return JarvisResponse(
                    success=False,
                    message=(
                        f"Conflict — '{conflict_title}' is already at that time. "
                        "What other time works?"
                    ),
                    latency_ms=(_tcal.time()-_scal)*1000,
                )

            result = await self.calendar.create_event(
                title=title,
                start_time=start_time,
                end_time=end_time,
                attendees=attendees or None,
            )
            self._pending_meeting = None

            if result.get("success"):
                dur_str = f"{duration_mins} minutes" if duration_mins != 60 else "1 hour"
                msg = f"✅ '{title}' scheduled{attendee_str} on {start_fmt} for {dur_str}."
                if result.get("link"):
                    msg += " View: " + result["link"]
            else:
                msg = f"Could not create event: {result.get('error', 'unknown error')}"

            asyncio.ensure_future(self.memory.store_task_result(
                user_request, "schedule_meeting",
                result.get("success", False), msg[:100],
            ))
            return JarvisResponse(
                success=result.get("success", False),
                message=msg,
                latency_ms=(_tcal.time()-_scal)*1000,
            )

        return None  # No shortcut — proceed with full pipeline

    async def _morning_briefing(self, voice_mode: bool = False) -> str:
        """
        Morning brief — fresh implementation (rewritten 29 May 2026).

        TEXT MODE  → bold-headed, ~20-30 line digest. Sections, in order:
                     Weather, Prayer Times, Your Day (calendar), Inbox,
                     Top News, Sports, Markets.
        VOICE MODE → 2-3 sentence spoken summary, no markdown, no lists.

        All data is fetched live via asyncio.gather (~3-5s wall-clock).
        Every section is independently optional — a failed API call just
        drops that section, the rest still ships.

        Routing: intercepted at the TOP of handle()/handle_stream() before
        the router runs, so the LLM never sees the request and can't
        hallucinate a fake briefing.
        """
        import random
        now = datetime.now()
        hour = now.hour
        date_str = now.strftime("%A, %d %B %Y")

        # ── Greeting ───────────────────────────────────────────────────────
        # Time-of-day aware. Voice mode keeps it simple; text mode allows a
        # touch of personality ("champ"/"legend"/"boss").
        if hour < 12:
            time_phrase = "Good morning"
        elif hour < 17:
            time_phrase = "Good afternoon"
        else:
            time_phrase = "Good evening"

        # ── Profile-driven personalisation ─────────────────────────────────
        # Which sections to render (and in concept, order) comes from the
        # user profile. `_on(name)` gates each section; with no profile, or
        # the default "all", every section shows.
        _prof = getattr(self, "profile", None)
        _secs = set(_prof.enabled_sections()) if _prof else set()
        def _on(name: str) -> bool:
            return (not _secs) or (name in _secs)
        _pref_name = _prof.preferred_name if _prof else "Abdullah"

        # ── Parallel fetch everything ──────────────────────────────────────
        from config.settings import FAVOURITE_TEAMS, FAVOURITE_FOOTBALL_LEAGUE, FAVOURITE_BASKETBALL_LEAGUE
        _fav_teams = (_prof.favourite_teams if _prof and _prof.favourite_teams else FAVOURITE_TEAMS)

        # All six tools run concurrently. return_exceptions=True means a
        # single failed fetch (e.g. Gmail OAuth expired) doesn't crash the
        # brief — that section just renders empty.
        (weather_data, prayer_data, cal_data, email_data,
         news_data, sports_pl, sports_rm, sports_nba,
         sports_cricket, market_data) = await asyncio.gather(
            self.weather.get_current(),
            self.prayer.get_times(),
            self.calendar.search_events(),
            self.gmail.get_inbox(max_results=5),
            self.news.get_headlines(max_stories=5),
            self.sports.get_scores(FAVOURITE_FOOTBALL_LEAGUE, limit=10),
            self.sports.search_team("Real Madrid", "la_liga"),
            self.sports.get_scores(FAVOURITE_BASKETBALL_LEAGUE, limit=8),
            self.sports.get_scores("cricket_psl", limit=4),
            self.markets.get_all(),
            return_exceptions=True,
        )

        # ════════════════════════════════════════════════════════════════
        # VOICE MODE — short spoken summary (2-3 sentences, no markdown)
        # ════════════════════════════════════════════════════════════════
        if voice_mode:
            bits: list[str] = []

            # Sentence 1: greeting + weather
            if isinstance(weather_data, dict) and weather_data.get("success"):
                w = weather_data
                bits.append(
                    f"{time_phrase}. It's {w.get('temperature_c','')} degrees "
                    f"and {str(w.get('condition','')).lower()} in {w.get('location','High Wycombe')}."
                )
            else:
                bits.append(f"{time_phrase}.")

            # Sentence 2: schedule + inbox
            ev_count = 0
            if isinstance(cal_data, dict) and cal_data.get("success"):
                ev_count = len(cal_data.get("events", []))
            unread = 0
            if isinstance(email_data, dict) and email_data.get("success"):
                unread = email_data.get("count", 0) or 0
            schedule_bit = (
                f"You have {ev_count} event{'s' if ev_count != 1 else ''} today"
                if ev_count else "Nothing on the calendar today"
            )
            inbox_bit = (
                f"and {unread} unread email{'s' if unread != 1 else ''}"
                if unread else "and a clean inbox"
            )
            bits.append(f"{schedule_bit} {inbox_bit}.")

            # Sentence 3: one news headline + market mood
            top_headline = ""
            if isinstance(news_data, dict) and news_data.get("success"):
                stories = news_data.get("stories", [])
                if stories:
                    top_headline = str(stories[0].get("title", "")).strip()
                    # Strip trailing punctuation so we can chain it
                    if top_headline.endswith("."):
                        top_headline = top_headline[:-1]
            market_bit = ""
            if isinstance(market_data, dict) and market_data.get("success"):
                try:
                    tickers = market_data.get("tickers") or market_data.get("data") or []
                    ups = sum(1 for t in tickers if isinstance(t, dict) and (t.get("change_pct") or 0) > 0)
                    downs = sum(1 for t in tickers if isinstance(t, dict) and (t.get("change_pct") or 0) < 0)
                    if ups + downs > 0:
                        market_bit = "markets are mostly up" if ups > downs else "markets are mostly down" if downs > ups else "markets are mixed"
                except Exception:
                    pass

            if top_headline and market_bit:
                bits.append(f"Top story: {top_headline}. And {market_bit}.")
            elif top_headline:
                bits.append(f"Top story: {top_headline}.")
            elif market_bit:
                bits.append(f"On the markets, {market_bit}.")

            return " ".join(bits)

        # ════════════════════════════════════════════════════════════════
        # TEXT MODE — bold-headed, 20-30 line digest
        # ════════════════════════════════════════════════════════════════
        lines: list[str] = [
            f"{time_phrase}, {_pref_name}. Here's your brief for {date_str}.",
            "",
        ]

        # ── Weather ────────────────────────────────────────────────────────
        if _on("weather") and isinstance(weather_data, dict) and weather_data.get("success"):
            w = weather_data
            cond = w.get("condition", "—")
            temp = w.get("temperature_c", "—")
            feels = w.get("feels_like_c", "—")
            loc = w.get("location", "High Wycombe")
            hum = w.get("humidity_pct", "—")
            wind = w.get("wind_kph", "—")
            lines.append("**Weather**")
            lines.append(f"  {cond}, {temp}°C (feels {feels}°C) in {loc}.")
            lines.append(f"  Humidity {hum}%, wind {wind} km/h.")
            lines.append("")

        # ── Prayer Times ───────────────────────────────────────────────────
        if _on("prayer") and isinstance(prayer_data, dict) and prayer_data.get("success"):
            lines.append("**Prayer Times**")
            ptimes = prayer_data.get("times") or {}
            # Compact one-liner: Fajr 02:57 · Zuhr 13:00 · Asr 17:19 · Maghrib 21:08 · Isha 23:04
            wanted = [("Fajr", "fajr"), ("Zuhr", "zuhr"),
                      ("Asr", "asr"), ("Maghrib", "maghrib"), ("Isha", "isha")]
            row = []
            for label, key in wanted:
                t = ptimes.get(key)
                if t:
                    row.append(f"{label} {t}")
            if row:
                lines.append("  " + " · ".join(row))
            try:
                _next = self.prayer.get_next_prayer(prayer_data)
                if _next:
                    lines.append(f"  Next: {_next}")
            except Exception:
                pass
            lines.append("")

        # ── Your Day (calendar) ───────────────────────────────────────────
        if _on("calendar") and isinstance(cal_data, dict) and cal_data.get("success"):
            events = cal_data.get("events", []) or []
            lines.append("**Your Day**")
            if events:
                lines.append(f"  {len(events)} event{'s' if len(events) != 1 else ''} scheduled:")
                for e in events[:5]:
                    title = e.get("title", "Untitled")
                    start_raw = (e.get("start") or "")[:16]
                    start_fmt = start_raw.replace("T", " at ") if start_raw else "—"
                    lines.append(f"  • {title} — {start_fmt}")
            else:
                lines.append("  Nothing scheduled — a free day.")
            lines.append("")
        elif _on("calendar") and (not isinstance(cal_data, dict) or cal_data.get("error")):
            # Calendar is disconnected (mock mode). Surface it honestly
            # rather than silently dropping the section.
            lines.append("**Your Day**")
            lines.append("  Calendar offline — connect Google Calendar to see events.")
            lines.append("")

        # ── Inbox ──────────────────────────────────────────────────────────
        if _on("email") and isinstance(email_data, dict) and email_data.get("success"):
            emails = email_data.get("emails", []) or []
            count = email_data.get("count", 0) or 0
            lines.append("**Inbox**")
            lines.append(f"  {count} unread email{'s' if count != 1 else ''}.")
            for em in emails[:3]:
                subj = (em.get("subject") or "(no subject)").strip()
                sender = (em.get("from") or "").split("<")[0].strip()[:40]
                if len(subj) > 65:
                    subj = subj[:62] + "…"
                lines.append(f"  • {subj} — {sender}")
            lines.append("")
        elif _on("email") and (not isinstance(email_data, dict) or email_data.get("error")):
            lines.append("**Inbox**")
            lines.append("  Gmail offline — connect Google to see your inbox.")
            lines.append("")

        # ── Top News (5 stories with one-line description) ────────────────
        if _on("news") and isinstance(news_data, dict) and news_data.get("success"):
            stories = news_data.get("stories", []) or []
            if stories:
                lines.append("**Top News**")
                for i, story in enumerate(stories[:5], 1):
                    title = (story.get("title") or "—").strip()
                    sources = story.get("sources") or []
                    src = sources[0] if sources else ""
                    src_tag = f" [{src}]" if src else ""
                    lines.append(f"  {i}. {title}{src_tag}")
                    desc = (story.get("description") or "").strip()
                    if desc:
                        if len(desc) > 140:
                            desc = desc[:137] + "…"
                        lines.append(f"     {desc}")
                lines.append("")

        # ── Sports ─────────────────────────────────────────────────────────
        sports_block: list[str] = []
        fav_lower = [t.lower() for t in _fav_teams]

        def _is_fav(g: dict) -> bool:
            home = (g.get("home_team", "") or "").lower()
            away = (g.get("away_team", "") or "").lower()
            return any(
                any(w in home or w in away for w in fav.split())
                for fav in fav_lower
            )

        def _add_league(label: str, data, max_games: int = 2):
            if not isinstance(data, dict) or not data.get("success"):
                return
            games = data.get("games", []) or []
            live = [g for g in games if g.get("status") == "live"]
            done = [g for g in games if g.get("status") == "final"]
            picks: list[dict] = []
            # Priority: live games > favourite-team finished games > most recent
            for g in live[:2]:
                picks.append(g)
            for g in done:
                if len(picks) >= max_games:
                    break
                if _is_fav(g) and g not in picks:
                    picks.append(g)
            for g in done:
                if len(picks) >= max_games:
                    break
                if g not in picks:
                    picks.append(g)
            for g in picks[:max_games]:
                star = " ★" if _is_fav(g) else ""
                clock = f" ({g.get('clock','live')})" if g.get("status") == "live" else ""
                ht = g.get("home_team", "?")
                at = g.get("away_team", "?")
                hs = g.get("home_score", "")
                as_ = g.get("away_score", "")
                sports_block.append(
                    f"  {label}: {ht} {hs} - {as_} {at}{clock}{star}"
                )

        if _on("sports"):
            _add_league("PL", sports_pl)
            _add_league("La Liga", sports_rm, max_games=1)
            _add_league("NBA", sports_nba)
            _add_league("Cricket", sports_cricket, max_games=1)

        if sports_block:
            lines.append("**Sports**")
            lines.extend(sports_block)
            lines.append("")

        # ── Markets ────────────────────────────────────────────────────────
        if _on("markets") and isinstance(market_data, dict) and market_data.get("success"):
            tickers = market_data.get("tickers") or market_data.get("data") or []
            if tickers:
                lines.append("**Markets**")
                for t in tickers[:8]:
                    if not isinstance(t, dict):
                        continue
                    sym = t.get("symbol") or t.get("name") or "—"
                    price = t.get("price")
                    chg = t.get("change_pct")
                    arrow = "↗" if (chg is not None and chg > 0) else ("↘" if (chg is not None and chg < 0) else "·")
                    price_str = f"${price:,.2f}" if isinstance(price, (int, float)) else (str(price) if price else "—")
                    chg_str = f"{chg:+.2f}%" if isinstance(chg, (int, float)) else ""
                    lines.append(f"  {sym:<8} {arrow} {price_str}  {chg_str}".rstrip())
                lines.append("")

        # ── Sign-off (deterministic — no LLM) ──────────────────────────────
        if _on("closing"):
            closings = [
                "Make it count today.",
                "Have a good one.",
                "You've got this.",
                "Go well today.",
            ]
            lines.append(random.choice(closings))

        lines_out = lines

        return chr(10).join(lines_out)

    # ──────────────────────────────────────────────────────────────────────
    #  Multi-ACTION decomposition & execution
    # ──────────────────────────────────────────────────────────────────────

    async def _split_into_actions(self, user_request: str) -> List[str]:
        """
        Decompose a compound request into an ordered list of atomic commands.

        Primary path: a fast LLM call that returns a JSON list of standalone
        commands. Fallback: the deterministic rule-based splitter in
        BriefingHandler. Returns a list with <2 items when the request is a
        single action (the caller then lets it flow through the normal path).
        """
        rule_segments = self.briefing.split_compound(user_request)

        system = (
            "You split a user's request into separate standalone commands. "
            "Return ONLY JSON: {\"actions\": [\"...\", \"...\"]}.\n"
            "Rules:\n"
            "- Each action must be a complete, self-contained instruction that "
            "could be run on its own (carry over any shared context like times "
            "or names into each).\n"
            "- Preserve the original order.\n"
            "- Do NOT split a single instruction that merely contains the word "
            "'and' (e.g. 'salt and pepper', 'Pakistan and Australia', "
            "'pros and cons').\n"
            "- If the request is really just ONE command or one question, "
            "return it as a single-element list."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f'Request: "{user_request}"'},
        ]
        try:
            data = await self.llm.chat_json(
                messages, model=OLLAMA_ROUTER_MODEL, max_tokens=200
            )
            actions = data.get("actions") if isinstance(data, dict) else None
            if isinstance(actions, list):
                cleaned = [a.strip() for a in actions
                           if isinstance(a, str) and a.strip()]
                if len(cleaned) >= 2:
                    return cleaned[:5]
        except Exception as exc:
            print(f"⚠️  Multi-action LLM split failed: {exc} — using rule split")

        return rule_segments[:5] if len(rule_segments) >= 2 else []

    async def _handle_multi_action(
        self,
        user_request: str,
        *,
        voice_mode: bool = False,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Optional[str]:
        """
        Split a compound action request and run each sub-command through the
        normal routing + shortcut path, then aggregate the replies.

        Returns the aggregated message, or None if the request didn't actually
        decompose into 2+ actions (so the caller falls back to normal flow).
        """
        segments = await self._split_into_actions(user_request)
        if len(segments) < 2:
            return None

        print(f"🧩 Multi-action: {len(segments)} commands → {segments}")

        results: List[tuple] = []
        for seg in segments:
            try:
                decision = await self.router.route(seg)
                resp = await self._try_shortcut(
                    decision.primary_agent,
                    seg,
                    voice_mode=voice_mode,
                    conversation_history=conversation_history,
                )
                if resp is None:
                    resp = await self._mini_answer(seg, decision.primary_agent)
            except Exception as exc:
                resp = JarvisResponse(
                    success=False, message=f"Couldn't complete this: {exc}"
                )
            results.append((seg, resp))

        return self._format_multi_action(results)

    async def _mini_answer(self, segment: str, agent: AgentRole) -> JarvisResponse:
        """
        Produce a short answer for a multi-action sub-command that has no
        tier-1 shortcut (e.g. an embedded question). Kept deliberately compact
        — one tool lookup (if relevant) plus one short LLM call.
        """
        context_str = ""
        try:
            if agent == AgentRole.WEBSEARCH:
                data = await self.websearch.search(segment)
                context_str = self.websearch.format_results(data)
            elif agent == AgentRole.NEWS:
                data = await self.news.get_headlines(query=segment, max_items=5)
                return JarvisResponse(success=True, message=self.news.format_headlines(data))
        except Exception:
            pass

        user_content = (
            f"Answer this concisely in 1-2 sentences: {segment}"
            + (f"\n\nSource material:\n{context_str}" if context_str else "")
        )
        try:
            ans = await self.llm.chat(
                [{"role": "user", "content": user_content}], max_tokens=160
            )
            return JarvisResponse(success=True, message=ans.strip() or "Done.")
        except Exception as exc:
            return JarvisResponse(success=False, message=f"Couldn't answer that: {exc}")

    def _format_multi_action(self, results: List[tuple]) -> str:
        """Aggregate per-command results into one readable reply."""
        lines: List[str] = []
        for seg, resp in results:
            ok = getattr(resp, "success", False)
            mark = "✅" if ok else "⚠️"
            label = seg[0].upper() + seg[1:] if seg else seg
            msg = (getattr(resp, "message", "") or "").strip()
            lines.append(f"{mark} {label}")
            if msg:
                # Indent the body so multi-line results (e.g. an email draft)
                # stay visually grouped under their command.
                lines.append("\n".join(f"   {ln}" for ln in msg.splitlines()))
            lines.append("")
        return "\n".join(lines).strip()

    # ──────────────────────────────────────────────────────────────────────
    #  Long-term memory: remember / forget / recall
    # ──────────────────────────────────────────────────────────────────────

    async def _try_memory_command(self, user_request: str):
        """
        Deterministic intercept for explicit memory commands:
          • "remember that I prefer short replies"  → store a SEMANTIC fact
          • "forget what I said about X"             → delete matching facts
          • "what do you know about me"              → list stored facts

        Returns a JarvisResponse if handled, else None. Runs before the router
        so these never get misrouted to web search.
        """
        import time as _tm
        q = user_request.strip()
        ql = q.lower()

        # ── Recall: "what do you know/remember about me" ────────────────────
        if re.search(r"\b(what do you (know|remember)( about me)?|"
                     r"what have you remembered|what do you know about me|"
                     r"show (me )?(my )?(memories|what you know about me))\b", ql):
            _s = _tm.time()
            facts = await self.memory.recall_facts("everything about the user", k=20)
            if not facts:
                msg = ("I haven't been told anything to remember yet. Say "
                       "\"remember that …\" and I'll keep it in mind.")
            else:
                lines = "\n".join(f"  • {m.content}" for m in facts)
                msg = f"Here's what I'm keeping in mind about you:\n{lines}"
            return JarvisResponse(success=True, message=msg, latency_ms=(_tm.time()-_s)*1000)

        # ── Forget ──────────────────────────────────────────────────────────
        m = re.match(r"^\s*(?:please\s+)?forget\b(?:\s+(?:that|about|what i (?:said|told you) about))?\s*(.+)$",
                     q, re.IGNORECASE)
        if m:
            target = m.group(1).strip(" .!?")
            if len(target) >= 2:
                _s = _tm.time()
                n = await self.memory.forget(target)
                msg = (f"Done — I've forgotten {('that' if n==1 else f'{n} things')} "
                       f"about \"{target}\"." if n else
                       f"I didn't have anything remembered about \"{target}\".")
                return JarvisResponse(success=True, message=msg, latency_ms=(_tm.time()-_s)*1000)

        # ── Remember ─────────────────────────────────────────────────────────
        m = re.match(r"^\s*(?:please\s+)?(?:remember|note|keep in mind|don'?t forget)\b"
                     r"\s*(?:that\s+|:\s*)?(.+)$", q, re.IGNORECASE)
        if m:
            fact = m.group(1).strip()
            # "remember TO …" is a reminder, not a fact — let it route normally.
            if fact.lower().startswith("to ") or len(fact) < 3:
                return None
            _s = _tm.time()
            await self.memory.remember(fact)
            return JarvisResponse(
                success=True,
                message=f"Noted — I'll remember that {fact.rstrip('.')}.",
                latency_ms=(_tm.time()-_s)*1000,
            )
        return None

    async def _recall_block(self, query: str) -> str:
        """Return a short bullet block of remembered facts relevant to `query`,
        or "" if none. Best-effort — never raises into the request path."""
        try:
            facts = await self.memory.recall_facts(query, k=4)
        except Exception:
            return ""
        if not facts:
            return ""
        return "\n".join(f"- {m.content}" for m in facts)

    async def _handle_multi_query(self, user_request: str, intents: list) -> str:
        """
        Handle compound queries by running multiple intents in parallel.
        e.g. "what's the weather and latest news and premier league scores?"
        """
        tasks = {}

        if "weather" in intents:
            location = self._extract_location(user_request)
            if location:
                tasks["weather"] = self.weather.get_current_for_location(location)
            else:
                tasks["weather"] = self.weather.get_current()

        if "news" in intents:
            tasks["news"] = self.news.get_headlines(query=user_request, max_stories=4)

        if "sports" in intents:
            league_key = self.sports.detect_league(user_request) or "premier_league"
            tasks["sports"] = self.sports.get_scores(league_key)

        if "calendar" in intents:
            tasks["calendar"] = self.calendar.search_events()

        if "email" in intents:
            tasks["email"] = self.gmail.get_inbox(max_results=5)

        if not tasks:
            return ""

        # Execute all in parallel
        keys = list(tasks.keys())
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        result_map = dict(zip(keys, results))

        sections = []

        if "weather" in result_map:
            data = result_map["weather"]
            if isinstance(data, dict) and data.get("success"):
                sections.append("WEATHER")
                sections.append("  " + self.weather.format_current(data))
                sections.append("")

        if "calendar" in result_map:
            data = result_map["calendar"]
            if isinstance(data, dict) and data.get("success"):
                events = data.get("events", [])
                sections.append("CALENDAR")
                if events:
                    for e in events[:4]:
                        start = e.get("start", "")[:16].replace("T", " at ")
                        sections.append(f"  • {e.get('title','')} — {start}")
                else:
                    sections.append("  No upcoming events.")
                sections.append("")

        if "email" in result_map:
            data = result_map["email"]
            if isinstance(data, dict) and data.get("success"):
                count = data.get("count", 0)
                sections.append("EMAILS")
                sections.append(f"  {count} unread email(s).")
                sections.append("")

        if "news" in result_map:
            data = result_map["news"]
            if isinstance(data, dict) and data.get("success"):
                stories = data.get("stories", [])
                sections.append("NEWS")
                for i, story in enumerate(stories[:4], 1):
                    sources = story.get("sources", [])
                    src_str = f" [{', '.join(sources[:2])}]" if len(sources) > 1 else ""
                    sections.append(f"  {i}. {story['title']}{src_str}")
                sections.append("")

        if "sports" in result_map:
            data = result_map["sports"]
            if isinstance(data, dict) and data.get("success"):
                sections.append("SPORTS")
                sections.append(self.sports.format_scores(data))
                sections.append("")

        return chr(10).join(sections).strip()


    def _time_greeting(self, now: datetime) -> str:
        hour = now.hour
        if hour < 12:
            return "Good morning"
        elif hour < 17:
            return "Good afternoon"
        else:
            return "Good evening"

    def _extract_location(self, user_request: str) -> str:
        """
        Extract a city name from a weather request.
        Returns empty string if no specific location found.
        Strips noise words like 'today', 'now', 'currently', country names.
        """
        import re as _reloc
        req = user_request.lower()

        # Noise words to strip from end of location
        noise_words = {
            "today", "now", "currently", "tonight", "tomorrow",
            "this", "week", "weekend", "morning", "evening",
            "afternoon", "night", "please", "for", "me",
        }

        # Country names to strip
        common_countries = {
            "pakistan","india","usa","uk","france","germany","china",
            "japan","australia","canada","brazil","italy","spain",
            "mexico","russia","nigeria","egypt","turkey","argentina",
            "bangladesh","indonesia","kenya","ghana","iran","iraq",
            "vietnam","thailand","malaysia","singapore","uae","qatar",
            "england","scotland","wales","ireland","netherlands","sweden",
            "norway","denmark","finland","switzerland","austria","belgium",
            "portugal","greece","poland","czech","romania","hungary",
            "southafrica","newzealand","saudiarabia",
        }

        for phrase in [
            "what's the weather in", "what is the weather in",
            "whats the weather in", "weather in", "weather for",
            "weather at", "weather today in", "weather forecast for",
            "forecast for", "forecast in", "temperature in",
            "how hot is it in", "how cold is it in", "how warm is it in",
            "what's it like in", "whats it like in",
        ]:
            if phrase in req:
                raw = user_request[req.index(phrase) + len(phrase):].strip()
                raw = raw.rstrip("?!., ")
                if not raw:
                    return ""

                # Split into words and strip noise/country from the end
                parts = raw.split()
                while parts and parts[-1].lower() in noise_words:
                    parts.pop()
                while parts and parts[-1].lower() in common_countries:
                    parts.pop()

                city = " ".join(parts).strip()
                return city if city else ""

        return ""