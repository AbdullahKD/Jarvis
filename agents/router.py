"""
Router Agent
Classifies every incoming user request and decides which specialist
agents should handle it. This is the entry point of every interaction.
"""

from __future__ import annotations

import re
from typing import List, Optional

from config.llm_client import OllamaClient
from config.models import AgentRole, RouterDecision
from config.settings import OLLAMA_ROUTER_MODEL


# All intents the router recognises
INTENT_MAP = {
    # Calendar
    "schedule_meeting":    AgentRole.CALENDAR,
    "check_calendar":      AgentRole.CALENDAR,
    "cancel_meeting":      AgentRole.CALENDAR,
    "reschedule_meeting":  AgentRole.CALENDAR,
    # Email
    "send_email":          AgentRole.EMAIL,
    "read_email":          AgentRole.EMAIL,
    "draft_email":         AgentRole.EMAIL,
    # Reminders
    "set_reminder":        AgentRole.REMINDER,
    "list_reminders":      AgentRole.REMINDER,
    # Notes
    "create_note":         AgentRole.NOTES,
    "search_notes":        AgentRole.NOTES,
    # Information
    "web_search":          AgentRole.WEBSEARCH,
    "research_topic":      AgentRole.RESEARCH,
    "get_news":            AgentRole.NEWS,
    "get_weather":         AgentRole.WEATHER,
    # System
    "mac_control":         AgentRole.MAC,
    "spotify_control":     AgentRole.SPOTIFY,
    "open_file":           AgentRole.FILE,
    "read_document":       AgentRole.DOCUMENT,
    # Finance
    "finex_query":         AgentRole.FINEX,
    # Meta
    "summarise":           AgentRole.SUMMARISER,
    "general_chat":        AgentRole.PLANNER,
}

SYSTEM_PROMPT = """You are the Router for Jarvis, an AI executive assistant.
Your only job is to classify the user's request and decide which agents should handle it.

Available intents and their primary agents:
- schedule_meeting, check_calendar, cancel_meeting, reschedule_meeting → calendar
- send_email, read_email, draft_email → email
- set_reminder, list_reminders → reminder
- create_note, search_notes → notes
- web_search → websearch (ANY factual question: who, what, when, where, why, how, which)
- research_topic → research (deep multi-source research on a topic)

IMPORTANT: Questions starting with who/what/when/where/why/how/which should ALWAYS route to websearch unless they clearly match another category (weather, calendar, email etc).
- get_news → news
- get_weather → weather
- mac_control (open apps, volume, brightness) → mac
- spotify_control (play, pause, skip, search) → spotify
- open_file, read_document → file or document
- summarise → summariser
- general_chat → planner

Tier classification rules (add "tier" field to your JSON):
- tier 1 (tool-only, instant, NO LLM response needed): weather, get_news, spotify_control, mac_control, get_sports, get_markets, check_calendar, list_reminders, prayer_times, open_file
- tier 2 (single LLM call, simple conversational): general_chat, web_search, summarise, set_reminder, send_email, draft_email
- tier 3 (full pipeline, complex/multi-step): research_topic, morning_briefing, read_document, schedule_meeting, any multi-step task

Respond with valid JSON only:
{
  "intent": "<one of the intents above>",
  "primary_agent": "<agent name>",
  "supporting_agents": ["memory"],
  "confidence": 0.95,
  "tier": 1,
  "reasoning": "Brief explanation"
}

Rules:
- Always include "memory" in supporting_agents
- Add "summariser" to supporting_agents if the response will be long
- confidence is 0.0–1.0
- If you are unsure, use general_chat intent with tier 2
"""


class RouterAgent:
    """
    Classifies incoming requests and decides agent dispatch.

    This is what makes Jarvis a proper MAS rather than a monolith —
    every request goes through structured classification before any
    action is taken.
    """

    def __init__(self, llm_client: OllamaClient | None = None):
        self.llm = llm_client or OllamaClient()
        print("🔀 RouterAgent ready")

    # ── Deterministic pre-routing (0ms, no LLM) ───────────────────────────

    def _deterministic_route(self, req: str) -> Optional[RouterDecision]:
        """
        Fast rule-based routing for common unambiguous patterns.
        Returns a RouterDecision if confident, None to fall through to LLM.
        """
        r = req.lower().strip()

        def _decision(agent: AgentRole, tier: int = 1) -> RouterDecision:
            return RouterDecision(
                primary_agent=agent,
                supporting_agents=[AgentRole.MEMORY],
                confidence=0.99,
                reasoning="deterministic rule",
                tier=tier,
            )

        # ── Email patterns ────────────────────────────────────────────────
        _email_send = re.compile(
            r'\b(send|write|draft|compose|email|message)\b.*\b(email|message|mail)\b|'
            r'\bemail\s+to\b|\btell\s+\w+\s+that\b|\bsend\s+\w+\s+an?\s+email\b',
            re.I,
        )
        _email_inbox = re.compile(
            r'\b(inbox|my emails?|show emails?|check emails?|unread|new emails?)\b', re.I
        )
        _email_action = re.compile(
            r'\b(read|reply|archive|mark)\s+(email|mail)\s*\d*\b|'
            r'\breply\s+to\s+(email|mail|that|it|this)\b|'
            r'\bmark\s+all\s+as\s+read\b|'
            r'\b(read|reply|archive)\s+email\s+\d+\b',
            re.I,
        )

        if _email_inbox.search(r):
            return _decision(AgentRole.EMAIL, tier=1)
        if _email_action.search(r):
            return _decision(AgentRole.EMAIL, tier=1)
        if _email_send.search(r):
            # Tier 1 so the orchestrator's _try_shortcut runs the
            # EmailComposer + GmailAgent flow (draft, preview, confirm, send)
            # instead of streaming a free-form LLM draft that never gets sent.
            return _decision(AgentRole.EMAIL, tier=1)

        # ── Calendar patterns ──────────────────────────────────────────────
        _cal_create = re.compile(
            r'\b(schedule|book|create|add|set|make|put)\b.{0,30}'
            r'\b(meeting|appointment|event|call|session)\b|'
            r'\b(meeting|appointment|event)\b.{0,20}\b(at|on|for|tomorrow|tonight)\b',
            re.I,
        )
        _cal_check = re.compile(
            r'\b(what.s on|whats on|my calendar|my schedule|my day|do i have|'
            r'any meetings?|any events?|check calendar)\b',
            re.I,
        )
        _reminder = re.compile(
            r'\b(remind me|set a reminder|reminder|alert me)\b', re.I
        )

        if _cal_check.search(r):
            return _decision(AgentRole.CALENDAR, tier=1)
        if _cal_create.search(r):
            # MUST be tier=1 — the real calendar-booking flow lives in
            # _try_shortcut, which only runs at tier 1 (or tier 3, never
            # tier 2). Previously this was tier=2 which skipped the
            # shortcut and let the LLM hallucinate fake "I've scheduled
            # it" responses without ever touching Google Calendar.
            return _decision(AgentRole.CALENDAR, tier=1)
        if _reminder.search(r):
            # Same reasoning — reminders have a shortcut handler that
            # needs tier=1 to fire. The shortcut falls back to tier=2
            # automatically if it doesn't match.
            return _decision(AgentRole.REMINDER, tier=1)

        # ── FinEx (financial-statement Q&A) patterns ──────────────────────
        # Must come before the generic factual-question rule, otherwise a
        # query like "what was Bestway's revenue?" gets routed to websearch.
        # Tier 2 — single LLM call inside FinEx (no planner needed).
        _finex_rx = re.compile(
            r'\b(finex|bestway|hbl|annual report|financial statement|'
            r'balance sheet|income statement|cash flow|p\s*&\s*l|'
            r'gross margin|operating margin|profit margin|ebitda|ebit|'
            r'(?:net|gross|operating)\s+profit|revenue|turnover|'
            r'eps|earnings per share|dividend per share|'
            r'(?:current|debt[-\s]to[-\s]equity|quick)\s+ratio|'
            r'\broa\b|\broe\b|\broce\b|\broic\b)\b',
            re.I,
        )
        if _finex_rx.search(r):
            return RouterDecision(
                primary_agent=AgentRole.FINEX,
                supporting_agents=[AgentRole.MEMORY],
                confidence=0.95,
                reasoning="deterministic finex rule (financial statement keyword)",
                tier=2,
            )

        # ── Factual-question patterns (who/what/when/etc.) ────────────────
        # These short-circuit the router LLM call (saves ~1–3s) and route
        # straight to the websearch agent. Excludes patterns owned by other
        # tier 1 agents (weather/calendar/email already handled above).
        _factual_q = re.compile(
            r'^\s*(who|what|when|where|why|how|which)\b.*\b(is|are|was|were|did|does|do|will)\b',
            re.I,
        )
        _tell_about = re.compile(
            r'^\s*(tell me|explain|describe|define)\b', re.I,
        )
        if _factual_q.search(r) or _tell_about.search(r):
            # Tier 2 = single LLM call after a websearch lookup
            return RouterDecision(
                primary_agent=AgentRole.WEBSEARCH,
                supporting_agents=[AgentRole.MEMORY],
                confidence=0.95,
                reasoning="deterministic factual-question rule",
                tier=2,
            )

        return None

    async def route(self, user_request: str) -> RouterDecision:
        """
        Classify a user request and return routing decision.

        Args:
            user_request: Raw natural language input from user

        Returns:
            RouterDecision with primary agent, supporting agents, confidence
        """
        # Fast deterministic pass first — no LLM needed
        fast = self._deterministic_route(user_request)
        if fast is not None:
            print(
                f"🔀 Routed (deterministic) → {fast.primary_agent.value} "
                f"(tier: {fast.tier})"
            )
            return fast

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Classify this request: "{user_request}"'},
        ]

        try:
            data = await self.llm.chat_json(messages, model=OLLAMA_ROUTER_MODEL, max_tokens=80)
        except Exception as exc:
            # Graceful fallback — never crash on routing
            print(f"⚠️  Router LLM error: {exc} — falling back to planner")
            return self._fallback_decision(user_request)

        primary = self._parse_agent(data.get("primary_agent", "planner"))
        supporting = [
            self._parse_agent(a)
            for a in data.get("supporting_agents", ["memory"])
            if self._parse_agent(a) is not None
        ]

        # Memory is always included
        if AgentRole.MEMORY not in supporting:
            supporting.insert(0, AgentRole.MEMORY)

        # Parse tier — clamp to 1-3
        raw_tier = data.get("tier", 3)
        try:
            tier = max(1, min(3, int(raw_tier)))
        except (TypeError, ValueError):
            tier = 3

        decision = RouterDecision(
            primary_agent=primary,
            supporting_agents=supporting,
            confidence=float(data.get("confidence", 0.8)),
            reasoning=data.get("reasoning", ""),
            tier=tier,
        )

        print(
            f"🔀 Routed → {decision.primary_agent.value} "
            f"(confidence: {decision.confidence:.2f}, tier: {decision.tier})"
        )
        return decision

    # ── Helpers ────────────────────────────────────────────────────────────

    def _parse_agent(self, name: str) -> AgentRole:
        try:
            return AgentRole(name.lower().strip())
        except ValueError:
            return AgentRole.PLANNER

    def _fallback_decision(self, request: str) -> RouterDecision:
        """Rule-based fallback when LLM is unavailable."""
        req = request.lower()
        words = req.split()
        first_word = words[0] if words else ""

        if any(w in req for w in ["meeting", "calendar", "schedule", "appointment"]):
            primary = AgentRole.CALENDAR
        elif any(w in req for w in ["email", "mail", "send", "inbox"]):
            primary = AgentRole.EMAIL
        elif any(w in req for w in ["weather", "forecast", "temperature"]):
            primary = AgentRole.WEATHER
        elif any(w in req for w in ["news", "headline", "latest news"]):
            primary = AgentRole.NEWS
        elif any(w in req for w in ["play", "pause", "spotify", "music", "song", "track"]):
            primary = AgentRole.SPOTIFY
        elif any(w in req for w in ["remind", "reminder", "alert"]):
            primary = AgentRole.REMINDER
        elif any(w in req for w in ["note", "write down", "jot", "save a note"]):
            primary = AgentRole.NOTES
        elif any(w in req for w in ["open", "launch", "volume", "brightness", "screenshot",
                                     "dark mode", "battery", "wifi", "clipboard", "quit",
                                     "mute", "lock", "sleep"]):
            primary = AgentRole.MAC
        elif any(w in req for w in ["folder", "directory", "create folder", "make folder",
                                     "new folder", "create file", "new file", "delete file",
                                     "rename", "move file", "find file", "list files",
                                     "what's on my desktop", "browse files"]):
            primary = AgentRole.FILE
        elif first_word in ("what", "who", "when", "where", "why", "how", "which",
                            "tell", "explain", "describe", "define", "search",
                            "find", "look", "google", "research", "investigate"):
            primary = AgentRole.WEBSEARCH
        elif any(w in req for w in ["research", "investigate", "analyse", "analyze"]):
            primary = AgentRole.RESEARCH
        else:
            primary = AgentRole.PLANNER

        # Assign tier for fallback routes
        tier1_agents = {AgentRole.WEATHER, AgentRole.NEWS, AgentRole.SPOTIFY,
                        AgentRole.MAC, AgentRole.REMINDER, AgentRole.FILE}
        tier3_agents = {AgentRole.RESEARCH}
        if primary in tier1_agents:
            tier = 1
        elif primary in tier3_agents:
            tier = 3
        else:
            tier = 2

        return RouterDecision(
            primary_agent=primary,
            supporting_agents=[AgentRole.MEMORY],
            confidence=0.5,
            reasoning="Fallback rule-based routing",
            tier=tier,
        )