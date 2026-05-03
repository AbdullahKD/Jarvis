"""
Router Agent
Classifies every incoming user request and decides which specialist
agents should handle it. This is the entry point of every interaction.
"""

from __future__ import annotations

from typing import List

from config.llm_client import OllamaClient
from config.models import AgentRole, RouterDecision


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

Respond with valid JSON only:
{
  "intent": "<one of the intents above>",
  "primary_agent": "<agent name>",
  "supporting_agents": ["memory"],
  "confidence": 0.95,
  "reasoning": "Brief explanation"
}

Rules:
- Always include "memory" in supporting_agents
- Add "summariser" to supporting_agents if the response will be long
- confidence is 0.0–1.0
- If you are unsure, use general_chat intent
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

    async def route(self, user_request: str) -> RouterDecision:
        """
        Classify a user request and return routing decision.

        Args:
            user_request: Raw natural language input from user

        Returns:
            RouterDecision with primary agent, supporting agents, confidence
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Classify this request: "{user_request}"'},
        ]

        try:
            data = await self.llm.chat_json(messages, max_tokens=80)
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

        decision = RouterDecision(
            primary_agent=primary,
            supporting_agents=supporting,
            confidence=float(data.get("confidence", 0.8)),
            reasoning=data.get("reasoning", ""),
        )

        print(
            f"🔀 Routed → {decision.primary_agent.value} "
            f"(confidence: {decision.confidence:.2f})"
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

        return RouterDecision(
            primary_agent=primary,
            supporting_agents=[AgentRole.MEMORY],
            confidence=0.5,
            reasoning="Fallback rule-based routing",
        )