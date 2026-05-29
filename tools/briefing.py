"""
Morning Briefing & Multi-Query Handler
Detects compound requests and executes multiple agents in parallel.
Examples:
  - "Good morning Jarvis"
  - "Give me a morning briefing"
  - "What's the weather and latest news?"
  - "Check my emails and calendar"
  - "What are the premier league scores and tech news?"
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class BriefingHandler:
    """
    Detects multi-part queries and orchestrates parallel execution.
    Used by the orchestrator as a pre-flight check before routing.
    """

    # Compound query connectors
    CONNECTORS = [
        " and ", " also ", " as well as ", " plus ", " along with ",
        " + ", " & ", ", and ", " then ", " also get ", " with "
    ]

    # Morning briefing triggers. Be generous — missing one here means the
    # request falls through to Tier-2 LLM chat, which then HALLUCINATES a
    # briefing from training data (stale news, wrong PM, fictional scores).
    # Order matters: longer phrases first so we don't false-positive on
    # something like "morning email".
    MORNING_TRIGGERS = [
        # Exact phrases (highest priority)
        "morning briefing", "morning brief", "give me a brief",
        "give me a morning brief", "give me a morning briefing",
        "give me my morning brief", "give me my morning briefing",
        "good morning", "morning jarvis", "morning update",
        "daily briefing", "daily update", "daily summary", "morning summary",
        "morning digest", "morning rundown",
        "start my day", "begin my day", "wake up",
        "what do i have today", "what's on today",
        "todays briefing", "today's briefing", "today's brief", "todays brief",
        "brief me", "morning brief me",
        # Single-word fallback (only "briefing" — bare "brief" is too greedy
        # and false-positives on phrases like "let me be brief").
        "briefing",
    ]

    # Intent detection map — what sub-queries map to
    INTENT_KEYWORDS = {
        "weather":  ["weather", "temperature", "forecast", "rain", "sunny", "cold", "hot", "raining"],
        "news":     ["news", "headlines", "latest", "current events", "what's happening", "stories"],
        "sports":   ["scores", "results", "fixtures", "premier league", "football", "nba", "nfl",
                    "champions league", "f1", "formula", "basketball", "sports"],
        "calendar": ["calendar", "schedule", "meetings", "appointments", "what do i have",
                    "what's on", "my day", "agenda"],
        "email":    ["email", "emails", "inbox", "messages", "mail", "unread"],
        "research": ["research", "tell me about", "explain", "what is", "who is"],
    }

    def is_morning_briefing(self, query: str) -> bool:
        """Check if query is a morning briefing request."""
        q = query.lower().strip()
        return any(trigger in q for trigger in self.MORNING_TRIGGERS)

    def detect_intents(self, query: str) -> List[str]:
        """
        Detect all intents in a compound query.
        Returns list of intent keys.
        """
        q = query.lower()
        found = []
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                found.append(intent)
        return found

    def is_compound(self, query: str) -> bool:
        """Check if query contains multiple distinct requests."""
        q = query.lower()
        # Check connectors
        has_connector = any(c in q for c in self.CONNECTORS)
        # Check multiple intents
        intents = self.detect_intents(query)
        return has_connector or len(intents) >= 2