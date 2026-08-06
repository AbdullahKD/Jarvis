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
import re
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

    # ──────────────────────────────────────────────────────────────────────
    #  Multi-ACTION detection & splitting
    #
    #  The intent map above is tuned for *information* requests (weather, news,
    #  sports…). Compound *action* requests — "dim the screen, play Despacito
    #  and remind me at 5" — need a different gate: they're characterised by
    #  multiple imperative action verbs joined by connectors. We detect that
    #  here so the orchestrator can decide whether to split the request into
    #  separate commands and run each one.
    # ──────────────────────────────────────────────────────────────────────

    # Verbs that signal an imperative action (not a question). Used to decide
    # whether a compound sentence is "do X and do Y" vs a single question that
    # merely happens to contain " and " (e.g. "what's the link between X and Y").
    ACTION_VERBS = {
        "play", "pause", "resume", "stop", "skip", "next", "previous",
        "send", "email", "draft", "write", "compose", "reply", "tell",
        "message", "remind", "set", "add", "create", "make", "schedule",
        "book", "open", "launch", "close", "quit", "turn", "increase",
        "decrease", "lower", "raise", "dim", "brighten", "mute", "unmute",
        "lock", "sleep", "search", "find", "look", "show", "check", "read",
        "list", "delete", "remove", "move", "rename", "save", "note",
        "jot", "put", "take", "screenshot", "summarise", "summarize",
    }

    # Connectors that genuinely separate two commands. Deliberately tighter
    # than CONNECTORS (no bare " with "/" & ") to reduce false splits.
    _SPLIT_CONNECTORS = [
        " then ", ", and then ", " and then ", ", then ", ", and ", " and also ",
        " also ", " plus ", " as well as ", " after that ", ";", ", ",
        " and ",
    ]

    def _segment_has_action(self, seg: str) -> bool:
        words = re.findall(r"[a-z']+", seg.lower())
        return any(w in self.ACTION_VERBS for w in words[:4]) or any(
            w in self.ACTION_VERBS for w in words
        )

    def looks_multi_action(self, query: str) -> bool:
        """
        Cheap pre-filter: does this look like 2+ imperative commands?

        Used by the orchestrator to decide whether it's worth spending an LLM
        call to split the request. Conservative on purpose — when in doubt it
        returns False and the request flows through the normal single path.
        """
        q = query.strip()
        if len(q) < 8:
            return False
        ql = q.lower()
        # Pure questions are never multi-action commands.
        if ql.split()[0] in {"what", "who", "when", "where", "why", "how",
                              "which", "is", "are", "does", "do", "can", "could",
                              "would", "should", "tell"} and "?" in q:
            return False
        if not any(c in ql for c in self._SPLIT_CONNECTORS):
            return False
        segments = self.split_compound(query)
        action_segments = [s for s in segments if self._segment_has_action(s)]
        return len(action_segments) >= 2

    def split_compound(self, query: str) -> List[str]:
        """
        Rule-based split of a compound request into ordered sub-requests.

        This is the deterministic fallback used when the LLM splitter is
        unavailable. It splits on action-boundary connectors but guards
        against splitting noun phrases ("salt and pepper", two teams in one
        fixture query) by requiring the resulting tail to contain an action
        verb before accepting an " and " split.
        """
        text = query.strip()
        # Normalise the strongest separators to a single delimiter first.
        marker = "|JARVIS_SPLIT|"
        work = text
        for conn in (" and then ", ", and then ", " then ", ", then ",
                     " after that ", " also get ", " also ", " as well as ",
                     " plus ", ", and ", ";"):
            work = re.sub(re.escape(conn), marker, work, flags=re.IGNORECASE)

        rough = [p.strip(" ,.;") for p in work.split(marker) if p.strip(" ,.;")]

        # Now consider splitting any remaining " and " / bare commas — but only
        # where BOTH sides look like actions, so we don't break "black and
        # white" or "Pakistan, Australia and India" inside one query.
        final: List[str] = []
        for chunk in rough:
            final.extend(self._split_on_action_boundary(chunk))

        return [s for s in (seg.strip(" ,.;") for seg in final) if s]

    def _split_on_action_boundary(self, chunk: str) -> List[str]:
        """Recursively split a chunk on ' and ' / ', ' only at action boundaries.

        Splits at the EARLIEST separator (comma or 'and') so that
        "turn brightness to 10, play Despacito and remind me" decomposes fully
        rather than only at the first 'and'.
        """
        candidates = []
        for pat in (r"\s+and\s+", r"\s*,\s+"):
            m = re.search(pat, chunk, re.IGNORECASE)
            if m:
                candidates.append(m)
        if candidates:
            m = min(candidates, key=lambda mm: mm.start())
            left = chunk[:m.start()].strip()
            right = chunk[m.end():].strip()
            if (left and right
                    and self._segment_has_action(left)
                    and self._segment_has_action(right)):
                # Recurse into BOTH sides to catch 3+ way splits.
                return [*self._split_on_action_boundary(left),
                        *self._split_on_action_boundary(right)]
        return [chunk]