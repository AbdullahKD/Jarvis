"""
Jarvis User Profile
───────────────────
A single, editable source of truth for who Jarvis is assisting.

Historically the user's details were scattered across env vars (favourite
teams, location) and the persona prompt, while the memory store only logged
task *results*. This module centralises a living profile that is:

  • loaded from data/profile.json (created from defaults on first run),
  • injected as a compact summary into every chat system prompt, and
  • read by the morning-briefing builder to decide which sections to show.

Edit data/profile.json directly, or update via code. The summary() method is
what actually reaches the model, so keep it short and factual.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List

from config.settings import (
    DATA_DIR,
    DEFAULT_LOCATION_NAME,
    FAVOURITE_TEAMS,
)

PROFILE_PATH = Path(DATA_DIR) / "profile.json"

# Canonical list of briefing sections + their default order. The profile's
# `briefing_sections` picks from these; "all" (default) means every section.
BRIEFING_SECTIONS = [
    "greeting",       # time-aware greeting + date
    "prayer",         # prayer times / next prayer
    "weather",        # local weather
    "calendar",       # today's events
    "email",          # important unread email
    "reminders",      # due/!upcoming reminders
    "sports",         # favourite teams' fixtures/results
    "markets",        # tracked markets
    "news",           # headlines
    "closing",        # motivational closing line
]


@dataclass
class UserProfile:
    # ── Identity ────────────────────────────────────────────────────────────
    name: str = "Abdullah Khan Durrani"
    preferred_name: str = "Abdullah"
    occupation: str = "University student (BNU — COM6001 dissertation project)"
    location: str = DEFAULT_LOCATION_NAME
    timezone: str = "Europe/London"
    languages: List[str] = field(default_factory=lambda: ["English", "Urdu"])
    faith: str = "Muslim"                 # drives prayer-time relevance / fasting awareness
    observes_prayer_times: bool = True

    # ── Interaction preferences ──────────────────────────────────────────────
    # tone: one of "butler-formal", "concise", "casual", "professional"
    tone: str = "butler-formal"
    email_signoff: str = "Jarvis, on behalf of Abdullah"

    # ── Interests (used for sports/markets/news personalisation) ─────────────
    favourite_teams: List[str] = field(default_factory=lambda: list(FAVOURITE_TEAMS))
    tracked_markets: List[str] = field(default_factory=lambda: ["Bestway Cement", "PSX", "Bitcoin"])
    interests: List[str] = field(default_factory=lambda: ["finance", "technology", "sports"])

    # ── Routine ──────────────────────────────────────────────────────────────
    working_hours: str = "09:00–18:00"
    # "all" → every section in canonical order; otherwise an explicit ordered list.
    briefing_sections: Any = "all"
    briefing_time: str = "07:30"          # used if a daily brief is scheduled

    # ── People Jarvis should know (name → relationship/notes) ────────────────
    key_people: Dict[str, str] = field(default_factory=dict)

    # ── Free-form extra notes the user wants Jarvis to always keep in mind ───
    notes: List[str] = field(default_factory=list)

    # ──────────────────────────────────────────────────────────────────────
    def enabled_sections(self) -> List[str]:
        """Resolve `briefing_sections` to a concrete ordered list."""
        if self.briefing_sections == "all" or not self.briefing_sections:
            return list(BRIEFING_SECTIONS)
        # Keep canonical order, but only the sections the user enabled.
        wanted = set(self.briefing_sections)
        ordered = [s for s in BRIEFING_SECTIONS if s in wanted]
        return ordered or list(BRIEFING_SECTIONS)

    def summary(self) -> str:
        """Compact one-block summary injected into the system prompt.

        Kept terse and factual — this is prepended to every chat call, so
        verbosity here is paid on every request.
        """
        tone_map = {
            "butler-formal": "polished, butler-like, professional yet warm",
            "concise": "concise and direct, minimal fluff",
            "casual": "friendly, relaxed and conversational",
            "professional": "professional and neutral",
        }
        tone_desc = tone_map.get(self.tone, self.tone)
        teams = ", ".join(self.favourite_teams) if self.favourite_teams else ""
        markets = ", ".join(self.tracked_markets) if self.tracked_markets else ""
        langs = " and ".join(self.languages) if self.languages else "English"

        parts = [
            f"You are assisting {self.name} (address him as {self.preferred_name}).",
            f"He is a {self.occupation}, based in {self.location} ({self.timezone}).",
            f"He speaks {langs}.",
        ]
        if self.faith:
            parts.append(
                f"He is {self.faith}"
                + (" — surface prayer times where relevant and be mindful of "
                   "prayer and fasting (e.g. Ramadan) when scheduling." if self.observes_prayer_times else ".")
            )
        parts.append(f"Preferred assistant tone: {tone_desc}.")
        if teams:
            parts.append(f"Teams he follows: {teams}.")
        if markets:
            parts.append(f"Markets/companies he tracks: {markets}.")
        parts.append(
            f"When you send email on his behalf, sign off exactly as: \"{self.email_signoff}\"."
        )
        if self.key_people:
            ppl = "; ".join(f"{n} ({r})" for n, r in self.key_people.items())
            parts.append(f"People he knows: {ppl}.")
        for note in self.notes:
            parts.append(note.rstrip(".") + ".")
        return "User profile — " + " ".join(parts)

    # ── Persistence ───────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path = PROFILE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        # Only accept known fields so a stale/extended JSON never crashes load.
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


def load_profile(path: Path = PROFILE_PATH) -> UserProfile:
    """Load the profile from JSON, creating it from defaults on first run."""
    try:
        if path.exists():
            return UserProfile.from_dict(json.loads(path.read_text()))
    except Exception as exc:  # corrupt/partial file → fall back to defaults
        print(f"⚠️  Could not load profile ({exc}); using defaults")
    profile = UserProfile()
    try:
        profile.save(path)
        print(f"👤 Created default user profile at {path}")
    except Exception as exc:
        print(f"⚠️  Could not write default profile: {exc}")
    return profile
