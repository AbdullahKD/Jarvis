"""
Voice persona — the "JARVIS" character layer.

The orchestrator returns neutral, factual answers. ElevenLabs then reads
whatever text we hand it. This module sits in between: it takes the
orchestrator's plain reply and rewrites it in JARVIS's voice (calm, witty,
polite, addresses the user as "sir") before it reaches the TTS.

Architectural note for the report: this is a deliberate separation of
*what to say* (the orchestrator) from *how to say it* (the persona). The
same brain can be presented through multiple personas without retraining —
useful if we ever want a "professional" mode, a "casual" mode, etc.

Usage:

    stylizer = PersonaStylizer(llm_client=jarvis.llm)
    spoken_reply = await stylizer.stylize_async(transcript, plain_reply)
    tts.speak(spoken_reply)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("jarvis.voice.persona")


# ── System prompt — JARVIS's character + self-knowledge ────────────────────

DEFAULT_PERSONA = """You are JARVIS — a personal AI assistant modelled on Mr Stark's butler.

CHARACTER:
- Calm, witty, supremely competent, unfailingly polite.
- Address the user as "sir" once per reply (usually at the end). Don't say "sir" in every sentence.
- Concise. Speak naturally — no markdown, no bullet points, no headings.
- Dry humour is welcome. Never sycophantic. Never apologise unless something actually failed.

WHAT YOU ARE:
You are the voice of Abdullah's personal AI agent, running on his MacBook. You operate through a multi-agent orchestrator with eight specialist agents (Router, Memory, Planner, Critic, Executor, Evaluator, Summariser, plus Calendar and Gmail agents) and a suite of tools.

WHAT YOU CAN DO (only list these when asked):
- Read and manage Google Calendar (check availability, list events)
- Read, search, and summarise Gmail
- Control Spotify (play, pause, skip, search music)
- Control macOS (open apps, system actions)
- Search the web and fetch news headlines
- Weather forecasts for any location
- Set reminders and recall them later
- Store conversational memories and recall them in future turns
- Multi-step task planning with self-critique before execution

TASK:
Rewrite the "plain reply" below as YOU would say it aloud. Stay under 60 words.
Do not invent facts that aren't in the plain reply. Do not read out URLs, IDs, or
markdown formatting verbatim. Sound natural — this will be spoken by a TTS engine.
If the plain reply already sounds like you, only lightly polish it."""


class PersonaStylizer:
    """Wraps an Ollama-style chat client to apply the JARVIS persona to replies."""

    def __init__(
        self,
        llm_client=None,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 140,
        enabled: bool = True,
    ):
        self.llm = llm_client
        self.system_prompt = system_prompt or DEFAULT_PERSONA
        self.max_tokens = max_tokens
        self.enabled = enabled

    # ── Public API ─────────────────────────────────────────────────────────

    async def stylize_async(self, transcript: str, plain_reply: str) -> str:
        """
        Rewrite `plain_reply` in JARVIS's voice. If the LLM call fails or
        persona is disabled, falls back to a lightweight deterministic
        transform that at least appends ", sir." so the persona is never
        completely absent.
        """
        if not self.enabled or self.llm is None:
            return self._fallback(plain_reply)

        if not plain_reply or not plain_reply.strip():
            return self._fallback(plain_reply)

        user_msg = (
            f'The user said: "{transcript}"\n\n'
            f'Plain reply from the orchestrator: "{plain_reply}"\n\n'
            f"Rewrite as JARVIS, under 60 words. End naturally — include "
            f'"sir" once."'
        )

        try:
            chunks: list[str] = []
            async for chunk in self.llm.chat_stream(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=self.max_tokens,
            ):
                chunks.append(chunk)
            out = "".join(chunks).strip()
            # Strip wrapping quotes the model sometimes adds
            out = out.strip('"').strip("'").strip()
            # Remove any "Rewritten:" / "JARVIS:" prefix the model might leak
            for prefix in ("JARVIS:", "Jarvis:", "Rewritten:", "Reply:"):
                if out.startswith(prefix):
                    out = out[len(prefix):].strip()
            return out or self._fallback(plain_reply)
        except Exception as exc:  # noqa: BLE001
            log.warning("persona stylize failed: %s — using fallback", exc)
            return self._fallback(plain_reply)

    # ── Internals ──────────────────────────────────────────────────────────

    def _fallback(self, plain_reply: str) -> str:
        """
        Lightweight deterministic touch-up. Used when Ollama is unavailable
        or the stylize call failed. Just guarantees the reply ends with
        "sir" so the persona never disappears entirely.
        """
        text = (plain_reply or "").strip()
        if not text:
            return "I'm afraid I don't have a response for that, sir."

        # Don't double-tag if "sir" is already there in the last 8 chars
        if "sir" in text[-8:].lower():
            return text

        # If it ends with punctuation, slot "sir" in before the punctuation;
        # otherwise append ", sir."
        if text[-1] in ".!?":
            return text[:-1] + ", sir" + text[-1]
        return text + ", sir."


# ── Factory honouring .env ─────────────────────────────────────────────────


def make_stylizer(llm_client=None) -> PersonaStylizer:
    """Build a PersonaStylizer using settings from `.env`."""
    enabled = os.getenv("VOICE_PERSONA_ENABLED", "true").strip().lower() in (
        "1", "true", "yes", "on",
    )
    max_tokens = int(os.getenv("VOICE_PERSONA_MAX_TOKENS", "140"))
    system_prompt = os.getenv("VOICE_PERSONA_SYSTEM_PROMPT") or None
    return PersonaStylizer(
        llm_client=llm_client,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        enabled=enabled,
    )
