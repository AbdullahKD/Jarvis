"""
Jarvis Query Parser
Universal Query Parser — uses local Ollama.

Replaces the manual is_simple / is_research / _max_tok / length_instruction
logic with a single structured parse that handles:
  - Mode selection  (QUICK / BALANCED / DEEP / ELI5 / EXPERT)
  - Intent taxonomy (16 sub-types)
  - Domain tagging
  - Confidence      (HIGH / MEDIUM / LOW)
  - Compound query decomposition via sub_questions
  - Flags           (TIME-SENSITIVE, TRICK-QUESTION, OPINION-AS-FACT, etc.)
  - Ambiguity detection

Usage in orchestrator:
    from tools.query_parser import QueryParser, infer_mode, Mode
    parser = QueryParser(llm_client)
    mode   = infer_mode(user_request)
    result = await parser.parse_and_answer(query, search_snippets, mode)
    msg    = result.answer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config.llm_client import OllamaClient


# ─────────────────────────────────────────────────────────────
#  ENUMS
# ─────────────────────────────────────────────────────────────

class Mode(str, Enum):
    QUICK    = "QUICK"     # 4–6 lines, key facts only
    BALANCED = "BALANCED"  # 1–3 clear paragraphs (default)
    DEEP     = "DEEP"      # Full breakdown, examples, edge cases
    ELI5     = "ELI5"      # Plain language, analogies, zero jargon
    EXPERT   = "EXPERT"    # Dense, technical, no hand-holding


class Confidence(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


class Ambiguity(str, Enum):
    NONE  = "none"
    MINOR = "minor"
    MAJOR = "major"


# ─────────────────────────────────────────────────────────────
#  RESULT DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class ParseResult:
    answer: str
    intent:          list[str]  = field(default_factory=list)
    domains:         list[str]  = field(default_factory=list)
    complexity:      int        = 3
    user_level:      str        = "Intermediate"
    confidence:      Confidence = Confidence.MEDIUM
    sub_questions:   list[str]  = field(default_factory=list)
    flags:           list[str]  = field(default_factory=list)
    ambiguity:       Ambiguity  = Ambiguity.NONE
    ambiguity_note:  str        = ""
    response_format: str        = "prose"
    mode_used:       Mode       = Mode.BALANCED

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def is_compound(self) -> bool:
        return len(self.sub_questions) > 1

    @property
    def has_flags(self) -> bool:
        return bool(self.flags)

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence == Confidence.LOW

    @property
    def is_time_sensitive(self) -> bool:
        return "TIME-SENSITIVE" in self.flags

    @property
    def is_opinion(self) -> bool:
        return any("OPINION" in i for i in self.intent)

    @property
    def max_tokens(self) -> int:
        """Recommended token budget for this mode."""
        return {
            Mode.QUICK:    140,
            Mode.BALANCED: 300,
            Mode.DEEP:     480,
            Mode.ELI5:     240,
            Mode.EXPERT:   420,
        }.get(self.mode_used, 300)


# ─────────────────────────────────────────────────────────────
#  SYSTEM PROMPT  (tuned for llama3.2 — concise and explicit)
# ─────────────────────────────────────────────────────────────

PARSER_SYSTEM_PROMPT = """You are the Jarvis Query Parser and Answer Engine.
Analyse the query deeply, then return ONLY a valid JSON object — no markdown, no preamble.

STEP 1 — INTENT (pick all that apply):
  FACTUAL/lookup  FACTUAL/calculation  FACTUAL/verification
  EXPLANATORY/concept  EXPLANATORY/mechanism  EXPLANATORY/cause-effect
  COMPARATIVE/ranking  COMPARATIVE/pros-cons  COMPARATIVE/similarity
  PROCEDURAL/how-to  PROCEDURAL/troubleshooting
  PREDICTIVE/forecast  PREDICTIVE/probability
  OPINION/debate  OPINION/recommendation
  CREATIVE/generate  CONVERSATIONAL  AMBIGUOUS

STEP 2 — DOMAINS (all that apply):
  Mathematics  Physics  Chemistry  Biology  Computer Science  Engineering  Astronomy
  Economics  Finance  History  Geography  Psychology  Law  Politics  Philosophy
  Literature  Music  Film  Language  Religion
  Sports  Health  Nutrition  Technology  Travel  General

STEP 3 — COMPOUND: if multiple questions are bundled, list each as "Q1: ...", "Q2: ..."

STEP 4 — FLAGS (list any that apply):
  TIME-SENSITIVE  TRICK-QUESTION  LOADED-QUESTION  OPINION-AS-FACT
  UNANSWERABLE  VAGUE-REFERENCE  MISSING-CONSTRAINTS

STEP 5 — COMPLEXITY: 1=trivial  2=simple  3=moderate  4=complex  5=expert

STEP 6 — USER LEVEL: Novice / Intermediate / Advanced / Expert
  Infer from vocabulary and assumed knowledge in the query.

STEP 7 — CONFIDENCE:
  HIGH   = established fact / scientific consensus
  MEDIUM = general consensus, some debate or regional variation
  LOW    = uncertain, conflicting sources, near knowledge cutoff, rapidly changing

STEP 8 — FORMAT: prose / bullets / steps / table

MODES — apply exactly as specified:
  QUICK    = one paragraph max, key facts only, no headers
  BALANCED = 1–3 paragraphs, clear structure
  DEEP     = 3–5 paragraphs, examples, nuance, common misconceptions
  ELI5     = plain language, analogies, zero jargon, like explaining to a curious teenager
  EXPERT   = dense, technical, assumes full domain knowledge, no hand-holding

Return ONLY this JSON — no text before or after:
{
  "parse": {
    "intent":          ["INTENT/subtype"],
    "domains":         ["Domain"],
    "complexity":      3,
    "user_level":      "Intermediate",
    "confidence":      "HIGH",
    "sub_questions":   [],
    "flags":           [],
    "ambiguity":       "none",
    "ambiguity_note":  "",
    "response_format": "prose"
  },
  "answer": "Full answer as a plain string. Use \\n for structure. Label sub-answers Q1:, Q2: if compound."
}

SPECIAL HANDLING:
- Never use ** or # markdown in the answer field
- Sports opinions → give a reasoned perspective based on form, history, tactics
- Time-sensitive topics → flag it; state knowledge may be outdated
- Trick/loaded questions → correct the premise before answering, respectfully
- Medical/legal → general information only; recommend professional consultation
- Compound queries → answer each sub-question before synthesising
- If SOURCE MATERIAL is provided → use it to ground the answer; do not invent facts"""


# ─────────────────────────────────────────────────────────────
#  MODE INFERENCE  (rule-based, 0ms — no LLM)
# ─────────────────────────────────────────────────────────────

_ELI5_TRIGGERS   = {
    "eli5", "explain like", "simple terms", "for a beginner", "layman",
    "easy to understand", "simply put", "like i'm 5", "like im 5",
    "dumbed down", "in simple words", "for dummies",
}
_EXPERT_TRIGGERS = {
    "technical", "advanced", "under the hood", "low level", "low-level",
    "architecture of", "implementation of", "deep technical", "expert level",
    "from first principles", "formally", "mathematically",
}
_DEEP_TRIGGERS   = {
    "research", "deep dive", "full overview", "everything about",
    "in depth", "in-depth", "comprehensive", "give me a full",
    "walk me through", "break it down", "break down", "elaborate",
    "explain fully", "full breakdown", "investigate", "analyse", "analyze",
    "detailed explanation", "detailed overview", "tell me everything",
}
_QUICK_TRIGGERS  = {
    "who is", "who was", "what is", "what's", "when did", "when was",
    "where is", "where was", "capital of", "ceo of", "founder of",
    "born in", "how old", "what year", "who won", "how tall",
    "how many", "what country", "what language", "define ",
    "who invented", "when was", "how much is",
}


def infer_mode(query: str) -> Mode:
    """
    Rule-based mode detection — runs in 0ms, no LLM needed.
    Call this before parse_and_answer() to set the expected depth.
    """
    q = query.lower().strip()
    if any(t in q for t in _ELI5_TRIGGERS):
        return Mode.ELI5
    if any(t in q for t in _EXPERT_TRIGGERS):
        return Mode.EXPERT
    if any(t in q for t in _DEEP_TRIGGERS):
        return Mode.DEEP
    if any(t in q for t in _QUICK_TRIGGERS):
        return Mode.QUICK
    return Mode.BALANCED


# ─────────────────────────────────────────────────────────────
#  QUERY PARSER
# ─────────────────────────────────────────────────────────────

class QueryParser:
    """
    Jarvis-native query parser and answer engine.
    Uses the local Ollama model — no external API required.

    Replaces the manual token-budget / length-instruction logic with
    a single structured LLM call that returns both the answer and
    rich parse metadata for downstream use.
    """

    # Token budget per mode — tuned for plain chat() output on llama3.2 3B
    # Keep budgets tight: the model is fast at ~10-15 tok/s so every 100 tokens ≈ 7-10s
    _TOKEN_MAP: dict[Mode, int] = {
        Mode.QUICK:    160,   # 4-6 lines  (~10-15s)
        Mode.BALANCED: 280,   # 1-3 paragraphs (~20-30s)
        Mode.DEEP:     380,   # 3-5 paragraphs (~30-45s)
        Mode.ELI5:     240,   # plain language, analogies (~20s)
        Mode.EXPERT:   360,   # dense technical (~30-40s)
    }

    # ── Mode-specific answer instructions ─────────────────────────────────

    _MODE_INSTRUCTIONS: dict = {
        Mode.QUICK:    (
            "Answer in exactly 4-6 lines. Be direct and factual. "
            "One short paragraph only. No lists, no headers."
        ),
        Mode.BALANCED: (
            "Answer in 1-3 clear paragraphs. Cover the essential facts. "
            "Well-structured and concise."
        ),
        Mode.DEEP:     (
            "Give a thorough response: 3-5 paragraphs covering background, "
            "key facts, current state, significance, and any nuance or common misconceptions. "
            "Use numbered points (1. 2. 3.) only where genuinely useful."
        ),
        Mode.ELI5:     (
            "Explain in plain language with analogies. Zero jargon. "
            "Write as if explaining to a curious 14-year-old. Keep it engaging and clear."
        ),
        Mode.EXPERT:   (
            "Dense, technical response. Assume full domain knowledge. "
            "No hand-holding, no simplifications. Cover edge cases and implementation details."
        ),
    }

    def __init__(self, llm_client: "OllamaClient"):
        self.llm = llm_client
        print("🧠 QueryParser ready (Ollama-backed)")

    # ── Main entry point ───────────────────────────────────────────────────

    async def parse_and_answer(
        self,
        query: str,
        search_snippets: str = "",
        mode: Mode = Mode.BALANCED,
    ) -> ParseResult:
        """
        Produce a mode-calibrated answer using plain chat (not JSON).

        Uses chat() instead of chat_json() to avoid JSON truncation on smaller
        models (llama3.2 3B). Mode behaviour is enforced via the system prompt.
        Parse metadata fields default — use classify() if you need intent tagging.

        Args:
            query:           The user's original request (cleaned).
            search_snippets: Web search results to ground the answer (optional).
            mode:            Depth mode. Use infer_mode(query) to auto-detect.

        Returns:
            ParseResult with .answer populated and .mode_used set.
        """
        instruction = self._MODE_INSTRUCTIONS.get(mode, self._MODE_INSTRUCTIONS[Mode.BALANCED])
        max_tok = self._TOKEN_MAP.get(mode, 340)

        system = (
            "You are Jarvis, an AI executive assistant. "
            "Answer the user's question accurately and concisely.\n\n"
            f"Style: {instruction}\n\n"
            "Rules:\n"
            "- No markdown (* bold * or # headers)\n"
            "- Write in natural prose\n"
            "- Never invent facts not present in the source material\n"
            "- You are Jarvis — never refer to yourself as the user"
        )

        user_msg = f"Question: {query}"
        if search_snippets and search_snippets.strip():
            user_msg += (
                f"\n\nSource material "
                f"(ground your answer in this — do not invent facts):\n{search_snippets}"
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ]

        try:
            answer = await self.llm.chat(messages, max_tokens=max_tok)
            answer = answer.strip()
            if not answer:
                return self._fallback(query, mode, "Empty answer from LLM")
            return ParseResult(answer=answer, mode_used=mode)
        except Exception as exc:
            print(f"⚠️  QueryParser LLM error: {exc}")
            return self._fallback(query, mode, str(exc))

    # ── Convenience: classify-only (no answer) — used by router ──────────

    async def classify(self, query: str) -> dict:
        """
        Lightweight classification — returns parse metadata only, no answer.
        Used by the router's _llm_classify() to get richer intent data.
        """
        messages = [
            {"role": "system", "content": PARSER_SYSTEM_PROMPT},
            {"role": "user",   "content": (
                f"MODE: QUICK\n\nQUERY: {query}\n\n"
                "Return ONLY the JSON parse block. "
                "The answer field can be an empty string."
            )},
        ]
        try:
            data = await self.llm.chat_json(messages, max_tokens=120)
            return data.get("parse", {})
        except Exception as exc:
            print(f"⚠️  QueryParser classify error: {exc}")
            return {}

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _fallback(query: str, mode: Mode, reason: str) -> ParseResult:
        """Return a safe degraded result when the LLM fails."""
        return ParseResult(
            answer=(
                "I had trouble processing that query. "
                "Please try rephrasing or ask again."
            ),
            flags=["UNANSWERABLE"],
            confidence=Confidence.LOW,
            mode_used=mode,
        )
