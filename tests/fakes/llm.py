"""A fake OllamaClient.

Substitutable for ``config.llm_client.OllamaClient`` anywhere one is injected.
It answers deterministically, never opens a socket, and returns instantly — so
tests exercise the *orchestration* rather than the model.

Two design choices worth stating:

**Scripted, not random.** Responses are chosen by matching the prompt against
registered rules. A test that wants a specific plan registers that plan; a test
that doesn't care gets a sensible default. Nothing depends on what a 8B model
happens to say today, which is what makes these tests deterministic where a
live-Ollama test never can be.

**It records everything.** ``client.calls`` holds every request, so a test can
assert on what the orchestrator *asked* — that memory context reached the
planner, that the router got the raw user text, that a replan actually
re-prompted. Those are the interesting assertions and they're invisible if you
only look at the final answer.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Pattern, Union

# Matches the real client: a deterministic pseudo-embedding when the model is
# unavailable. Kept the same dimension so ChromaDB collections are compatible.
EMBED_DIM = 768


@dataclass
class LLMCall:
    """One request the code under test made."""

    messages: List[Dict[str, str]]
    model: Optional[str] = None
    expect_json: bool = False
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        """All message content joined — what the rules match against."""
        return "\n".join(m.get("content", "") for m in self.messages)

    @property
    def system(self) -> str:
        return "\n".join(m.get("content", "") for m in self.messages
                         if m.get("role") == "system")

    @property
    def user(self) -> str:
        return "\n".join(m.get("content", "") for m in self.messages
                         if m.get("role") == "user")


Responder = Union[str, Dict[str, Any], Callable[[LLMCall], Any]]


class FakeOllamaClient:
    """Drop-in replacement for OllamaClient.

    ``when(pattern, response)`` registers a rule; the first rule whose pattern
    is found in the prompt wins, so register specific rules before general
    ones. With no rule matched, ``default_response`` is returned (or
    ``default_json`` for a JSON call).
    """

    #: The orchestrator assigns this at construction time; agents print it and
    #: some branch on it, so the fake carries it like the real client does.
    JARVIS_SYSTEM_PROMPT = "You are Jarvis."

    def __init__(
        self,
        *,
        model: str = "qwen3:8b",
        default_response: str = "Understood.",
        default_json: Optional[Dict[str, Any]] = None,
        healthy: bool = True,
        available_models: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.calls: List[LLMCall] = []
        self.rules: List[tuple[Pattern[str], Responder]] = []
        self.default_response = default_response
        self.default_json = default_json if default_json is not None else {}
        self.healthy = healthy
        self.available_models = available_models or ["qwen3:8b", "qwen3:1.7b"]
        # Set to raise from the next call, to test degradation paths.
        self.raise_next: Optional[Exception] = None
        self.embed_fails = False

    # ── Scripting ───────────────────────────────────────────────────────────

    def when(self, pattern: str, response: Responder) -> "FakeOllamaClient":
        """Register a rule. `pattern` is a regex searched in the whole prompt."""
        self.rules.append((re.compile(pattern, re.IGNORECASE | re.DOTALL), response))
        return self

    def _resolve(self, call: LLMCall) -> Any:
        for pattern, response in self.rules:
            if pattern.search(call.prompt):
                return response(call) if callable(response) else response
        return self.default_json if call.expect_json else self.default_response

    # ── OllamaClient surface ────────────────────────────────────────────────

    async def chat(self, messages, model=None, temperature=None, expect_json=False,
                   inject_system=True, max_tokens=None, num_ctx=None) -> str:
        call = LLMCall(messages=list(messages), model=model, expect_json=expect_json,
                       kwargs={"temperature": temperature, "max_tokens": max_tokens,
                               "num_ctx": num_ctx, "inject_system": inject_system})
        self.calls.append(call)
        self._maybe_raise()
        out = self._resolve(call)
        return out if isinstance(out, str) else json.dumps(out)

    async def chat_json(self, messages, model=None, max_tokens=None) -> Dict[str, Any]:
        call = LLMCall(messages=list(messages), model=model, expect_json=True,
                       kwargs={"max_tokens": max_tokens})
        self.calls.append(call)
        self._maybe_raise()
        out = self._resolve(call)
        if isinstance(out, str):
            try:
                return json.loads(out)
            except json.JSONDecodeError:
                # Mirrors the real client, which returns {} when the model
                # emits prose instead of JSON. Several code paths depend on
                # that being an empty dict rather than an exception.
                return {}
        return out

    async def chat_stream(self, messages, model=None, max_tokens=None):
        text = await self.chat(messages, model=model, max_tokens=max_tokens)
        for word in text.split(" "):
            yield word + " "

    async def embed(self, text: str) -> List[float]:
        if self.embed_fails:
            # The real client falls back to _hash_embed and flags degraded.
            raise RuntimeError("fake embedding endpoint unavailable")
        return self._hash_embed(text)

    def _hash_embed(self, text: str, dim: int = EMBED_DIM) -> List[float]:
        """Deterministic pseudo-embedding, same idea as the real fallback."""
        digest = hashlib.sha256(text.encode("utf-8", "ignore")).digest()
        raw = (digest * ((dim // len(digest)) + 1))[:dim]
        return [(b - 127.5) / 127.5 for b in raw]

    async def list_models(self) -> List[str]:
        return list(self.available_models)

    async def is_model_available(self, model: str) -> bool:
        return model in self.available_models

    async def health_check(self) -> bool:
        return self.healthy

    async def close(self) -> None:
        pass

    # ── Helpers for assertions ──────────────────────────────────────────────

    def _maybe_raise(self) -> None:
        if self.raise_next is not None:
            exc, self.raise_next = self.raise_next, None
            raise exc

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def prompts_containing(self, needle: str) -> List[LLMCall]:
        return [c for c in self.calls if needle.lower() in c.prompt.lower()]

    def assert_asked_about(self, needle: str) -> LLMCall:
        hits = self.prompts_containing(needle)
        assert hits, (f"no LLM call mentioned {needle!r}; "
                      f"saw {[c.prompt[:60] for c in self.calls]}")
        return hits[0]

    def reset(self) -> None:
        self.calls.clear()


# ── Ready-made response builders ────────────────────────────────────────────


def route(intent: str, agent: str = "general", confidence: float = 0.9,
          **extra) -> Dict[str, Any]:
    """A RouterAgent-shaped response."""
    return {"intent": intent, "agent": agent, "confidence": confidence, **extra}


def plan(*subtasks: Dict[str, Any], intent: str = "test_intent",
         **extra) -> Dict[str, Any]:
    """A PlannerAgent-shaped response.

    Each subtask needs id/agent/action; params and depends_on default sensibly,
    which keeps test plans readable::

        plan(subtask("s1", "weather", "get_current"))
    """
    return {"intent": intent, "subtasks": list(subtasks), **extra}


def subtask(sid: str, agent: str, action: str,
            params: Optional[Dict[str, Any]] = None,
            depends_on: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "id": sid, "agent": agent, "action": action,
        "params": params or {}, "depends_on": depends_on or [],
    }


__all__ = ["FakeOllamaClient", "LLMCall", "route", "plan", "subtask", "EMBED_DIM"]
