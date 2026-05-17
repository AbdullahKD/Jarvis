"""
finex/_llm_helper.py — Backend-aware sync LLM helper for FinEx.

The FinEx extraction + Q&A code (extract_pdf.py and LLM_SQL.py) was written
against a local Ollama server using `requests` directly. In cloud deployments
that URL doesn't exist, so every call hangs on the 90-second timeout.

This helper provides one function — `chat_sync` — that:
  * Talks to Groq's OpenAI-compatible API when LLM_BACKEND=groq or GROQ_API_KEY
    is set (the cloud path).
  * Falls back to the original local Ollama call shape otherwise (so local
    dev is unchanged).

It uses synchronous `requests` (not aiohttp) because the FinEx pipeline runs
in a thread executor, not on the event loop.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Optional

import requests


# ── Backend detection ──────────────────────────────────────────────────────
def _backend() -> str:
    """Return 'groq' or 'ollama' based on env."""
    if os.getenv("LLM_BACKEND", "").lower() == "groq":
        return "groq"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return "ollama"


# ── Groq config ────────────────────────────────────────────────────────────
_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
_GROQ_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
_GROQ_ROUTER_MODEL = os.getenv("GROQ_ROUTER_MODEL", "llama-3.1-8b-instant")

# ── Ollama config (unchanged from FinEx defaults) ──────────────────────────
_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
_OLLAMA_MODEL = os.getenv("FINEX_MODEL", "llama3.2:latest")

# Persistent session to avoid TCP handshakes on every call.
_session = requests.Session()


def chat_sync(
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    num_ctx: int = 4096,
    timeout: int = 60,
    use_router_model: bool = False,
) -> str:
    """
    Synchronous chat call that routes to the active backend.

    Returns the assistant message content as a string. On failure, returns
    a "[LLM error: ...]" string — never raises. This matches the original
    FinEx ask_llm semantics so callers don't need to be updated.
    """
    backend = _backend()
    if backend == "groq":
        return _chat_groq(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            use_router_model=use_router_model,
        )
    return _chat_ollama(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        num_ctx=num_ctx,
        timeout=timeout,
    )


def health_check() -> bool:
    """Quick liveness probe for the active backend."""
    backend = _backend()
    if backend == "groq":
        if not _GROQ_API_KEY:
            return False
        try:
            r = _session.get(
                f"{_GROQ_BASE_URL}/models",
                headers={"Authorization": f"Bearer {_GROQ_API_KEY}"},
                timeout=5,
            )
            return r.status_code == 200
        except Exception:
            return False
    try:
        r = _session.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Internal: Groq path ────────────────────────────────────────────────────
def _chat_groq(
    messages: List[Dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
    use_router_model: bool,
) -> str:
    if not _GROQ_API_KEY:
        return "[LLM error: GROQ_API_KEY is not set]"
    model = _GROQ_ROUTER_MODEL if use_router_model else _GROQ_MODEL
    try:
        resp = _session.post(
            f"{_GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {_GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            return f"[LLM error: HTTP {resp.status_code} — {resp.text[:200]}]"
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return "[LLM error: empty response]"
        return (choices[0].get("message") or {}).get("content", "").strip()
    except requests.exceptions.Timeout:
        return f"[LLM timeout after {timeout}s — try again.]"
    except requests.exceptions.ConnectionError as e:
        return f"[LLM connection error: {e}]"
    except Exception as e:
        return f"[LLM error: {e}]"


# ── Internal: Ollama path (preserves original semantics) ───────────────────
def _chat_ollama(
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    num_ctx: int,
    timeout: int,
) -> str:
    try:
        resp = _session.post(
            _OLLAMA_URL,
            json={
                "model": _OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": num_ctx,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "[Ollama is not running. Please start Ollama and try again.]"
    except requests.exceptions.Timeout:
        return f"[LLM timeout after {timeout}s — model may be loading. Try again.]"
    except Exception as e:
        return f"[LLM error: {e}]"
