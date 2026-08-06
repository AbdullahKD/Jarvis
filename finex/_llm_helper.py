"""
finex/_llm_helper.py — sync LLM helper for FinEx.

The FinEx extraction + Q&A code (extract_pdf.py and LLM_SQL.py) was written
against a local Ollama server using `requests` directly. This helper wraps that
call shape in one function — `chat_sync` — so the two callers share a single
place for timeouts, retries and error strings.

It uses synchronous `requests` (not aiohttp) because the FinEx pipeline runs
in a thread executor, not on the event loop.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Optional

import requests


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
    Synchronous chat call to the local Ollama server.

    Returns the assistant message content as a string. On failure, returns
    a "[LLM error: ...]" string — never raises. This matches the original
    FinEx ask_llm semantics so callers don't need to be updated.

    `use_router_model` is accepted and ignored: it selected the smaller of two
    cloud models, and Ollama's FinEx model is configured by FINEX_MODEL alone.
    """
    return _chat_ollama(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        num_ctx=num_ctx,
        timeout=timeout,
    )


def health_check() -> bool:
    """Quick liveness probe for the local Ollama server."""
    try:
        r = _session.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


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
