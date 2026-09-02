"""
finex/_llm_helper.py — sync LLM helper for FinEx.

The FinEx extraction + Q&A code (extract_pdf.py and LLM_SQL.py) was written
against a local Ollama server using `requests` directly. This helper wraps that
call shape in one function — `chat_sync` — so the two callers share a single
place for timeouts, retries, model selection and error strings.

It uses synchronous `requests` (not aiohttp) because the FinEx pipeline runs
in a thread executor, not on the event loop. That is also why it does not reuse
``config.llm_client.OllamaClient``: that client is async, and adapting it would
mean an event loop per worker thread for no functional gain over what is here.

Three things this file is responsible for that it previously was not:

* **Thinking-model suppression.** FinEx now runs the same model as the main
  agent (qwen3:8b by default), and qwen3 emits chain-of-thought by default —
  Ollama returns it in ``message.thinking``, or inline as ``<think>…</think>``.
  Left alone the model spends the whole ``num_predict`` budget reasoning and
  the visible answer arrives empty or truncated mid-sentence. Worse for FinEx
  than for chat: ``_llm_classify_rows`` and ``_llm_extract_fields`` parse JSON
  out of the reply, so a reasoning preamble doesn't raise — it fails the parse
  and silently degrades extraction to "field not found". ``config/llm_client``
  already solved this for the async path; this is the same fix for the sync one.

* **Model selection follows the main agent.** ``FINEX_MODEL`` still wins if set,
  but the fallback is now ``OLLAMA_CHAT_MODEL`` rather than a hardcoded
  llama3.2, so FinEx tracks whatever the rest of Jarvis is running instead of
  drifting away from it silently.

* **keep_alive matches the rest of Jarvis.** This used to send a hardcoded
  "10m", which evicted an 8B model from RAM ten minutes after the last FinEx
  question while the main agent had the same weights pinned. The next question
  then paid a full model load. It now honours ``OLLAMA_KEEP_ALIVE``.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import requests


# ── Endpoint ───────────────────────────────────────────────────────────────
_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
_OLLAMA_TAGS_URL = os.getenv(
    "OLLAMA_TAGS_URL", "http://localhost:11434/api/tags"
)

# Persistent session to avoid TCP handshakes on every call.
_session = requests.Session()


# ── Model selection ────────────────────────────────────────────────────────
# Resolved per call rather than pinned at import, so a .env change needs a
# process restart but not a code edit, and tests can monkeypatch the env.
#
# Precedence: FINEX_MODEL (explicit override) → OLLAMA_CHAT_MODEL (whatever the
# main Jarvis agent is using) → qwen3:8b. The final fallback is only reached
# when neither variable is set; it names the family these prompts are tuned for
# rather than a model nothing else in the repo runs.
_DEFAULT_MODEL = "qwen3:8b"


def resolve_model() -> str:
    """The model FinEx will call. Exposed so callers can log or display it."""
    return (
        os.getenv("FINEX_MODEL")
        or os.getenv("OLLAMA_CHAT_MODEL")
        or _DEFAULT_MODEL
    ).strip()


# ── Context window ─────────────────────────────────────────────────────────
# ONE size for every FinEx call — extraction and Q&A alike.
#
# num_ctx is a load-time parameter: Ollama re-loads the model when it changes.
# FinEx used to ask for four different values (256 when warming, 2048 for L1
# lookups, 4096 for extraction and most handlers, more for the analytical
# levels), so uploading a PDF and then asking a question about it paid a full
# model reload for the switch — and warm_model() pre-loaded at 256 only for the
# first real question to discard it.
#
# 8192 is sized for the largest prompt FinEx builds (L6: financial context plus
# ~4000 characters of report text plus history) and matches LLM_NUM_CTX for the
# main agent, so both share one resident configuration of the same weights.
_DEFAULT_NUM_CTX = 8192


def resolve_num_ctx() -> int:
    try:
        return int(os.getenv("FINEX_NUM_CTX", str(_DEFAULT_NUM_CTX)))
    except (TypeError, ValueError):
        return _DEFAULT_NUM_CTX


# ── keep_alive ─────────────────────────────────────────────────────────────
# Ollama's JSON API accepts a duration STRING ("30m", "24h") or a NUMBER
# (seconds; -1 = pinned forever). A bare "-1" from the env arrives as a string
# and fails Go's duration parsing with 'missing unit in duration "-1"', so
# purely-numeric values are coerced to int. Same treatment as config/settings.
def _keep_alive() -> Any:
    raw = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return raw


# ── Thinking-model handling ────────────────────────────────────────────────
# Kept in sync with _THINKING_FAMILIES in config/llm_client.py. Duplicated
# rather than imported because importing that module pulls aiohttp and
# config.settings, and config.settings creates directories at import time —
# an unwanted side effect for the standalone `python -m finex.extract_pdf` CLI.
# If you add a family there, add it here.
_THINKING_FAMILIES = ("qwen3", "deepseek-r1", "magistral")

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.S)


def _wants_think_off(model: str) -> bool:
    m = (model or "").lower()
    return any(fam in m for fam in _THINKING_FAMILIES)


def _strip_think(text: str) -> str:
    """Remove <think>…</think> blocks, including a dangling unclosed one.

    Belt-and-braces: `think: False` should mean this never fires, but older
    Ollama builds ignore the flag for some models and emit the reasoning
    inline in `content` anyway.
    """
    if not text or "<think>" not in text:
        return text
    return _THINK_RE.sub("", text).strip()


def chat_sync(
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    num_ctx: Optional[int] = None,
    timeout: int = 60,
    json_mode: bool = False,
    use_router_model: bool = False,
) -> str:
    """
    Synchronous chat call to the local Ollama server.

    Returns the assistant message content as a string. On failure, returns a
    string beginning with "[LLM" — never raises. Callers already test for that
    prefix, so an empty or thinking-only reply is reported the same way rather
    than being passed downstream as a valid (empty) answer.

    `num_ctx` defaults to resolve_num_ctx() — pass a value only with a reason,
    since a differing context size forces Ollama to re-load the model.

    `json_mode=True` asks Ollama to constrain output to valid JSON. Used by the
    extraction call sites, where the reply is parsed rather than displayed.

    `use_router_model` is accepted and ignored: it selected the smaller of two
    cloud models, and the local model is chosen by resolve_model() alone.
    """
    return _chat_ollama(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        num_ctx=resolve_num_ctx() if num_ctx is None else num_ctx,
        timeout=timeout,
        json_mode=json_mode,
    )


def health_check() -> bool:
    """Quick liveness probe for the local Ollama server."""
    try:
        r = _session.get(_OLLAMA_TAGS_URL, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Internal: Ollama path ──────────────────────────────────────────────────
def _chat_ollama(
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    num_ctx: int,
    timeout: int,
    json_mode: bool = False,
) -> str:
    model = resolve_model()

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": _keep_alive(),
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }
    # The whole point of this module's existence — see the header.
    if _wants_think_off(model):
        payload["think"] = False
    if json_mode:
        payload["format"] = "json"

    # One retry budget, spent only on a 400 that names a key this server build
    # doesn't understand. `think` and `format` are both relatively recent
    # additions to the Ollama API; an older daemon rejects the request outright
    # rather than ignoring the unknown field, and falling back is strictly
    # better than failing the call.
    for attempt in (1, 2):
        try:
            resp = _session.post(_OLLAMA_URL, json=payload, timeout=timeout)

            if resp.status_code == 400 and attempt == 1:
                body = (resp.text or "").lower()
                dropped = False
                if "think" in payload and "think" in body:
                    payload.pop("think", None)
                    dropped = True
                if "format" in payload and "format" in body:
                    payload.pop("format", None)
                    dropped = True
                if dropped:
                    continue  # retry once without the unsupported key(s)

            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            return "[Ollama is not running. Please start Ollama and try again.]"
        except requests.exceptions.Timeout:
            return (
                f"[LLM timeout after {timeout}s — model may be loading. "
                "Try again.]"
            )
        except Exception as e:
            return f"[LLM error: {e}]"

        content = _strip_think((data.get("message", {}) or {}).get("content", "") or "").strip()

        if not content:
            # Distinguish the thinking-budget failure from a real empty answer.
            # Without this the caller receives "" and treats it as a valid
            # response, which is how a broken model config becomes a blank
            # financial answer rather than a visible error.
            if (data.get("message", {}) or {}).get("thinking"):
                return (
                    f"[LLM error: {model} returned only reasoning and no answer. "
                    f"Raise num_predict (currently {max_tokens}) or confirm this "
                    "Ollama build honours think:false.]"
                )
            return f"[LLM error: {model} returned an empty response.]"

        return content

    return f"[LLM error: {model} rejected the request after a retry.]"


__all__ = ["chat_sync", "health_check", "resolve_model", "resolve_num_ctx"]
