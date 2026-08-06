"""
Ollama LLM Client
Wraps the Ollama REST API for async chat completions and embeddings.
Supports model switching for benchmarking (llama3, mistral, etc.)
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any, Dict, List, Optional

import aiohttp

from config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_KEEP_ALIVE,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_BACKOFF,
)


import os as _os_dim

# Dimension of the deterministic fallback embedding produced by
# OllamaClient._hash_embed. MUST equal the live embed model's output dim so a
# fallback vector can coexist with real vectors in the same ChromaDB collection.
# nomic-embed-text = 768. Override via EMBED_FALLBACK_DIM if you switch models.
EMBED_FALLBACK_DIM = int(_os_dim.getenv("EMBED_FALLBACK_DIM", "768"))


class OllamaError(Exception):
    pass


# ── Thinking-model handling ─────────────────────────────────────────────────
# Model families that emit chain-of-thought by default (Ollama puts it in
# message.thinking, or inline <think>...</think> tags). Left alone, they burn
# the whole num_predict budget "thinking" and the visible reply arrives empty
# — the UI shows "(no response)". We ask Ollama to disable thinking for these
# families, and additionally strip any inline think tags as a safety net.
_THINKING_FAMILIES = ("qwen3", "deepseek-r1", "magistral")

_THINK_RE_BLOCK = None  # compiled lazily


def _wants_think_off(model: str) -> bool:
    m = (model or "").lower()
    return any(fam in m for fam in _THINKING_FAMILIES)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks (and a dangling unclosed <think>)
    that some thinking models emit inline in content."""
    if "<think>" not in text:
        return text
    global _THINK_RE_BLOCK
    if _THINK_RE_BLOCK is None:
        import re as _re
        _THINK_RE_BLOCK = _re.compile(r"<think>.*?(?:</think>|$)", _re.S)
    return _THINK_RE_BLOCK.sub("", text).strip()


class OllamaClient:
    """
    Async client for the Ollama local LLM server.

    Usage:
        client = OllamaClient()
        response = await client.chat([{"role": "user", "content": "Hello"}])
        embedding = await client.embed("some text")
    """

    def __init__(
        self,
        model: str = OLLAMA_CHAT_MODEL,
        embed_model: str = OLLAMA_EMBED_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = LLM_TEMPERATURE,
        timeout: int = LLM_TIMEOUT,
    ):
        self.model = model
        self.embed_model = embed_model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        # Streaming uses an INTER-CHUNK (sock_read) timeout, not a total cap.
        # A total cap kills long but healthy generations (e.g. "deep research"
        # / "elaborate") the moment they run past `timeout` seconds, even while
        # tokens are still flowing. With sock_read we only abort if NO token
        # arrives for STREAM_IDLE_TIMEOUT seconds (covers a slow first token /
        # cold model), and otherwise let the full answer stream to completion.
        _stream_idle = int(_os_dim.getenv("STREAM_IDLE_TIMEOUT", str(max(timeout, 120))))
        self.stream_timeout = aiohttp.ClientTimeout(
            total=None, sock_connect=15, sock_read=_stream_idle
        )

    # ── Chat ──────────────────────────────────────────────────────────────

    # Jarvis system prompt — injected into every general chat call
    JARVIS_SYSTEM_PROMPT = (
        "You are J.A.R.V.I.S — a Multi-Agent AI Executive Assistant built for Abdullah Khan Durrani. "
        "You were created as part of Abdullah's dissertation project at BNU (COM6001). "
        "Your name is Jarvis. You are NOT Abdullah. You are NOT the user. "
        "You act on Abdullah's behalf — when writing emails or messages, you write AS Abdullah, "
        "but you never confuse yourself with him. "
        "Only introduce yourself if the user explicitly asks who you are. "
        "Never begin an answer to a query by introducing yourself or stating your name. "
        "Never refer to yourself as Abdullah, never sign off as Abdullah, "
        "and never assume you ARE the user in any context. "
        "Your capabilities include: managing Abdullah's Gmail and Google Calendar, "
        "checking weather, browsing the web, reading news from 25+ sources, "
        "tracking sports scores (Premier League, La Liga, UCL, NBA, Cricket), "
        "monitoring financial markets (Bitcoin, NVDA, AAPL, TSLA, MSFT), "
        "controlling his Mac (open apps, volume, brightness, screenshots, dark mode), "
        "managing files on his Desktop, Documents and Downloads, "
        "fetching prayer times, playing Spotify, running morning briefings, "
        "and handling complex multi-step tasks through a multi-agent pipeline. "
        "Respond in clean, natural prose — no markdown asterisks (*), no raw bullet symbols. "
        "Keep a professional, warm, and direct tone. "
        "RESPONSE STYLE RULES — STRICT: "
        "Default answer length is EXACTLY ONE PARAGRAPH of 3–5 sentences. "
        "Never produce more than one paragraph unless the user explicitly asks to "
        "'elaborate', 'expand', 'tell me more', 'go into detail', 'in detail', "
        "'break it down', or otherwise requests a longer response. "
        "Never use bullet points (* or -), numbered lists, headers (#), or markdown formatting in a default answer. "
        "Never include blank lines that split your answer into multiple paragraphs. "
        "Never cite, mention, or reference sources, URLs, or where you found information — just answer directly. "
        "Only provide sources if the user explicitly asks for them. "
        "Never end with follow-up offers like 'Would you like to know more?' — the user knows they can ask."
    )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        expect_json: bool = False,
        inject_system: bool = True,
        max_tokens: int = 1024,
        num_ctx: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request to Ollama using streaming for speed.

        Args:
            messages:       OpenAI-style message list
            model:          Override the default model (for benchmarking)
            temperature:    Override default temperature
            expect_json:    If True, add JSON format instruction
            inject_system:  If True, prepend Jarvis system prompt (skip for JSON calls)
            max_tokens:     Token budget — use 50-100 for routing/classification calls

        Returns:
            Model response as a string (or JSON string if expect_json)
        """
        # Inject system prompt for conversational calls (not JSON/router calls)
        if inject_system and not expect_json:
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages = [
                    {"role": "system", "content": self.JARVIS_SYSTEM_PROMPT}
                ] + list(messages)

        options: Dict[str, Any] = {
            "temperature": temperature or self.temperature,
            "num_predict": max_tokens,
        }
        # num_ctx caps the prompt window. Per-call override wins (short
        # body-only calls like the email composer pass a small value to save
        # memory); otherwise use the configured default (LLM_NUM_CTX, 8192 on
        # the M5/24GB machine — up from Ollama's implicit 4096).
        options["num_ctx"] = num_ctx if num_ctx is not None else LLM_NUM_CTX

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            # Keep the model resident between calls (env-configurable via
            # OLLAMA_KEEP_ALIVE; "-1" = pinned forever, the right call on the
            # 24GB M5). Ollama's default 5 min reliably bit demos whenever the
            # user paused to introduce a feature.
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": options,
        }

        if expect_json:
            payload["format"] = "json"
            payload["stream"] = False

        # Thinking models (qwen3 etc.): answer directly, don't reason first.
        # _post_with_retry auto-drops this key if the server rejects it.
        if _wants_think_off(payload["model"]):
            payload["think"] = False

        return await self._post_with_retry("/api/chat", payload, expect_json)

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ):
        """
        Async generator that yields text chunks as they arrive from Ollama.
        Use this for real-time streaming to the WebSocket.

        Usage:
            async for chunk in client.chat_stream(messages):
                await ws.send_text(json.dumps({"type":"chunk","text":chunk}))
        """
        if not any(m.get("role") == "system" for m in messages):
            messages = [
                {"role": "system", "content": self.JARVIS_SYSTEM_PROMPT}
            ] + list(messages)

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": OLLAMA_KEEP_ALIVE,  # see note in `chat()` — match its keep_alive
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
                "num_ctx": LLM_NUM_CTX,
            },
        }
        if _wants_think_off(payload["model"]):
            payload["think"] = False

        # Inline <think> tag filter state (safety net for models that emit
        # reasoning in content rather than the separate thinking field).
        _in_think = False

        for _attempt in (1, 2):
            try:
                async with aiohttp.ClientSession(timeout=self.stream_timeout) as session:
                    async with session.post(
                        f"{self.base_url}/api/chat", json=payload
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            # Older Ollama servers reject the `think` key —
                            # drop it and retry once rather than failing.
                            if (resp.status == 400 and "think" in payload
                                    and "think" in text.lower() and _attempt == 1):
                                payload.pop("think", None)
                                break  # out of session; loop retries
                            raise OllamaError(f"HTTP {resp.status}: {text[:200]}")
                        async for raw_line in resp.content:
                            line = raw_line.strip()
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                # Skip inline <think>...</think> spans.
                                if _in_think:
                                    if "</think>" in content:
                                        content = content.split("</think>", 1)[1]
                                        _in_think = False
                                    else:
                                        content = ""
                                if "<think>" in content:
                                    pre, _, rest = content.partition("<think>")
                                    if "</think>" in rest:
                                        content = pre + rest.split("</think>", 1)[1]
                                    else:
                                        content = pre
                                        _in_think = True
                                if content:
                                    yield content
                            if chunk.get("done"):
                                return
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                raise OllamaError(f"Stream error: {exc}") from exc

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Convenience: chat and parse JSON response."""
        raw = await self.chat(messages, model=model, expect_json=True, max_tokens=max_tokens)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            # Try to extract JSON block from response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(raw[start:end])
            raise OllamaError(f"Model returned invalid JSON: {raw[:200]}") from exc

    # ── Embeddings ────────────────────────────────────────────────────────

    async def embed(self, text: str) -> List[float]:
        """
        Generate an embedding vector for text using Ollama.
        Falls back to a simple TF-IDF-style hash if embed model unavailable.
        """
        payload = {
            "model": self.embed_model,
            "prompt": text,
            "keep_alive": OLLAMA_KEEP_ALIVE,
        }
        try:
            result = await self._post_with_retry("/api/embeddings", payload)
            data = json.loads(result)
            return data["embedding"]
        except Exception:
            # Fallback: use nomic-embed-text via the generate endpoint
            # or return a deterministic hash vector for testing
            return self._hash_embed(text)

    def _hash_embed(self, text: str, dim: Optional[int] = None) -> List[float]:
        """
        Deterministic fallback embedding, used only when the Ollama embed
        model is unavailable or times out.

        Produces a ``dim``-dimensional unit vector. ``dim`` MUST match the live
        embed model's output dimension (nomic-embed-text = 768, configurable
        via the EMBED_FALLBACK_DIM env var). This matters: a fallback vector of
        a different length than the real embeddings is rejected by ChromaDB with
        a dimension-mismatch error, which silently breaks memory store/retrieve
        whenever an embedding briefly falls back (e.g. a cold/slow Ollama). The
        old implementation emitted only 8 dims and could contain NaN/inf from
        raw-byte float decoding — both of which corrupted the memory collection.

        Not semantic — similarity between two hash embeddings is meaningless;
        this only keeps the vector store consistent so it never errors.
        """
        import hashlib

        if dim is None:
            dim = EMBED_FALLBACK_DIM

        vec: List[float] = []
        counter = 0
        while len(vec) < dim:
            digest = hashlib.sha256(f"{text}#{counter}".encode()).digest()
            for b in digest:
                # Map each byte [0, 255] → [-1.0, 1.0]; no NaN/inf possible.
                vec.append((b / 127.5) - 1.0)
                if len(vec) >= dim:
                    break
            counter += 1
        magnitude = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / magnitude for x in vec]

    # ── Model management ──────────────────────────────────────────────────

    async def list_models(self) -> List[str]:
        """Return names of all models pulled in Ollama."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(f"{self.base_url}/api/tags") as resp:
                data = await resp.json()
                return [m["name"] for m in data.get("models", [])]

    async def is_model_available(self, model: str) -> bool:
        """Check if a specific model is available locally."""
        models = await self.list_models()
        return any(model in m for m in models)

    async def health_check(self) -> bool:
        """Ping Ollama server."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
                async with s.get(f"{self.base_url}/api/tags") as r:
                    return r.status == 200
        except Exception:
            return False

    # ── Internal ──────────────────────────────────────────────────────────

    async def _post_with_retry(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        parse_content: bool = False,
    ) -> str:
        last_exc: Exception = OllamaError("No attempts made")
        delay = RETRY_BASE_DELAY
        is_streaming = payload.get("stream", False)

        for attempt in range(MAX_RETRIES + 1):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.post(
                        f"{self.base_url}{endpoint}",
                        json=payload,
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            # Older Ollama servers reject the `think` key —
                            # drop it and retry immediately instead of failing.
                            if (resp.status == 400 and "think" in payload
                                    and "think" in text.lower()):
                                payload.pop("think", None)
                                raise OllamaError("retry-without-think")
                            raise OllamaError(f"HTTP {resp.status}: {text[:200]}")

                        if is_streaming:
                            # Collect streamed NDJSON chunks
                            chunks = []
                            async for raw_line in resp.content:
                                line = raw_line.strip()
                                if not line:
                                    continue
                                try:
                                    chunk = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                content = chunk.get("message", {}).get("content", "")
                                if content:
                                    chunks.append(content)
                                if chunk.get("done"):
                                    break
                            return _strip_think("".join(chunks))
                        else:
                            data = await resp.json()
                            # /api/chat non-streamed response
                            if "message" in data:
                                return _strip_think(data["message"]["content"])
                            # /api/embeddings response
                            return json.dumps(data)

            except (aiohttp.ClientError, asyncio.TimeoutError, OllamaError) as exc:
                last_exc = exc
                if str(exc) == "retry-without-think":
                    # Not a real failure — the server just didn't understand
                    # the `think` key (now removed from payload). Retry
                    # immediately without consuming a retry attempt.
                    continue
                if attempt < MAX_RETRIES:
                    jitter = random.uniform(0, delay * 0.1)
                    await asyncio.sleep(delay + jitter)
                    delay *= RETRY_BACKOFF

        raise last_exc


# ── Backend selection ───────────────────────────────────────────────────────
# Ollama is the only backend. Jarvis previously carried a swappable cloud
# client selected by LLM_BACKEND=groq; that path is gone, so the alias below
# exists purely so the handful of call sites that ask for the local client by
# name keep resolving.
_LocalOllamaClient = OllamaClient