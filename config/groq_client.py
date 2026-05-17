"""
Cloud LLM Client — Groq for chat, sentence-transformers for embeddings.

Drop-in replacement for OllamaClient with identical public API so the rest
of the codebase doesn't need to change. Selected via `LLM_BACKEND=groq` env
var (see the rebind at the bottom of config/llm_client.py).

Groq:
    - OpenAI-compatible chat completions at https://api.groq.com/openai/v1
    - Free tier, very high tokens/sec on Llama 3.x models
    - No embeddings endpoint — that's why we run sentence-transformers locally

Embeddings:
    - sentence-transformers/all-MiniLM-L6-v2 (~80 MB, 384-dim)
    - Loaded once per process, called via thread executor to keep async sane
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Any, Dict, List, Optional

import aiohttp


# ── Settings ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# Default chat model — Llama 3.3 70B is the strongest free Groq model
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
# Faster, lighter model for routing/classification calls
GROQ_ROUTER_MODEL = os.getenv("GROQ_ROUTER_MODEL", "llama-3.1-8b-instant")

# Local embedding model (downloaded into image at build time)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "0.5"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "2.0"))


class GroqError(Exception):
    pass


# Reuse the system prompt verbatim from OllamaClient so behaviour is identical.
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


# ── Embedding model singleton ───────────────────────────────────────────────
_embed_model = None
_embed_model_lock = asyncio.Lock()


async def _get_embed_model():
    """Lazy-load the sentence-transformers model. Called from any coroutine."""
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    async with _embed_model_lock:
        if _embed_model is not None:
            return _embed_model
        # Heavy import — defer until first use.
        from sentence_transformers import SentenceTransformer
        loop = asyncio.get_event_loop()
        _embed_model = await loop.run_in_executor(
            None, lambda: SentenceTransformer(EMBED_MODEL_NAME)
        )
    return _embed_model


class GroqClient:
    """
    Cloud LLM client. Public API matches OllamaClient exactly so the
    rest of the codebase is unchanged.

    Methods:
        chat()         — non-streaming chat
        chat_stream()  — async generator, yields text chunks
        chat_json()    — chat + parse JSON
        embed()        — local sentence-transformers embedding
        health_check() — simple ping
        list_models()  — Groq model catalogue
        is_model_available()
    """

    JARVIS_SYSTEM_PROMPT = JARVIS_SYSTEM_PROMPT

    def __init__(
        self,
        model: str = GROQ_CHAT_MODEL,
        embed_model: str = EMBED_MODEL_NAME,
        base_url: str = GROQ_BASE_URL,
        temperature: float = LLM_TEMPERATURE,
        timeout: int = LLM_TIMEOUT,
        api_key: Optional[str] = None,
    ):
        # Normalize any Ollama-style tag (e.g. "llama3", "llama3.2:1b") to a
        # real Groq model. Without this, callers that pass model="llama3" — and
        # there are several in the codebase (PlannerAgent, JarvisOrchestrator,
        # the warmup task) — would 404 on every Groq call.
        self.model = self._normalize_model(model)
        self.embed_model = embed_model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            # Don't crash on import — many tests construct the client without
            # ever calling chat(). We'll raise lazily in _post_*.
            pass

    @staticmethod
    def _normalize_model(name: Optional[str]) -> str:
        """
        Translate Ollama-style tags to Groq model IDs.

        - Anything that already looks like a Groq model ID (e.g. "llama-3.3-
          70b-versatile", "mixtral-8x7b-32768") is passed through.
        - Tiny / router-class Ollama tags ("llama3.2:1b", "llama3.2:3b") map
          to GROQ_ROUTER_MODEL.
        - Anything else falls back to GROQ_CHAT_MODEL.
        """
        if not name:
            return GROQ_CHAT_MODEL
        n = name.strip().lower()
        # Already a Groq-shape model name — pass through.
        if n.startswith("llama-3") or n.startswith("mixtral-") or n.startswith("gemma"):
            return name
        # Router-class small models
        if any(t in n for t in (":1b", ":3b", "1b-instant", "router")):
            return GROQ_ROUTER_MODEL
        # Default — everything else (llama3, llama3:latest, mistral, etc.)
        return GROQ_CHAT_MODEL

    # ── Internal: headers ─────────────────────────────────────────────────
    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise GroqError(
                "GROQ_API_KEY is not set. Get one at https://console.groq.com/keys "
                "and export it (or fly secrets set GROQ_API_KEY=...)"
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ── Chat ──────────────────────────────────────────────────────────────
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        expect_json: bool = False,
        inject_system: bool = True,
        max_tokens: int = 1024,
        num_ctx: Optional[int] = None,  # accepted but ignored — Groq has fixed ctx
    ) -> str:
        if inject_system and not expect_json:
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages = [
                    {"role": "system", "content": self.JARVIS_SYSTEM_PROMPT}
                ] + list(messages)

        payload: Dict[str, Any] = {
            "model": self._normalize_model(model) if model else self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if expect_json:
            payload["response_format"] = {"type": "json_object"}

        return await self._post_chat_with_retry(payload)

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 512,
    ):
        """Async generator yielding text chunks. SSE format from Groq."""
        if not any(m.get("role") == "system" for m in messages):
            messages = [
                {"role": "system", "content": self.JARVIS_SYSTEM_PROMPT}
            ] + list(messages)

        payload: Dict[str, Any] = {
            "model": self._normalize_model(model) if model else self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise GroqError(f"HTTP {resp.status}: {text[:200]}")
                    async for raw_line in resp.content:
                        line = raw_line.strip()
                        if not line:
                            continue
                        # SSE: lines start with "data: "
                        if not line.startswith(b"data:"):
                            continue
                        data = line[len(b"data:"):].strip()
                        if data == b"[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content", "")
                        if content:
                            yield content
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise GroqError(f"Stream error: {exc}") from exc

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Chat with JSON response_format and parse the result."""
        raw = await self.chat(messages, model=model, expect_json=True, max_tokens=max_tokens)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(raw[start:end])
            raise GroqError(f"Model returned invalid JSON: {raw[:200]}") from exc

    # ── Embeddings (local sentence-transformers) ──────────────────────────
    async def embed(self, text: str) -> List[float]:
        """
        Generate a 384-dim embedding using sentence-transformers/all-MiniLM-L6-v2.
        Runs in a thread executor so it doesn't block the event loop.
        """
        try:
            model = await _get_embed_model()
            loop = asyncio.get_event_loop()
            vec = await loop.run_in_executor(
                None, lambda: model.encode(text, normalize_embeddings=True).tolist()
            )
            return vec
        except Exception:
            return self._hash_embed(text)

    def _hash_embed(self, text: str) -> List[float]:
        """Deterministic fallback if sentence-transformers fails to load."""
        import hashlib
        import struct
        h = hashlib.sha256(text.encode()).digest()
        floats = [struct.unpack("f", h[i:i + 4])[0] for i in range(0, min(len(h), 64), 4)]
        magnitude = sum(x ** 2 for x in floats) ** 0.5 or 1.0
        return [x / magnitude for x in floats]

    # ── Model management ──────────────────────────────────────────────────
    async def list_models(self) -> List[str]:
        """Fetch Groq's model catalogue."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                async with session.get(
                    f"{self.base_url}/models", headers=self._headers()
                ) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
                    return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    async def is_model_available(self, model: str) -> bool:
        models = await self.list_models()
        return any(model in m for m in models)

    async def health_check(self) -> bool:
        """Ping the Groq API."""
        if not self.api_key:
            return False
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(
                    f"{self.base_url}/models", headers=self._headers()
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    # ── Internal: chat POST with retry ────────────────────────────────────
    async def _post_chat_with_retry(self, payload: Dict[str, Any]) -> str:
        last_exc: Exception = GroqError("No attempts made")
        delay = RETRY_BASE_DELAY
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=self._headers(),
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            raise GroqError(f"HTTP {resp.status}: {text[:200]}")
                        data = await resp.json()
                        choices = data.get("choices") or []
                        if not choices:
                            raise GroqError(f"Empty response: {data}")
                        return choices[0]["message"]["content"]
            except (aiohttp.ClientError, asyncio.TimeoutError, GroqError) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    jitter = random.uniform(0, delay * 0.1)
                    await asyncio.sleep(delay + jitter)
                    delay *= RETRY_BACKOFF
        raise last_exc
