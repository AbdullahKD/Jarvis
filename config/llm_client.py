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
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_BACKOFF,
)


class OllamaError(Exception):
    pass


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

    # ── Chat ──────────────────────────────────────────────────────────────

    # Jarvis system prompt — injected into every general chat call
    JARVIS_SYSTEM_PROMPT = (
        "You are J.A.R.V.I.S — a Multi-Agent AI Executive Assistant built for Abdullah Khan Durrani. "
        "You were created as part of Abdullah's dissertation project at BNU (COM6001). "
        "Your name is Jarvis. You are NOT Abdullah. You are NOT the user. "
        "You act on Abdullah's behalf — when writing emails or messages, you write AS Abdullah, "
        "but you never confuse yourself with him. "
        "When introducing yourself, say you are Jarvis, Abdullah's AI assistant. "
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
        "When listing items, use a numbered list (1. 2. 3.) or flowing sentences. "
        "Keep a professional, warm, and direct tone."
    )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        expect_json: bool = False,
        inject_system: bool = True,
        max_tokens: int = 1024,
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

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens,
            },
        }

        if expect_json:
            payload["format"] = "json"
            payload["stream"] = False

        return await self._post_with_retry("/api/chat", payload, expect_json)

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
        payload = {"model": self.embed_model, "prompt": text}
        try:
            result = await self._post_with_retry("/api/embeddings", payload)
            data = json.loads(result)
            return data["embedding"]
        except Exception:
            # Fallback: use nomic-embed-text via the generate endpoint
            # or return a deterministic hash vector for testing
            return self._hash_embed(text)

    def _hash_embed(self, text: str) -> List[float]:
        """
        Deterministic fallback embedding (for when embed model not pulled).
        Not semantic — only used if Ollama embed model is unavailable.
        """
        import hashlib
        import struct
        h = hashlib.sha256(text.encode()).digest()
        # Produce 256-dim float vector from hash bytes
        floats = [struct.unpack("f", h[i:i+4])[0] for i in range(0, min(len(h), 64), 4)]
        # Normalise to unit vector
        magnitude = sum(x**2 for x in floats) ** 0.5 or 1.0
        return [x / magnitude for x in floats]

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
                            return "".join(chunks)
                        else:
                            data = await resp.json()
                            # /api/chat non-streamed response
                            if "message" in data:
                                return data["message"]["content"]
                            # /api/embeddings response
                            return json.dumps(data)

            except (aiohttp.ClientError, asyncio.TimeoutError, OllamaError) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    jitter = random.uniform(0, delay * 0.1)
                    await asyncio.sleep(delay + jitter)
                    delay *= RETRY_BACKOFF

        raise last_exc