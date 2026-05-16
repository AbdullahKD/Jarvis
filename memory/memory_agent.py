"""
Memory Agent
Persistent semantic memory using ChromaDB + Ollama embeddings.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from config.llm_client import OllamaClient
from config.models import MemoryItem, MemoryType
from config.settings import (
    CHROMA_DIR,
    MEMORY_COLLECTION_NAME,
    MEMORY_SIMILARITY_THRESHOLD,
    MEMORY_TOP_K,
)

# Embeddings should never block the request path. If Ollama is slow, fall
# back to the deterministic hash embedding rather than waiting on the full
# LLM_TIMEOUT (60s) — which would itself blow the WS budget.
_EMBED_TIMEOUT = 8.0


class MemoryAgent:
    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.llm = llm_client or OllamaClient()
        self.chroma = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"🧠 MemoryAgent ready — {self.collection.count()} memories loaded")

    async def _get_embedding(self, text: str) -> List[float]:
        try:
            emb = await asyncio.wait_for(self.llm.embed(text), timeout=_EMBED_TIMEOUT)
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            return [float(x) for x in emb]
        except asyncio.TimeoutError:
            print(f"⚠️  Embedding timed out after {_EMBED_TIMEOUT}s — using hash fallback")
            return self.llm._hash_embed(text)
        except Exception as e:
            print(f"⚠️  Embedding failed: {e} — using hash fallback")
            return self.llm._hash_embed(text)

    async def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        if not content or not content.strip():
            content = "empty memory"

        mem_id = f"mem_{uuid.uuid4().hex[:12]}"
        now = datetime.now()
        embedding = await self._get_embedding(content)

        chroma_metadata = {
            "memory_type": memory_type.value,
            "created_at": now.isoformat(),
            **(metadata or {}),
        }
        chroma_metadata = {
            k: str(v) if not isinstance(v, (str, int, float, bool)) else v
            for k, v in chroma_metadata.items()
        }

        try:
            self.collection.add(
                ids=[mem_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[chroma_metadata],
            )
            print(f"💾 Memory stored [{memory_type.value}]: {content[:60]}...")
        except Exception as e:
            print(f"⚠️  ChromaDB store failed: {e}")

        return MemoryItem(
            id=mem_id,
            content=content,
            memory_type=memory_type,
            metadata=chroma_metadata,
            created_at=now,
        )

    async def retrieve(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        k: int = MEMORY_TOP_K,
        threshold: float = MEMORY_SIMILARITY_THRESHOLD,
    ) -> List[MemoryItem]:
        if self.collection.count() == 0:
            return []

        query_embedding = await self._get_embedding(query)
        where_filter = {"memory_type": memory_type.value} if memory_type else None

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"⚠️  ChromaDB query failed: {e}")
            return []

        memories = []
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - distance
            if similarity < threshold:
                continue
            memories.append(MemoryItem(
                id=meta.get("id", ""),
                content=doc,
                memory_type=MemoryType(meta.get("memory_type", "episodic")),
                metadata=meta,
                created_at=datetime.fromisoformat(
                    meta.get("created_at", datetime.now().isoformat())
                ),
                relevance_score=round(similarity, 4),
            ))
        return memories

    async def store_task_result(
        self, user_request: str, intent: str, success: bool, summary: str
    ) -> None:
        content = (
            f"Task: {user_request} | Intent: {intent} | "
            f"Success: {success} | Summary: {summary}"
        )
        await self.store(content, memory_type=MemoryType.EPISODIC,
                        metadata={"intent": intent, "success": str(success)})

    def get_count(self) -> int:
        return self.collection.count()

    async def clear(self) -> None:
        self.chroma.delete_collection(MEMORY_COLLECTION_NAME)
        self.collection = self.chroma.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("🗑️  Memory cleared")