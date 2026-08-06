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

    async def _get_embedding(self, text: str) -> tuple:
        """Return (vector, degraded). degraded=True means the hash fallback
        fired — the vector is deterministic noise, not a real embedding, and
        the memory it produces should be tagged so it can be re-embedded
        later rather than silently polluting semantic search."""
        try:
            emb = await asyncio.wait_for(self.llm.embed(text), timeout=_EMBED_TIMEOUT)
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            return [float(x) for x in emb], False
        except asyncio.TimeoutError:
            print(f"⚠️  Embedding timed out after {_EMBED_TIMEOUT}s — using hash fallback")
            return self.llm._hash_embed(text), True
        except Exception as e:
            print(f"⚠️  Embedding failed: {e} — using hash fallback")
            return self.llm._hash_embed(text), True

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
        embedding, degraded = await self._get_embedding(content)

        chroma_metadata = {
            "memory_type": memory_type.value,
            "created_at": now.isoformat(),
            # Tag hash-fallback vectors so they're identifiable — a periodic
            # maintenance pass (or manual script) can re-embed them properly.
            **({"embedding_degraded": True} if degraded else {}),
            # Persist the id in metadata too so retrieve() can surface it,
            # which is what makes targeted forget()/deletion possible.
            "id": mem_id,
            **(metadata or {}),
        }
        chroma_metadata = {
            k: str(v) if not isinstance(v, (str, int, float, bool)) else v
            for k, v in chroma_metadata.items()
        }

        try:
            # ChromaDB is synchronous (sqlite + hnswlib under the hood) — run
            # it off the event loop so a slow disk write can't stall the
            # server's WebSocket/voice/reminder loops.
            await asyncio.to_thread(
                self.collection.add,
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
        # Memory retrieval is an enrichment step on the request path — a hiccup
        # in the vector store must never fail an otherwise-good turn, so every
        # store interaction here degrades gracefully to "no memories".
        try:
            count = await asyncio.to_thread(self.collection.count)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  ChromaDB count failed: {e}")
            return []
        if count == 0:
            return []

        query_embedding, query_degraded = await self._get_embedding(query)

        # A hash-fallback embedding is deterministic noise: cosine similarity
        # between two of them is meaningless. Querying WITH one returns
        # arbitrary memories that look like genuine matches, and memories
        # STORED with one can never be found by a real query. Both directions
        # are excluded rather than silently returning nonsense.
        #
        # store() already tags these `degraded`; nothing consumed the tag until
        # now, so one slow Ollama during a conversation permanently poisoned
        # that slice of memory with no symptom beyond Jarvis quietly recalling
        # irrelevant things.
        if query_degraded:
            print("⚠️  Query embedding degraded — skipping retrieval rather than "
                  "returning arbitrary matches")
            return []

        # Filtered in Python below, not with a `where` clause: store() only
        # writes `embedding_degraded` when it IS degraded, and a Chroma
        # {"$ne": True} predicate on a key most documents don't have excludes
        # those documents — which would have dropped every healthy memory.
        where_filter = {"memory_type": memory_type.value} if memory_type else None

        try:
            # Off-loop for the same reason as store() — see comment there.
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding],
                # Over-fetch: degraded memories are dropped below, and
                # without headroom a few of them would shrink an otherwise
                # healthy result set.
                n_results=min(k * 2, count),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"⚠️  ChromaDB query failed: {e}")
            return []

        memories = []
        try:
            for doc, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                similarity = 1.0 - distance
                if similarity < threshold:
                    continue
                meta = meta or {}
                try:
                    created = datetime.fromisoformat(
                        meta.get("created_at", datetime.now().isoformat())
                    )
                except (ValueError, TypeError):
                    created = datetime.now()
                memories.append(MemoryItem(
                    id=meta.get("id", ""),
                    content=doc,
                    memory_type=MemoryType(meta.get("memory_type", "episodic")),
                    metadata=meta,
                    created_at=created,
                    relevance_score=round(similarity, 4),
                ))
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  ChromaDB result parse failed: {e}")
            return self._drop_degraded(memories)[:k]
        return self._drop_degraded(memories)[:k]

    async def store_task_result(
        self, user_request: str, intent: str, success: bool, summary: str
    ) -> None:
        # This is fired-and-forgotten via asyncio.ensure_future from the
        # request path, so it MUST never raise — an unretrieved exception here
        # would surface as a noisy "Task exception was never retrieved" log and
        # serves no user-facing purpose. Episodic memory is best-effort.
        try:
            content = (
                f"Task: {user_request} | Intent: {intent} | "
                f"Success: {success} | Summary: {summary}"
            )
            await self.store(content, memory_type=MemoryType.EPISODIC,
                            metadata={"intent": intent, "success": str(success)})
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Episodic memory store skipped: {e}")

    # ── User-facts memory (preferences the user explicitly asks to keep) ──────
    async def remember(self, fact: str) -> "MemoryItem":
        """Store an explicit user fact/preference as a SEMANTIC memory."""
        return await self.store(
            fact.strip(),
            memory_type=MemoryType.SEMANTIC,
            metadata={"source": "user_remember"},
        )

    async def recall_facts(self, query: str, k: int = 4) -> List["MemoryItem"]:
        """Retrieve SEMANTIC (preference/fact) memories relevant to a query."""
        return await self.retrieve(query, memory_type=MemoryType.SEMANTIC, k=k)

    async def forget(self, query: str, k: int = 5, threshold: float = 0.45) -> int:
        """Delete SEMANTIC memories matching a query. Returns count removed."""
        matches = await self.retrieve(
            query, memory_type=MemoryType.SEMANTIC, k=k, threshold=threshold
        )
        ids = [m.metadata.get("id") for m in matches if m.metadata.get("id")]
        if not ids:
            return 0
        try:
            await asyncio.to_thread(self.collection.delete, ids=ids)
        except Exception as e:
            print(f"⚠️  ChromaDB delete failed: {e}")
            return 0
        return len(ids)

    def _drop_degraded(self, items: List[MemoryItem]) -> List[MemoryItem]:
        """Remove memories stored with a hash-fallback embedding."""
        kept = [m for m in items
                if not (m.metadata or {}).get("embedding_degraded")]
        dropped = len(items) - len(kept)
        if dropped:
            print(f"⚠️  Skipped {dropped} memory/memories with a degraded "
                  f"embedding — they are unreachable by semantic search")
        return kept

    def degraded_count(self) -> int:
        """How many stored memories carry a hash-fallback embedding.

        These are unreachable by semantic search. Surfaced on the health
        endpoint so a degraded store is visible rather than inferred from
        Jarvis seeming forgetful; re-embedding them is a follow-up.
        """
        try:
            got = self.collection.get(
                where={"embedding_degraded": True}, include=[])
            return len(got.get("ids", []))
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Could not count degraded memories: {exc}")
            return 0

    def get_count(self) -> int:
        return self.collection.count()

    async def clear(self) -> None:
        self.chroma.delete_collection(MEMORY_COLLECTION_NAME)
        self.collection = self.chroma.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("🗑️  Memory cleared")