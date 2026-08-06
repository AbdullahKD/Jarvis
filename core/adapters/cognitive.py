"""
Adapters — the orchestrator's own faculties (memory, summariser, FinEx, and the
small internal helpers the Planner can call).

These aren't external services, but they're things the Planner emits subtasks
for, so they need to be in the registry too. Otherwise ``_dispatch`` keeps a
special-case branch for them and the "one lookup, no if/elif" property is lost
the moment a plan mentions memory.

``MemoryAgent.retrieve()`` never raises by design — a vector-store hiccup must
not fail an otherwise-good turn. That's the right call, but it means an empty
result is ambiguous: nothing relevant, or the store is down? The adapter
resolves it by reporting store health separately, so the dashboard can show
"memory degraded" while requests keep succeeding.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from core.tool import (
    Action,
    BaseTool,
    HealthReport,
    HealthStatus,
    ToolInputError,
    ToolUpstreamError,
)

logger = logging.getLogger("jarvis.adapters.cognitive")


# ── Memory ──────────────────────────────────────────────────────────────────


class MemoryAdapter(BaseTool):
    _name = "memory"
    _description = "Semantic long-term memory: store facts and retrieve relevant context."

    def __init__(self, agent: Any) -> None:
        self._a = agent
        super().__init__()

    def _register_actions(self) -> None:
        self.add_action(Action(
            name="retrieve_context",
            description="Find stored memories relevant to a query.",
            input_schema={"properties": {
                "query": {"type": "string", "minLength": 1},
                "k": {"type": "integer", "minimum": 1, "maximum": 25, "default": 5},
            }, "required": ["query"]},
            handler=self._retrieve, timeout=20.0,
        ))
        self.add_action(Action(
            name="store_fact", description="Remember a fact for future turns.",
            input_schema={"properties": {"content": {"type": "string", "minLength": 1}},
                          "required": ["content"]},
            handler=self._store, timeout=20.0, read_only=False,
        ))
        self.add_action(Action(
            name="recall_facts", description="Recall stored facts matching a query.",
            input_schema={"properties": {
                "query": {"type": "string", "minLength": 1},
                "k": {"type": "integer", "minimum": 1, "maximum": 25, "default": 5},
            }, "required": ["query"]},
            handler=self._recall, timeout=20.0,
        ))

    async def _retrieve(self, query: str, k: int = 5):
        mems = await self._a.retrieve(query, k=k)
        contents = [m.content for m in mems]
        return (
            {"memories": contents, "count": len(contents)},
            "\n".join(f"• {c}" for c in contents) if contents
            else "Nothing relevant in memory.",
        )

    async def _recall(self, query: str, k: int = 5):
        facts = await self._a.recall_facts(query, k=k)
        items = [getattr(f, "content", str(f)) for f in facts]
        return ({"facts": items, "count": len(items)},
                "\n".join(f"• {c}" for c in items) if items else "No matching facts.")

    async def _store(self, content: str):
        item = await self._a.store(content)
        degraded = bool(getattr(item, "metadata", {}) and
                        getattr(item, "metadata", {}).get("degraded"))
        from core.tool import ToolResult
        return ToolResult.ok(
            self.name, "store_fact",
            data={"id": getattr(item, "id", None), "content": content},
            message="Noted.", degraded=degraded,
            # A degraded embedding is deterministic noise, not a real vector.
            # The memory is stored but will never be found by similarity —
            # worth surfacing rather than silently poisoning the store.
            meta={"embedding": "hash-fallback"} if degraded else {},
        )

    async def _check_health(self) -> HealthReport:
        try:
            import asyncio
            count = await asyncio.to_thread(self._a.get_count)
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name,
                                f"vector store unreachable: {type(exc).__name__}: {exc}")
        # A probe query distinguishes "empty" from "broken" — retrieve()
        # swallows its own errors, so an empty list alone tells us nothing.
        try:
            await self._a.retrieve("health probe", k=1)
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.DEGRADED, self.name,
                                f"{count} memories stored, retrieval failing: {exc}")
        return HealthReport(HealthStatus.OK, self.name, f"{count} memories")


# ── Summariser ──────────────────────────────────────────────────────────────


class SummariserAdapter(BaseTool):
    _name = "summariser"
    _description = "Condense long text to a target length."

    def __init__(self, agent: Any) -> None:
        self._a = agent
        super().__init__()

    def _register_actions(self) -> None:
        self.add_action(Action(
            name="summarise", description="Summarise a block of text.",
            input_schema={"properties": {
                "text": {"type": "string", "minLength": 1},
                "max_words": {"type": "integer", "minimum": 10, "maximum": 2000,
                              "default": 150},
            }, "required": ["text"]},
            handler=self._summarise, timeout=90.0,
        ))

    async def _summarise(self, text: str, max_words: int = 150):
        out = await self._a.summarise(text, max_words=max_words)
        if not out:
            raise ToolUpstreamError("summariser returned nothing")
        return {"summary": out, "input_chars": len(text)}, out

    async def _check_health(self) -> HealthReport:
        llm = getattr(self._a, "llm", None)
        if llm is None:
            return HealthReport(HealthStatus.ERROR, self.name, "no LLM client attached")
        try:
            ok = await llm.health_check()
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name,
                                f"{type(exc).__name__}: {exc}")
        return (HealthReport(HealthStatus.OK, self.name, "llm reachable") if ok
                else HealthReport(HealthStatus.ERROR, self.name, "llm not reachable"))


# ── FinEx ───────────────────────────────────────────────────────────────────


class FinExAdapter(BaseTool):
    _name = "finex"
    _description = "Question answering over ingested company financial statements."
    # FinEx runs a multi-step LLM + SQL pipeline; the default 30s is far too
    # tight and produced spurious timeouts.
    _health_timeout = 20.0

    def __init__(self, get_agent: Callable[[], Any],
                 default_company: str = "Bestway Cement") -> None:
        # A callable, not the agent: importing FinExAgent pulls psycopg2 and
        # ChromaDB, so it stays lazy exactly as the orchestrator property does.
        self._get = get_agent
        self._default_company = default_company
        super().__init__()

    def _register_actions(self) -> None:
        self.add_action(Action(
            name="chat", description="Ask a question about a company's financials.",
            input_schema={"properties": {
                "question": {"type": "string", "minLength": 1},
                "company": {"type": "string", "default": self._default_company},
            }, "required": ["question"]},
            handler=self._chat, timeout=180.0,
        ))

    async def _chat(self, question: str, company: Optional[str] = None):
        if not question.strip():
            raise ToolInputError("question must not be empty")
        result = await self._get().chat(
            question=question, company=company or self._default_company)
        answer = (result or {}).get("answer", "")
        if not answer:
            raise ToolUpstreamError("FinEx returned no answer")
        return result, answer

    async def _check_health(self) -> HealthReport:
        # Deliberately shallow: a real probe runs the full SQL+LLM pipeline,
        # which is far too expensive for a heartbeat. Constructing the agent
        # is enough to prove psycopg2/Chroma imported and the DB is reachable.
        try:
            agent = self._get()
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.UNAVAILABLE, self.name,
                                f"cannot construct: {type(exc).__name__}: {exc}")
        return HealthReport(HealthStatus.OK, self.name,
                            f"{type(agent).__name__} ready")


# ── Internal helpers ────────────────────────────────────────────────────────


class InternalAdapter(BaseTool):
    """Orchestrator-owned helpers the Planner emits subtasks for.

    ``resolve_temporal`` and ``validate_output`` were branches in the dispatcher
    that switched on ``action`` inside a chain otherwise switching on ``agent`` —
    a latent mis-dispatch waiting for the right combination. Making them a
    named tool removes the ambiguity.
    """

    _name = "internal"
    _description = "Planner helpers: resolve relative dates, validate step output."

    def __init__(self, resolve_temporal: Callable[[str], Any]) -> None:
        self._resolve_temporal = resolve_temporal
        super().__init__()

    def _register_actions(self) -> None:
        self.add_action(Action(
            name="resolve_temporal",
            description="Turn a relative phrase ('next Tuesday', 'in 2 hours') into a datetime.",
            input_schema={"properties": {"phrase": {"type": "string", "minLength": 1}},
                          "required": ["phrase"]},
            handler=self._temporal, timeout=5.0,
        ))
        self.add_action(Action(
            name="validate_output",
            description="No-op validation step some plans include.",
            input_schema={"properties": {}}, handler=self._validate, timeout=5.0,
        ))

    async def _temporal(self, phrase: str):
        resolved = self._resolve_temporal(phrase)
        if resolved is None:
            from core.tool import ToolNotFoundError
            raise ToolNotFoundError(f"could not resolve the phrase {phrase!r} to a date")
        return {"phrase": phrase, "resolved": resolved}, str(resolved)

    async def _validate(self):
        return {"validated": True}, ""

    async def _check_health(self) -> HealthReport:
        return HealthReport(HealthStatus.OK, self.name, "pure functions, always available")


__all__ = ["MemoryAdapter", "SummariserAdapter", "FinExAdapter", "InternalAdapter"]
