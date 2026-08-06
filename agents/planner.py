"""
Planner Agent
Decomposes natural language requests into structured task DAGs
using ReAct-style reasoning. Fully local via Ollama.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.llm_client import OllamaClient
from config.models import MemoryItem, Subtask, TaskPlan
from config.settings import OLLAMA_CHAT_MODEL


SYSTEM_PROMPT = """You are the Planner agent for Jarvis, an AI executive assistant.
Your role is to decompose user requests into structured, executable subtasks.

Use ReAct reasoning:
1. THOUGHT: What does the user actually want?
2. OBSERVATION: What context and memories are relevant?
3. THOUGHT: What are the atomic steps needed?
4. OUTPUT: Structured JSON plan

Available agents and their actions:
- calendar:   create_event, search_events, check_conflicts, delete_event
- email:      send_email, read_emails, draft_email, search_emails
- reminder:   set_reminder, list_reminders, cancel_reminder
- notes:      create_note, search_notes, update_note
- websearch:  search_web, scrape_page
- research:   deep_research, summarise_research
- news:       get_headlines, search_news
- weather:    get_current, get_forecast
- mac:        open_app, set_volume (level 0-100), set_brightness (level 0-100), get_clipboard, send_notification
- spotify:    play_track, pause, skip, search_tracks, get_now_playing
- file:       find_file, read_file, move_file
- document:   extract_text, summarise_document
- summariser: summarise_text
- memory:     retrieve_context, store_fact
- finex:      chat   (params: {"question": "<finance question>", "company": "Bestway Cement"})

Output MUST be valid JSON:
{
  "intent": "schedule_meeting",
  "reasoning": "Step-by-step thought process using ReAct...",
  "subtasks": [
    {
      "id": "subtask_1",
      "action": "retrieve_context",
      "agent": "memory",
      "params": {"query": "user meeting preferences"},
      "depends_on": []
    },
    {
      "id": "subtask_2",
      "action": "check_conflicts",
      "agent": "calendar",
      "params": {"datetime": "{subtask_1.result.preferred_time}"},
      "depends_on": ["subtask_1"]
    },
    {
      "id": "subtask_3",
      "action": "create_event",
      "agent": "calendar",
      "params": {"title": "Meeting", "start_time": "{subtask_2.result.datetime}"},
      "depends_on": ["subtask_2"]
    }
  ]
}

Rules:
- Use {subtask_id.result.field} template syntax for dependency injection
- Keep subtasks atomic — one action per subtask
- Always start with memory retrieval when context helps
- Use ISO 8601 for all datetimes
- Maximum 8 subtasks per plan
"""


class PlannerAgent:
    """
    Transforms natural language into a structured DAG of subtasks.

    Key improvement over the original: runs fully locally on Ollama
    and produces validated, dependency-aware plans with ReAct reasoning
    traces that are stored and surfaced in the dissertation benchmarks.
    """

    def __init__(self, llm_client: OllamaClient | None = None):
        self.llm = llm_client or OllamaClient()
        print(f"🎯 PlannerAgent ready — model: {self.llm.model}")

    async def plan(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
        memory_context: Optional[List[MemoryItem]] = None,
        model_override: Optional[str] = None,
    ) -> TaskPlan:
        """
        Create a structured task plan using ReAct reasoning.

        Args:
            user_request:   Natural language task
            context:        Current datetime, timezone, user_id
            memory_context: Relevant memories retrieved by MemoryAgent
            model_override: Use a specific model (for benchmarking)

        Returns:
            TaskPlan with subtasks and dependency graph
        """
        ctx = context or {
            "current_datetime": datetime.now().isoformat(),
            "timezone": "Europe/London",
            "user_id": "user_001",
        }

        user_prompt = self._build_prompt(user_request, ctx, memory_context)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        model = model_override or self.llm.model
        data = await self.llm.chat_json(messages, model=model)

        plan = self._parse(data, user_request, model)
        print(
            f"🎯 Plan created: {plan.intent} — "
            f"{len(plan.subtasks)} subtasks"
        )
        return plan

    # ── Prompt building ────────────────────────────────────────────────────

    def _build_prompt(
        self,
        request: str,
        context: Dict[str, Any],
        memories: Optional[List[MemoryItem]],
    ) -> str:
        mem_block = ""
        if memories:
            mem_block = "\n\nRelevant memories:\n" + "\n".join(
                f"  - [{m.memory_type.value}] {m.content} "
                f"(relevance: {m.relevance_score:.2f})"
                for m in memories
            )

        return (
            f'User request: "{request}"\n\n'
            f"Current datetime: {context.get('current_datetime')}\n"
            f"Timezone: {context.get('timezone', 'UTC')}\n"
            f"User ID: {context.get('user_id', 'unknown')}"
            f"{mem_block}\n\n"
            "Create a task plan for this request."
        )

    # ── Parsing ────────────────────────────────────────────────────────────

    def _parse(
        self,
        data: Dict[str, Any],
        user_request: str,
        model: str,
    ) -> TaskPlan:
        task_id = f"task_{uuid.uuid4().hex[:10]}"

        subtasks: List[Subtask] = []
        for i, st in enumerate(data.get("subtasks", [])):
            subtasks.append(
                Subtask(
                    id=st.get("id", f"subtask_{i+1}"),
                    action=st.get("action", "unknown"),
                    agent=st.get("agent", "executor"),
                    params=st.get("params", {}),
                    depends_on=st.get("depends_on", []),
                )
            )

        return TaskPlan(
            task_id=task_id,
            user_request=user_request,
            intent=data.get("intent", "unknown"),
            subtasks=subtasks,
            reasoning=data.get("reasoning", ""),
            created_at=datetime.now(),
            model_used=model,
        )
