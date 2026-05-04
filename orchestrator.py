"""
Jarvis Orchestrator
The central coordinator. Receives a user request and drives the full
pipeline: Router → Memory → Planner → Critic → Executor → Evaluator.

This is what makes Jarvis a proper Multi-Agent System.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.critic import CriticAgent
from agents.evaluator import EvaluatorAgent
from agents.planner import PlannerAgent
from agents.router import RouterAgent
from agents.summariser import SummariserAgent
from tools.contacts import ContactBook
from tools.email_composer import EmailComposer, EmailDraft
from agents.calendar_agent import CalendarAgent
from agents.gmail_agent import GmailAgent
from config.llm_client import OllamaClient
from config.models import (
    AgentRole,
    JarvisResponse,
    MemoryType,
    Subtask,
    TaskPlan,
    TaskStatus,
)
from config.settings import OLLAMA_CHAT_MODEL
from memory.memory_agent import MemoryAgent
from tools.document import DocumentTool
from tools.sports import SportsTool
from tools.markets import MarketsTool
from tools.prayer_times import PrayerTimesTool
from tools.briefing import BriefingHandler
from tools.mac_control import MacControlTool
from tools.news import NewsTool
from tools.spotify import SpotifyTool
from tools.weather import WeatherTool
from tools.web_search import WebSearchTool
from tools.file_manager import FileManagerTool, PendingFileOp
from tools.reminders import ReminderStore

MAX_REPLAN_ATTEMPTS = 2


class JarvisOrchestrator:
    """
    Coordinates all Jarvis agents to execute user requests.

    Pipeline per request:
    1.  Router    — classify intent, decide primary agent
    2.  Memory    — retrieve relevant context
    3.  Planner   — decompose into subtask DAG
    4.  Critic    — review plan quality, trigger replan if needed
    5.  Executor  — run subtasks respecting dependency order
    6.  Critic    — review results
    7.  Evaluator — score and persist benchmark data
    8.  Memory    — store episodic memory of this interaction
    """

    def __init__(self, model: str = OLLAMA_CHAT_MODEL):
        self.model = model

        # Shared LLM client (all agents can use same instance)
        self.llm = OllamaClient(model=model)

        # Core agents
        self.router    = RouterAgent(self.llm)
        self.memory    = MemoryAgent(self.llm)
        self.planner   = PlannerAgent(self.llm)
        self.critic    = CriticAgent(self.llm)
        self.evaluator = EvaluatorAgent()
        self.summariser = SummariserAgent(self.llm)
        self.calendar  = CalendarAgent()
        self.gmail     = GmailAgent()
        self.contacts  = ContactBook()
        self.composer  = EmailComposer(self.llm)
        # Pending confirmation states
        self._pending_email: EmailDraft | None = None
        self._pending_meeting: dict | None = None
        self._pending_file_op: PendingFileOp | None = None

        # Tool instances
        self.weather    = WeatherTool()
        self.websearch  = WebSearchTool()
        self.news       = NewsTool()
        self.mac        = MacControlTool()
        self.spotify    = SpotifyTool(llm=self.llm)
        self.document   = DocumentTool()
        self.sports     = SportsTool()
        self.markets    = MarketsTool()
        self.prayer     = PrayerTimesTool()
        self.briefing   = BriefingHandler()
        self.files      = FileManagerTool()
        self.reminders  = ReminderStore()

        print(f"\n🤖 Jarvis Orchestrator ready — model: {model}")
        print(f"   Agents: Router, Memory, Planner, Critic, Evaluator, Summariser, Calendar, Gmail")
        print(f"   Tools:  Weather, WebSearch, News, Mac, Spotify, Document, FileManager\n")

    # ── Main entry point ───────────────────────────────────────────────────

    async def handle(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
        _routing=None,  # pre-computed RouterDecision — skips router step when provided
    ) -> JarvisResponse:
        """
        Handle a user request end-to-end.

        Args:
            user_request:   Natural language request from user
            context:        Optional dict with datetime, timezone, user_id
            model_override: Force a specific model (for benchmarking)

        Returns:
            JarvisResponse with message, evaluation, and task data
        """
        start_time = time.time()
        model = model_override or self.model

        print(f"\n{'='*60}")
        print(f"📨 Request: {user_request}")
        print(f"   Model: {model}")
        print(f"{'='*60}")

        ctx = context or {
            "current_datetime": datetime.now().isoformat(),
            "timezone": "Europe/London",
            "user_id": "user_001",
        }

        def _t(label: str, t0: float):
            elapsed = time.time() - t0
            print(f"[JARVIS] ⏱  {label}: {elapsed:.2f}s")
            return elapsed

        try:
            # ── Step 1: Route ──────────────────────────────────────────────
            if _routing is not None:
                routing = _routing
                print(f"🔀 Router skipped — reusing pre-computed routing ({routing.primary_agent.value}, tier {routing.tier})")
            else:
                t0 = time.time()
                routing = await self.router.route(user_request)
                _t("router", t0)
            tier = routing.tier

            # ── Tier 1: tool-only — skip memory, run shortcut, return fast ─
            if tier == 1:
                t0 = time.time()
                shortcut = await self._try_shortcut(routing.primary_agent, user_request)
                _t("tool_shortcut", t0)
                if shortcut is not None:
                    print(f"[JARVIS] ⚡ Tier 1 complete — total: {time.time()-start_time:.2f}s")
                    return shortcut

            # ── Step 2: Memory retrieval (Tier 2 + 3 only) ────────────────
            t0 = time.time()
            memories = await self.memory.retrieve(user_request)
            _t("memory_retrieve", t0)
            print(f"🧠 Retrieved {len(memories)} relevant memories")

            # ── Tier 1 fallback: shortcut missed, treat as Tier 2 ─────────
            if tier == 1:
                t0 = time.time()
                shortcut = await self._try_shortcut(routing.primary_agent, user_request)
                _t("tool_shortcut_fallback", t0)
                if shortcut is not None:
                    return shortcut
                tier = 2  # escalate

            # ── Tier 2: single LLM call, skip Planner/Critic/Evaluator ────
            if tier == 2:
                t0 = time.time()
                context_str = ""
                agent = routing.primary_agent
                if agent == AgentRole.WEBSEARCH:
                    data = await self.websearch.search(user_request)
                    context_str = self.websearch.format_results(data)
                    _t("websearch_tool", t0)
                elif agent == AgentRole.NEWS:
                    data = await self.news.get_headlines(query=user_request, max_items=5)
                    context_str = self.news.format_headlines(data)
                    _t("news_tool", t0)

                user_content = (
                    f"Context:\n{context_str}\n\nUser: {user_request}"
                    if context_str else user_request
                )
                msgs = [{"role": "user", "content": user_content}]

                t0 = time.time()
                llm_response = await self.llm.chat(msgs, max_tokens=200)
                _t("llm_single_call", t0)

                total_ms = (time.time() - start_time) * 1000
                print(f"[JARVIS] ⚡ Tier 2 complete — total: {total_ms/1000:.2f}s")
                asyncio.ensure_future(self.memory.store_task_result(
                    user_request=user_request, intent=routing.primary_agent.value,
                    success=True, summary=llm_response[:100]
                ))
                return JarvisResponse(
                    success=True, message=llm_response,
                    latency_ms=total_ms,
                )

            # ── Tier 3: full pipeline ──────────────────────────────────────
            # Short-circuit for deterministic tools even in Tier 3
            t0 = time.time()
            shortcut = await self._try_shortcut(routing.primary_agent, user_request)
            _t("shortcut_check", t0)
            if shortcut is not None:
                return shortcut

            # ── Step 3: Plan ───────────────────────────────────────────────
            t0 = time.time()
            plan = await self.planner.plan(
                user_request, ctx, memories, model_override=model
            )
            _t("planner", t0)

            # ── Step 4: Critic ─────────────────────────────────────────────
            _needs_critic = (
                routing.confidence < 0.82 or
                len(plan.subtasks) > 2 or
                any(a in routing.primary_agent.value for a in ["planner", "research"])
            )
            planning_score = 0.8

            if _needs_critic:
                t0 = time.time()
                plan_verdict = await self.critic.review_plan(plan)
                planning_score = plan_verdict.score
                _t("critic_plan", t0)
                replan_attempts = 0
                while plan_verdict.replan_needed and replan_attempts < MAX_REPLAN_ATTEMPTS:
                    replan_attempts += 1
                    print(f"🔄 Replanning (attempt {replan_attempts})...")
                    feedback_ctx = {
                        **ctx,
                        "critic_feedback": "; ".join(plan_verdict.issues),
                        "critic_suggestions": "; ".join(plan_verdict.suggestions),
                    }
                    t0 = time.time()
                    plan = await self.planner.plan(
                        user_request, feedback_ctx, memories, model_override=model
                    )
                    plan.replan_count = replan_attempts
                    plan_verdict = await self.critic.review_plan(plan)
                    planning_score = max(planning_score, plan_verdict.score)
                    _t(f"replan_{replan_attempts}", t0)
            else:
                print(f"⚡ Critic skipped — high confidence ({routing.confidence:.2f})")

            # ── Step 5: Execute ────────────────────────────────────────────
            t0 = time.time()
            results = await self._execute_dag(plan, routing.primary_agent)
            _t("execute_dag", t0)

            # ── Step 6: Critic result review ───────────────────────────────
            if _needs_critic:
                t0 = time.time()
                result_verdict = await self.critic.review_result(plan, results)
                _t("critic_result", t0)

            # ── Step 7: Evaluate ───────────────────────────────────────────
            t0 = time.time()
            evaluation = self.evaluator.evaluate(
                plan, results, start_time, planning_score=planning_score
            )
            _t("evaluator", t0)

            # ── Step 8: Store episodic memory ──────────────────────────────
            asyncio.ensure_future(self.memory.store_task_result(
                user_request=user_request,
                intent=plan.intent,
                success=evaluation.success,
                summary=evaluation.feedback,
            ))

            # ── Build response ─────────────────────────────────────────────
            t0 = time.time()
            message = self._build_response_message(
                user_request, plan, results, routing.primary_agent
            )
            _t("build_response", t0)

            total_ms = (time.time() - start_time) * 1000
            print(f"[JARVIS] ✅ Tier 3 complete — total: {total_ms/1000:.2f}s")

            return JarvisResponse(
                success=evaluation.success,
                message=message,
                task_plan=plan.to_dict(),
                evaluation=evaluation.to_dict(),
                latency_ms=evaluation.latency_ms,
            )

        except Exception as exc:
            latency_ms = (time.time() - start_time) * 1000
            print(f"❌ Orchestrator error: {exc}")
            import traceback
            traceback.print_exc()
            return JarvisResponse(
                success=False,
                message=f"I encountered an error: {exc}",
                error=str(exc),
                latency_ms=latency_ms,
            )

    # ── Streaming entry point ─────────────────────────────────────────────

    async def handle_stream(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
    ):
        """
        Async generator version of handle().
        Yields dicts: {"type":"thinking"}, {"type":"chunk","text":"..."}, {"type":"response",...}

        Tier 1 → single {"type":"response"} (instant, no LLM)
        Tier 2 → {"type":"thinking"} then streamed {"type":"chunk"} tokens then {"type":"response"}
        Tier 3 → {"type":"thinking"} then full pipeline result as {"type":"response"}
        """
        start_time = time.time()
        model = model_override or self.model
        ctx = context or {
            "current_datetime": datetime.now().isoformat(),
            "timezone": "Europe/London",
            "user_id": "user_001",
        }

        def _t(label: str, t0: float):
            elapsed = time.time() - t0
            print(f"[JARVIS] ⏱  {label}: {elapsed:.2f}s")

        try:
            # Router (always needed — 1b model, fast)
            t0 = time.time()
            routing = await self.router.route(user_request)
            _t("router", t0)
            tier = routing.tier

            # ── Tier 1: instant tool response ─────────────────────────────
            if tier == 1:
                t0 = time.time()
                shortcut = await self._try_shortcut(routing.primary_agent, user_request)
                _t("tool_shortcut", t0)
                if shortcut is not None:
                    print(f"[JARVIS] ⚡ Stream Tier 1 — {(time.time()-start_time):.2f}s")
                    yield {
                        "type": "response",
                        "message": shortcut.message,
                        "success": shortcut.success,
                        "latency_ms": (time.time() - start_time) * 1000,
                    }
                    return
                tier = 2  # escalate if no shortcut matched

            # ── Tier 2: stream LLM response token by token ────────────────
            if tier == 2:
                yield {"type": "thinking"}

                context_str = ""
                agent = routing.primary_agent
                if agent == AgentRole.WEBSEARCH:
                    t0 = time.time()
                    data = await self.websearch.search(user_request)
                    context_str = self.websearch.format_results(data)
                    _t("websearch_tool", t0)
                elif agent == AgentRole.NEWS:
                    t0 = time.time()
                    data = await self.news.get_headlines(query=user_request, max_items=5)
                    context_str = self.news.format_headlines(data)
                    _t("news_tool", t0)

                user_content = (
                    f"Context:\n{context_str}\n\nUser: {user_request}"
                    if context_str else user_request
                )
                msgs = [{"role": "user", "content": user_content}]

                full_text = ""
                t0 = time.time()
                async for chunk in self.llm.chat_stream(msgs, model=model, max_tokens=200):
                    full_text += chunk
                    yield {"type": "chunk", "text": chunk}
                _t("llm_stream", t0)

                total_ms = (time.time() - start_time) * 1000
                print(f"[JARVIS] ⚡ Stream Tier 2 — {total_ms/1000:.2f}s")
                asyncio.ensure_future(self.memory.store_task_result(
                    user_request, routing.primary_agent.value, True, full_text[:100]
                ))
                yield {
                    "type": "response",
                    "message": full_text,
                    "success": True,
                    "latency_ms": total_ms,
                }
                return

            # ── Tier 3: full pipeline, show thinking indicator ─────────────
            yield {"type": "thinking"}

            # Pass pre-computed routing to avoid double-routing (saves ~2-3s)
            response = await self.handle(user_request, context=ctx, model_override=model_override, _routing=routing)
            yield {
                "type": "response",
                "message": response.message,
                "success": response.success,
                "latency_ms": response.latency_ms,
            }

        except Exception as exc:
            print(f"❌ Stream error: {exc}")
            import traceback; traceback.print_exc()
            yield {
                "type": "response",
                "message": f"I encountered an error: {exc}",
                "success": False,
                "latency_ms": (time.time() - start_time) * 1000,
            }

    # ── DAG Execution ──────────────────────────────────────────────────────

    async def _execute_dag(
        self,
        plan: TaskPlan,
        primary_agent: AgentRole,
    ) -> Dict[str, Any]:
        """
        Execute subtasks in dependency order (topological sort).
        Subtasks whose dependencies have all completed are eligible to run.
        """
        completed: Dict[str, Any] = {}
        pending = {st.id: st for st in plan.subtasks}
        max_iterations = len(pending) * 2

        for _ in range(max_iterations):
            if not pending:
                break

            executed_this_round = []

            for st_id, subtask in list(pending.items()):
                deps_done = all(d in completed for d in subtask.depends_on)
                if not deps_done:
                    # Check for failed deps → block this subtask
                    failed_deps = [
                        d for d in subtask.depends_on
                        if d in completed and not completed[d].get("success")
                    ]
                    if failed_deps:
                        subtask.status = TaskStatus.BLOCKED
                        completed[st_id] = {
                            "success": False,
                            "error": f"Blocked by failed deps: {failed_deps}",
                        }
                        executed_this_round.append(st_id)
                    continue

                # Execute
                subtask.started_at = datetime.now()
                subtask.status = TaskStatus.IN_PROGRESS
                result = await self._dispatch(subtask, completed, primary_agent)
                subtask.completed_at = datetime.now()
                subtask.status = (
                    TaskStatus.COMPLETED if result.get("success")
                    else TaskStatus.FAILED
                )
                subtask.result = result
                completed[st_id] = result
                executed_this_round.append(st_id)

                icon = "✅" if result.get("success") else "❌"
                print(f"   {icon} [{st_id}] {subtask.agent}.{subtask.action}")

            for st_id in executed_this_round:
                pending.pop(st_id, None)

            # Circular dependency check
            if not executed_this_round and pending:
                print("⚠️  Circular dependency detected")
                for st_id in pending:
                    completed[st_id] = {
                        "success": False, "error": "Circular dependency"
                    }
                break

        return completed

    # ── Dispatcher ────────────────────────────────────────────────────────

    async def _dispatch(
        self,
        subtask: Subtask,
        completed: Dict[str, Any],
        primary_agent: AgentRole,
    ) -> Dict[str, Any]:
        """Route a subtask to the correct tool/agent based on agent field."""
        params = self._inject_deps(subtask.params, subtask.depends_on, completed)
        agent = subtask.agent.lower()
        action = subtask.action.lower()

        try:
            # ── Memory ──────────────────────────────────────────────────────
            if agent == "memory":
                if action == "retrieve_context":
                    mems = await self.memory.retrieve(params.get("query", ""))
                    return {"success": True, "result": [m.content for m in mems]}
                elif action == "store_fact":
                    await self.memory.store(
                        params.get("content", ""),
                        memory_type=MemoryType.SEMANTIC,
                    )
                    return {"success": True}

            # ── Weather ─────────────────────────────────────────────────────
            elif agent == "weather":
                if action == "get_current":
                    data = await self.weather.get_current()
                    return {"success": True, "result": data,
                            "message": self.weather.format_current(data)}
                elif action == "get_forecast":
                    data = await self.weather.get_forecast()
                    return {"success": True, "result": data,
                            "message": self.weather.format_forecast(data)}

            # ── Web search ──────────────────────────────────────────────────
            elif agent == "websearch":
                data = await self.websearch.search(params.get("query", ""))
                return {"success": data.get("success", False), "result": data,
                        "message": self.websearch.format_results(data)}

            # ── News ────────────────────────────────────────────────────────
            elif agent == "news":
                data = await self.news.get_headlines(
                    source=params.get("source", "bbc"),
                    topic=params.get("topic"),
                    max_items=params.get("max_items", 5),
                )
                return {"success": data.get("success", False), "result": data,
                        "message": self.news.format_headlines(data)}

            # ── Mac control ─────────────────────────────────────────────────
            elif agent == "mac":
                if action == "open_app":
                    return await self.mac.open_app(params.get("app", ""))
                elif action == "set_volume":
                    return await self.mac.set_volume(int(params.get("level", 50)))
                elif action == "set_brightness":
                    return await self.mac.set_brightness(float(params.get("level", 0.5)))
                elif action == "send_notification":
                    return await self.mac.send_notification(
                        params.get("message", ""),
                        params.get("title", "Jarvis"),
                    )
                elif action == "get_battery":
                    return await self.mac.get_battery()
                elif action == "lock_screen":
                    return await self.mac.lock_screen()

            # ── Spotify ─────────────────────────────────────────────────────
            elif agent == "spotify":
                if action == "search_tracks":
                    return await self.spotify.search(params.get("query", ""))
                elif action == "play_track":
                    return await self.spotify.play(params.get("uri"))
                elif action == "pause":
                    return await self.spotify.pause()
                elif action == "skip":
                    return await self.spotify.skip()
                elif action == "set_volume":
                    return await self.spotify.set_volume(int(params.get("level", 50)))
                elif action == "get_now_playing":
                    data = await self.spotify.get_now_playing()
                    return {"success": True, "result": data,
                            "message": self.spotify.format_now_playing(data)}

            # ── Document ────────────────────────────────────────────────────
            elif agent in ("document", "file"):
                data = await self.document.extract(params.get("path", ""))
                return {"success": data.get("success", False), "result": data}

            # ── Summariser ──────────────────────────────────────────────────
            elif agent == "summariser":
                text = params.get("text", "")
                summary = await self.summariser.summarise(
                    text, max_words=params.get("max_words", 150)
                )
                return {"success": True, "result": summary, "message": summary}

            # ── Temporal resolution ─────────────────────────────────────────
            elif action == "resolve_temporal":
                resolved = self._resolve_temporal(params.get("phrase", ""))
                return {"success": True, "result": resolved}

            # ── Validation ──────────────────────────────────────────────────
            elif action == "validate_output":
                return {"success": True, "result": {"validated": True}}

            # ── Calendar ────────────────────────────────────────────────
            elif agent == "calendar":
                if action == "create_event":
                    result = await self.calendar.create_event(
                        title=params.get("title", "Untitled"),
                        start_time=params.get("start_time", ""),
                        end_time=params.get("end_time", ""),
                        attendees=params.get("attendees"),
                        description=params.get("description"),
                        location=params.get("location"),
                    )
                    return result
                elif action in ("search_events", "get_events"):
                    result = await self.calendar.search_events(
                        start_date=params.get("start_date"),
                        end_date=params.get("end_date"),
                        query=params.get("query"),
                    )
                    return result
                elif action == "check_conflicts":
                    result = await self.calendar.check_conflicts(
                        params.get("start_time", ""),
                        params.get("end_time", ""),
                    )
                    return result
                elif action == "delete_event":
                    return await self.calendar.delete_event(params.get("event_id", ""))

            # ── Gmail ────────────────────────────────────────────────────────
            elif agent == "email":
                if action in ("read_emails", "get_inbox"):
                    result = await self.gmail.get_inbox(
                        max_results=params.get("max_results", 5),
                        query=params.get("query", "is:unread"),
                    )
                    return result
                elif action == "send_email":
                    result = await self.gmail.send_email(
                        to=params.get("to", ""),
                        subject=params.get("subject", ""),
                        body=params.get("body", ""),
                        cc=params.get("cc"),
                    )
                    return result
                elif action == "draft_email":
                    result = await self.gmail.draft_email(
                        to=params.get("to", ""),
                        subject=params.get("subject", ""),
                        body=params.get("body", ""),
                    )
                    return result
                elif action == "search_emails":
                    result = await self.gmail.search_emails(
                        query=params.get("query", ""),
                        max_results=params.get("max_results", 5),
                    )
                    return result

            # ── Reminders ───────────────────────────────────────────────────
            elif agent == "reminder":
                if action == "add_reminder":
                    rid = self.reminders.add(
                        title=params.get("title", "Reminder"),
                        body=params.get("body", ""),
                        due_at=params.get("due_at"),
                        recurring_minutes=params.get("recurring_minutes"),
                        offset_minutes=params.get("offset_minutes"),
                    )
                    due = params.get("due_at") or f"in {params.get('offset_minutes', 5)} minutes"
                    return {"success": True, "id": rid,
                            "message": f"Reminder set: '{params.get('title', 'Reminder')}' — {due}"}
                elif action == "list_reminders":
                    pending = self.reminders.list_pending()
                    return {"success": True, "result": pending,
                            "message": self.reminders.format_list(pending)}
                elif action == "complete_reminder":
                    ok = self.reminders.complete(params.get("id", ""))
                    return {"success": ok, "message": "Reminder marked complete." if ok else "Reminder not found."}
                elif action == "delete_reminder":
                    ok = self.reminders.delete(params.get("id", ""))
                    return {"success": ok, "message": "Reminder deleted." if ok else "Reminder not found."}

            # ── Fallback ────────────────────────────────────────────────────
            else:
                return {
                    "success": False,
                    "error": f"Unknown agent/action: {agent}.{action}",
                }

        except Exception as exc:
            return {"success": False, "error": str(exc)}
        # Safety: should never reach here
        return {"success": False, "error": f"Unhandled: {agent}.{action}"}

    # ── Dependency injection ───────────────────────────────────────────────

    def _inject_deps(
        self,
        params: Dict[str, Any],
        depends_on: List[str],
        completed: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Replace {subtask_id.result.field} templates with actual values."""
        if not depends_on:
            return params

        enriched = params.copy()
        for key, value in enriched.items():
            if not isinstance(value, str) or "{" not in value:
                continue
            for dep_id in depends_on:
                if dep_id not in completed:
                    continue
                dep_result = completed[dep_id].get("result", {})
                if isinstance(dep_result, dict):
                    for field, field_val in dep_result.items():
                        placeholder = f"{{{dep_id}.result.{field}}}"
                        if placeholder in value:
                            enriched[key] = value.replace(placeholder, str(field_val))
        return enriched

    # ── Temporal resolution ────────────────────────────────────────────────

    def _resolve_temporal(self, phrase: str) -> Dict[str, str]:
        """Convert natural language time phrases to ISO datetime with timezone."""
        import re
        from datetime import timedelta, timezone
        from zoneinfo import ZoneInfo

        # Use local timezone
        try:
            local_tz = ZoneInfo("Europe/London")
        except Exception:
            local_tz = timezone.utc

        now = datetime.now(local_tz)
        target = now
        p = phrase.lower()

        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
        }

        # Date resolution
        if "tomorrow" in p:
            target = now + timedelta(days=1)
        elif "next week" in p:
            target = now + timedelta(days=7)
        elif "today" in p or "tonight" in p:
            target = now
        else:
            for day_name, day_num in day_map.items():
                if day_name in p:
                    days_ahead = (day_num - now.weekday()) % 7 or 7
                    target = now + timedelta(days=days_ahead)
                    break

        # Time resolution — find the LAST time mention in the phrase
        # to avoid picking up times from earlier context
        time_matches = list(re.finditer(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", p))
        if not time_matches:
            # Try without am/pm but only if a clear time word is nearby
            time_matches = list(re.finditer(r"at\s+(\d{1,2})(?::(\d{2}))?(?!\s*(?:am|pm))", p))
            if time_matches:
                # Reformat match groups
                m = time_matches[-1]
                h = int(m.group(1))
                mins = int(m.group(2) or 0)
                # Default: if hour < 8, assume pm (e.g. "at 2" = 2pm)
                if h < 8:
                    h += 12
                target = target.replace(hour=h, minute=mins, second=0, microsecond=0)
            else:
                # No time found — default to 9am
                target = target.replace(hour=9, minute=0, second=0, microsecond=0)
        else:
            m = time_matches[-1]
            h = int(m.group(1))
            mins = int(m.group(2) or 0)
            period = m.group(3)
            if period == "pm" and h != 12:
                h += 12
            elif period == "am" and h == 12:
                h = 0
            target = target.replace(hour=h, minute=mins, second=0, microsecond=0)

        end = target + timedelta(hours=1)

        return {
            "datetime": target.isoformat(),
            "date": target.date().isoformat(),
            "time": target.strftime("%H:%M"),
            "end_datetime": end.isoformat(),
            "timezone": str(local_tz),
        }

    # ── Response building ─────────────────────────────────────────────────

    def _build_response_message(
        self,
        request: str,
        plan: TaskPlan,
        results: Dict[str, Any],
        primary_agent: AgentRole,
    ) -> str:
        """Build a natural language response from execution results."""
        messages = []
        for result in results.values():
            if result.get("success") and result.get("message"):
                messages.append(result["message"])

        if messages:
            return " ".join(messages)

        # Fallback: summarise what happened
        successes = sum(1 for r in results.values() if r.get("success"))
        total = len(results)
        return (
            f"Completed {successes}/{total} steps for: {plan.intent.replace('_', ' ')}."
        )


    # ── Shortcut handler ──────────────────────────────────────────────────

    async def _try_shortcut(
        self,
        primary_agent: AgentRole,
        user_request: str,
    ):
        """
        Bypass LLM planning for deterministic single-tool intents.
        Returns a JarvisResponse if handled, None otherwise.
        """
        import time as _time
        import re
        start = _time.time()

        # Also catch weather requests the router misclassified
        req_lower = user_request.lower()

        # ── Battery (must be before weather — "temperature" could collide) ──
        if any(kw in req_lower for kw in ["battery", "battery level", "how much battery"]):
            import time as _tbat2
            _sbat2 = _tbat2.time()
            result = await self.mac.get_battery()
            if result.get("success"):
                pct = result.get("battery_pct", "unknown")
                msg = f"Battery is at {pct}%."
            else:
                msg = "Could not read battery level."
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tbat2.time()-_sbat2)*1000)

        weather_keywords = ["weather", "temperature", "forecast", "humid", "rain", "sunny", "cloudy", "wind speed"]
        is_weather_request = (
            primary_agent == AgentRole.WEATHER or
            any(kw in req_lower for kw in weather_keywords)
        )

        if is_weather_request:
            req = req_lower
            is_forecast = any(w in req for w in ["forecast", "this week", "next week", "7 day", "seven day", "weekly"])

            # Detect if user specified a location other than the default
            location = self._extract_location(user_request)

            if location:
                if is_forecast:
                    data = await self.weather.get_forecast_for_location(location)
                else:
                    data = await self.weather.get_current_for_location(location)
            else:
                if is_forecast:
                    data = await self.weather.get_forecast()
                else:
                    data = await self.weather.get_current()

            msg = self.weather.format_forecast(data) if is_forecast else self.weather.format_current(data)
            asyncio.ensure_future(self.memory.store_task_result(
                user_request, "get_weather", data.get("success", False), msg[:100]
            ))
            print(f"Jarvis shortcut: weather — {(_time.time()-start)*1000:.0f}ms")
            return JarvisResponse(
                success=data.get("success", False),
                message=msg,
                latency_ms=(_time.time() - start) * 1000,
            )

        # News shortcut — smart category/source/topic detection
        news_keywords = [
            "news", "headlines", "latest news", "top stories", "breaking",
            "what's happening", "whats happening", "current events",
            "sports news", "tech news", "business news", "world news",
            "science news", "ai news", "uk news", "football news",
        ]
        if any(kw in req_lower for kw in news_keywords):
            import time as _t
            _s = _t.time()
            detailed = any(w in req_lower for w in ["detailed", "detail", "more info", "tell me more"])
            data = await self.news.get_headlines(
                query=user_request,
                max_items=6,
            )
            msg = self.news.format_headlines(data, detailed=detailed)
            asyncio.ensure_future(self.memory.store_task_result(user_request, "get_news", True, msg[:100]))
            print(f"Jarvis shortcut: news — {(_t.time()-_s)*1000:.0f}ms")
            return JarvisResponse(success=True, message=msg, latency_ms=(_t.time()-_s)*1000)

        # ── Early Spotify intercept — catches "play X" BEFORE sports routing ──
        _early_play = (
            req_lower.startswith("play ") and
            not any(kw in req_lower for kw in ["play premier", "play match", "play game",
                                                "play the game", "play the match"])
        )
        if _early_play:
            import time as _tep
            _sep = _tep.time()
            _query_ep = re.sub(r'^play\s+', '', user_request, flags=re.IGNORECASE).strip()
            _query_ep = re.sub(r'\s+on spotify$', '', _query_ep, flags=re.IGNORECASE).strip()
            if _query_ep and _query_ep.lower() not in ("music", "something", "anything", "spotify"):
                result = await self.spotify.play_by_name(_query_ep)
                msg = self.spotify.format_play_result(result)
            else:
                result = await self.spotify.play()
                msg = "Resumed playback." if result.get("success") else result.get("error", "Could not resume.")
            return JarvisResponse(success=result.get("success", False), message=msg,
                                  latency_ms=(_tep.time()-_sep)*1000)

        # ── Sports shortcuts ──────────────────────────────────────────────────
        sports_keywords = [
            "scores", "results", "fixtures", "standings", "table",
            "premier league", "champions league", "la liga", "serie a",
            "bundesliga", "nfl", "nba", "nhl", "mlb", "f1", "formula 1",
            "football scores", "football results", "basketball scores",
            "sports results", "match results", "game scores", "football score",
            "who won", "vs", "final score", "match score",
        ]
        # ── Calendar pre-check — must come before sports to avoid "call" collision ──
        _has_meeting = any(w in req_lower for w in ["meeting", "appointment", "event", "session"])
        _has_schedule = any(w in req_lower for w in [
            "schedule", "book", "create", "add", "set", "arrange",
            "block", "put", "plan", "organise", "organize", "new"
        ])
        _explicit_cal = any(kw in req_lower for kw in ["add to calendar", "calendar event", "add event"])
        if (_has_meeting and _has_schedule) or _explicit_cal:
            # Jump straight to calendar section below
            pass
        else:
            sports_context = any(kw in req_lower for kw in sports_keywords)
        from config.settings import FAVOURITE_TEAMS
        team_sports = [
            # Premier League
            "arsenal", "chelsea", "liverpool", "manchester", "spurs",
            "tottenham", "city", "united", "west ham", "newcastle",
            "aston villa", "brighton", "everton", "wolves", "fulham",
            "brentford", "crystal palace", "bournemouth", "ipswich",
            "leicester", "southampton",
            # European
            "real madrid", "barcelona", "atletico", "juventus", "inter",
            "ac milan", "napoli", "bayern", "dortmund", "psg",
            # NBA
            "lakers", "celtics", "warriors", "bulls", "heat", "nets",
            "knicks", "clippers", "suns", "bucks", "nuggets", "76ers",
            # NFL
            "patriots", "chiefs", "cowboys", "packers", "eagles",
            # Cricket
            "pakistan", "india", "england cricket", "australia cricket",
            "west indies", "south africa cricket", "new zealand cricket",
        ] + [t.lower() for t in FAVOURITE_TEAMS]
        team_mentioned = any(t in req_lower for t in team_sports)

        sports_context = any(kw in req_lower for kw in sports_keywords)
        # Exclude clear music/Spotify requests from sports routing
        _is_music_req = (
            req_lower.startswith("play ") or
            any(kw in req_lower for kw in ["play song", "play track", "play music", "by michael", "by drake",
                                            "by the weeknd", "by kanye", "by taylor", "by eminem",
                                            "on spotify", "spotify", "pause music", "skip track"])
        )
        if (sports_context or team_mentioned) and not ((_has_meeting and _has_schedule) or _explicit_cal) and not _is_music_req:
            import time as _tsp
            _ssp = _tsp.time()

            # Team → league mapping (used when detect_league returns nothing)
            _team_league_map = {
                # La Liga
                "real madrid": "la_liga", "barcelona": "la_liga", "atletico": "la_liga",
                "sevilla": "la_liga", "villarreal": "la_liga", "real sociedad": "la_liga",
                # Serie A
                "juventus": "serie_a", "inter": "serie_a", "ac milan": "serie_a",
                "napoli": "serie_a", "roma": "serie_a", "lazio": "serie_a",
                # Bundesliga
                "bayern": "bundesliga", "dortmund": "bundesliga", "leverkusen": "bundesliga",
                "leipzig": "bundesliga",
                # Ligue 1
                "psg": "ligue_1", "paris saint": "ligue_1", "marseille": "ligue_1",
                # NBA
                "lakers": "nba", "celtics": "nba", "warriors": "nba", "bulls": "nba",
                "heat": "nba", "nets": "nba", "knicks": "nba", "clippers": "nba",
                "suns": "nba", "bucks": "nba", "nuggets": "nba", "76ers": "nba",
                # Cricket
                "pakistan": "cricket", "india": "cricket", "west indies": "cricket",
                # Premier League teams stay as default
            }

            # Detect league from request, then from team name
            league_key = self.sports.detect_league(user_request)
            if not league_key:
                for team_kw, mapped_league in _team_league_map.items():
                    if team_kw in req_lower:
                        league_key = mapped_league
                        break

            # If team mentioned, search for that team
            if team_mentioned and not any(kw in req_lower for kw in ["table", "standings"]):
                if not league_key:
                    league_key = "premier_league"
                data = await self.sports.search_team(user_request, league_key)
                if data.get("success") and data.get("games"):
                    msg = self.sports.format_scores(data)
                else:
                    # Fallback to full league scores
                    data = await self.sports.get_scores(league_key or "premier_league")
                    msg = self.sports.format_scores(data)
            elif any(kw in req_lower for kw in ["table", "standings", "top of", "who is top", "who leads"]):
                if not league_key:
                    league_key = "premier_league"
                data = await self.sports.get_standings(league_key)
                msg = self.sports.format_standings(data)
            else:
                if not league_key:
                    league_key = "premier_league"
                data = await self.sports.get_scores(league_key)
                msg = self.sports.format_scores(data)

            asyncio.ensure_future(self.memory.store_task_result(user_request, "sports", True, msg[:100]))
            return JarvisResponse(success=True, message=msg, latency_ms=(_tsp.time()-_ssp)*1000)

        # News digest — all categories
        if any(kw in req_lower for kw in ["morning briefing", "daily briefing", "news digest", "all news", "full news"]):
            import time as _td
            _sd = _td.time()
            digest = await self.news.get_all_categories(max_per_category=2)
            lines = ["Your news digest:"]
            for cat, items in digest.get("digest", {}).items():
                if items:
                    lines.append(f"**{cat.title()}**")
                    for item in items:
                        lines.append(f"  • {item['title']}")
                    lines.append("")
            msg = chr(10).join(lines)
            return JarvisResponse(success=True, message=msg, latency_ms=(_td.time()-_sd)*1000)

        # List available news sources
        if any(kw in req_lower for kw in ["news sources", "available news", "what news sources"]):
            return JarvisResponse(success=True, message=self.news.list_sources())

        if primary_agent == AgentRole.NEWS:
            data = await self.news.get_headlines(max_items=5)
            msg = self.news.format_headlines(data)
            asyncio.ensure_future(self.memory.store_task_result(
                user_request, "get_news", True, msg[:100]
            ))
            return JarvisResponse(
                success=True,
                message=msg,
                latency_ms=(_time.time() - start) * 1000,
            )

        # Email shortcut — read inbox
        email_read_keywords = ["check my email", "read my email", "my inbox", "any emails", "new emails", "unread"]
        if any(kw in req_lower for kw in email_read_keywords):
            import time as _t2
            _s2 = _t2.time()
            result = await self.gmail.get_inbox(max_results=5)
            msg = result.get("message", "Could not read inbox.")
            asyncio.ensure_future(self.memory.store_task_result(user_request, "read_email", True, msg[:100]))
            return JarvisResponse(success=True, message=msg, latency_ms=(_t2.time()-_s2)*1000)

        # Email shortcut — confirmation check first
        confirm_keywords = ["yes send it", "yes", "send it", "confirm", "go ahead", "yeah send"]
        if self._pending_email and any(kw in req_lower for kw in confirm_keywords):
            import time as _tc
            _sc = _tc.time()
            draft = self._pending_email
            self._pending_email = None
            result = await self.gmail.send_email(
                to=draft.recipient_email,
                subject=draft.subject,
                body=draft.body,
            )
            msg = result.get("message", f"Email sent to {draft.recipient_email}")
            # Auto-save contact after successful send
            if result.get("success") and draft.recipient_email and draft.recipient_name:
                self.contacts.add(draft.recipient_name, draft.recipient_email)
            asyncio.ensure_future(self.memory.store_task_result(user_request, "send_email", result.get("success", False), msg[:100]))
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tc.time()-_sc)*1000)

        # Cancel pending email
        cancel_keywords = ["no", "cancel", "don't send", "do not send", "abort"]
        if self._pending_email and any(kw in req_lower for kw in cancel_keywords):
            self._pending_email = None
            return JarvisResponse(success=True, message="Email cancelled. No email was sent.")

        # ── Email address reply — MUST be first, before everything ─────────────
        # ── Email address reply — MUST be first, before everything ─────────────
        import re as _reemail
        # Strip markdown link format if present e.g. [email](mailto:email)
        _ur_clean = _reemail.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", user_request.strip())
        _email_m = _reemail.search(r"[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}", _ur_clean)
        _is_email_reply = _email_m and self._pending_email and getattr(self._pending_email, "needs_email", False)
        if _is_email_reply:
            import time as _teer
            _seer = _teer.time()
            self._pending_email.recipient_email = _email_m.group(0)
            self._pending_email.needs_email = False
            # Auto-save this new contact so we remember them next time
            if self._pending_email.recipient_name and self._pending_email.recipient_email:
                self.contacts.add(self._pending_email.recipient_name, self._pending_email.recipient_email)
            msg = self.composer.format_draft_for_confirmation(self._pending_email)
            return JarvisResponse(success=True, message=msg, latency_ms=(_teer.time()-_seer)*1000)

        # ── Edit pending email ─────────────────────────────────────────────────
        _edit_kws = ["edit", "change", "modify", "rewrite", "update", "tell her",
                     "tell him", "add that", "also say", "mention", "make it"]
        if self._pending_email and not getattr(self._pending_email, "needs_email", False) and any(kw in req_lower for kw in _edit_kws):
            import time as _tedit
            _sedit = _tedit.time()
            ep = "Edit this email body based on the instruction.\n\nCurrent body:\n" + self._pending_email.body + "\n\nInstruction: " + user_request + "\n\nReturn ONLY the updated body."
            try:
                nb = await self.llm.chat([{"role": "user", "content": ep}])
                self._pending_email.body = nb.strip()
                msg = "Updated! " + self.composer.format_draft_for_confirmation(self._pending_email)
            except Exception as e:
                msg = "Could not edit: " + str(e)
            return JarvisResponse(success=True, message=msg, latency_ms=(_tedit.time()-_sedit)*1000)


        # ── Morning Briefing ──────────────────────────────────────────────────
        if self.briefing.is_morning_briefing(user_request):
            import time as _tbf
            _sbf = _tbf.time()
            msg = await self._morning_briefing()
            return JarvisResponse(success=True, message=msg, latency_ms=(_tbf.time()-_sbf)*1000)
            intents = self.briefing.detect_intents(user_request)
            if len(intents) >= 2:
                import time as _tmq
                _smq = _tmq.time()
                msg = await self._handle_multi_query(user_request, intents)
                if msg:
                    return JarvisResponse(success=True, message=msg, latency_ms=(_tmq.time()-_smq)*1000)

        # ── File Manager — confirmation flow ──────────────────────────────────
        import time as _tfile
        _sfile = _tfile.time()

        # Handle pending file operation confirmation
        if self._pending_file_op:
            _low = req_lower.strip()
            if _low in ("confirm", "yes", "do it", "go ahead", "proceed", "ok"):
                op = self._pending_file_op
                self._pending_file_op = None
                if op.operation == "delete":
                    result = self.files.execute_delete(op)
                elif op.operation == "move":
                    result = self.files.execute_move(op)
                elif op.operation == "rename":
                    result = self.files.execute_rename(op)
                else:
                    result = {"success": False, "error": "Unknown operation"}
                msg = result.get("message", "Done.") if result.get("success") else f"Failed: {result.get('error')}"
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tfile.time()-_sfile)*1000)
            elif _low in ("cancel", "no", "stop", "abort", "nevermind", "never mind"):
                self._pending_file_op = None
                return JarvisResponse(success=True, message="Cancelled.",
                                      latency_ms=(_tfile.time()-_sfile)*1000)

        # Detect file intent keywords
        _file_kw = [
            "folder", "directory", "file", "files", "desktop", "documents", "downloads",
            "create folder", "make folder", "new folder", "create file", "new file",
            "delete file", "delete folder", "remove file", "remove folder",
            "rename", "move to", "find file", "search file", "list files",
            "show files", "browse", "what's on my desktop", "whats on my desktop",
            "what's on my documents", "whats on my documents",
            "show me what's on", "show me whats on", "show me my files",
            "what files", "what do i have on my", "list my files",
        ]
        _is_file_request = any(kw in req_lower for kw in _file_kw)

        if _is_file_request:
            parsed = self.files.parse_request(user_request)
            action = parsed.get("action")
            loc = parsed.get("location", "desktop")
            path = parsed.get("path")
            name = parsed.get("name")
            dest = parsed.get("destination")

            # ── List directory ──────────────────────────────────────────────
            if action == "list" or (not action and any(w in req_lower for w in ["list", "show", "browse", "what's on", "whats on"])):
                target = loc  # always use the location key ('desktop','documents','downloads','all')
                data = self.files.list_directory(target)
                msg = self.files.format_listing(data)
                return JarvisResponse(success=data.get("success", False), message=msg,
                                      latency_ms=(_tfile.time()-_sfile)*1000)

            # ── Search ──────────────────────────────────────────────────────
            elif action == "search" and path:
                data = self.files.search(path, location=loc)
                msg = self.files.format_search(data)
                return JarvisResponse(success=True, message=msg,
                                      latency_ms=(_tfile.time()-_sfile)*1000)

            # ── Read file ───────────────────────────────────────────────────
            elif action == "read" and path:
                data = self.files.read_file(path)
                if data.get("success"):
                    content = data.get("content", "")
                    trunc = "\n\n[File truncated at 50KB]" if data.get("truncated") else ""
                    # Let LLM summarise if large
                    if len(content) > 2000:
                        prompt = (
                            f"The user wants to read this file: {data['name']}\n\n"
                            f"Content:\n{content[:4000]}\n\n"
                            f"Give a brief summary of what this file contains, then show the first ~30 lines."
                        )
                        summary = await self.llm.chat([{"role": "user", "content": prompt}])
                        msg = summary.strip() + trunc
                    else:
                        msg = f"{data['display_path']}:\n\n{content}{trunc}"
                else:
                    msg = data.get("error", "Could not read file.")
                return JarvisResponse(success=data.get("success", False), message=msg,
                                      latency_ms=(_tfile.time()-_sfile)*1000)

            # ── Create folder ───────────────────────────────────────────────
            elif action == "create_folder" and name:
                # Build full path using detected location
                from pathlib import Path as _P
                from tools.file_manager import ALLOWED_ROOTS as _ROOTS
                target_root = _ROOTS.get(loc, _ROOTS["desktop"])
                full_path = str(target_root / name)
                data = self.files.create_folder(full_path)
                msg = data.get("message", data.get("error", "Could not create folder."))
                return JarvisResponse(success=data.get("success", False), message=msg,
                                      latency_ms=(_tfile.time()-_sfile)*1000)

            # ── Create file ─────────────────────────────────────────────────
            elif action == "create_file" and name:
                from pathlib import Path as _P
                from tools.file_manager import ALLOWED_ROOTS as _ROOTS
                target_root = _ROOTS.get(loc, _ROOTS["desktop"])
                full_path = str(target_root / name)
                data = self.files.create_file(full_path)
                msg = data.get("message", data.get("error", "Could not create file."))
                return JarvisResponse(success=data.get("success", False), message=msg,
                                      latency_ms=(_tfile.time()-_sfile)*1000)

            # ── Delete — requires approval ───────────────────────────────────
            elif action == "delete" and path:
                from tools.file_manager import ALLOWED_ROOTS as _ROOTS
                # Try direct resolve first, then search by name in detected location
                op = self.files.prepare_delete(path)
                if isinstance(op, dict):
                    # Not found directly — search by name in detected root
                    search_root = _ROOTS.get(loc, None)
                    found_path = None
                    search_roots = [search_root] if search_root else list(_ROOTS.values())
                    for root in search_roots:
                        for candidate in root.rglob("*"):
                            if candidate.name.lower() == path.lower() or \
                               candidate.name.lower().replace(" ", "") == path.lower().replace(" ", ""):
                                found_path = candidate
                                break
                        if found_path:
                            break
                    if found_path:
                        op = self.files.prepare_delete(str(found_path))
                    else:
                        return JarvisResponse(success=False,
                            message=f"Could not find \"{path}\" on your {loc}. Try: find {path}",
                            latency_ms=(_tfile.time()-_sfile)*1000)
                if isinstance(op, dict):
                    return JarvisResponse(success=False, message=op.get("error", "Could not prepare delete."),
                                          latency_ms=(_tfile.time()-_sfile)*1000)
                self._pending_file_op = op
                return JarvisResponse(success=True, message=op.summary(),
                                      latency_ms=(_tfile.time()-_sfile)*1000)

            # ── Rename — requires approval ───────────────────────────────────
            elif action == "rename" and path and name:
                op = self.files.prepare_rename(path, name)
                if isinstance(op, dict):
                    return JarvisResponse(success=False, message=op.get("error", "Could not prepare rename."),
                                          latency_ms=(_tfile.time()-_sfile)*1000)
                self._pending_file_op = op
                return JarvisResponse(success=True, message=op.summary(),
                                      latency_ms=(_tfile.time()-_sfile)*1000)

            # ── Move — requires approval ─────────────────────────────────────
            elif action == "move" and path and dest:
                op = self.files.prepare_move(path, dest)
                if isinstance(op, dict):
                    return JarvisResponse(success=False, message=op.get("error", "Could not prepare move."),
                                          latency_ms=(_tfile.time()-_sfile)*1000)
                self._pending_file_op = op
                return JarvisResponse(success=True, message=op.summary(),
                                      latency_ms=(_tfile.time()-_sfile)*1000)

        # Pending email — user replied with just an email address
        import re as _re2
        email_only_match = _re2.search(r'^\s*[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}\s*$', user_request)
        if self._pending_email and self._pending_email.needs_email and email_only_match:
            import time as _ter
            _ser = _ter.time()
            email_addr = email_only_match.group(0).strip()
            self._pending_email.recipient_email = email_addr
            self._pending_email.needs_email = False
            msg = self.composer.format_draft_for_confirmation(self._pending_email)
            return JarvisResponse(success=True, message=msg, latency_ms=(_ter.time()-_ser)*1000)

        # Email shortcut — send email (Level 2-4 pipeline)
        send_keywords = ["send an email", "send email", "email to", "send a message to", "write an email", "draft an email"]
        if any(kw in req_lower for kw in send_keywords):
            import time as _ts
            _ss = _ts.time()

            # Inject context so LLM writes on behalf of Abdullah, not as itself
            _jarvis_context = (
                "Write this email on behalf of Abdullah. "
                "Do NOT make up facts not stated in the request — only use information explicitly given. "
                "Write professionally in first person as Abdullah.\n\n"
                f"Email request: {user_request}"
            )
            draft = await self.composer.compose(_jarvis_context, self.contacts)

            # Level 3: Contact not found — ask for email
            if draft.needs_email:
                self._pending_email = draft
                msg = (
                    f"I don't have an email address for '{draft.recipient_name}' in your contacts.\n"
                    f"What is their email address? (or say 'add [name] [email]' to save them)"
                )
                return JarvisResponse(success=True, message=msg, latency_ms=(_ts.time()-_ss)*1000)

            # No recipient at all
            if not draft.recipient_email:
                return JarvisResponse(success=False, message="I need an email address to send to. Who would you like to email?", latency_ms=(_ts.time()-_ss)*1000)

            # Level 4: Show draft for confirmation
            self._pending_email = draft
            msg = self.composer.format_draft_for_confirmation(draft)
            return JarvisResponse(success=True, message=msg, latency_ms=(_ts.time()-_ss)*1000)

        # Add contact shortcut
        add_contact_match = re.search(r'add\s+(\w+)\s+([\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,})', req_lower)
        if add_contact_match:
            name = add_contact_match.group(1).capitalize()
            email = add_contact_match.group(2)
            self.contacts.add(name, email)
            # If we have a pending email waiting for this contact
            if self._pending_email and self._pending_email.needs_email:
                self._pending_email.recipient_email = email
                self._pending_email.needs_email = False
                msg = self.composer.format_draft_for_confirmation(self._pending_email)
                return JarvisResponse(success=True, message=f"Contact saved! {msg}")
            return JarvisResponse(success=True, message=f"Contact saved: {name} → {email}")

        # List contacts shortcut
        if any(kw in req_lower for kw in ["my contacts", "list contacts", "show contacts"]):
            return JarvisResponse(success=True, message=self.contacts.format_list())

        # ── Spotify shortcuts ──────────────────────────────────────────────────
        import time as _tsp2
        _ssp2 = _tsp2.time()

        _spotify_kw = [
            "play", "pause", "skip", "next song", "previous song", "last song",
            "spotify", "music", "song", "track", "artist", "playlist",
            "volume up", "volume down", "shuffle", "repeat",
            "what's playing", "whats playing", "now playing", "currently playing",
            "queue", "add to queue",
        ]
        _is_spotify = any(kw in req_lower for kw in _spotify_kw)

        # Avoid collision with mac volume/open app shortcuts
        _is_mac_vol = bool(re.search(r'(?:set\s+)?(?:volume|vol)\s+(?:to\s+)?\d+', req_lower))
        _is_open    = req_lower.startswith("open ") or req_lower.startswith("launch ")

        if _is_spotify and not _is_mac_vol and not _is_open:

            # Now playing
            if any(kw in req_lower for kw in ["what's playing", "whats playing", "now playing", "currently playing", "what song"]):
                data = await self.spotify.get_now_playing()
                msg = self.spotify.format_now_playing(data)
                return JarvisResponse(success=data.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # Pause
            if any(kw in req_lower for kw in ["pause", "stop music", "stop playing", "stop spotify"]):
                result = await self.spotify.pause()
                msg = "Paused." if result.get("success") else result.get("error", "Could not pause.")
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # Skip / next
            if any(kw in req_lower for kw in ["skip", "next song", "next track", "next please"]):
                result = await self.spotify.skip()
                msg = "Skipped to next track." if result.get("success") else result.get("error", "Could not skip.")
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # Previous
            if any(kw in req_lower for kw in ["previous", "last song", "go back", "prev track"]):
                result = await self.spotify.previous()
                msg = "Going back to previous track." if result.get("success") else result.get("error", "Could not go back.")
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # Shuffle
            if "shuffle" in req_lower:
                state = "off" not in req_lower
                result = await self.spotify.shuffle(state)
                msg = f"Shuffle {'on' if state else 'off'}." if result.get("success") else result.get("error", "Could not set shuffle.")
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # Repeat
            if "repeat" in req_lower:
                if "off" in req_lower:
                    mode = "off"
                elif "track" in req_lower or "song" in req_lower:
                    mode = "track"
                else:
                    mode = "context"
                result = await self.spotify.repeat(mode)
                msg = f"Repeat set to {mode}." if result.get("success") else result.get("error", "Could not set repeat.")
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # Spotify volume
            _spvol = re.search(r'(?:spotify\s+)?volume\s+(?:to\s+)?(\d+)', req_lower)
            if _spvol:
                level = int(_spvol.group(1))
                result = await self.spotify.set_volume(level)
                msg = f"Spotify volume set to {level}%." if result.get("success") else result.get("error", "Could not set volume.")
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # My playlists
            if any(kw in req_lower for kw in ["my playlists", "show playlists", "list playlists"]):
                data = await self.spotify.get_playlists()
                if data.get("success"):
                    plists = data.get("playlists", [])
                    if plists:
                        lines = ["Your Spotify playlists:\n"]
                        for i, p in enumerate(plists, 1):
                            lines.append(f"{i}. {p['name']} ({p['tracks']} tracks)")
                        msg = "\n".join(lines)
                    else:
                        msg = "No playlists found."
                else:
                    msg = data.get("error", "Could not get playlists.")
                return JarvisResponse(success=data.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

            # Play / resume — with or without a song name
            _play_match = re.search(
                r'play\s+(?:some\s+)?(?:me\s+)?(?:the\s+)?(?:song\s+|track\s+|artist\s+|playlist\s+)?'
                r'["\']?(.+?)["\']?\s*(?:on spotify|by .+)?$',
                user_request, re.IGNORECASE
            )
            if "play" in req_lower:
                if _play_match:
                    query = _play_match.group(1).strip()
                    # Strip trailing "on spotify"
                    query = re.sub(r'\s+on spotify$', '', query, flags=re.IGNORECASE).strip()
                    if query and query.lower() not in ("spotify", "music", "something", "anything"):
                        result = await self.spotify.play_by_name(query)
                        msg = self.spotify.format_play_result(result)
                    else:
                        # Resume
                        result = await self.spotify.play()
                        msg = "Resumed playback." if result.get("success") else result.get("error", "Could not resume.")
                else:
                    # Just "play" or "resume"
                    result = await self.spotify.play()
                    msg = "Resumed playback." if result.get("success") else result.get("error", "Could not resume.")
                return JarvisResponse(success=result.get("success", False), message=msg,
                                      latency_ms=(_tsp2.time()-_ssp2)*1000)

        # Calendar shortcut — check schedule
        cal_read_keywords = ["what's on my calendar", "my schedule", "my meetings", "what do i have", "events today", "events this week"]
        if any(kw in req_lower for kw in cal_read_keywords):
            import time as _t3
            _s3 = _t3.time()
            result = await self.calendar.search_events()
            msg = result.get("message", "No events found.")
            asyncio.ensure_future(self.memory.store_task_result(user_request, "check_calendar", True, msg[:100]))
            return JarvisResponse(success=True, message=msg, latency_ms=(_t3.time()-_s3)*1000)

        # ── Mac Control shortcuts ──────────────────────────────────────────
        import re as _rem

        # Open app
        if any(kw in req_lower for kw in ["open", "launch", "start"]):
            import time as _to
            _so = _to.time()
            # Known app aliases
            app_aliases = {
                "chrome": "Google Chrome", "google chrome": "Google Chrome",
                "safari": "Safari", "firefox": "Firefox",
                "spotify": "Spotify", "music": "Music",
                "terminal": "Terminal", "iterm": "iTerm",
                "vscode": "Visual Studio Code", "vs code": "Visual Studio Code",
                "code": "Visual Studio Code", "visual studio": "Visual Studio Code",
                "notes": "Notes", "mail": "Mail", "calendar": "Calendar",
                "slack": "Slack", "zoom": "Zoom", "discord": "Discord",
                "finder": "Finder", "calculator": "Calculator",
                "messages": "Messages", "facetime": "FaceTime",
                "photos": "Photos", "maps": "Maps", "word": "Microsoft Word",
                "excel": "Microsoft Excel", "powerpoint": "Microsoft PowerPoint",
                "app": "__app__", "system preferences": "System Preferences",
                "system settings": "System Settings", "settings": "System Settings",
            }
            # Try to find a known app name in the request
            app = None
            for alias, real_name in app_aliases.items():
                if alias in req_lower:
                    app = real_name
                    break
            # Fallback: extract word after open/launch/start
            if not app:
                open_match = _rem.search(r'(?:open|launch|start)\s+([a-zA-Z][a-zA-Z0-9]+)', user_request, _rem.IGNORECASE)
                if open_match:
                    app = open_match.group(1).strip().title()
            if app:
                new_window = any(kw in req_lower for kw in ["new window", "new tab", "open new", "new session"])

                # App — open native app
                # App — open native app, new window via Cmd+N
                if app == "__app__":
                    script = (
                        "tell application \"App\" to activate\n"
                        "delay 0.5\n"
                        "tell application \"System Events\"\n"
                        "    keystroke \"n\" using {command down}\n"
                        "end tell"
                    )
                    await self.mac._async_script(script)
                    msg = "Opening new App window." if new_window else "Opening App."
                    return JarvisResponse(success=True, message=msg, latency_ms=(_to.time()-_so)*1000)

                # VSCode new window — Cmd+Shift+N
                if app == "Visual Studio Code" and new_window:
                    script = (
                        "tell application \"Visual Studio Code\" to activate\n"
                        "delay 0.5\n"
                        "tell application \"System Events\"\n"
                        "    keystroke \"n\" using {command down, shift down}\n"
                        "end tell"
                    )
                    await self.mac._async_script(script)
                    return JarvisResponse(success=True, message="Opening new VSCode window.", latency_ms=(_to.time()-_so)*1000)

                result = await self.mac.open_app(app, new_window=new_window)
                action = "Opening new window in" if new_window else "Opening"
                msg = f"{action} {app}." if result.get("success") else f"Could not open {app}: {result.get('error')}"
                return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_to.time()-_so)*1000)

        # Set volume
        vol_match = _rem.search(r'(?:set\s+)?(?:volume|vol)\s+(?:to\s+)?(\d+)', req_lower)
        if vol_match:
            import time as _tv
            _sv = _tv.time()
            level = int(vol_match.group(1))
            result = await self.mac.set_volume(level)
            msg = f"Volume set to {level}." if result.get("success") else f"Could not set volume: {result.get('error')}"
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tv.time()-_sv)*1000)

        # Mute / unmute
        if any(kw in req_lower for kw in ["mute", "silence", "quiet"]) and "volume" not in req_lower:
            import time as _tm
            _sm = _tm.time()
            result = await self.mac.mute()
            return JarvisResponse(success=result.get("success", False), message="Muted.", latency_ms=(_tm.time()-_sm)*1000)

        if any(kw in req_lower for kw in ["unmute", "unsilence", "turn sound on"]):
            import time as _tum
            _sum = _tum.time()
            result = await self.mac.unmute()
            return JarvisResponse(success=result.get("success", False), message="Unmuted.", latency_ms=(_tum.time()-_sum)*1000)

        # Set brightness
        bright_match = _rem.search(r'(?:set\s+)?brightness\s+(?:to\s+)?(\d+)', req_lower)
        if bright_match:
            import time as _tb
            _sb = _tb.time()
            level = min(100, int(bright_match.group(1))) / 100.0
            result = await self.mac.set_brightness(level)
            msg = f"Brightness set to {int(level*100)}%." if result.get("success") else f"Could not set brightness: {result.get('error')}"
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tb.time()-_sb)*1000)

        # Battery
        if any(kw in req_lower for kw in ["battery", "battery level", "how much battery"]):
            import time as _tbat
            _sbat = _tbat.time()
            result = await self.mac.get_battery()
            if result.get("success"):
                pct = result.get("battery_pct", "unknown")
                msg = f"Battery is at {pct}%."
            else:
                msg = "Could not read battery level."
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tbat.time()-_sbat)*1000)

        # Lock screen
        if any(kw in req_lower for kw in ["lock screen", "lock my screen", "lock the screen"]):
            import time as _tl
            _sl = _tl.time()
            result = await self.mac.lock_screen()
            return JarvisResponse(success=result.get("success", False), message="Screen locked.", latency_ms=(_tl.time()-_sl)*1000)

        # Get clipboard
        if any(kw in req_lower for kw in ["clipboard", "what did i copy", "whats in my clipboard"]):
            import time as _tcb
            _scb = _tcb.time()
            result = await self.mac.get_clipboard()
            text = result.get("text", "")
            msg = f"Clipboard contains: {text[:200]}" if text else "Clipboard is empty."
            return JarvisResponse(success=True, message=msg, latency_ms=(_tcb.time()-_scb)*1000)

        # Send Mac notification
        notif_match = _rem.search(r'(?:send|show|give me)\s+(?:a\s+)?notification[:\s]+(.+)', user_request, _rem.IGNORECASE)
        if notif_match:
            import time as _tn
            _sn = _tn.time()
            message = notif_match.group(1).strip()
            result = await self.mac.send_notification(message)
            msg = f"Notification sent: {message}" if result.get("success") else "Could not send notification."
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tn.time()-_sn)*1000)

        # ── Information tool shortcuts ─────────────────────────────────────

        # ── Search / Research — router-driven, no keyword list needed ──────
        # If the router said websearch or research, we search.
        # Also catch common question patterns the router might miss.
        question_starters = (
            'what', 'who', 'when', 'where', 'why', 'how', 'which',
            'tell', 'explain', 'describe', 'define', 'search', 'find',
            'look', 'google', 'research', 'investigate', 'analyse',
            'analyze', 'give me', 'show me', 'can you find',
        )

        already_handled = any(kw in req_lower for kw in [
            'weather', 'battery', 'wifi', 'volume', 'brightness',
            'screenshot', 'dark mode', 'trash', 'news', 'headlines',
            'schedule', 'remind', 'open ', 'quit ', 'close ',
            'sleep the mac', 'lock screen', 'check my email',
            'send email', 'send an email', 'my calendar', 'my emails',
            'mute', 'unmute', 'clipboard', 'whats on my calendar',
        ])

        # Use primary_agent passed in to detect search intent
        primary_agent_val = primary_agent.value if primary_agent else ''

        is_search = (
            primary_agent_val in ('websearch', 'research') or
            (req_lower.split()[0] in question_starters if req_lower.split() else False)
        ) and not already_handled

        is_research = (
            primary_agent_val == 'research' or
            any(kw in req_lower for kw in [
                'research', 'deep dive', 'detailed', 'comprehensive',
                'everything about', 'investigate', 'analyse', 'analyze',
                'in depth', 'give me a full', 'full overview',
            ])
        ) and not already_handled

        if is_search or is_research:
            import time as _tws
            _sws = _tws.time()

            # Use the search tool query parser to clean the query
            query = self.websearch.parse_query(user_request)
            # Also strip research/investigate triggers for cleaner queries
            for trigger in ['research ', 'investigate ', 'analyse ', 'analyze ']:
                if query.lower().startswith(trigger):
                    query = query[len(trigger):].strip()
                    break
            if not query or len(query) < 2:
                query = user_request

            msg = "Could not find information about: " + query

            # Detect query complexity for adaptive length
            _simple_triggers = ["who is", "what is the", "when did", "where is",
                                 "capital of", "ceo of", "founder of", "born in",
                                 "how old is", "what year", "who won", "who owns"]
            _elaborate_triggers = ["elaborate", "explain in detail", "tell me more",
                                   "comprehensive", "deep dive", "in depth", "full overview",
                                   "everything about", "give me a full"]
            _req_low = user_request.lower()
            is_elaborate = any(t in _req_low for t in _elaborate_triggers)
            is_simple = (not is_research and not is_elaborate and
                         any(_req_low.startswith(t) or t in _req_low for t in _simple_triggers))

            if is_simple:
                length_instruction = "- Answer in 3-5 sentences. Be direct and factual. No lists needed."
            elif is_elaborate or is_research:
                length_instruction = "- Give a thorough response of 200-350 words. Cover background, key facts, current state, and significance. Use numbered lists where appropriate."
            else:
                length_instruction = "- Aim for 5-10 sentences. Cover the key facts and essential context. Use a numbered list only if there are genuinely multiple distinct items."

            if is_research:
                # Run all 3 searches IN PARALLEL instead of sequentially
                search_queries = [query, query + " explained", query + " overview"]
                search_results = await asyncio.gather(
                    *[self.websearch.search(q, max_results=3) for q in search_queries],
                    return_exceptions=True
                )
                all_results = []
                for data in search_results:
                    if isinstance(data, Exception): continue
                    if data.get("success") and data.get("results"):
                        for r in data.get("results", []):
                            if r not in all_results:
                                all_results.append(r)
                if len(all_results) < 3:
                    wiki_data = await self.websearch._wiki(query, 3)
                    if wiki_data.get("success"):
                        for r in wiki_data.get("results", []):
                            if r not in all_results:
                                all_results.append(r)
                if all_results:
                    snips = [r.get("snippet", "")[:300] for r in all_results[:8]]
                    combined2 = " ".join(snips)
                    rp = (
                        f"Answer this question: {query}\n\n"
                        f"Source material:\n{combined2}\n\n"
                        f"Instructions:\n"
                        f"{length_instruction}\n"
                        f"- Organise logically. Use numbered points (1. 2. 3.) for lists.\n"
                        f"- Write in clear prose. NO markdown asterisks (*) or hash (#).\n"
                        f"- You are Jarvis, an AI assistant. Never refer to yourself as the user."
                    )
                    report = await self.llm.chat([{"role": "user", "content": rp}])
                    msg = report.strip()
                else:
                    msg = "Could not find enough information about: " + query
            else:
                # Single search
                data = await self.websearch.search(query, max_results=5)
                if data.get("success") and data.get("results"):
                    snips2 = [r.get("snippet", "")[:400] for r in data["results"][:5]]
                    ct = " ".join(snips2)
                    ap = (
                        f"Answer this question: {query}\n\n"
                        f"Source material:\n{ct}\n\n"
                        f"Instructions:\n"
                        f"{length_instruction}\n"
                        f"- Write in natural prose. NO markdown asterisks (*) or hash (#) symbols.\n"
                        f"- You are Jarvis, an AI assistant. Never refer to yourself as the user."
                    )
                    summary = await self.llm.chat([{"role": "user", "content": ap}])
                    msg = summary.strip()
                else:
                    msg = "Could not find results for: " + query

            asyncio.ensure_future(self.memory.store_task_result(user_request, "web_search", True, msg[:100]))
            return JarvisResponse(success=True, message=msg, latency_ms=(_tws.time()-_sws)*1000)
            app_aliases_q = {
                "chrome": "Google Chrome", "safari": "Safari",
                "spotify": "Spotify", "vscode": "Visual Studio Code",
                "vs code": "Visual Studio Code", "slack": "Slack",
                "zoom": "Zoom", "discord": "Discord", "mail": "Mail",
                "terminal": "Terminal", "finder": "Finder",
            }
            raw_app = quit_match.group(1).strip().lower()
            app_q = app_aliases_q.get(raw_app, quit_match.group(1).strip().title())
            result = await self.mac.quit_app(app_q)
            msg = f"Closed {app_q}." if result.get("success") else f"Could not close {app_q}: {result.get('error')}"
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tq.time()-_sq)*1000)

        # Volume up/down
        vol_up = any(kw in req_lower for kw in ["volume up", "turn up", "louder", "increase volume", "raise volume"])
        vol_down = any(kw in req_lower for kw in ["volume down", "turn down", "quieter", "decrease volume", "lower volume"])
        if vol_up or vol_down:
            import time as _tvd
            _svd = _tvd.time()
            amount_match = _remac.search(r'(\d+)', req_lower)
            amount = int(amount_match.group(1)) if amount_match else 10
            result = await self.mac.adjust_volume("up" if vol_up else "down", amount)
            direction = "up" if vol_up else "down"
            msg = f"Volume turned {direction} to {result.get('volume', '?')}%."
            return JarvisResponse(success=True, message=msg, latency_ms=(_tvd.time()-_svd)*1000)

        # Screenshot
        if any(kw in req_lower for kw in ["screenshot", "take a screenshot", "capture screen", "screen capture"]):
            import time as _tss
            _sss = _tss.time()
            result = await self.mac.take_screenshot()
            return JarvisResponse(success=result.get("success", False), message=result.get("message", "Screenshot taken."), latency_ms=(_tss.time()-_sss)*1000)

        # Dark mode toggle
        if any(kw in req_lower for kw in ["dark mode", "light mode", "toggle dark", "toggle light", "switch to dark", "switch to light"]):
            import time as _tdm
            _sdm = _tdm.time()
            if "off" in req_lower or "light mode" in req_lower or "switch to light" in req_lower:
                # Force light mode
                result = await self.mac.get_dark_mode()
                if result.get("dark_mode"):
                    result = await self.mac.toggle_dark_mode()
                msg = "Switched to light mode."
            elif "on" in req_lower or "dark mode" in req_lower or "switch to dark" in req_lower:
                result = await self.mac.get_dark_mode()
                if not result.get("dark_mode"):
                    result = await self.mac.toggle_dark_mode()
                msg = "Switched to dark mode."
            else:
                result = await self.mac.toggle_dark_mode()
                msg = "Toggled dark/light mode."
            return JarvisResponse(success=True, message=msg, latency_ms=(_tdm.time()-_sdm)*1000)

        # System info
        if any(kw in req_lower for kw in ["system info", "disk space", "storage", "cpu usage", "ram usage", "memory usage", "how much storage"]):
            import time as _tsi
            _ssi = _tsi.time()
            result = await self.mac.get_system_info()
            msg = result.get("message", "Could not get system info.")
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tsi.time()-_ssi)*1000)

        # WiFi info
        if any(kw in req_lower for kw in ["wifi", "wi-fi", "network", "internet connection", "what network", "connected to"]):
            import time as _twifi
            _swifi = _twifi.time()
            result = await self.mac.get_wifi_info()
            msg = result.get("message", "Could not get WiFi info.")
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_twifi.time()-_swifi)*1000)

        # Empty trash
        if any(kw in req_lower for kw in ["empty trash", "clear trash", "delete trash"]):
            import time as _ttr
            _str = _ttr.time()
            result = await self.mac.empty_trash()
            msg = "Trash emptied." if result.get("success") else f"Could not empty trash: {result.get('error')}"
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_ttr.time()-_str)*1000)

        # Sleep Mac
        if any(kw in req_lower for kw in ["sleep", "put to sleep", "sleep the mac", "sleep my mac"]) and "reminder" not in req_lower:
            import time as _tslp
            _sslp = _tslp.time()
            result = await self.mac.sleep()
            return JarvisResponse(success=result.get("success", False), message="Putting Mac to sleep.", latency_ms=(_tslp.time()-_sslp)*1000)

        # ── Calendar shortcuts ────────────────────────────────────────────────
        import re as _recal

        # Pending meeting — user gave a new time after conflict
        if self._pending_meeting and self._pending_meeting.get("needs_new_time"):
            time_match = _recal.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', req_lower)
            if not time_match:
                time_match = _recal.search(r'at\s+(\d{1,2})(?::(\d{2}))?', req_lower)
            if time_match or any(kw in req_lower for kw in ["tomorrow", "monday","tuesday","wednesday","thursday","friday","saturday","sunday"]):
                import time as _tnt
                _snt = _tnt.time()

                # Preserve the original date if user only gave a new time
                original_start = self._pending_meeting.get("start_time", "")
                has_date_word = any(kw in req_lower for kw in [
                    "tomorrow", "today", "monday","tuesday","wednesday",
                    "thursday","friday","saturday","sunday","next week"
                ])

                if has_date_word:
                    # User gave a full new date+time — resolve normally
                    new_temporal = self._resolve_temporal(user_request)
                    new_start = new_temporal.get("datetime", "")
                else:
                    # User only gave a new time — keep original date, just change time
                    from datetime import datetime as _dtfix, timezone
                    from zoneinfo import ZoneInfo
                    local_tz = ZoneInfo("Europe/London")
                    orig_dt = _dtfix.fromisoformat(original_start)
                    # Extract new time from user request
                    new_temporal = self._resolve_temporal(user_request)
                    new_time_str = new_temporal.get("time", "")
                    if new_time_str:
                        h, m = map(int, new_time_str.split(":"))
                        new_dt = orig_dt.replace(hour=h, minute=m, second=0, microsecond=0)
                        new_start = new_dt.isoformat()
                    else:
                        new_start = new_temporal.get("datetime", "")
                if new_start:
                    from datetime import timedelta, datetime as _dtnt
                    duration_mins = self._pending_meeting.get("duration_mins", 60)
                    new_end = (_dtnt.fromisoformat(new_start) + timedelta(minutes=duration_mins)).isoformat()

                    # Check conflicts again
                    conflict2 = await self.calendar.check_conflicts(new_start, new_end)
                    if conflict2.get("has_conflict"):
                        ct = conflict2["conflicts"][0].get("title", "another event")
                        return JarvisResponse(success=False,
                            message=f"That time also conflicts with '{ct}'. What other time works?",
                            latency_ms=(_tnt.time()-_snt)*1000)

                    # Book it
                    result = await self.calendar.create_event(
                        title=self._pending_meeting["title"],
                        start_time=new_start,
                        end_time=new_end,
                        attendees=self._pending_meeting.get("attendees") or None,
                    )
                    self._pending_meeting = None
                    if result.get("success"):
                        start_fmt = new_start[:16].replace("T", " at ")
                        dur_str = f"{duration_mins} minutes" if duration_mins != 60 else "1 hour"
                        msg = f"Done! Rescheduled to {start_fmt} for {dur_str}."
                        if result.get("link"):
                            msg += " View: " + result["link"]
                    else:
                        msg = f"Could not create event: {result.get('error')}"
                    return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tnt.time()-_snt)*1000)

        # Schedule/create event — pending duration confirmation
        if self._pending_meeting and any(kw in req_lower for kw in
            ["minute", "hour", "min", "hr", "30", "45", "60", "90", "15", "yes", "confirm", "book it", "1 hour", "2 hour"]):
            import time as _tconf
            _sconf = _tconf.time()
            meeting = self._pending_meeting

            # Extract duration from reply — handle all common formats
            duration_mins = 60  # default 1 hour
            dur_match = _recal.search(r'(\d+)\s*(?:hours?|hrs?|minutes?|mins?|m)', req_lower)
            if dur_match:
                val = int(dur_match.group(1))
                is_hours = any(u in req_lower[dur_match.start():dur_match.end()+2] for u in ["hour", "hr"])
                duration_mins = val * 60 if is_hours else val
            elif "half" in req_lower:
                duration_mins = 30
            elif "quarter" in req_lower:
                duration_mins = 15
            elif "one hour" in req_lower or "an hour" in req_lower:
                duration_mins = 60
            elif "two hour" in req_lower:
                duration_mins = 120
            # Clamp to sensible range
            duration_mins = max(15, min(480, duration_mins))

            from datetime import timedelta
            from datetime import datetime as _dt
            start_time = meeting["start_time"]
            start_dt = _dt.fromisoformat(start_time)
            end_dt = start_dt + timedelta(minutes=duration_mins)
            end_time = end_dt.isoformat()

            # Check conflicts with actual duration
            conflict = await self.calendar.check_conflicts(start_time, end_time)
            if conflict.get("has_conflict"):
                conflict_title = conflict["conflicts"][0].get("title", "another event")
                # Keep pending meeting alive so user can pick new time
                self._pending_meeting["duration_mins"] = duration_mins
                self._pending_meeting["needs_new_time"] = True
                return JarvisResponse(
                    success=False,
                    message=f"Conflict detected — '{conflict_title}' is already at that time. What time would you like instead?",
                    latency_ms=(_tconf.time()-_sconf)*1000
                )

            # Create the event
            result = await self.calendar.create_event(
                title=meeting["title"],
                start_time=start_time,
                end_time=end_time,
                attendees=meeting.get("attendees") or None,
            )
            self._pending_meeting = None

            if result.get("success"):
                attendee_str = f" with {', '.join(meeting['attendees'])}" if meeting.get("attendees") else ""
                start_fmt = start_time[:16].replace("T", " at ")
                dur_str = f"{duration_mins} minutes" if duration_mins != 60 else "1 hour"
                msg = f"✅ '{meeting['title']}' scheduled{attendee_str} on {start_fmt} for {dur_str}."
                if result.get("link"):
                    msg += ' View: ' + result['link']
            else:
                msg = f"Could not create event: {result.get('error', 'unknown error')}"

            asyncio.ensure_future(self.memory.store_task_result(user_request, "schedule_meeting", result.get("success", False), msg[:100]))
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tconf.time()-_sconf)*1000)

        # Cancel pending meeting
        if self._pending_meeting and any(kw in req_lower for kw in ["cancel", "no", "don't book", "abort", "never mind"]):
            self._pending_meeting = None
            return JarvisResponse(success=True, message="Meeting cancelled. Nothing was added to your calendar.")

        # Schedule/create event shortcut
        # Use broader schedule detection — check individual trigger words + meeting context
        has_meeting_word = any(w in req_lower for w in ["meeting", "call", "appointment", "event", "session"])
        has_action_word = any(w in req_lower for w in [
            "schedule", "book", "create", "add", "set", "arrange", "block",
            "put", "plan", "organise", "organize", "new"
        ])
        explicit_schedule = any(kw in req_lower for kw in [
            "add to calendar", "calendar event", "add event",
        ])

        if (has_meeting_word and has_action_word) or explicit_schedule:
            import time as _tcal
            _scal = _tcal.time()

            # Extract title — look for "called X", "named X", "titled X", "call it X"
            # Extract title using clean patterns
            title = None
            req_lower_t = user_request.lower()

            # Pattern: call it X / called X / named X / titled X
            import re as _ret
            m1 = _ret.search(r"call(?:ed)?\s+it\s+(.+?)(?:\s+(?:for|on|at)\s+\d|\s*$)", user_request, _ret.IGNORECASE)
            m2 = _ret.search(r"(?:called|named|titled)\s+(.+?)(?:\s+(?:at|on)\s+\d|\s+(?:today|tomorrow|next|for)\s|\s*$)", user_request, _ret.IGNORECASE)
            m3 = _ret.search(r"about\s+([\w\s]+?)(?:\s+(?:at|on|for|today|tomorrow)|$)", user_request, _ret.IGNORECASE)

            if m1:
                title = m1.group(1).strip()
            elif m2:
                title = m2.group(1).strip()
            elif m3:
                title = m3.group(1).strip().title()
            else:
                title = "Meeting"

            # Strip time/date noise from title
            noise = ["today","tomorrow","tonight","monday","tuesday","wednesday",
                     "thursday","friday","saturday","sunday","next week",
                     "9pm","8pm","7pm","6pm","5pm","4pm","3pm","2pm","1pm",
                     "12pm","11am","10am","9am","8am","am","pm"]
            for n in noise:
                title = _ret.sub(r"\b" + n + r"\b", "", title, flags=_ret.IGNORECASE).strip()
            title = title.strip(". ").title() or "Meeting"
            if attendee_match:
                names = attendee_match.group(1).strip()
                for name in _recal.split(r'[,\s]+(?:and\s+)?', names):
                    name = name.strip()
                    if name:
                        contact = self.contacts.find(name)
                        if contact:
                            attendees.append(contact["email"])

            # Resolve time
            temporal = self._resolve_temporal(user_request)
            start_time = temporal.get("datetime", "")

            if not start_time:
                return JarvisResponse(success=False, message="I couldn't figure out when to schedule the meeting. Could you specify a date and time?", latency_ms=(_tcal.time()-_scal)*1000)

            # Store pending meeting and ask for duration
            self._pending_meeting = {
                "title": title,
                "start_time": start_time,
                "attendees": attendees,
            }

            start_fmt = start_time[:16].replace("T", " at ")
            attendee_str = f" with {', '.join(attendees)}" if attendees else ""
            attendee_str = f" with {", ".join(attendees)}" if attendees else ""
            msg = (
                f"I will schedule {title!r}{attendee_str} on {start_fmt}. "
                "How long should the meeting be? (e.g. 30 minutes, 1 hour, 45 minutes)"
            )
            return JarvisResponse(success=True, message=msg, latency_ms=(_tcal.time()-_scal)*1000)

        return None  # No shortcut — proceed with full pipeline

    async def _morning_briefing(self) -> str:
        """
        Fully personalised morning briefing for Abdullah.
        Sections: greeting, weather, prayer times, calendar, inbox,
                  news (detailed), tech, sports (fav teams), markets, quote.
        All fetched in parallel for speed.
        """
        import random
        now = datetime.now()
        hour = now.hour
        date_str = now.strftime("%A, %d %B %Y")

        # ── Dynamic name + greeting ────────────────────────────────────────
        names_morning   = ["champ", "legend", "boss", "big man", "chief"]
        names_afternoon = ["mate", "boss", "legend", "G"]
        names_evening   = ["night owl", "champ", "boss"]

        if hour < 12:
            time_phrase = "Good morning"
            name = random.choice(names_morning)
        elif hour < 17:
            time_phrase = "Good afternoon"
            name = random.choice(names_afternoon)
        else:
            time_phrase = "Good evening"
            name = random.choice(names_evening)

        # Check memory for mood
        mood_memories = await self.memory.retrieve("mood feeling tired stressed", k=2)
        mood_note = ""
        if mood_memories:
            last_mood = mood_memories[0].content
            if any(w in last_mood.lower() for w in ["tired", "stressed", "rough", "bad"]):
                mood_note = " Hope you're feeling better today."
            elif any(w in last_mood.lower() for w in ["good", "great", "happy", "productive"]):
                mood_note = " Glad to hear you were in good form."

        # ── Parallel fetch everything ──────────────────────────────────────
        from config.settings import FAVOURITE_TEAMS, FAVOURITE_FOOTBALL_LEAGUE, FAVOURITE_BASKETBALL_LEAGUE

        (weather_data, prayer_data, news_data, tech_data,
         cal_data, email_data, sports_pl, sports_ucl,
         sports_nba, market_data) = await asyncio.gather(
            self.weather.get_current(),
            self.prayer.get_times(),
            self.news.get_headlines(max_stories=5),
            self.news.get_headlines(category="technology", max_stories=2),
            self.calendar.search_events(),
            self.gmail.get_inbox(max_results=5),
            self.sports.get_scores(FAVOURITE_FOOTBALL_LEAGUE, limit=10),
            self.sports.get_scores("champions_league", limit=8),
            self.sports.get_scores(FAVOURITE_BASKETBALL_LEAGUE, limit=8),
            self.markets.get_all(),
            return_exceptions=True
        )

        lines_out = []

        # ── Greeting ───────────────────────────────────────────────────────
        lines_out.append(f"{time_phrase}, {name}!{mood_note} Here is your briefing for {date_str}.")
        lines_out.append("")

        # ── Weather ────────────────────────────────────────────────────────
        if isinstance(weather_data, dict) and weather_data.get("success"):
            w = weather_data
            lines_out.append("WEATHER")
            lines_out.append(
                f"  {w.get('condition','')}, {w.get('temperature_c','')}°C "
                f"(feels like {w.get('feels_like_c','')}°C) in {w.get('location','')}. "
                f"Humidity {w.get('humidity_pct','')}%, wind {w.get('wind_kph','')} km/h."
            )
            lines_out.append("")

        # ── Prayer times ───────────────────────────────────────────────────
        if isinstance(prayer_data, dict) and prayer_data.get("success"):
            lines_out.append(self.prayer.format_times(prayer_data))
            lines_out.append("  " + self.prayer.get_next_prayer(prayer_data))
            lines_out.append("")

        # ── Calendar ───────────────────────────────────────────────────────
        if isinstance(cal_data, dict) and cal_data.get("success"):
            events = cal_data.get("events", [])
            lines_out.append("YOUR DAY")
            if events:
                lines_out.append(f"  {len(events)} event(s) scheduled:")
                for e in events[:5]:
                    start_t = e.get("start", "")[:16].replace("T", " at ")
                    lines_out.append(f"  • {e.get('title','Event')} — {start_t}")
            else:
                lines_out.append("  Nothing in the calendar today. A free day!")
            lines_out.append("")

        # ── Email ──────────────────────────────────────────────────────────
        if isinstance(email_data, dict) and email_data.get("success"):
            emails = email_data.get("emails", [])
            count = email_data.get("count", 0)
            lines_out.append("INBOX")
            lines_out.append(f"  {count} unread email(s).")
            if emails:
                subj = emails[0].get("subject", "(no subject)")
                sender = emails[0].get("from", "")[:50]
                lines_out.append("  Latest: " + repr(subj) + " from " + sender)
            lines_out.append("")

        # ── Top News (detailed) ────────────────────────────────────────────
        if isinstance(news_data, dict) and news_data.get("success"):
            stories = news_data.get("stories", [])
            lines_out.append("TOP NEWS")
            for i, story in enumerate(stories[:5], 1):
                sources = story.get("sources", [])
                if len(sources) > 1:
                    src_str = f"[{', '.join(sources)}] — {len(sources)} outlets"
                else:
                    src_str = f"[{sources[0]}]" if sources else ""
                lines_out.append(f"  {i}. {story['title']}")
                lines_out.append(f"     {src_str}")
                if story.get("description"):
                    lines_out.append(f"     {story['description'][:180]}")
            lines_out.append("")

        # ── Tech & AI ──────────────────────────────────────────────────────
        if isinstance(tech_data, dict) and tech_data.get("success"):
            tech_stories = tech_data.get("stories", [])
            if tech_stories:
                lines_out.append("TECH & AI")
                for story in tech_stories[:2]:
                    sources = story.get("sources", [])
                    src_str = f" [{sources[0]}]" if sources else ""
                    lines_out.append(f"  • {story['title']}{src_str}")
                lines_out.append("")

        # ── Sports ────────────────────────────────────────────────────────
        fav_lower = [t.lower() for t in FAVOURITE_TEAMS]

        def is_fav(game):
            home = game.get("home_team", "").lower()
            away = game.get("away_team", "").lower()
            return any(
                any(word in home or word in away for word in fav.lower().split())
                for fav in fav_lower
            )

        def is_big_game(game):
            big_teams = ["manchester city", "real madrid", "barcelona", "liverpool",
                        "arsenal", "chelsea", "Bayern", "psg", "juventus",
                        "lakers", "celtics", "heat", "nuggets"]
            home = game.get("home_team", "").lower()
            away = game.get("away_team", "").lower()
            return sum(1 for t in big_teams if t in home or t in away) >= 2

        sports_lines = []

        for league_data, league_label in [
            (sports_pl, "PREMIER LEAGUE"),
            (sports_ucl, "CHAMPIONS LEAGUE"),
            (sports_nba, "NBA"),
        ]:
            if not isinstance(league_data, dict) or not league_data.get("success"):
                continue
            games = league_data.get("games", [])
            finished = [g for g in games if g.get("status") == "final"]
            live = [g for g in games if g.get("status") == "live"]

            fav_games = [g for g in finished if is_fav(g)]
            big_games = [g for g in finished if is_big_game(g) and not is_fav(g)]
            show = fav_games + big_games[:2]

            if live:
                sports_lines.append(f"{league_label} — LIVE")
                for g in live[:2]:
                    fav = " ★" if is_fav(g) else ""
                    sports_lines.append(
                        f"  {g['home_team']} {g['home_score']} - "
                        f"{g['away_score']} {g['away_team']} [{g.get('clock','')}]{fav}"
                    )

            if show:
                if not live:
                    sports_lines.append(f"{league_label}")
                for g in show[:4]:
                    fav = " ★" if is_fav(g) else ""
                    sports_lines.append(
                        f"  {g['home_team']} {g['home_score']} - "
                        f"{g['away_score']} {g['away_team']}{fav}"
                    )

        if sports_lines:
            lines_out.extend(sports_lines)
            lines_out.append("")

        # ── Markets ───────────────────────────────────────────────────────
        if isinstance(market_data, dict) and market_data.get("success"):
            lines_out.append(self.markets.format_prices(market_data))
            lines_out.append("")

        # ── Witty + inspirational quote (LLM generated, unique daily) ─────
        quote_prompt = (
            f"Generate a single witty yet genuinely inspiring quote or observation "
            f"that is relevant to someone starting their {date_str}. "
            f"It should feel fresh, not cliché. Max 2 sentences. "
            f"No quotation marks needed, just the text."
        )
        try:
            quote = await self.llm.chat([{"role": "user", "content": quote_prompt}])
            lines_out.append("THOUGHT FOR THE DAY")
            lines_out.append(f"  {quote.strip()}")
            lines_out.append("")
        except Exception:
            pass

        # ── Closing ────────────────────────────────────────────────────────
        closings = [
            "Make it count today.",
            "Go get it.",
            "Let's have a good one.",
            "You've got this.",
            "Make it a productive one.",
        ]
        lines_out.append(random.choice(closings))

        return chr(10).join(lines_out)

    async def _handle_multi_query(self, user_request: str, intents: list) -> str:
        """
        Handle compound queries by running multiple intents in parallel.
        e.g. "what's the weather and latest news and premier league scores?"
        """
        tasks = {}

        if "weather" in intents:
            location = self._extract_location(user_request)
            if location:
                tasks["weather"] = self.weather.get_current_for_location(location)
            else:
                tasks["weather"] = self.weather.get_current()

        if "news" in intents:
            tasks["news"] = self.news.get_headlines(query=user_request, max_stories=4)

        if "sports" in intents:
            league_key = self.sports.detect_league(user_request) or "premier_league"
            tasks["sports"] = self.sports.get_scores(league_key)

        if "calendar" in intents:
            tasks["calendar"] = self.calendar.search_events()

        if "email" in intents:
            tasks["email"] = self.gmail.get_inbox(max_results=5)

        if not tasks:
            return ""

        # Execute all in parallel
        keys = list(tasks.keys())
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        result_map = dict(zip(keys, results))

        sections = []

        if "weather" in result_map:
            data = result_map["weather"]
            if isinstance(data, dict) and data.get("success"):
                sections.append("WEATHER")
                sections.append("  " + self.weather.format_current(data))
                sections.append("")

        if "calendar" in result_map:
            data = result_map["calendar"]
            if isinstance(data, dict) and data.get("success"):
                events = data.get("events", [])
                sections.append("CALENDAR")
                if events:
                    for e in events[:4]:
                        start = e.get("start", "")[:16].replace("T", " at ")
                        sections.append(f"  • {e.get('title','')} — {start}")
                else:
                    sections.append("  No upcoming events.")
                sections.append("")

        if "email" in result_map:
            data = result_map["email"]
            if isinstance(data, dict) and data.get("success"):
                count = data.get("count", 0)
                sections.append("EMAILS")
                sections.append(f"  {count} unread email(s).")
                sections.append("")

        if "news" in result_map:
            data = result_map["news"]
            if isinstance(data, dict) and data.get("success"):
                stories = data.get("stories", [])
                sections.append("NEWS")
                for i, story in enumerate(stories[:4], 1):
                    sources = story.get("sources", [])
                    src_str = f" [{', '.join(sources[:2])}]" if len(sources) > 1 else ""
                    sections.append(f"  {i}. {story['title']}{src_str}")
                sections.append("")

        if "sports" in result_map:
            data = result_map["sports"]
            if isinstance(data, dict) and data.get("success"):
                sections.append("SPORTS")
                sections.append(self.sports.format_scores(data))
                sections.append("")

        return chr(10).join(sections).strip()


    def _time_greeting(self, now: datetime) -> str:
        hour = now.hour
        if hour < 12:
            return "Good morning"
        elif hour < 17:
            return "Good afternoon"
        else:
            return "Good evening"

    def _extract_location(self, user_request: str) -> str:
        """
        Extract a city name from a weather request.
        Returns empty string if no specific location found.
        Strips noise words like 'today', 'now', 'currently', country names.
        """
        import re as _reloc
        req = user_request.lower()

        # Noise words to strip from end of location
        noise_words = {
            "today", "now", "currently", "tonight", "tomorrow",
            "this", "week", "weekend", "morning", "evening",
            "afternoon", "night", "please", "for", "me",
        }

        # Country names to strip
        common_countries = {
            "pakistan","india","usa","uk","france","germany","china",
            "japan","australia","canada","brazil","italy","spain",
            "mexico","russia","nigeria","egypt","turkey","argentina",
            "bangladesh","indonesia","kenya","ghana","iran","iraq",
            "vietnam","thailand","malaysia","singapore","uae","qatar",
            "england","scotland","wales","ireland","netherlands","sweden",
            "norway","denmark","finland","switzerland","austria","belgium",
            "portugal","greece","poland","czech","romania","hungary",
            "southafrica","newzealand","saudiarabia",
        }

        for phrase in [
            "what's the weather in", "what is the weather in",
            "whats the weather in", "weather in", "weather for",
            "weather at", "weather today in", "weather forecast for",
            "forecast for", "forecast in", "temperature in",
            "how hot is it in", "how cold is it in", "how warm is it in",
            "what's it like in", "whats it like in",
        ]:
            if phrase in req:
                raw = user_request[req.index(phrase) + len(phrase):].strip()
                raw = raw.rstrip("?!., ")
                if not raw:
                    return ""

                # Split into words and strip noise/country from the end
                parts = raw.split()
                while parts and parts[-1].lower() in noise_words:
                    parts.pop()
                while parts and parts[-1].lower() in common_countries:
                    parts.pop()

                city = " ".join(parts).strip()
                return city if city else ""

        return ""