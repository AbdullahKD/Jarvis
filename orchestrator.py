"""
Jarvis Orchestrator
The central coordinator. Receives a user request and drives the full
pipeline: Router → Memory → Planner → Critic → Executor → Evaluator.

This is what makes Jarvis a proper Multi-Agent System.
"""

from __future__ import annotations

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
from tools.mac_control import MacControlTool
from tools.news import NewsTool
from tools.spotify import SpotifyTool
from tools.weather import WeatherTool
from tools.web_search import WebSearchTool

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
        # Pending email confirmation state
        self._pending_email: EmailDraft | None = None

        # Tool instances
        self.weather    = WeatherTool()
        self.websearch  = WebSearchTool()
        self.news       = NewsTool()
        self.mac        = MacControlTool()
        self.spotify    = SpotifyTool()
        self.document   = DocumentTool()

        print(f"\n🤖 Jarvis Orchestrator ready — model: {model}")
        print(f"   Agents: Router, Memory, Planner, Critic, Evaluator, Summariser, Calendar, Gmail")
        print(f"   Tools:  Weather, WebSearch, News, Mac, Spotify, Document\n")

    # ── Main entry point ───────────────────────────────────────────────────

    async def handle(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
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

        try:
            # ── Step 1: Route ──────────────────────────────────────────────
            routing = await self.router.route(user_request)

            # ── Step 2: Memory retrieval ───────────────────────────────────
            memories = await self.memory.retrieve(user_request)
            print(f"🧠 Retrieved {len(memories)} relevant memories")

            # ── Step 2.5: Short-circuit for simple deterministic tools ───
            shortcut = await self._try_shortcut(routing.primary_agent, user_request)
            if shortcut is not None:
                return shortcut

            # ── Step 3: Plan ───────────────────────────────────────────────
            plan = await self.planner.plan(
                user_request, ctx, memories, model_override=model
            )

            # ── Step 4: Critic reviews plan (with replan loop) ────────────
            plan_verdict = await self.critic.review_plan(plan)
            planning_score = plan_verdict.score

            replan_attempts = 0
            while plan_verdict.replan_needed and replan_attempts < MAX_REPLAN_ATTEMPTS:
                replan_attempts += 1
                print(f"🔄 Replanning (attempt {replan_attempts})...")

                # Inject critic feedback into context
                feedback_ctx = {
                    **ctx,
                    "critic_feedback": "; ".join(plan_verdict.issues),
                    "critic_suggestions": "; ".join(plan_verdict.suggestions),
                }
                plan = await self.planner.plan(
                    user_request, feedback_ctx, memories, model_override=model
                )
                plan.replan_count = replan_attempts
                plan_verdict = await self.critic.review_plan(plan)
                planning_score = max(planning_score, plan_verdict.score)

            # ── Step 5: Execute subtasks ───────────────────────────────────
            results = await self._execute_dag(plan, routing.primary_agent)

            # ── Step 6: Critic reviews results ────────────────────────────
            result_verdict = await self.critic.review_result(plan, results)

            # ── Step 7: Evaluate ───────────────────────────────────────────
            evaluation = self.evaluator.evaluate(
                plan, results, start_time, planning_score=planning_score
            )

            # ── Step 8: Store episodic memory ──────────────────────────────
            await self.memory.store_task_result(
                user_request=user_request,
                intent=plan.intent,
                success=evaluation.success,
                summary=evaluation.feedback,
            )

            # ── Build response ─────────────────────────────────────────────
            message = self._build_response_message(
                user_request, plan, results, routing.primary_agent
            )

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
        """Convert natural language time phrases to ISO datetime."""
        import re
        from datetime import timedelta

        now = datetime.now()
        target = now
        p = phrase.lower()

        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
        }

        if "tomorrow" in p:
            target = now + timedelta(days=1)
        elif "next week" in p:
            target = now + timedelta(days=7)
        else:
            for day_name, day_num in day_map.items():
                if day_name in p:
                    days_ahead = (day_num - now.weekday()) % 7 or 7
                    target = now + timedelta(days=days_ahead)
                    break

        time_match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", p)
        if time_match:
            h = int(time_match.group(1))
            m = int(time_match.group(2) or 0)
            period = time_match.group(3)
            if period == "pm" and h != 12:
                h += 12
            elif period == "am" and h == 12:
                h = 0
            target = target.replace(hour=h, minute=m, second=0, microsecond=0)

        end = target + __import__("datetime").timedelta(hours=1)
        return {
            "datetime": target.isoformat(),
            "date": target.date().isoformat(),
            "time": target.strftime("%H:%M"),
            "end_datetime": end.isoformat(),
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
        weather_keywords = ["weather", "temperature", "forecast", "humid", "rain", "sunny", "cloudy", "wind speed"]
        is_weather_request = (
            primary_agent == AgentRole.WEATHER or
            any(kw in req_lower for kw in weather_keywords)
        )

        if is_weather_request:
            req = req_lower
            is_forecast = any(w in req for w in ["forecast", "week", "tomorrow", "tonight"])

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
            await self.memory.store_task_result(
                user_request, "get_weather", data.get("success", False), msg[:100]
            )
            print(f"Jarvis shortcut: weather — {(_time.time()-start)*1000:.0f}ms")
            return JarvisResponse(
                success=data.get("success", False),
                message=msg,
                latency_ms=(_time.time() - start) * 1000,
            )

        # News shortcut
        news_keywords = ["news", "headlines", "latest news", "top stories", "breaking"]
        if any(kw in req_lower for kw in news_keywords):
            import time as _t
            _s = _t.time()
            # Try to detect topic filter
            topic = None
            for kw in ["about", "on", "regarding"]:
                if kw in req_lower:
                    topic = req_lower.split(kw, 1)[-1].strip().rstrip("?!. ")
                    break
            data = await self.news.get_headlines(topic=topic, max_items=5)
            msg = self.news.format_headlines(data)
            await self.memory.store_task_result(user_request, "get_news", True, msg[:100])
            print(f"Jarvis shortcut: news — {(_t.time()-_s)*1000:.0f}ms")
            return JarvisResponse(success=True, message=msg, latency_ms=(_t.time()-_s)*1000)

        if primary_agent == AgentRole.NEWS:
            data = await self.news.get_headlines(max_items=5)
            msg = self.news.format_headlines(data)
            await self.memory.store_task_result(
                user_request, "get_news", True, msg[:100]
            )
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
            await self.memory.store_task_result(user_request, "read_email", True, msg[:100])
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
            await self.memory.store_task_result(user_request, "send_email", result.get("success", False), msg[:100])
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tc.time()-_sc)*1000)

        # Cancel pending email
        cancel_keywords = ["no", "cancel", "don't send", "do not send", "abort"]
        if self._pending_email and any(kw in req_lower for kw in cancel_keywords):
            self._pending_email = None
            return JarvisResponse(success=True, message="Email cancelled. No email was sent.")

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
            draft = await self.composer.compose(user_request, self.contacts)

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


        # Calendar shortcut — check schedule
        cal_read_keywords = ["what's on my calendar", "my schedule", "my meetings", "what do i have", "events today", "events this week"]
        if any(kw in req_lower for kw in cal_read_keywords):
            import time as _t3
            _s3 = _t3.time()
            result = await self.calendar.search_events()
            msg = result.get("message", "No events found.")
            await self.memory.store_task_result(user_request, "check_calendar", True, msg[:100])
            return JarvisResponse(success=True, message=msg, latency_ms=(_t3.time()-_s3)*1000)

        return None  # No shortcut — proceed with full pipeline

    def _extract_location(self, user_request: str) -> str:
        """
        Extract a city name from a weather request.
        Returns empty string if no specific location found.
        Strips country names — geocoding API works best with city only.
        """
        req = user_request.lower()

        for phrase in [
            "what's the weather in", "what is the weather in",
            "weather in", "weather for", "whats the weather in",
            "weather at", "weather today in", "weather forecast for",
            "forecast for", "forecast in", "temperature in",
        ]:
            if phrase in req:
                raw = user_request[req.index(phrase) + len(phrase):].strip()
                raw = raw.rstrip("?!. ")
                if not raw:
                    return ""
                # Strip trailing country name if 2+ words
                # e.g. "karachi pakistan" -> "karachi"
                # but "new york" stays "new york"
                # We use a simple heuristic: if last word is a known country indicator, drop it
                parts = raw.split()
                if len(parts) >= 2:
                    # Check if last word looks like a country (capitalised, not a city suffix)
                    common_countries = {
                        "pakistan","india","usa","uk","france","germany","china",
                        "japan","australia","canada","brazil","italy","spain",
                        "mexico","russia","nigeria","egypt","turkey","argentina",
                        "bangladesh","indonesia","kenya","ghana","iran","iraq",
                        "vietnam","thailand","malaysia","singapore","uae","qatar"
                    }
                    if parts[-1].lower() in common_countries:
                        city = " ".join(parts[:-1])
                    else:
                        city = raw
                else:
                    city = raw
                return city.strip()
        return ""