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
        # Pending confirmation states
        self._pending_email: EmailDraft | None = None
        self._pending_meeting: dict | None = None

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
            if is_research:
                searches = [query, query + " explained", query + " overview"]
                all_results = []
                searches = [query, query + " explained", query + " overview", query + " AI"]
                all_results = []
                for q in searches[:3]:
                    data = await self.websearch.search(q, max_results=3)
                    if data.get("success") and data.get("results"):
                        for r in data.get("results", []):
                            if r not in all_results:
                                all_results.append(r)
                # Also try direct Wikipedia for comprehensive coverage
                if len(all_results) < 3:
                    wiki_data = await self.websearch._wiki(query, 3)
                    if wiki_data.get("success"):
                        for r in wiki_data.get("results", []):
                            if r not in all_results:
                                all_results.append(r)
                if all_results:
                    snips = [r.get("snippet", "")[:200] for r in all_results[:6]]
                    combined2 = " ".join(snips)
                    rp = "Summarise what you know about " + query + " using these sources: " + combined2 + ". Use bullet points."
                    report = await self.llm.chat([{"role": "user", "content": rp}])
                    msg = "Here is what I found about " + query + ": " + report.strip()
                else:
                    msg = "Could not find enough information about: " + query
            else:
                # Single search
                data = await self.websearch.search(query, max_results=5)
                if data.get("success") and data.get("results"):
                    snips2 = [r.get("snippet", "")[:300] for r in data["results"][:3]]
                    ct = " ".join(snips2)
                    ap = "Answer this question: " + query + ". Based on: " + ct + ". Be concise and clear."
                    summary = await self.llm.chat([{"role": "user", "content": ap}])
                    msg = summary.strip()
                else:
                    msg = "Could not find results for: " + query

            await self.memory.store_task_result(user_request, "web_search", True, msg[:100])
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

            await self.memory.store_task_result(user_request, "schedule_meeting", result.get("success", False), msg[:100])
            return JarvisResponse(success=result.get("success", False), message=msg, latency_ms=(_tconf.time()-_sconf)*1000)

        # Cancel pending meeting
        if self._pending_meeting and any(kw in req_lower for kw in ["cancel", "no", "don't book", "abort", "never mind"]):
            self._pending_meeting = None
            return JarvisResponse(success=True, message="Meeting cancelled. Nothing was added to your calendar.")

        # Schedule/create event shortcut
        schedule_keywords = ["schedule", "book a meeting", "create an event", "add to calendar", "set up a meeting", "arrange a meeting"]
        if any(kw in req_lower for kw in schedule_keywords):
            import time as _tcal
            _scal = _tcal.time()

            # Extract title
            title_match = _recal.search(r"(?:called|named|titled|about)\s+(.+?)\s+(?:on|at|tomorrow|next|this|for)", user_request, _recal.IGNORECASE)
            if not title_match:
                title_match = _recal.search(r"(?:meeting|event|appointment)\s+(?:with\s+\w+\s+)?(?:called|named|about)?\s*(.+?)\s*(?:at|on|tomorrow)", user_request, _recal.IGNORECASE)
            title = title_match.group(1).strip() if title_match else "Meeting"

            # Extract attendees
            attendee_match = _recal.search(r'with\s+([\w\s,and]+?)\s+(?:at|on|tomorrow|next|called|about)', user_request, _recal.IGNORECASE)
            attendees = []
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