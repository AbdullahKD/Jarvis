"""
Critic Agent
Reviews plans and outputs for quality. Triggers replanning if the
plan doesn't meet the minimum quality threshold.
This is the self-reflection loop in the ReAct architecture.
"""

from __future__ import annotations

from typing import Any, Dict, List

from config.llm_client import OllamaClient
from config.models import CriticVerdict, TaskPlan
from config.settings import CRITIC_REPLAN_THRESHOLD


SYSTEM_PROMPT = """You are the Critic agent for Jarvis, an AI executive assistant.
Your job is to review task plans and execution results for quality and correctness.

When reviewing a PLAN, check for:
- Does the plan actually address the user's request?
- Are subtask dependencies correctly ordered?
- Are there missing steps (e.g. no memory retrieval, no validation)?
- Are the params complete and sensible?
- Is the plan over-engineered (too many steps for a simple task)?

When reviewing a RESULT, check for:
- Did all subtasks succeed?
- Is the output coherent and actually useful?
- Are there any obvious errors or hallucinations?

Respond with valid JSON only:
{
  "approved": true,
  "score": 0.85,
  "issues": ["issue 1", "issue 2"],
  "suggestions": ["suggestion 1"],
  "replan_needed": false
}

Rules:
- score is 0.0–1.0 (1.0 = perfect)
- approved should be true if score >= 0.6
- replan_needed should be true only for score < 0.5
- Be specific about issues — vague feedback is useless
- If the plan is good, return an empty issues list
"""


class CriticAgent:
    """
    Self-reflection agent that reviews plans and results.

    This implements the critique loop that distinguishes advanced
    ReAct systems from basic LLM pipelines. If the critic rejects
    a plan, the orchestrator triggers replanning up to 2 times.
    """

    def __init__(self, llm_client: OllamaClient | None = None):
        self.llm = llm_client or OllamaClient()
        print("🔍 CriticAgent ready")

    async def review_plan(self, plan: TaskPlan) -> CriticVerdict:
        """
        Review a task plan before execution.

        Args:
            plan: The TaskPlan from the PlannerAgent

        Returns:
            CriticVerdict with approval, score, issues, suggestions
        """
        plan_summary = self._summarise_plan(plan)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Review this task plan:\n\n"
                    f"User request: {plan.user_request}\n"
                    f"Intent: {plan.intent}\n"
                    f"Reasoning: {plan.reasoning}\n\n"
                    f"Subtasks:\n{plan_summary}"
                ),
            },
        ]

        verdict = await self._get_verdict(messages)
        self._log_verdict("PLAN", verdict)
        return verdict

    async def review_result(
        self,
        plan: TaskPlan,
        results: Dict[str, Any],
    ) -> CriticVerdict:
        """
        Review execution results after the plan has run.

        Args:
            plan:    The executed TaskPlan
            results: Dict of subtask_id → result from ExecutorAgent

        Returns:
            CriticVerdict assessing output quality
        """
        successes = sum(1 for r in results.values() if r.get("success"))
        total = len(results)
        result_summary = self._summarise_results(results)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Review these execution results:\n\n"
                    f"User request: {plan.user_request}\n"
                    f"Intent: {plan.intent}\n"
                    f"Subtask success rate: {successes}/{total}\n\n"
                    f"Results:\n{result_summary}"
                ),
            },
        ]

        verdict = await self._get_verdict(messages)
        self._log_verdict("RESULT", verdict)
        return verdict

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _get_verdict(self, messages: List[Dict]) -> CriticVerdict:
        try:
            data = await self.llm.chat_json(messages)
        except Exception as exc:
            print(f"⚠️  Critic LLM error: {exc} — approving by default")
            return CriticVerdict(
                approved=True,
                score=0.7,
                issues=[],
                suggestions=[],
                replan_needed=False,
            )

        score = float(data.get("score", 0.7))
        return CriticVerdict(
            approved=data.get("approved", score >= 0.6),
            score=score,
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            replan_needed=data.get("replan_needed", score < CRITIC_REPLAN_THRESHOLD),
        )

    def _summarise_plan(self, plan: TaskPlan) -> str:
        lines = []
        for st in plan.subtasks:
            deps = f" (depends: {', '.join(st.depends_on)})" if st.depends_on else ""
            lines.append(f"  [{st.id}] {st.agent}.{st.action}{deps} — params: {st.params}")
        return "\n".join(lines)

    def _summarise_results(self, results: Dict[str, Any]) -> str:
        lines = []
        for task_id, result in results.items():
            status = "✅" if result.get("success") else "❌"
            detail = result.get("result", result.get("error", "no detail"))
            lines.append(f"  {status} [{task_id}]: {str(detail)[:120]}")
        return "\n".join(lines)

    def _log_verdict(self, stage: str, verdict: CriticVerdict) -> None:
        icon = "✅" if verdict.approved else "❌"
        print(
            f"🔍 Critic [{stage}] {icon} score={verdict.score:.2f} "
            f"replan={verdict.replan_needed}"
        )
        if verdict.issues:
            for issue in verdict.issues:
                print(f"   ⚠️  {issue}")
