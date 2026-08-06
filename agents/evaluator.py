"""
Evaluator Agent
Scores every task execution. Generates real benchmark data across
multiple models — this is the primary source of benchmark evidence
answering "by which metrics will the MAS be measured?"
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.models import EvaluationResult, TaskPlan
from config.settings import EVALUATOR_MIN_SCORE, SQLITE_PATH


class EvaluatorAgent:
    """
    Scores every task on multiple dimensions and persists results.

    Metrics (all answering the lecturer's question directly):
      - planning_score: quality of the generated plan (0–1)
      - execution_score: % of subtasks that succeeded (0–1)
      - latency_ms: total wall-clock time
      - replan_count: how many times critic triggered replanning
      - overall_score: weighted combination

    All results are stored in SQLite so you can:
      1. Run the benchmark suite → get a CSV/JSON
      2. Drop it into your benchmark Results section
      3. Plot model comparisons (llama3 vs mistral vs gpt-4o)
    """

    def __init__(self):
        self._init_db()
        print(f"📊 EvaluatorAgent ready — results stored at {SQLITE_PATH}")

    # ── Evaluation ─────────────────────────────────────────────────────────

    def _score(
        self,
        plan: TaskPlan,
        results: Dict[str, Any],
        start_time: float,
        planning_score: float = 1.0, # from CriticAgent
    ) -> EvaluationResult:
        """
        Score a completed task execution.

        Args:
            plan: The executed TaskPlan
            results: Dict of subtask_id → result
            start_time: time.time() when execution began
            planning_score: Score from CriticAgent (0–1)

        Returns:
            EvaluationResult with all metrics
        """
        latency_ms = (time.time() - start_time) * 1000

        # Execution score: proportion of subtasks that succeeded
        total = len(results)
        successes = sum(1 for r in results.values() if r.get("success"))
        execution_score = successes / total if total > 0 else 0.0

        # Overall weighted score
        overall = (
            0.4 * planning_score +
            0.4 * execution_score +
            0.2 * (1.0 if execution_score > 0 else 0.0) # any success bonus
        )

        success = overall >= EVALUATOR_MIN_SCORE

        # Human-readable feedback
        feedback = self._generate_feedback(
            execution_score, planning_score, successes, total, plan.replan_count
        )

        result = EvaluationResult(
            task_id=plan.task_id,
            model=plan.model_used,
            intent=plan.intent,
            success=success,
            score=round(overall, 4),
            planning_score=round(planning_score, 4),
            execution_score=round(execution_score, 4),
            latency_ms=round(latency_ms, 2),
            subtask_count=total,
            replan_count=plan.replan_count,
            feedback=feedback,
        )
        return result

    def evaluate(
        self,
        plan: TaskPlan,
        results: Dict[str, Any],
        start_time: float,
        planning_score: float = 1.0,
    ) -> EvaluationResult:
        """Synchronous score-and-persist. Prefer evaluate_async() from async
        code — this blocks the event loop on the SQLite write."""
        result = self._score(plan, results, start_time, planning_score)
        self._store(result)
        self._print(result)
        return result



    async def evaluate_async(
        self,
        plan: TaskPlan,
        results: Dict[str, Any],
        start_time: float,
        planning_score: float = 1.0,
    ) -> EvaluationResult:
        """Score and persist without blocking the event loop.

        evaluate() writes to SQLite inline, and it is called from the async
        request path — so every turn stalled the loop for the write, and for
        the whole busy_timeout whenever the reminder scheduler held the file.
        """
        import asyncio as _asyncio
        result = self._score(plan, results, start_time, planning_score)
        await _asyncio.to_thread(self._store, result)
        self._print(result)
        return result

    # ── Benchmark export ───────────────────────────────────────────────────

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Return all evaluation results as a list of dicts."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM evaluations ORDER BY timestamp DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_model_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Return per-model aggregate stats.
        Perfect for a benchmark comparison table.
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    model,
                    COUNT(*) as total_tasks,
                    ROUND(AVG(score), 4) as avg_score,
                    ROUND(AVG(planning_score), 4) as avg_planning,
                    ROUND(AVG(execution_score), 4) as avg_execution,
                    ROUND(AVG(latency_ms), 2) as avg_latency_ms,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                    ROUND(AVG(replan_count), 2) as avg_replans
                FROM evaluations
                GROUP BY model
            """).fetchall()
            return {row["model"]: dict(row) for row in rows}

    def export_json(self, path: Optional[Path] = None) -> Path:
        """Export all results to JSON for benchmark appendix."""
        out = path or (SQLITE_PATH.parent / "benchmark_results.json")
        data = {
            "exported_at": datetime.now().isoformat(),
            "model_summary": self.get_model_summary(),
            "all_results": self.get_all_results(),
        }
        out.write_text(json.dumps(data, indent=2))
        print(f"📊 Benchmark results exported → {out}")
        return out

    def export_csv(self, path: Optional[Path] = None) -> Path:
        """Export all results to CSV for benchmark charts."""
        import csv
        out = path or (SQLITE_PATH.parent / "benchmark_results.csv")
        rows = self.get_all_results()
        if not rows:
            print("⚠️ No results to export")
            return out
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"📊 CSV exported → {out}")
        return out

    async def store_async(self, r: EvaluationResult) -> None:
        """Persist off the event loop.

        _store() is called from the async request path. sqlite3 is blocking, so
        every evaluation stalled the loop for the duration of the write — and
        under contention, for the whole busy_timeout.
        """
        import asyncio as _asyncio
        await _asyncio.to_thread(self._store, r)

    # ── Internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _connect():
        """Open a connection with the pragmas this database needs.

        WAL lets the reminder scheduler read while an evaluation is written,
        instead of both contending for a whole-file lock; busy_timeout turns a
        collision into a short wait rather than an immediate
        "database is locked". ReminderStore already sets WAL on the same file —
        this connection was still opening in the default rollback-journal mode.
        """
        conn = sqlite3.connect(SQLITE_PATH, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    task_id TEXT PRIMARY KEY,
                    model TEXT,
                    intent TEXT,
                    success INTEGER,
                    score REAL,
                    planning_score REAL,
                    execution_score REAL,
                    latency_ms REAL,
                    subtask_count INTEGER,
                    replan_count INTEGER,
                    feedback TEXT,
                    timestamp TEXT
                )
            """)

    def _store(self, r: EvaluationResult) -> None:
        with self._connect() as conn:
            # Columns named explicitly: positional VALUES silently writes into
            # the wrong fields the moment a column is added to the schema.
            conn.execute("""
                INSERT OR REPLACE INTO evaluations (
                    task_id, model, intent, success, score,
                    planning_score, execution_score, latency_ms,
                    subtask_count, replan_count, feedback, timestamp
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                r.task_id, r.model, r.intent, int(r.success),
                r.score, r.planning_score, r.execution_score,
                r.latency_ms, r.subtask_count, r.replan_count,
                r.feedback, r.timestamp.isoformat(),
            ))

    def _generate_feedback(
        self,
        exec_score: float,
        plan_score: float,
        successes: int,
        total: int,
        replans: int,
    ) -> str:
        parts = [f"{successes}/{total} subtasks succeeded"]
        if exec_score == 1.0:
            parts.append("all subtasks completed successfully")
        elif exec_score >= 0.5:
            parts.append("majority of subtasks completed")
        else:
            parts.append("most subtasks failed")
        if plan_score < 0.6:
            parts.append("plan quality was below threshold")
        if replans > 0:
            parts.append(f"required {replans} replan(s)")
        return "; ".join(parts)

    def _print(self, r: EvaluationResult) -> None:
        icon = "✅" if r.success else "❌"
        print(
            f"📊 Evaluation {icon} | score={r.score:.3f} | "
            f"plan={r.planning_score:.2f} | exec={r.execution_score:.2f} | "
            f"latency={r.latency_ms:.0f}ms | model={r.model}"
        )
