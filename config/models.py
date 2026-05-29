"""
Core data models for Jarvis.
All agents share these types — import from here, never redefine.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enums ──────────────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING    = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED  = "completed"
    FAILED     = "failed"
    BLOCKED    = "blocked"
    REPLANNING = "replanning"


class MemoryType(str, Enum):
    EPISODIC   = "episodic"    # things that happened
    SEMANTIC   = "semantic"    # facts and preferences
    PROCEDURAL = "procedural"  # how to do things


class AgentRole(str, Enum):
    ROUTER     = "router"
    PLANNER    = "planner"
    MEMORY     = "memory"
    EXECUTOR   = "executor"
    CRITIC     = "critic"
    EVALUATOR  = "evaluator"
    SUMMARISER = "summariser"
    CALENDAR   = "calendar"
    EMAIL      = "email"
    REMINDER   = "reminder"
    NOTES      = "notes"
    WEBSEARCH  = "websearch"
    RESEARCH   = "research"
    NEWS       = "news"
    WEATHER    = "weather"
    MAC        = "mac"
    SPOTIFY    = "spotify"
    FILE       = "file"
    DOCUMENT   = "document"
    FINEX      = "finex"   # Financial-statement Q&A sub-agent (Bestway, HBL, …)


# ── Core data structures ───────────────────────────────────────────────────

@dataclass
class Subtask:
    id: str
    action: str
    agent: str
    params: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None


@dataclass
class TaskPlan:
    task_id: str
    user_request: str
    intent: str
    subtasks: List[Subtask]
    reasoning: str
    created_at: datetime
    model_used: str = ""
    replan_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "user_request": self.user_request,
            "intent": self.intent,
            "reasoning": self.reasoning,
            "model_used": self.model_used,
            "replan_count": self.replan_count,
            "created_at": self.created_at.isoformat(),
            "subtasks": [
                {
                    "id": st.id,
                    "action": st.action,
                    "agent": st.agent,
                    "params": st.params,
                    "depends_on": st.depends_on,
                    "status": st.status.value,
                    "result": st.result,
                    "error": st.error,
                    "duration_ms": st.duration_ms,
                }
                for st in self.subtasks
            ],
        }


@dataclass
class MemoryItem:
    id: str
    content: str
    memory_type: MemoryType
    metadata: Dict[str, Any]
    created_at: datetime
    relevance_score: Optional[float] = None


@dataclass
class EvaluationResult:
    task_id: str
    model: str
    intent: str
    success: bool
    score: float                    # 0.0 – 1.0 overall
    planning_score: float           # quality of the plan
    execution_score: float          # % subtasks succeeded
    latency_ms: float               # total wall-clock time
    subtask_count: int
    replan_count: int
    feedback: str                   # human-readable explanation
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "intent": self.intent,
            "success": self.success,
            "score": round(self.score, 4),
            "planning_score": round(self.planning_score, 4),
            "execution_score": round(self.execution_score, 4),
            "latency_ms": round(self.latency_ms, 2),
            "subtask_count": self.subtask_count,
            "replan_count": self.replan_count,
            "feedback": self.feedback,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CriticVerdict:
    approved: bool
    score: float           # 0.0 – 1.0
    issues: List[str]      # what's wrong
    suggestions: List[str] # how to fix it
    replan_needed: bool


@dataclass
class RouterDecision:
    primary_agent: AgentRole
    supporting_agents: List[AgentRole]
    confidence: float
    reasoning: str
    tier: int = 3  # 1=tool-only, 2=simple LLM, 3=full pipeline


@dataclass
class JarvisResponse:
    """Final response returned to the user."""
    success: bool
    message: str                          # human-readable response
    task_plan: Optional[Dict] = None
    evaluation: Optional[Dict] = None
    data: Optional[Dict] = None           # raw data if needed
    error: Optional[str] = None
    latency_ms: float = 0.0
