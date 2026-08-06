"""
Jarvis — the common tool contract.

Every capability Jarvis has (weather, sports, Gmail, mac control, …) implements
``Tool``. One interface, one result shape, one health signal.

Design notes that matter downstream:

* ``input_schema`` is **JSON Schema**, not an ad-hoc dict. That is deliberate:
  MCP's ``tools/list`` returns JSON Schema verbatim, so in Phase 3 a Tool's
  schema becomes an MCP tool declaration with no translation layer. The same
  schema is what validates params here.

* ``execute()`` **does not raise** for expected failures — a 404 from an API, a
  missing OAuth token, a malformed user param all come back as
  ``ToolResult(success=False, ...)`` carrying a typed ``error_type``. It raises
  only for programmer error (unknown action, broken adapter). This is what lets
  the orchestrator, the MCP gateway and the telemetry layer treat every tool
  identically instead of each caller inventing its own try/except.

* ``health_check()`` is separate from ``execute()`` and cheap, so the Phase 4
  heartbeat can poll a tool that has been idle for an hour and still know
  whether it works.

* Latency is measured here, once, in the base class — not in 14 adapters.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional

try:
    from jsonschema import Draft202012Validator
    from jsonschema import ValidationError as _JSONSchemaValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on a broken install
    JSONSCHEMA_AVAILABLE = False
    Draft202012Validator = None  # type: ignore[assignment]
    _JSONSchemaValidationError = Exception  # type: ignore[misc,assignment]


# ── Errors ──────────────────────────────────────────────────────────────────
#
# These are the *typed* failure modes. They exist so telemetry can aggregate
# ("Gmail failed 12 times this week, 11 of them AUTH") instead of storing 12
# distinct free-text strings, and so the Phase 3 circuit breaker can decide
# what is worth retrying: UPSTREAM and TIMEOUT are transient, INPUT and AUTH
# are not — retrying them just burns the budget.


class ErrorType(str, Enum):
    """Why a tool call failed. Retryable-ness is a property of the type."""

    INPUT = "input"              # bad params — caller's fault, never retry
    AUTH = "auth"                # missing/expired credentials — never retry blindly
    UNAVAILABLE = "unavailable"  # tool can't run here (e.g. macOS tool on Linux)
    TIMEOUT = "timeout"          # exceeded its budget — retryable
    UPSTREAM = "upstream"        # third-party API failed — retryable
    NOT_FOUND = "not_found"      # the thing asked for doesn't exist
    INTERNAL = "internal"        # bug in the tool — not retryable, should alert

    @property
    def retryable(self) -> bool:
        return self in (ErrorType.TIMEOUT, ErrorType.UPSTREAM)


class ToolError(Exception):
    """Base for errors a tool raises internally. Converted to a ToolResult by
    ``BaseTool.execute`` — adapters raise these, callers never see them."""

    error_type: ErrorType = ErrorType.INTERNAL

    def __init__(self, message: str, *, detail: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


class ToolInputError(ToolError):
    error_type = ErrorType.INPUT


class ToolAuthError(ToolError):
    error_type = ErrorType.AUTH


class ToolUnavailableError(ToolError):
    error_type = ErrorType.UNAVAILABLE


class ToolTimeoutError(ToolError):
    error_type = ErrorType.TIMEOUT


class ToolUpstreamError(ToolError):
    error_type = ErrorType.UPSTREAM


class ToolNotFoundError(ToolError):
    error_type = ErrorType.NOT_FOUND


class UnknownActionError(ValueError):
    """Raised when an action isn't declared by the tool. This is a programmer
    error (a bad plan, a typo in the registry), not a runtime failure, so it
    propagates rather than becoming a ToolResult."""


# ── Results ─────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ToolResult:
    """The single return shape for every tool call in Jarvis.

    ``data`` is the machine-readable payload (what dependent subtasks read).
    ``message`` is the human/LLM-readable rendering. Keeping them separate is
    what lets the DAG executor pass structured data between subtasks while the
    UI still gets prose — today those are conflated in most tools.
    """

    success: bool
    tool: str
    action: str
    data: Any = None
    message: str = ""
    error: Optional[str] = None
    error_type: Optional[ErrorType] = None
    latency_ms: float = 0.0
    degraded: bool = False           # succeeded, but on a fallback path
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, tool: str, action: str, data: Any = None, message: str = "",
           **kw: Any) -> "ToolResult":
        return cls(success=True, tool=tool, action=action, data=data,
                   message=message, **kw)

    @classmethod
    def fail(cls, tool: str, action: str, error: str,
             error_type: ErrorType = ErrorType.INTERNAL, **kw: Any) -> "ToolResult":
        return cls(success=False, tool=tool, action=action, error=error,
                   error_type=error_type, **kw)

    @property
    def retryable(self) -> bool:
        return bool(self.error_type and self.error_type.retryable)

    def to_dict(self) -> Dict[str, Any]:
        """Legacy-compatible dict.

        The orchestrator's existing code reads ``result["success"]``,
        ``result["message"]``, ``result["error"]`` and (for DAG dependency
        injection) ``result["result"]``. Emitting those exact keys means the
        registry can be dropped in underneath the current dispatcher without
        touching every call site — the migration stays incremental.
        """
        out: Dict[str, Any] = {
            "success": self.success,
            "tool": self.tool,
            "action": self.action,
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.data is not None:
            out["result"] = self.data
        if self.message:
            out["message"] = self.message
        if self.error:
            out["error"] = self.error
        if self.error_type:
            out["error_type"] = self.error_type.value
        if self.degraded:
            out["degraded"] = True
        if self.meta:
            out["meta"] = self.meta
        return out


class HealthStatus(str, Enum):
    OK = "ok"                    # fully working
    DEGRADED = "degraded"        # working on a fallback (cache, secondary API)
    UNAVAILABLE = "unavailable"  # can't work here — missing creds, wrong OS
    ERROR = "error"              # should work, doesn't
    UNKNOWN = "unknown"          # never checked


@dataclass(slots=True)
class HealthReport:
    status: HealthStatus
    tool: str
    detail: str = ""
    latency_ms: float = 0.0
    checked_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def healthy(self) -> bool:
        return self.status in (HealthStatus.OK, HealthStatus.DEGRADED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "status": self.status.value,
            "healthy": self.healthy,
            "detail": self.detail,
            "latency_ms": round(self.latency_ms, 2),
            "checked_at": self.checked_at.isoformat(),
        }


# ── Action declaration ──────────────────────────────────────────────────────


@dataclass(slots=True)
class Action:
    """One callable operation on a tool.

    A tool is not a single function — ``SportsTool`` alone has scores,
    standings and team search, with different params each. Declaring actions
    individually (rather than one blob schema per tool) is what makes the MCP
    conversion 1 action = 1 MCP tool, which is the granularity MCP expects.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any]
    timeout: float = 30.0
    destructive: bool = False   # requires confirmation before execution
    read_only: bool = True

    def __post_init__(self) -> None:
        self.input_schema.setdefault("type", "object")
        self.input_schema.setdefault("properties", {})
        # additionalProperties defaults open: the Planner LLM routinely emits
        # extra keys, and rejecting the whole call over a stray param would be
        # a regression against today's behaviour. Tools that need strictness
        # set it False explicitly.
        self.input_schema.setdefault("additionalProperties", True)

    def mcp_declaration(self, tool_name: str) -> Dict[str, Any]:
        """This action as an MCP ``tools/list`` entry. Used in Phase 3."""
        return {
            "name": f"{tool_name}.{self.name}",
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": {
                "readOnlyHint": self.read_only,
                "destructiveHint": self.destructive,
            },
        }


# ── The interface ───────────────────────────────────────────────────────────


class Tool(ABC):
    """The contract every Jarvis capability implements."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier, lowercase, no spaces (e.g. ``weather``)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One line. Shown to the Planner LLM and in MCP tool listings."""

    @property
    @abstractmethod
    def actions(self) -> Mapping[str, Action]:
        """Action name → Action."""

    @abstractmethod
    async def execute(self, action: str, params: Optional[Dict[str, Any]] = None,
                      *, timeout: Optional[float] = None) -> ToolResult:
        ...

    @abstractmethod
    async def health_check(self) -> HealthReport:
        ...


class BaseTool(Tool):
    """Concrete base doing the work no adapter should repeat: schema
    validation, timeout enforcement, latency measurement, and turning
    exceptions into ToolResults.

    Subclasses declare ``_name``, ``_description``, build ``_actions``, and
    optionally override ``_check_health``. Handlers are plain async callables
    that return ``(data, message)``, a ``ToolResult``, or raise a ``ToolError``.
    """

    _name: str = "unnamed"
    _description: str = ""
    _default_timeout: float = 30.0
    # Heartbeats (Phase 4) poll this on a schedule, so it must be bounded and
    # short. Tools with a slow probe lower it rather than raising it.
    _health_timeout: float = 10.0

    def __init__(self) -> None:
        self._action_map: Dict[str, Action] = {}
        self._register_actions()
        if JSONSCHEMA_AVAILABLE:
            self._validators = {
                n: Draft202012Validator(a.input_schema)
                for n, a in self._action_map.items()
            }
        else:
            self._validators = {}

    # ── Subclass hooks ──────────────────────────────────────────────────────

    def _register_actions(self) -> None:
        """Populate ``self._action_map``. Subclasses override."""

    async def _check_health(self) -> HealthReport:
        """Default: healthy if the tool declares at least one action. Tools
        with a real backing service (Gmail, Ollama, Spotify) override this
        with an actual cheap probe."""
        status = HealthStatus.OK if self._action_map else HealthStatus.ERROR
        detail = f"{len(self._action_map)} actions" if self._action_map else "no actions registered"
        return HealthReport(status=status, tool=self.name, detail=detail)

    # ── Interface ───────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def actions(self) -> Mapping[str, Action]:
        return dict(self._action_map)

    def add_action(self, action: Action) -> None:
        self._action_map[action.name] = action

    async def execute(self, action: str, params: Optional[Dict[str, Any]] = None,
                      *, timeout: Optional[float] = None) -> ToolResult:
        params = dict(params or {})
        spec = self._action_map.get(action)
        if spec is None:
            # Programmer error, not a runtime failure — surfacing it as a
            # ToolResult would let a typo'd action name masquerade as a
            # transient tool outage in the telemetry.
            raise UnknownActionError(
                f"{self.name!r} has no action {action!r}. "
                f"Available: {sorted(self._action_map)}"
            )

        budget = timeout if timeout is not None else spec.timeout
        started = time.perf_counter()

        def _elapsed() -> float:
            return (time.perf_counter() - started) * 1000

        # ── Validate ────────────────────────────────────────────────────────
        validator = self._validators.get(action)
        if validator is not None:
            errors = sorted(validator.iter_errors(params), key=lambda e: list(e.path))
            if errors:
                first = errors[0]
                where = ".".join(str(p) for p in first.path) or "(root)"
                return ToolResult.fail(
                    self.name, action,
                    error=f"invalid params at {where}: {first.message}",
                    error_type=ErrorType.INPUT,
                    latency_ms=_elapsed(),
                    meta={"violations": [e.message for e in errors[:5]]},
                )

        # ── Run ─────────────────────────────────────────────────────────────
        try:
            raw = await asyncio.wait_for(spec.handler(**params), timeout=budget)
        except asyncio.TimeoutError:
            return ToolResult.fail(
                self.name, action,
                error=f"timed out after {budget:.0f}s",
                error_type=ErrorType.TIMEOUT, latency_ms=_elapsed(),
            )
        except asyncio.CancelledError:
            # Never swallow cancellation — it belongs to the caller's task,
            # and eating it here is how event loops end up un-shutdown-able.
            raise
        except ToolError as exc:
            return ToolResult.fail(
                self.name, action, error=exc.message,
                error_type=exc.error_type, latency_ms=_elapsed(),
                meta={"detail": exc.detail} if exc.detail else {},
            )
        except TypeError as exc:
            # Almost always "unexpected keyword argument" — the schema let a
            # param through that the handler doesn't accept. That's an input
            # problem, and reporting it as INTERNAL would page us for a
            # malformed LLM plan.
            if "unexpected keyword argument" in str(exc) or "positional argument" in str(exc):
                return ToolResult.fail(
                    self.name, action, error=f"parameter mismatch: {exc}",
                    error_type=ErrorType.INPUT, latency_ms=_elapsed(),
                )
            return self._internal_failure(action, exc, _elapsed())
        except Exception as exc:  # noqa: BLE001 - deliberate boundary
            return self._internal_failure(action, exc, _elapsed())

        return self._normalise(action, raw, _elapsed())

    async def health_check(self) -> HealthReport:
        started = time.perf_counter()
        try:
            report = await asyncio.wait_for(self._check_health(),
                                            timeout=self._health_timeout)
        except asyncio.TimeoutError:
            return HealthReport(
                status=HealthStatus.ERROR, tool=self.name,
                detail=f"health check timed out after {self._health_timeout:.0f}s",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            return HealthReport(
                status=HealthStatus.ERROR, tool=self.name,
                detail=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        if not report.latency_ms:
            report.latency_ms = (time.perf_counter() - started) * 1000
        return report

    # ── MCP bridge (Phase 3) ────────────────────────────────────────────────

    def mcp_declarations(self) -> list[Dict[str, Any]]:
        return [a.mcp_declaration(self.name) for a in self._action_map.values()]

    # ── Internals ───────────────────────────────────────────────────────────

    def _internal_failure(self, action: str, exc: BaseException,
                          elapsed: float) -> ToolResult:
        import traceback
        return ToolResult.fail(
            self.name, action,
            error=f"{type(exc).__name__}: {exc}",
            error_type=ErrorType.INTERNAL, latency_ms=elapsed,
            # Keeping the traceback on the result (rather than printing it and
            # moving on, which is what the codebase does in ~340 places today)
            # is what makes the Phase 4 telemetry table actually diagnostic.
            meta={"traceback": traceback.format_exc(limit=8)},
        )

    def _normalise(self, action: str, raw: Any, elapsed: float) -> ToolResult:
        """Coerce whatever a handler returned into a ToolResult."""
        if isinstance(raw, ToolResult):
            raw.latency_ms = raw.latency_ms or elapsed
            raw.tool = raw.tool or self.name
            raw.action = raw.action or action
            return raw

        if isinstance(raw, tuple) and len(raw) == 2:
            data, message = raw
            return ToolResult.ok(self.name, action, data=data,
                                 message=str(message or ""), latency_ms=elapsed)

        if isinstance(raw, dict):
            # Legacy shape: the existing tools return {"success": bool, ...}.
            # Honour it so adapters can pass a tool's native dict straight
            # through during migration instead of being rewritten first.
            if "success" in raw:
                success = bool(raw.get("success"))
                data = raw.get("result", {k: v for k, v in raw.items()
                                          if k not in ("success", "message", "error")})
                if success:
                    return ToolResult.ok(self.name, action, data=data,
                                         message=str(raw.get("message") or ""),
                                         latency_ms=elapsed,
                                         degraded=bool(raw.get("degraded")))
                return ToolResult.fail(
                    self.name, action,
                    error=str(raw.get("error") or "tool reported failure"),
                    error_type=ErrorType(raw["error_type"])
                    if raw.get("error_type") in ErrorType._value2member_map_
                    else ErrorType.UPSTREAM,
                    latency_ms=elapsed,
                )
            return ToolResult.ok(self.name, action, data=raw,
                                 message=str(raw.get("message") or ""),
                                 latency_ms=elapsed)

        return ToolResult.ok(self.name, action, data=raw,
                             message=str(raw) if isinstance(raw, str) else "",
                             latency_ms=elapsed)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.name!r} actions={len(self._action_map)}>"
