"""
Jarvis — the tool registry.

One lookup table replacing the 240-line ``if agent == ... elif agent == ...``
chain in ``orchestrator._dispatch``. Beyond removing the branching, it fixes a
real gap: six tools (sports, markets, prayer times, file manager, contacts,
brain) currently have no branch in that chain at all, so the Planner could
never reach them through the DAG path. Registration makes reachability
structural rather than something you have to remember to hand-wire.

Also the seam for Phase 3: ``MCPGateway`` will implement the same lookup
surface, so swapping a local tool for an MCP-backed one is a registration
change, not a call-site change.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

from core.tool import (
    Action,
    ErrorType,
    HealthReport,
    HealthStatus,
    Tool,
    ToolResult,
    UnknownActionError,
)

logger = logging.getLogger("jarvis.registry")


class ToolNotRegisteredError(KeyError):
    """Asked for a tool nobody registered."""


@dataclass(slots=True)
class Resolution:
    """The outcome of resolving an (agent, action) pair from a plan."""

    tool: Tool
    action: str
    alias_used: Optional[str] = None


class ToolRegistry:
    """Holds every Tool and resolves plan subtasks to them.

    Aliases exist because the Planner LLM is not consistent about names — it
    emits ``email``/``gmail``/``mail`` for the same tool, and ``get_events``
    vs ``search_events`` for the same action. Today that inconsistency is
    absorbed by hand-written ``elif action in ("search_events", "get_events")``
    clauses scattered through the dispatcher. Centralising it means one place
    to look, and a Planner prompt that can be generated from the registry
    instead of maintained in parallel with it.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._aliases: Dict[str, str] = {}
        self._action_aliases: Dict[str, Dict[str, str]] = {}

    # ── Registration ────────────────────────────────────────────────────────

    def register(self, tool: Tool, *, aliases: Iterable[str] = (),
                 action_aliases: Optional[Dict[str, str]] = None) -> None:
        key = tool.name.lower()
        if key in self._tools:
            raise ValueError(f"tool {key!r} already registered")
        if not tool.actions:
            raise ValueError(f"tool {key!r} declares no actions")

        self._tools[key] = tool
        for alias in aliases:
            a = alias.lower()
            if a in self._aliases and self._aliases[a] != key:
                raise ValueError(
                    f"alias {a!r} already maps to {self._aliases[a]!r}, "
                    f"cannot remap to {key!r}"
                )
            if a in self._tools:
                raise ValueError(f"alias {a!r} collides with registered tool name")
            self._aliases[a] = key

        if action_aliases:
            resolved = {k.lower(): v.lower() for k, v in action_aliases.items()}
            unknown = [v for v in resolved.values() if v not in tool.actions]
            if unknown:
                raise ValueError(
                    f"tool {key!r}: action aliases point at undeclared actions {unknown}"
                )
            self._action_aliases[key] = resolved

        logger.debug("registered tool %s (%d actions, %d aliases)",
                     key, len(tool.actions), len(list(aliases)))

    def unregister(self, name: str) -> None:
        key = name.lower()
        self._tools.pop(key, None)
        self._action_aliases.pop(key, None)
        for a, target in list(self._aliases.items()):
            if target == key:
                del self._aliases[a]

    # ── Lookup ──────────────────────────────────────────────────────────────

    def get(self, name: str) -> Tool:
        key = name.lower()
        key = self._aliases.get(key, key)
        try:
            return self._tools[key]
        except KeyError:
            raise ToolNotRegisteredError(
                f"no tool named {name!r}; registered: {sorted(self._tools)}"
            ) from None

    def try_get(self, name: str) -> Optional[Tool]:
        try:
            return self.get(name)
        except ToolNotRegisteredError:
            return None

    def resolve(self, tool_name: str, action: str) -> Resolution:
        """Resolve a plan's (agent, action) to a concrete tool + action name."""
        tool = self.get(tool_name)
        act = action.lower()
        alias_used = None
        if act not in tool.actions:
            mapped = self._action_aliases.get(tool.name.lower(), {}).get(act)
            if mapped:
                alias_used, act = act, mapped
            else:
                raise UnknownActionError(
                    f"tool {tool.name!r} has no action {action!r}. "
                    f"Available: {sorted(tool.actions)}"
                    + (f"; aliases: {sorted(self._action_aliases.get(tool.name.lower(), {}))}"
                       if self._action_aliases.get(tool.name.lower()) else "")
                )
        return Resolution(tool=tool, action=act, alias_used=alias_used)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        key = name.lower()
        return key in self._tools or key in self._aliases

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> List[str]:
        return sorted(self._tools)

    # ── Execution ───────────────────────────────────────────────────────────

    async def execute(self, tool_name: str, action: str,
                      params: Optional[Dict[str, Any]] = None,
                      *, timeout: Optional[float] = None) -> ToolResult:
        """Resolve and run. Unknown tool/action come back as a failed
        ToolResult here (rather than raising as they do on ``Tool.execute``)
        because the caller is the DAG executor acting on LLM-generated plans —
        a hallucinated tool name is expected traffic, not a crash."""
        try:
            res = self.resolve(tool_name, action)
        except ToolNotRegisteredError as exc:
            return ToolResult.fail(tool_name, action, error=str(exc),
                                   error_type=ErrorType.NOT_FOUND)
        except UnknownActionError as exc:
            return ToolResult.fail(tool_name, action, error=str(exc),
                                   error_type=ErrorType.NOT_FOUND)

        result = await res.tool.execute(res.action, params, timeout=timeout)
        if res.alias_used:
            result.meta.setdefault("action_alias", res.alias_used)
        return result

    # ── Health ──────────────────────────────────────────────────────────────

    async def health_check(self, *, concurrency: int = 6
                           ) -> Dict[str, HealthReport]:
        """Check every tool, bounded concurrency so a heartbeat doesn't open 14
        simultaneous network connections. ``Tool.health_check`` already
        swallows its own exceptions, but the gather is guarded anyway — a tool
        that raises on construction of its check shouldn't blank the dashboard.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _one(tool: Tool) -> tuple[str, HealthReport]:
            async with sem:
                try:
                    return tool.name, await tool.health_check()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("health check raised for %s: %s", tool.name, exc)
                    return tool.name, HealthReport(
                        status=HealthStatus.ERROR, tool=tool.name,
                        detail=f"{type(exc).__name__}: {exc}",
                    )

        pairs = await asyncio.gather(*(_one(t) for t in self._tools.values()))
        return dict(pairs)

    # ── Introspection ───────────────────────────────────────────────────────

    def describe(self) -> List[Dict[str, Any]]:
        """Registry contents, for the /health dashboard and for generating the
        Planner's tool list. Generating that prompt from here instead of
        maintaining it by hand is what stops the Planner emitting subtasks for
        tools that don't exist."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "aliases": sorted(a for a, t in self._aliases.items()
                                  if t == tool.name.lower()),
                "actions": [
                    {
                        "name": a.name,
                        "description": a.description,
                        "input_schema": a.input_schema,
                        "destructive": a.destructive,
                        "read_only": a.read_only,
                        "timeout": a.timeout,
                    }
                    for a in tool.actions.values()
                ],
            }
            for tool in self._tools.values()
        ]

    def mcp_declarations(self) -> List[Dict[str, Any]]:
        """Every action across every tool as MCP tool declarations."""
        out: List[Dict[str, Any]] = []
        for tool in self._tools.values():
            declare = getattr(tool, "mcp_declarations", None)
            if callable(declare):
                out.extend(declare())
            else:  # a Tool implemented without BaseTool
                out.extend(a.mcp_declaration(tool.name) for a in tool.actions.values())
        return out

    def planner_catalogue(self) -> str:
        """Compact text listing for the Planner system prompt."""
        lines: List[str] = []
        for tool in sorted(self._tools.values(), key=lambda t: t.name):
            aliases = sorted(a for a, t in self._aliases.items()
                             if t == tool.name.lower())
            head = tool.name + (f" (aka {', '.join(aliases)})" if aliases else "")
            lines.append(f"- {head}: {tool.description}")
            for act in tool.actions.values():
                required = act.input_schema.get("required", [])
                props = list(act.input_schema.get("properties", {}))
                sig = ", ".join(
                    f"{p}*" if p in required else p for p in props
                ) or "no params"
                flag = " [destructive]" if act.destructive else ""
                lines.append(f"    · {act.name}({sig}) — {act.description}{flag}")
        return "\n".join(lines)


# Module-level default registry. Tools register into this at import of
# ``core.bootstrap``; tests build their own ToolRegistry() instead of
# monkeypatching a global, which is why this is a plain object and not a
# singleton class.
default_registry = ToolRegistry()
