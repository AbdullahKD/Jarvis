"""Tests for core.registry — the replacement for orchestrator._dispatch's
240-line if/elif chain.

The behaviours worth pinning are the ones the old chain got wrong: unknown
tools returning something sane instead of falling through, action aliases
being resolvable in one place rather than scattered, and every registered tool
being reachable (the old chain silently omitted six of them).
"""

from __future__ import annotations

import pytest

from core.registry import Resolution, ToolNotRegisteredError, ToolRegistry
from core.tool import Action, BaseTool, ErrorType, HealthStatus, UnknownActionError


# ── Registration ────────────────────────────────────────────────────────────


def test_register_and_lookup(registry, fake_tool):
    assert len(registry) == 1
    assert registry.get("fake") is fake_tool
    assert registry.get("FAKE") is fake_tool          # case-insensitive
    assert "fake" in registry
    assert registry.names == ["fake"]


def test_aliases_resolve_to_the_same_tool(registry, fake_tool):
    assert registry.get("fakey") is fake_tool
    assert registry.get("double") is fake_tool
    assert "double" in registry


def test_duplicate_registration_is_rejected(registry, fake_tool):
    with pytest.raises(ValueError, match="already registered"):
        registry.register(fake_tool)


def test_tool_with_no_actions_is_rejected():
    class Empty(BaseTool):
        _name = "empty"
        _description = "declares nothing"

    with pytest.raises(ValueError, match="no actions"):
        ToolRegistry().register(Empty())


def test_conflicting_alias_is_rejected(fake_tool):
    from tests.conftest import FakeTool

    r = ToolRegistry()
    r.register(fake_tool, aliases=("shared",))

    other = FakeTool()
    other._name = "other"
    with pytest.raises(ValueError, match="already maps to"):
        r.register(other, aliases=("shared",))


def test_alias_colliding_with_a_real_tool_name_is_rejected(fake_tool):
    from tests.conftest import FakeTool

    r = ToolRegistry()
    r.register(fake_tool)
    other = FakeTool()
    other._name = "other"
    with pytest.raises(ValueError, match="collides"):
        r.register(other, aliases=("fake",))


def test_action_alias_pointing_at_a_missing_action_is_rejected(fake_tool):
    """Catching this at registration is the whole point — the old dispatcher
    hid the same mistake until a user happened to trigger that branch."""
    r = ToolRegistry()
    with pytest.raises(ValueError, match="undeclared actions"):
        r.register(fake_tool, action_aliases={"whatever": "does_not_exist"})


def test_unregister_removes_tool_and_its_aliases(registry):
    registry.unregister("fake")
    assert len(registry) == 0
    assert "fakey" not in registry
    assert registry.try_get("fake") is None


# ── Resolution ──────────────────────────────────────────────────────────────


def test_resolve_direct_action(registry, fake_tool):
    res = registry.resolve("fake", "echo")
    assert isinstance(res, Resolution)
    assert res.tool is fake_tool
    assert res.action == "echo"
    assert res.alias_used is None


def test_resolve_via_action_alias(registry):
    res = registry.resolve("fake", "say")
    assert res.action == "echo"
    assert res.alias_used == "say"


def test_resolve_is_case_insensitive(registry):
    assert registry.resolve("FAKE", "ECHO").action == "echo"


def test_resolve_unknown_tool_raises(registry):
    with pytest.raises(ToolNotRegisteredError, match="no tool named"):
        registry.resolve("nope", "echo")


def test_resolve_unknown_action_raises_and_lists_options(registry):
    with pytest.raises(UnknownActionError) as exc:
        registry.resolve("fake", "nonsense")
    assert "echo" in str(exc.value)
    assert "say" in str(exc.value)      # aliases listed too


# ── Execution ───────────────────────────────────────────────────────────────


async def test_execute_routes_to_the_tool(registry):
    r = await registry.execute("fake", "echo", {"text": "routed"})
    assert r.success is True
    assert r.data == {"echoed": "routed"}


async def test_execute_through_alias_records_it(registry):
    r = await registry.execute("fakey", "say", {"text": "hi"})
    assert r.success is True
    assert r.meta["action_alias"] == "say"


async def test_execute_unknown_tool_returns_failed_result_not_raise(registry):
    """A hallucinated tool name from the Planner is expected traffic. It must
    come back as a NOT_FOUND result the DAG can record, not an exception that
    unwinds the whole plan."""
    r = await registry.execute("teleporter", "engage")
    assert r.success is False
    assert r.error_type is ErrorType.NOT_FOUND
    assert "teleporter" in r.error


async def test_execute_unknown_action_returns_failed_result(registry):
    r = await registry.execute("fake", "nonsense")
    assert r.success is False
    assert r.error_type is ErrorType.NOT_FOUND


async def test_execute_propagates_timeout_override(registry):
    r = await registry.execute("fake", "slow", {"seconds": 0.05}, timeout=2.0)
    assert r.success is True


# ── Health ──────────────────────────────────────────────────────────────────


async def test_health_check_covers_every_registered_tool(registry):
    reports = await registry.health_check()
    assert set(reports) == {"fake"}
    assert reports["fake"].status is HealthStatus.OK


async def test_health_check_isolates_a_tool_that_raises(fake_tool):
    class Exploding(BaseTool):
        _name = "exploding"
        _description = "health_check itself raises"

        def _register_actions(self):
            self.add_action(Action(name="noop", description="",
                                   input_schema={}, handler=self._noop))

        async def _noop(self):
            return None

        async def health_check(self):          # bypasses BaseTool's guard
            raise RuntimeError("kaboom")

    r = ToolRegistry()
    r.register(fake_tool)
    r.register(Exploding())

    reports = await r.health_check()
    # One broken tool must not blank the other's status on the dashboard.
    assert reports["fake"].healthy is True
    assert reports["exploding"].status is HealthStatus.ERROR
    assert "kaboom" in reports["exploding"].detail


# ── Introspection ───────────────────────────────────────────────────────────


def test_describe_exposes_schemas_and_aliases(registry):
    (desc,) = registry.describe()
    assert desc["name"] == "fake"
    assert sorted(desc["aliases"]) == ["double", "fakey"]
    actions = {a["name"]: a for a in desc["actions"]}
    assert actions["echo"]["input_schema"]["required"] == ["text"]
    assert actions["wipe"]["destructive"] is True


def test_mcp_declarations_cover_all_actions(registry, fake_tool):
    decls = registry.mcp_declarations()
    assert len(decls) == len(fake_tool.actions)
    assert all(d["name"].startswith("fake.") for d in decls)


def test_planner_catalogue_marks_required_params_and_destructive(registry):
    cat = registry.planner_catalogue()
    assert "fake (aka double, fakey)" in cat
    assert "echo(text*)" in cat          # * marks required
    assert "[destructive]" in cat
    assert "wipe(no params)" in cat


async def test_every_registered_tool_is_actually_reachable(registry):
    """Guards the exact class of bug found in the audit: six tools existed but
    had no branch in the dispatcher, so nothing could ever call them."""
    for tool in registry:
        for action_name in tool.actions:
            res = registry.resolve(tool.name, action_name)
            assert res.tool is tool
            assert res.action == action_name
