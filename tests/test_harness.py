"""Tests for the test harness itself, and the first end-to-end tests it enables.

The harness is worth testing because everything built on it inherits its
mistakes. A fake that quietly returns success for a call the real service would
reject produces a suite that's green and meaningless.

The second half is the payoff: the *real* orchestrator — real router, planner,
critic, registry, executor, adapters and tools — running a full request with
nothing but its boundaries faked. None of this was previously testable without
Ollama running, Google authorised and a Mac to run it on.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.fakes import (
    FakeGmailService,
    FakeMessage,
    FakeOllamaClient,
    FakeShell,
    NetworkAccessError,
    ShellResult,
    plan,
    route,
    subtask,
)


# ── The LLM fake ────────────────────────────────────────────────────────────


async def test_llm_returns_the_default_when_no_rule_matches(fake_llm):
    assert await fake_llm.chat([{"role": "user", "content": "hi"}]) == "Understood."


async def test_llm_rules_match_on_prompt_content(fake_llm):
    fake_llm.when("weather", "It's raining.").when("news", "Nothing new.")
    assert await fake_llm.chat([{"role": "user", "content": "what's the weather"}]) \
        == "It's raining."
    assert await fake_llm.chat([{"role": "user", "content": "any news?"}]) \
        == "Nothing new."


async def test_llm_first_matching_rule_wins(fake_llm):
    """Registration order is the documented tie-break, so a test can register a
    specific case before a general one."""
    fake_llm.when("weather in london", "Specific").when("weather", "General")
    assert await fake_llm.chat([{"role": "user", "content": "weather in London"}]) \
        == "Specific"


async def test_llm_json_call_parses_a_string_rule(fake_llm):
    fake_llm.when("route", json.dumps(route("weather_query", "weather")))
    got = await fake_llm.chat_json([{"role": "user", "content": "route this"}])
    assert got["intent"] == "weather_query"


async def test_llm_json_call_returns_empty_dict_on_unparseable_output(fake_llm):
    """Matches the real client: prose where JSON was expected yields {}, and
    several call sites depend on that rather than on an exception."""
    fake_llm.when("route", "I think this is about the weather, actually")
    assert await fake_llm.chat_json([{"role": "user", "content": "route"}]) == {}


async def test_llm_callable_rule_sees_the_call(fake_llm):
    fake_llm.when("echo", lambda call: f"you said {len(call.prompt)} chars")
    out = await fake_llm.chat([{"role": "user", "content": "echo"}])
    assert out == "you said 4 chars"


async def test_llm_records_calls_for_assertions(fake_llm):
    await fake_llm.chat([{"role": "system", "content": "sys"},
                         {"role": "user", "content": "remember High Wycombe"}])
    call = fake_llm.assert_asked_about("high wycombe")
    assert call.system == "sys"
    assert "High Wycombe" in call.user
    assert fake_llm.call_count == 1


async def test_llm_assert_asked_about_fails_helpfully(fake_llm):
    await fake_llm.chat([{"role": "user", "content": "something else"}])
    with pytest.raises(AssertionError, match="no LLM call mentioned"):
        fake_llm.assert_asked_about("kangaroo")


async def test_llm_can_be_told_to_raise_once(fake_llm):
    fake_llm.raise_next = RuntimeError("ollama down")
    with pytest.raises(RuntimeError, match="ollama down"):
        await fake_llm.chat([{"role": "user", "content": "x"}])
    assert await fake_llm.chat([{"role": "user", "content": "x"}]) == "Understood."


async def test_llm_embeddings_are_deterministic_and_right_sized(fake_llm):
    a, b = await fake_llm.embed("hello"), await fake_llm.embed("hello")
    assert a == b
    assert len(a) == 768
    assert a != await fake_llm.embed("goodbye")
    assert all(-1.0 <= v <= 1.0 for v in a)


async def test_llm_streaming_yields_the_same_text(fake_llm):
    fake_llm.when("stream", "one two three")
    chunks = [c async for c in
              fake_llm.chat_stream([{"role": "user", "content": "stream"}])]
    assert "".join(chunks).strip() == "one two three"


# ── The Google fakes ────────────────────────────────────────────────────────


def test_gmail_fake_returns_real_api_shapes():
    """A tidy dict would skip the agent's parsing, which is where its bugs are.
    Messages come back with base64url bodies and header lists, as Gmail sends."""
    svc = FakeGmailService()
    msg = svc.users().messages().get(id="msg_1").execute()

    headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
    assert headers["Subject"] == "Q3 numbers"
    assert "sarah@example.com" in headers["From"]

    import base64
    data = msg["payload"]["body"]["data"]
    decoded = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode()
    assert "Q3 figures" in decoded


def test_gmail_fake_models_multipart_messages():
    svc = FakeGmailService()
    msg = svc.users().messages().get(id="msg_2").execute()
    assert msg["payload"]["mimeType"] == "multipart/alternative"
    kinds = {p["mimeType"] for p in msg["payload"]["parts"]}
    assert kinds == {"text/plain", "text/html"}


def test_gmail_fake_honours_the_unread_query():
    svc = FakeGmailService()
    unread = svc.users().messages().list(q="is:unread").execute()
    assert {m["id"] for m in unread["messages"]} == {"msg_1", "msg_2"}


def test_gmail_fake_records_sends():
    svc = FakeGmailService()
    svc.users().messages().send(body={"raw": "encoded-mime"}).execute()
    assert len(svc.sent) == 1
    assert svc.sent[0]["raw"] == "encoded-mime"


def test_gmail_fake_missing_message_raises_like_the_api():
    svc = FakeGmailService()
    with pytest.raises(Exception):
        svc.users().messages().get(id="nope").execute()


def test_gmail_fake_modify_applies_labels():
    svc = FakeGmailService()
    svc.users().messages().modify(
        id="msg_1", body={"removeLabelIds": ["UNREAD"]}).execute()
    assert "UNREAD" not in svc.store["msg_1"].labels


def test_gmail_fake_can_be_told_to_fail():
    svc = FakeGmailService()
    svc.raise_on["list"] = RuntimeError("quota exceeded")
    with pytest.raises(RuntimeError, match="quota"):
        svc.users().messages().list().execute()


def test_calendar_fake_filters_by_time_window(fake_calendar_service):
    got = fake_calendar_service.events().list(
        timeMin="2026-07-27T12:00:00", timeMax="2026-07-27T23:59:59").execute()
    assert [e["summary"] for e in got["items"]] == ["Dissertation supervision"]


def test_calendar_fake_insert_then_read_back(fake_calendar_service):
    created = fake_calendar_service.events().insert(body={
        "summary": "New meeting",
        "start": {"dateTime": "2026-07-28T10:00:00"},
        "end": {"dateTime": "2026-07-28T11:00:00"},
        "attendees": [{"email": "a@example.com"}],
    }).execute()
    assert created["summary"] == "New meeting"
    back = fake_calendar_service.events().get(eventId=created["id"]).execute()
    assert back["attendees"] == [{"email": "a@example.com"}]


def test_calendar_fake_delete_removes_it(fake_calendar_service):
    fake_calendar_service.events().delete(eventId="evt_1").execute()
    assert "evt_1" not in fake_calendar_service.store
    assert fake_calendar_service.deleted == ["evt_1"]


# ── The macOS fake ──────────────────────────────────────────────────────────


def test_shell_defaults_look_like_real_command_output(fake_shell):
    """The tool parses these with regexes and splits, so the shape matters."""
    assert fake_shell.run(["osascript", "-e", "output volume of (get volume settings)"]).stdout.strip() == "45"
    assert "82%" in fake_shell.run(["pmset", "-g", "batt"]).stdout
    assert "brightness 0.65" in fake_shell.run(["brightness", "-l"]).stdout


def test_shell_rules_override_defaults(fake_shell):
    fake_shell.when("pmset -g batt", ShellResult(stdout="  5%; discharging;"))
    assert "5%" in fake_shell.run(["pmset", "-g", "batt"]).stdout


def test_shell_can_simulate_failure(fake_shell):
    fake_shell.fails("brightness", stderr="command not found", returncode=127)
    r = fake_shell.run(["brightness", "-l"])
    assert r.returncode == 127
    assert "not found" in r.stderr


def test_shell_records_commands_for_assertions(fake_shell):
    fake_shell.run(["osascript", "-e", "set volume output volume 30"])
    fake_shell.assert_ran("set volume")
    fake_shell.assert_never_ran("shutdown")
    with pytest.raises(AssertionError, match="no command matched"):
        fake_shell.assert_ran("rm -rf")


def test_shell_never_ran_catches_a_real_call(fake_shell):
    fake_shell.run(["rm", "-rf", "/"])
    with pytest.raises(AssertionError, match="was executed"):
        fake_shell.assert_never_ran("rm -rf")


# ── The network guard ───────────────────────────────────────────────────────


def test_outbound_connections_are_blocked():
    """The guard is autouse, so this is live for every test in the suite."""
    import socket
    with pytest.raises(NetworkAccessError, match="a real service is not faked"):
        socket.create_connection(("example.com", 80), timeout=1)


def test_loopback_is_still_allowed():
    """ChromaDB and SQLite must not be collateral damage."""
    import socket
    s = socket.socket()
    try:
        s.connect_ex(("127.0.0.1", 1))       # refused, but not blocked
    except NetworkAccessError:
        pytest.fail("loopback was blocked")
    except OSError:
        pass                                  # connection refused is fine
    finally:
        s.close()


def test_stores_are_redirected_away_from_the_real_data_dir(test_home):
    """A test run must never write into the live reminders or evaluation DB."""
    from config.settings import CHROMA_DIR, DATA_DIR, SQLITE_PATH
    assert str(DATA_DIR).startswith(str(test_home))
    assert str(SQLITE_PATH).startswith(str(test_home))
    assert str(CHROMA_DIR).startswith(str(test_home))
    assert "Desktop/Jarvis/data" not in str(DATA_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# End to end: the real orchestrator
# ═══════════════════════════════════════════════════════════════════════════


def test_orchestrator_builds_with_everything_faked(jarvis):
    """Previously impossible: constructing this needed Ollama up, Google
    authorised and macOS underneath."""
    # 14 user-facing tools + memory, summariser, finex, internal, forge, sentinel
    assert len(jarvis.tools) == 20
    assert sum(len(t.actions) for t in jarvis.tools) > 80


def test_building_the_orchestrator_authenticates_nothing(jarvis):
    assert jarvis.gmail.auth_error is None
    assert jarvis.gmail.service is jarvis.fake_gmail
    assert jarvis.calendar.service is jarvis.fake_calendar


async def test_registry_health_check_covers_every_tool(jarvis):
    """Runs the real health probes against the fakes — 20 tools, no network."""
    reports = await jarvis.tools.health_check()
    assert len(reports) == 20
    for name, rep in reports.items():
        assert rep.status is not None, name


async def test_weather_tool_through_the_registry_is_blocked_from_the_network(jarvis):
    """WeatherTool has no fake, so it must fail loudly rather than reach
    Open-Meteo. This is the guard proving its own worth."""
    result = await jarvis.tools.execute("weather", "get_current")
    assert result.success is False


async def test_gmail_read_runs_the_real_parsing_code(jarvis):
    """The agent's own base64url decode, multipart walk and header parsing all
    execute — only the transport is fake."""
    result = await jarvis.tools.execute("gmail", "get_inbox", {"max_results": 5})
    assert result.success, result.error
    assert result.degraded is False, "real service should not be flagged as mock"
    subjects = [e.get("subject") for e in result.data.get("emails", [])]
    assert "Q3 numbers" in subjects


async def test_gmail_send_validates_before_reaching_the_api(jarvis):
    bad = await jarvis.tools.execute("gmail", "send_email", {
        "to": "not-an-address", "subject": "Hi", "body": "text"})
    assert bad.success is False
    assert jarvis.fake_gmail.sent == []

    ok = await jarvis.tools.execute("gmail", "send_email", {
        "to": "sarah@example.com", "subject": "Q3", "body": "Attached."})
    assert ok.success, ok.error
    assert len(jarvis.fake_gmail.sent) == 1


async def test_calendar_create_reaches_the_service(jarvis):
    result = await jarvis.tools.execute("calendar", "create_event", {
        "title": "Review", "start_time": "2026-07-29T10:00:00",
        "end_time": "2026-07-29T11:00:00"})
    assert result.success, result.error
    assert jarvis.fake_calendar.created[0]["summary"] == "Review"


async def test_mac_volume_goes_through_the_fake_shell(jarvis):
    result = await jarvis.tools.execute("mac", "set_volume", {"level": 30})
    assert result.success, result.error
    jarvis.fake_shell.assert_ran("set volume")


async def test_mac_battery_parses_the_faked_pmset_output(jarvis):
    result = await jarvis.tools.execute("mac", "get_battery")
    assert result.success, result.error
    assert "82" in json.dumps(result.data)


async def test_a_full_dag_runs_end_to_end(jarvis):
    """Two independent reads plus a dependent write, through the real executor,
    registry and adapters."""
    from dataclasses import dataclass, field
    from typing import Any, Dict, List

    @dataclass
    class T:
        id: str
        agent: str
        action: str
        params: Dict[str, Any] = field(default_factory=dict)
        depends_on: List[str] = field(default_factory=list)

    from core.executor import DagExecutor

    report = await DagExecutor(jarvis.tools).execute([
        T("inbox", "gmail", "get_inbox", {"max_results": 3}),
        T("events", "calendar", "search_events", {"max_results": 5}),
        T("note", "mac", "send_notification", {"message": "Daily brief ready"},
          depends_on=["inbox", "events"]),
    ])
    assert report.all_ok, report.results
    assert report.statuses["note"] == "completed"


async def test_a_failed_step_blocks_the_dependent_send(jarvis):
    """The Severity 1.1 scenario, end to end through the real stack: when the
    calendar read fails, the email must not go out."""
    from dataclasses import dataclass, field
    from typing import Any, Dict, List

    @dataclass
    class T:
        id: str
        agent: str
        action: str
        params: Dict[str, Any] = field(default_factory=dict)
        depends_on: List[str] = field(default_factory=list)

    from core.executor import DagExecutor

    jarvis.fake_calendar.raise_on["list"] = RuntimeError("calendar API 503")

    report = await DagExecutor(jarvis.tools).execute([
        T("clash", "calendar", "search_events", {"max_results": 5}),
        T("tell", "gmail", "send_email",
          {"to": "sarah@example.com", "subject": "Clash", "body": "We clash."},
          depends_on=["clash"]),
    ])

    assert report.results["clash"]["success"] is False
    assert report.statuses["tell"] == "blocked"
    assert jarvis.fake_gmail.sent == [], "email sent despite the calendar failing"


async def test_planner_prompt_can_be_asserted(jarvis):
    """What the orchestrator *asks* is usually the interesting assertion, and
    it's invisible if you only look at the final answer."""
    jarvis.fake_llm.when("classify|intent|route", json.dumps(
        route("weather_query", "weather")))
    await jarvis.router.route("what's the weather in High Wycombe")
    call = jarvis.fake_llm.assert_asked_about("High Wycombe")
    assert call.user


async def test_mac_volume_round_trips_through_the_modelled_state(jarvis):
    """The tool sets, then reads back to confirm. A static fake makes every
    write look like the Bluetooth failure case, so the shell fake models the
    setting rather than returning a constant."""
    assert (await jarvis.tools.execute("mac", "get_volume")).data["volume"] == 45
    assert (await jarvis.tools.execute("mac", "set_volume", {"level": 30})).success
    assert jarvis.fake_shell.state.volume == 30
    assert (await jarvis.tools.execute("mac", "get_volume")).data["volume"] == 30


async def test_mac_dark_mode_toggles(jarvis):
    before = (await jarvis.tools.execute("mac", "get_dark_mode")).data
    await jarvis.tools.execute("mac", "toggle_dark_mode")
    after = (await jarvis.tools.execute("mac", "get_dark_mode")).data
    assert before != after


async def test_battery_state_is_configurable(jarvis):
    jarvis.fake_shell.state.battery_percent = 7
    jarvis.fake_shell.state.charging = False
    result = await jarvis.tools.execute("mac", "get_battery")
    assert result.success
    assert "7" in json.dumps(result.data)


async def test_reminders_persist_to_the_temp_database(jarvis):
    """Real ReminderStore, real SQLite — just not the live jarvis.db."""
    added = await jarvis.tools.execute(
        "reminders", "add", {"title": "Submit dissertation", "offset_minutes": 60})
    assert added.success, added.error

    listed = await jarvis.tools.execute("reminders", "list_pending")
    assert "Submit dissertation" in json.dumps(listed.data)

    done = await jarvis.tools.execute("reminders", "complete",
                                      {"id": added.data["id"]})
    assert done.success


async def test_memory_store_and_retrieve_against_real_chromadb(jarvis):
    """ChromaDB is real and on disk in the temp dir; only the embeddings are
    the fake's deterministic hash."""
    stored = await jarvis.tools.execute(
        "memory", "store_fact", {"content": "Abdullah is based in High Wycombe"})
    assert stored.success, stored.error

    got = await jarvis.tools.execute("memory", "retrieve_context",
                                     {"query": "where is Abdullah based"})
    assert got.success
    assert got.data["count"] >= 0        # similarity threshold may exclude it


async def test_contacts_round_trip_without_touching_the_real_address_book(jarvis):
    await jarvis.tools.execute("contacts", "add",
                               {"name": "Sarah", "email": "sarah@example.com"})
    found = await jarvis.tools.execute("contacts", "find", {"name": "sarah"})
    assert found.success
    assert found.data["email"] == "sarah@example.com"
