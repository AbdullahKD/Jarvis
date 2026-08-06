"""Tests for the n8n workflow patcher.

Shaped around the real discovery workflow: two triggers (schedule + manual)
both feeding a Config node, then a chain into Google Sheets. The property that
matters is that the added webhook lands on the *same* run path as the manual
trigger — a webhook wired anywhere else would run a different workflow than the
one the user tests by hand.
"""

from __future__ import annotations

import json

import pytest

from tools.n8n_trigger import (
    WEBHOOK_TYPE,
    add_trigger,
    audit_secrets,
    classify_probe,
    downstream_of,
    existing_webhook,
    last_failure,
    make_webhook_node,
    parse_execution_error,
    registered_webhooks,
    webhook_workflow_ids,
)


def workflow(with_manual=True, with_schedule=True, connected=True):
    nodes, conns = [], {}
    if with_schedule:
        nodes.append({"id": "s1", "name": "Daily 7am", "type": "n8n-nodes-base.scheduleTrigger",
                      "typeVersion": 1.2, "position": [7600, 1808], "parameters": {}})
        if connected:
            conns["Daily 7am"] = {"main": [[{"node": "Config", "type": "main", "index": 0}]]}
    if with_manual:
        nodes.append({"id": "m1", "name": "Run manually", "type": "n8n-nodes-base.manualTrigger",
                      "typeVersion": 1, "position": [7600, 2000], "parameters": {}})
        if connected:
            conns["Run manually"] = {"main": [[{"node": "Config", "type": "main", "index": 0}]]}
    nodes.append({
        "id": "c1", "name": "Config", "type": "n8n-nodes-base.set", "typeVersion": 3.4,
        "position": [7824, 1904],
        "parameters": {"assignments": {"assignments": [
            {"name": "JOBS_SHEET_ID", "value": "1ptcU1E8rsAIdhd49DiFcCsHZjNmUcD1B"},
            {"name": "REED_API_KEY", "value": "f781e1ab-d5dd-4904-91b8-a1c5148deec4"},
            {"name": "CAREERJET_AFFID", "value": ""},
        ]}},
    })
    conns["Config"] = {"main": [[{"node": "Get existing jobs", "type": "main", "index": 0}]]}
    nodes.append({"id": "g1", "name": "Get existing jobs",
                  "type": "n8n-nodes-base.googleSheets", "typeVersion": 4.5,
                  "position": [8048, 1904], "parameters": {}})
    return {"nodes": nodes, "connections": conns, "pinData": {}, "meta": {}}


# ── Adding the trigger ──────────────────────────────────────────────────────


def test_adds_a_webhook_node():
    out, summary = add_trigger(workflow(), "discovery")
    hooks = [n for n in out["nodes"] if n["type"] == WEBHOOK_TYPE]
    assert len(hooks) == 1
    assert hooks[0]["parameters"]["path"] == "discovery"
    assert hooks[0]["parameters"]["httpMethod"] == "POST"
    assert "discovery" in summary


def test_webhook_targets_the_same_nodes_as_the_manual_trigger():
    """The whole point: firing the webhook must run exactly what "Test
    workflow" runs. Wiring it elsewhere would run a different workflow."""
    out, _ = add_trigger(workflow(), "discovery")
    manual_targets = downstream_of(out, "Run manually")
    hook_targets = downstream_of(out, "Jarvis trigger")
    assert hook_targets == manual_targets
    assert hook_targets[0]["node"] == "Config"


def test_falls_back_to_the_schedule_trigger_when_there_is_no_manual_one():
    out, summary = add_trigger(workflow(with_manual=False), "discovery")
    assert downstream_of(out, "Jarvis trigger")[0]["node"] == "Config"
    assert "Daily 7am" in summary


def test_responds_on_receipt_not_on_completion():
    """A multi-minute discovery run would otherwise hold the HTTP connection
    open until something in between times out and calls it a failure."""
    out, _ = add_trigger(workflow(), "discovery")
    hook = next(n for n in out["nodes"] if n["type"] == WEBHOOK_TYPE)
    assert hook["parameters"]["responseMode"] == "onReceived"


def test_is_idempotent():
    once, _ = add_trigger(workflow(), "discovery")
    twice, summary = add_trigger(once, "discovery")
    assert len([n for n in twice["nodes"] if n["type"] == WEBHOOK_TYPE]) == 1
    assert "already has" in summary


def test_a_second_distinct_path_is_added_alongside():
    once, _ = add_trigger(workflow(), "discovery")
    twice, _ = add_trigger(once, "responses", name="Jarvis trigger 2")
    paths = {n["parameters"]["path"] for n in twice["nodes"] if n["type"] == WEBHOOK_TYPE}
    assert paths == {"discovery", "responses"}


def test_does_not_mutate_the_input():
    wf = workflow()
    before = json.dumps(wf, sort_keys=True)
    add_trigger(wf, "discovery")
    assert json.dumps(wf, sort_keys=True) == before


def test_existing_nodes_and_connections_are_preserved():
    wf = workflow()
    out, _ = add_trigger(wf, "discovery")
    for name in ("Daily 7am", "Run manually", "Config", "Get existing jobs"):
        assert any(n["name"] == name for n in out["nodes"])
    assert out["connections"]["Config"] == wf["connections"]["Config"]
    assert out["connections"]["Run manually"] == wf["connections"]["Run manually"]


def test_webhook_gets_a_unique_id_and_webhook_id():
    a, _ = add_trigger(workflow(), "discovery")
    b, _ = add_trigger(workflow(), "discovery")
    ha = next(n for n in a["nodes"] if n["type"] == WEBHOOK_TYPE)
    hb = next(n for n in b["nodes"] if n["type"] == WEBHOOK_TYPE)
    assert ha["id"] != hb["id"]
    assert ha["webhookId"] != hb["webhookId"]


def test_placed_below_the_existing_triggers():
    out, _ = add_trigger(workflow(), "discovery")
    hook = next(n for n in out["nodes"] if n["type"] == WEBHOOK_TYPE)
    manual = next(n for n in out["nodes"] if n["name"] == "Run manually")
    assert hook["position"][1] > manual["position"][1]


def test_output_is_valid_json():
    out, _ = add_trigger(workflow(), "discovery")
    assert json.loads(json.dumps(out))["nodes"]


# ── Failure modes ───────────────────────────────────────────────────────────


def test_refuses_when_there_is_no_trigger_to_copy():
    wf = workflow(with_manual=False, with_schedule=False)
    with pytest.raises(SystemExit, match="No manual or schedule trigger"):
        add_trigger(wf, "discovery")


def test_refuses_when_the_trigger_goes_nowhere():
    wf = workflow(connected=False)
    with pytest.raises(SystemExit, match="isn't connected"):
        add_trigger(wf, "discovery")


# ── Secret audit ────────────────────────────────────────────────────────────


def test_flags_plaintext_keys_in_set_nodes():
    """Set-node values are exported verbatim, so they travel with every copy
    of the workflow file."""
    leaks = audit_secrets(workflow())
    assert any("REED_API_KEY" in l for l in leaks)


def test_redacts_the_value_it_reports():
    leaks = audit_secrets(workflow())
    assert not any("f781e1ab-d5dd-4904-91b8-a1c5148deec4" in l for l in leaks)
    assert any("***" in l for l in leaks)


def test_ignores_empty_and_non_secret_fields():
    leaks = audit_secrets(workflow())
    assert not any("CAREERJET_AFFID" in l for l in leaks)   # empty
    assert not any("JOBS_SHEET_ID" in l for l in leaks)     # not a credential name


def test_clean_workflow_reports_nothing():
    wf = workflow()
    wf["nodes"] = [n for n in wf["nodes"] if n["type"] != "n8n-nodes-base.set"]
    assert audit_secrets(wf) == []


# ── Reading n8n's answer to a probe ─────────────────────────────────────────
#
# These two bodies are the real thing, copied from n8n. They are the reason
# the first version of the probe was wrong: both say "not registered", but one
# means "you never imported this" and the other means "this works, just not
# over GET". Treating them the same reported every POST-only workflow as
# missing — which is what hid a correctly imported, active `responses`.

UNKNOWN_PATH = json.dumps({
    "code": 404,
    "message": 'The requested webhook "GET discovery" is not registered.',
    "hint": ("The workflow must be active for a production URL to run "
             "successfully. You can activate the workflow using the toggle "
             "in the top-right of the editor."),
})

WRONG_METHOD = json.dumps({
    "code": 404,
    "message": ("This webhook is not registered for GET requests. "
                "Did you mean to make a POST request?"),
    "hint": "",
})


def test_unknown_path_is_reported_dead():
    got = classify_probe(404, UNKNOWN_PATH)
    assert got["live"] is False
    assert "no webhook" in got["reason"]


def test_wrong_method_is_reported_live_as_post():
    """The bug. n8n is saying "this exists, use POST" — not "this is missing"."""
    got = classify_probe(404, WRONG_METHOD)
    assert got["live"] is True
    assert got["method"] == "POST"


def test_the_two_404s_do_not_classify_the_same_way():
    assert classify_probe(404, UNKNOWN_PATH)["live"] != \
           classify_probe(404, WRONG_METHOD)["live"]


def test_a_200_is_a_live_get_webhook():
    got = classify_probe(200, '{"jobs":[]}')
    assert got == {"live": True, "method": "GET"}


def test_405_without_a_body_is_still_live():
    """Some proxies swallow n8n's body and leave only the status code."""
    assert classify_probe(405, "")["live"] is True


def test_server_error_is_not_treated_as_live():
    assert classify_probe(502, "Bad Gateway")["live"] is False


def test_empty_body_is_tolerated():
    assert classify_probe(404, None)["live"] is False


# ── Reading n8n's own database ──────────────────────────────────────────────


def _n8n_db(tmp_path, rows):
    import sqlite3
    db = tmp_path / "database.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE webhook_entity ("
                "webhookPath TEXT, method TEXT, workflowId TEXT)")
    con.executemany("INSERT INTO webhook_entity VALUES (?,?,?)", rows)
    con.commit()
    con.close()
    return db


def test_reads_registered_paths_and_methods(tmp_path):
    db = _n8n_db(tmp_path, [("responses", "POST", "tSLltoeNlEoV8JlF"),
                            ("hud-data", "GET", "aaa")])
    assert registered_webhooks(db) == {"responses": "POST", "hud-data": "GET"}


def test_an_inactive_workflow_registers_nothing(tmp_path):
    """This is the distinction HTTP probing can't make: a saved-but-inactive
    workflow has rows in workflow_entity and none in webhook_entity, so it is
    indistinguishable over the wire from a workflow that was never imported."""
    db = _n8n_db(tmp_path, [("responses", "POST", "x")])
    assert "mark-applied" not in registered_webhooks(db)


def test_missing_database_returns_none_so_callers_can_fall_back(tmp_path):
    """None means "I don't know", not "nothing is registered" — n8n may be on
    another host or backed by Postgres."""
    assert registered_webhooks(tmp_path / "nope.sqlite") is None


def test_unreadable_database_returns_none(tmp_path):
    junk = tmp_path / "database.sqlite"
    junk.write_bytes(b"not a database")
    assert registered_webhooks(junk) is None


def test_opens_the_database_read_only(tmp_path):
    """n8n is running against this file. Opening it read-write risks a lock
    that stalls the live instance."""
    db = _n8n_db(tmp_path, [("responses", "POST", "x")])
    before = db.stat().st_mtime_ns
    registered_webhooks(db)
    assert db.stat().st_mtime_ns == before


# ── Reading why a run failed ────────────────────────────────────────────────
#
# The fixture below is the real shape n8n writes: a flattened array where an
# object's values are string indices into that same array. Reproduced from an
# actual failed execution (a revoked Google Sheets credential) rather than
# invented, because the whole point of this parser is tolerating n8n's format.

FLAT_ERROR = json.dumps([
    {"version": 1, "startData": "1", "resultData": "2"},          # 0 root
    {},                                                            # 1
    {"error": "3", "runData": "4", "lastNodeExecuted": "5"},       # 2 resultData
    {"description": "6", "name": "7", "node": "8", "message": "9"},  # 3 error
    {},                                                            # 4
    "Get Jobs (data)",                                             # 5
    "Access could not be refreshed because the connected account "
    "has revoked access.",                                         # 6
    "NodeApiError",                                                # 7
    {"name": "5", "type": "10"},                                   # 8 node object
    'The credential "Google Sheets account" needs to be reconnected.',  # 9
    "n8n-nodes-base.googleSheets",                                 # 10
])


def test_parses_the_message_out_of_a_flattened_execution():
    got = parse_execution_error(FLAT_ERROR)
    assert got["message"] == \
        'The credential "Google Sheets account" needs to be reconnected.'
    assert got["error_type"] == "NodeApiError"


def test_resolves_the_failing_node_through_its_object():
    """`error.node` is a whole node object, not a name — the name is one more
    pointer hop inside it. Reporting the wrong node sends you to the wrong
    place in a 21-node workflow."""
    assert parse_execution_error(FLAT_ERROR)["node"] == "Get Jobs (data)"


def test_carries_the_description_because_it_holds_the_remedy():
    """The message says what broke; the description says what to do about it."""
    assert "revoked access" in parse_execution_error(FLAT_ERROR)["description"]


def test_a_successful_execution_yields_no_error():
    ok = json.dumps([{"version": 1, "resultData": "1"}, {"runData": "2"}, {}])
    assert parse_execution_error(ok) is None


def test_unrecognised_shapes_return_none_rather_than_raising():
    """A diagnostic that raises is worse than one that shrugs — n8n is free to
    change this format, and it must not take the JAMS read path down with it."""
    for blob in ("", "null", "{}", "[]", "not json", json.dumps([1, 2, 3]),
                 json.dumps([{"version": 1, "resultData": "99"}])):
        assert parse_execution_error(blob) is None


def test_a_pointer_past_the_end_of_the_array_is_not_followed():
    blob = json.dumps([{"resultData": "1"}, {"error": "500"}])
    assert parse_execution_error(blob) is None


def test_reads_workflow_ids_alongside_paths(tmp_path):
    db = _n8n_db(tmp_path, [("hud-data", "GET", "VbhEKj2G1zApt0GL"),
                            ("discovery", "POST", "tSLltoeNlEoV8JlF")])
    assert webhook_workflow_ids(db) == {"hud-data": "VbhEKj2G1zApt0GL",
                                        "discovery": "tSLltoeNlEoV8JlF"}


def test_workflow_ids_missing_database_returns_none(tmp_path):
    assert webhook_workflow_ids(tmp_path / "nope.sqlite") is None


def _db_with_executions(tmp_path, rows):
    """rows: (id, workflowId, status, startedAt, data)"""
    import sqlite3
    db = tmp_path / "database.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE execution_entity ("
                "id INTEGER, workflowId TEXT, status TEXT, startedAt TEXT)")
    con.execute("CREATE TABLE execution_data (executionId INTEGER, data TEXT)")
    for r in rows:
        con.execute("INSERT INTO execution_entity VALUES (?,?,?,?)", r[:4])
        con.execute("INSERT INTO execution_data VALUES (?,?)", (r[0], r[4]))
    con.commit()
    con.close()
    return db


def test_last_failure_reports_the_most_recent_error(tmp_path):
    db = _db_with_executions(tmp_path, [
        (1, "wf", "error", "2026-07-21 11:05:04", FLAT_ERROR),
        (2, "wf", "success", "2026-07-27 13:56:01", "[]"),
        (3, "wf", "error", "2026-08-01 12:49:13", FLAT_ERROR),
    ])
    got = last_failure(db, "wf")
    assert got["execution_id"] == 3
    assert got["started_at"] == "2026-08-01 12:49:13"
    assert "Google Sheets account" in got["message"]


def test_a_failure_that_has_since_been_fixed_is_not_reported(tmp_path):
    """The exact sequence that happened live: hud-data failed at 12:49 on a
    revoked credential, the credential was reconnected, and 13:01 succeeded.
    Still flagging the 12:49 error makes a working workflow look broken, and a
    health field that cries wolf gets ignored like the old green tick did."""
    db = _db_with_executions(tmp_path, [
        (1434, "wf", "error", "2026-08-01 12:49:13", FLAT_ERROR),
        (1435, "wf", "success", "2026-08-01 13:01:27", "[]"),
    ])
    assert last_failure(db, "wf") is None


def test_a_success_before_the_failure_does_not_clear_it(tmp_path):
    """Only a *later* success resolves a failure — ordering by id, not just
    presence of any success row."""
    db = _db_with_executions(tmp_path, [
        (1, "wf", "success", "2026-07-27 13:56:01", "[]"),
        (2, "wf", "error", "2026-08-01 12:49:13", FLAT_ERROR),
    ])
    assert last_failure(db, "wf")["execution_id"] == 2


def test_another_workflows_success_does_not_clear_the_failure(tmp_path):
    db = _db_with_executions(tmp_path, [
        (1, "wf", "error", "2026-08-01 12:49:13", FLAT_ERROR),
        (2, "other", "success", "2026-08-01 13:01:27", "[]"),
    ])
    assert last_failure(db, "wf")["execution_id"] == 1


def test_last_failure_ignores_other_workflows(tmp_path):
    db = _db_with_executions(tmp_path, [
        (1, "other", "error", "2026-08-01 12:00:00", FLAT_ERROR),
    ])
    assert last_failure(db, "wf") is None


def test_last_failure_is_none_when_nothing_has_failed(tmp_path):
    db = _db_with_executions(tmp_path, [
        (1, "wf", "success", "2026-08-01 12:00:00", "[]"),
    ])
    assert last_failure(db, "wf") is None


def test_last_failure_survives_an_unparseable_blob(tmp_path):
    """Still worth reporting *that* it failed and when, even if the reason is
    in a shape we don't recognise."""
    db = _db_with_executions(tmp_path, [
        (7, "wf", "error", "2026-08-01 12:00:00", "garbage"),
    ])
    got = last_failure(db, "wf")
    assert got["execution_id"] == 7 and not got.get("message")


def test_last_failure_missing_database_returns_none(tmp_path):
    assert last_failure(tmp_path / "nope.sqlite", "wf") is None
