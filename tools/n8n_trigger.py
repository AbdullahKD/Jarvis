"""
Add a Jarvis-callable webhook trigger to an n8n workflow.

Why this exists: an n8n **manual trigger** has no URL. It can only be fired from
inside the n8n editor, so nothing outside n8n — Jarvis, curl, a cron job — can
ever start that workflow. A schedule trigger has the same property: it fires on
its own clock and can't be poked.

Only a **Webhook** node gives a workflow an HTTP entry point. This adds one,
wired to whatever the existing manual trigger already feeds, so the run path is
identical to clicking "Test workflow" by hand.

Deliberately operates on an exported JSON file rather than through n8n's API:
the public API can't add nodes, and this way the change is reviewable as a diff
before you import it.

    python -m tools.n8n_trigger discovery.json --path discovery
    python -m tools.n8n_trigger discovery.json --path discovery --in-place

The webhook responds the moment the run starts rather than when it finishes.
Discovery takes minutes — holding the HTTP connection open that long means any
proxy, browser or client timeout in between looks like a failure while the run
is actually fine.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WEBHOOK_TYPE = "n8n-nodes-base.webhook"
MANUAL_TYPES = {"n8n-nodes-base.manualTrigger"}
SCHEDULE_TYPES = {"n8n-nodes-base.scheduleTrigger", "n8n-nodes-base.cron"}

DEFAULT_NODE_NAME = "Jarvis trigger"


def find_nodes(wf: Dict[str, Any], types: set) -> List[Dict[str, Any]]:
    return [n for n in wf.get("nodes", []) if n.get("type") in types]


def downstream_of(wf: Dict[str, Any], node_name: str) -> List[Dict[str, Any]]:
    """The connection targets of a node, in n8n's connection format."""
    conns = wf.get("connections", {}).get(node_name, {}).get("main", [])
    out: List[Dict[str, Any]] = []
    for branch in conns:
        for link in branch or []:
            out.append(link)
    return out


def existing_webhook(wf: Dict[str, Any], path: str) -> Optional[Dict[str, Any]]:
    for n in find_nodes(wf, {WEBHOOK_TYPE}):
        if (n.get("parameters") or {}).get("path") == path:
            return n
    return None


def make_webhook_node(path: str, position: Tuple[int, int],
                      name: str = DEFAULT_NODE_NAME) -> Dict[str, Any]:
    return {
        "parameters": {
            "httpMethod": "POST",
            "path": path,
            # Respond as soon as the run starts. "lastNode" would hold the HTTP
            # connection open for the whole run; a multi-minute discovery pass
            # then trips whatever timeout sits in between and reads as a
            # failure even though the workflow completed.
            "responseMode": "onReceived",
            "options": {},
        },
        "id": str(uuid.uuid4()),
        "name": name,
        "type": WEBHOOK_TYPE,
        "typeVersion": 2,
        "position": list(position),
        "webhookId": str(uuid.uuid4()),
        "notes": ("Added so Jarvis can start this workflow over HTTP. "
                  "A manual trigger has no URL and can only be run from the "
                  "n8n editor."),
    }


def add_trigger(wf: Dict[str, Any], path: str,
                name: str = DEFAULT_NODE_NAME) -> Tuple[Dict[str, Any], str]:
    """Return (patched workflow, human summary). Idempotent."""
    wf = json.loads(json.dumps(wf))          # don't mutate the caller's dict

    if existing_webhook(wf, path):
        return wf, f"already has a webhook at '{path}' — nothing to do"

    manual = find_nodes(wf, MANUAL_TYPES)
    schedule = find_nodes(wf, SCHEDULE_TYPES)
    anchor = (manual or schedule or [None])[0]
    if anchor is None:
        raise SystemExit(
            "No manual or schedule trigger found, so there's nothing to copy "
            "the wiring from. Add the webhook node by hand in n8n.")

    targets = downstream_of(wf, anchor["name"])
    if not targets:
        raise SystemExit(
            f"'{anchor['name']}' isn't connected to anything — patch the "
            f"workflow in n8n first so there's a run path to attach to.")

    # Sit below the existing triggers so the canvas stays readable.
    ax, ay = (anchor.get("position") or [0, 0])[:2]
    lowest = max((n.get("position") or [0, 0])[1]
                 for n in (manual + schedule) or [anchor])
    node = make_webhook_node(path, (ax, lowest + 192), name)

    wf.setdefault("nodes", []).append(node)
    # Same downstream targets as the manual trigger: identical run path, so
    # what Jarvis fires is exactly what "Test workflow" does.
    wf.setdefault("connections", {})[name] = {"main": [targets]}

    target_names = ", ".join(t.get("node", "?") for t in targets)
    return wf, (f"added webhook POST /webhook/{path} -> {target_names} "
                f"(same path as '{anchor['name']}')")


def audit_secrets(wf: Dict[str, Any]) -> List[str]:
    """Flag plaintext-looking credentials in Set nodes.

    Set-node values are exported verbatim with the workflow, so anything put
    there travels with every backup, copy and paste of the file.
    """
    hits = []
    for n in wf.get("nodes", []):
        if n.get("type") != "n8n-nodes-base.set":
            continue
        assigns = ((n.get("parameters") or {}).get("assignments") or {}).get("assignments", [])
        for a in assigns:
            name, value = str(a.get("name", "")), str(a.get("value", ""))
            if not value or len(value) < 12:
                continue
            if any(k in name.upper() for k in ("KEY", "SECRET", "TOKEN", "PASSWORD", "APP_ID")):
                hits.append(f"{n.get('name')}.{name} = {value[:4]}***{value[-2:]}")
    return hits


# ── Discovering what n8n will actually answer ───────────────────────────────


def classify_probe(status: int, body: str) -> Dict[str, Any]:
    """Read n8n's answer to a GET probe of a webhook path.

    n8n uses the phrase "not registered" for two completely different things,
    which is what broke the first version of this:

        unknown path   The requested webhook "GET discovery" is not registered.
        wrong method   This webhook is not registered for GET requests.
                       Did you mean to make a POST request?

    The discriminator is "not registered **for**" — that form means the webhook
    exists and simply doesn't take GET, i.e. it's a live POST trigger. Keying
    off the bare phrase reported every POST-only workflow as missing, which is
    exactly what hid a correctly-imported `responses` workflow.
    """
    low = (body or "").lower()
    if status == 200:
        return {"live": True, "method": "GET"}
    if "not registered for" in low or "did you mean" in low:
        return {"live": True, "method": "POST"}
    if "not registered" in low or "not found" in low:
        return {"live": False, "reason": "no webhook registered at this path"}
    if status == 405:
        # Method Not Allowed can only come from a route that exists.
        return {"live": True, "method": "POST"}
    if status == 404:
        # A bare 404 with no body — something in front of n8n swallowed the
        # message. Nothing here says the path exists, so don't claim it does.
        return {"live": False, "reason": "HTTP 404 with no n8n message"}
    return {"live": status < 500, "method": "POST", "reason": f"HTTP {status}"}


def registered_webhooks(db_path: Path) -> Optional[Dict[str, str]]:
    """{path: method} straight from n8n's own database.

    Far more reliable than parsing prose over HTTP, and it can distinguish
    "no such workflow" from "workflow exists but is inactive" — an inactive
    workflow has rows in workflow_entity and none in webhook_entity. Returns
    None when the database isn't readable, so callers fall back to probing.
    """
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
        rows = con.execute(
            "SELECT webhookPath, method FROM webhook_entity").fetchall()
        con.close()
        return {r[0]: r[1] for r in rows}
    except Exception:  # noqa: BLE001
        return None


def webhook_workflow_ids(db_path: Path) -> Optional[Dict[str, str]]:
    """{path: workflowId} for every registered webhook.

    Kept separate from ``registered_webhooks`` rather than widening its return
    type: that function's {path: method} shape is what the probe logic reads,
    and callers of one rarely want the other.
    """
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
        rows = con.execute(
            "SELECT webhookPath, workflowId FROM webhook_entity").fetchall()
        con.close()
        return {r[0]: r[1] for r in rows}
    except Exception:  # noqa: BLE001
        return None


# ── Reading why a run failed ────────────────────────────────────────────────
#
# n8n stores execution data "flattened": a JSON array where an object's values
# are string indices into that same array, so repeated substrings are stored
# once. `{"resultData": "2"}` means "resultData is arr[2]". Walking it beats
# scanning the blob for error-shaped text — the paths below are stable, a
# substring search would happily match a node *named* "error".


def _deref(arr: List[Any], value: Any) -> Any:
    """Resolve one flattened pointer. Non-pointers pass through unchanged."""
    if isinstance(value, str) and value.isdigit():
        idx = int(value)
        if 0 <= idx < len(arr):
            return arr[idx]
    return value


def parse_execution_error(blob: str) -> Optional[Dict[str, Any]]:
    """Pull the failure out of one execution_data.data blob.

    Returns {message, description, node, error_type} or None if the blob
    doesn't describe a failure (or isn't in a shape we recognise — n8n is free
    to change this, and a diagnostic that raises is worse than one that
    shrugs).
    """
    try:
        arr = json.loads(blob)
        if not isinstance(arr, list) or not arr:
            return None
        result = _deref(arr, (arr[0] or {}).get("resultData"))
        if not isinstance(result, dict):
            return None
        err = _deref(arr, result.get("error"))
        if not isinstance(err, dict):
            return None

        node = _deref(arr, err.get("node"))
        if isinstance(node, dict):
            node = _deref(arr, node.get("name"))
        if not isinstance(node, str):
            node = _deref(arr, result.get("lastNodeExecuted"))

        def _text(key: str) -> str:
            v = _deref(arr, err.get(key))
            return v if isinstance(v, str) else ""

        message = _text("message")
        if not message:
            return None
        return {
            "message": message,
            "description": _text("description"),
            "node": node if isinstance(node, str) else "",
            "error_type": _text("name"),
        }
    except Exception:  # noqa: BLE001
        return None


def last_failure(db_path: Path, workflow_id: str) -> Optional[Dict[str, Any]]:
    """The workflow's *unresolved* failure — its latest error, but only if
    nothing has succeeded since.

    Deliberately not "the most recent error ever". A workflow that failed at
    12:49 and succeeded at 13:01 is working, and reporting the 12:49 error
    against it trains you to ignore the field — the same way a permanently
    green "✓ Updated" trained you to ignore the refresh button. Returns None
    once a later run succeeds.

    This is the piece that turns "0 jobs" into "the Google Sheets credential
    needs reconnecting": n8n knows exactly what went wrong, it just has no way
    to say so over a webhook whose Respond node never ran.
    """
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
        row = con.execute(
            "SELECT e.id, e.startedAt, d.data FROM execution_entity e "
            "JOIN execution_data d ON d.executionId = e.id "
            "WHERE e.workflowId = ? AND e.status = 'error' "
            "ORDER BY e.id DESC LIMIT 1", (workflow_id,)).fetchone()
        if not row:
            con.close()
            return None
        newer_success = con.execute(
            "SELECT 1 FROM execution_entity "
            "WHERE workflowId = ? AND status = 'success' AND id > ? LIMIT 1",
            (workflow_id, row[0])).fetchone()
        con.close()
    except Exception:  # noqa: BLE001
        return None
    if newer_success:
        return None
    parsed = parse_execution_error(row[2]) or {}
    return {"execution_id": row[0], "started_at": row[1], **parsed}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workflow", type=Path, help="exported n8n workflow .json")
    ap.add_argument("--path", required=True,
                    help="webhook path, e.g. 'discovery' -> /webhook/discovery")
    ap.add_argument("--name", default=DEFAULT_NODE_NAME, help="node name")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite instead of writing <name>.jarvis.json")
    args = ap.parse_args(argv)

    wf = json.loads(args.workflow.read_text())
    patched, summary = add_trigger(wf, args.path, args.name)

    out = args.workflow if args.in_place \
        else args.workflow.with_suffix(".jarvis.json")
    out.write_text(json.dumps(patched, indent=2))

    print(f"✓ {summary}")
    print(f"  written to {out}")
    print( "  Import it in n8n (Workflows → ⋯ → Import from File), then make")
    print( "  sure the workflow is ACTIVE — an inactive workflow registers no")
    print( "  webhook, and the URL 404s.")

    leaks = audit_secrets(patched)
    if leaks:
        print("\n⚠ plaintext credentials in Set nodes (exported with the file):")
        for l in leaks:
            print(f"    {l}")
        print("  Move these to n8n Credentials or $env and rotate them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
