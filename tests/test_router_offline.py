"""
Offline unit test for the FinEx deterministic query router.

Loads only the routing helpers from finex/LLM_SQL.py via AST so it runs without
psycopg2 / Ollama / a live database. Guards against the substring false-positive
class of bugs (e.g. "ratio" matching inside "operational") and verifies the
explicit level-override normalisation.

Run:  python -m tests.test_router_offline      (or: python tests/test_router_offline.py)
"""
from __future__ import annotations

import ast
import os
import re
from typing import Optional

_LLM_SQL = os.path.join(os.path.dirname(__file__), "..", "finex", "LLM_SQL.py")


def _load_router_namespace():
    src = open(_LLM_SQL, encoding="utf-8").read()
    mod = ast.parse(src)
    ns = {"re": re, "Optional": Optional}
    want = {"_term_rx", "_matches", "route_question", "_normalize_level"}
    for node in mod.body:
        if isinstance(node, ast.Assign):
            try:
                exec(compile(ast.Module([node], []), "<llm_sql>", "exec"), ns)
            except Exception:
                pass  # skip assignments that need runtime deps
        elif isinstance(node, ast.FunctionDef) and node.name in want:
            exec(compile(ast.Module([node], []), "<llm_sql>", "exec"), ns)
    return ns


ROUTE_CASES = {
    # The original misrouting bug: "operational" contains "ratio" as a substring.
    "give me a rundown of any major operational improvements during their last fiscal year": "TEXT",
    "what major operational improvements happened in the last year": "TEXT",
    "what are the operational highlights": "TEXT",
    "tell me about the auditor": "TEXT",
    # Retrieval
    "what was revenue in 2025": "L1",
    # Comparative
    "compare revenue 2024 vs 2025": "L2",
    # Ratios (must still route to L3 even with narrative words present)
    "gross profit margin": "L3",
    "what is the operating margin overview": "L3",
    "what is the current ratio": "L3",
    "calculate roe for 2025": "L3",
    "what is the debt to equity ratio": "L3",
    # Analytical / strategic / investor
    "why did net profit fall": "L4",
    "summarise the key risks": "L4",
    "should management expand capacity": "L6",
    "is this a healthy company to invest in": "L5",
}

# Substring false-positives that must NOT be classified as ratio analysis.
NEGATIVE_L3 = [
    "how does the company road map look",   # "road" must not match "roa"
    "what is the company strategy",          # no ratio cue
]

LEVEL_CASES = {
    None: None, "auto": None, 0: None, "0": None, "": None, 7: None,
    3: "L3", "3": "L3", "L4": "L4", "l6": "L6",
}


def main() -> int:
    ns = _load_router_namespace()
    rq = ns["route_question"]
    nl = ns["_normalize_level"]
    failures = []

    for q, expected in ROUTE_CASES.items():
        got = rq(q)
        if got != expected:
            failures.append(f"ROUTE  {q!r}: expected {expected}, got {got}")

    for q in NEGATIVE_L3:
        if rq(q) == "L3":
            failures.append(f"FALSE-POSITIVE  {q!r} wrongly routed to L3")

    for lv, expected in LEVEL_CASES.items():
        got = nl(lv)
        if got != expected:
            failures.append(f"LEVEL  {lv!r}: expected {expected}, got {got}")

    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"PASS — {len(ROUTE_CASES)} route cases, {len(NEGATIVE_L3)} negatives, "
          f"{len(LEVEL_CASES)} level cases all green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
