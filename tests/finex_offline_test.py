"""
Offline test harness for FinEx pure-Python logic.

Cannot run Postgres or Ollama from this sandbox, so we:
  - stub finex.db_query and ollama HTTP calls
  - exercise route_question across a corpus
  - exercise _fmt_value / enrich_with_derived / clean_sql
  - exercise extract_pdf identity-rule helpers (without pdfplumber)
"""

from __future__ import annotations
import os, sys, types, time, json, re

# Stub psycopg2 (already installed) — also stub the connection function so import-time db_query works
ROOT = "/sessions/eloquent-affectionate-volta/mnt/Jarvis"
sys.path.insert(0, ROOT)

# --- Monkeypatch db_insert.get_connection so importing db_query doesn't try to connect ---
import finex.db_insert as dbi
def _no_connect():
    raise RuntimeError("Postgres not available in sandbox — test should not call this")
dbi.get_connection = _no_connect

# Now import db_query — it imports get_connection but doesn't *call* it at import time
import finex.db_query as dq

# Stub run_query so anything that calls it gets a deterministic fixture
_FIXTURE_FINANCIALS = [
    {  # 2025 row
        "id": 1, "company": "Bestway Cement", "year": 2025, "period": "FY2025",
        "currency": "PKR", "unit_label": "millions",
        "revenue": 55_374_000_000.0,
        "gross_profit": 19_870_000_000.0,
        "operating_profit": 16_220_000_000.0,
        "profit_before_tax": 14_330_000_000.0,
        "net_profit": 9_660_000_000.0,
        "eps": 16.20,
        "dividend_per_share": 5.00,
        "cost_of_goods_sold": 35_504_000_000.0,
        "operating_expenses": 3_650_000_000.0,
        "depreciation": 2_500_000_000.0,
        "finance_cost": 1_890_000_000.0,
        "tax_expense": 4_670_000_000.0,
        "total_assets": 142_000_000_000.0,
        "current_assets": 23_000_000_000.0,
        "non_current_assets": 119_000_000_000.0,
        "cash_balance": 4_300_000_000.0,
        "trade_receivables": 2_800_000_000.0,
        "inventory": 7_100_000_000.0,
        "total_liabilities": 71_000_000_000.0,
        "current_liabilities": 22_000_000_000.0,
        "non_current_liabilities": 49_000_000_000.0,
        "total_equity": 71_000_000_000.0,
        "share_capital": 5_960_000_000.0,
        "long_term_debt": 33_000_000_000.0,
        "operating_cashflow": 12_500_000_000.0,
        "investing_cashflow": -7_400_000_000.0,
        "financing_cashflow": -4_300_000_000.0,
    },
    {  # 2024 row
        "id": 2, "company": "Bestway Cement", "year": 2024, "period": "FY2024",
        "currency": "PKR", "unit_label": "millions",
        "revenue": 48_120_000_000.0,
        "gross_profit": 14_300_000_000.0,
        "operating_profit": 11_400_000_000.0,
        "profit_before_tax": 9_780_000_000.0,
        "net_profit": 6_500_000_000.0,
        "eps": 10.90,
        "dividend_per_share": 3.50,
        "cost_of_goods_sold": 33_820_000_000.0,
        "operating_expenses": 2_900_000_000.0,
        "depreciation": 2_300_000_000.0,
        "finance_cost": 1_500_000_000.0,
        "tax_expense": 3_280_000_000.0,
        "total_assets": 134_000_000_000.0,
        "current_assets": 19_500_000_000.0,
        "non_current_assets": 114_500_000_000.0,
        "cash_balance": 3_100_000_000.0,
        "trade_receivables": 2_300_000_000.0,
        "inventory": 5_900_000_000.0,
        "total_liabilities": 70_500_000_000.0,
        "current_liabilities": 21_000_000_000.0,
        "non_current_liabilities": 49_500_000_000.0,
        "total_equity": 63_500_000_000.0,
        "share_capital": 5_960_000_000.0,
        "long_term_debt": 32_000_000_000.0,
        "operating_cashflow": 9_700_000_000.0,
        "investing_cashflow": -6_100_000_000.0,
        "financing_cashflow": -3_400_000_000.0,
    },
]
COLS = list(_FIXTURE_FINANCIALS[0].keys())

def _stub_run_query(sql, params=None):
    """Lightweight SQL stub. Reads year from `params` (post-injection-fix), not
    from the SQL string."""
    s = sql.strip().lower()
    if "select currency, unit_label" in s:
        return {"columns": ["currency", "unit_label"], "rows": [("PKR", "millions")]}
    if "select distinct company" in s:
        return {"columns": ["company", "year", "period"],
                "rows": [(r["company"], r["year"], r["period"]) for r in _FIXTURE_FINANCIALS]}
    # parameterised year filter
    if " year = %s" in s or "and year = %s" in s:
        year = (params[-1] if params else 2025)
        for r in _FIXTURE_FINANCIALS:
            if r["year"] == year:
                return {"columns": COLS, "rows": [tuple(r[c] for c in COLS)]}
        return {"columns": COLS, "rows": []}
    # IN-list multi-year
    if "year in" in s:
        return {"columns": COLS, "rows": [tuple(r[c] for c in COLS) for r in _FIXTURE_FINANCIALS]}
    # legacy interpolated form
    if "year =" in s and "2024" in s:
        return {"columns": COLS, "rows": [tuple(_FIXTURE_FINANCIALS[1][c] for c in COLS)]}
    if "year =" in s and "2025" in s:
        return {"columns": COLS, "rows": [tuple(_FIXTURE_FINANCIALS[0][c] for c in COLS)]}
    return {"columns": COLS, "rows": []}

def _stub_get_all_years(company="Bestway Cement"):
    return [2025, 2024]

dq.run_query = _stub_run_query
dq.get_all_years = _stub_get_all_years

# Stub ask_llm so handler-paths don't need Ollama. We trace call counts + prompt sizes.
import finex.LLM_SQL as ls
LLM_CALLS = []
def _stub_ask_llm(prompt, system="", **kwargs):
    LLM_CALLS.append({"prompt_len": len(prompt), "system_len": len(system),
                      "prompt_head": prompt[:120], "kwargs": kwargs})
    # Return a deterministic answer so we can sanity-check formatting paths
    return "[MOCK LLM RESPONSE]"
ls.ask_llm = _stub_ask_llm

# also make sure LLM_SQL points at our stubbed db_query funcs
ls.run_query = _stub_run_query
ls.get_all_years = _stub_get_all_years

# ──────────────────────────────────────────────────────────────────────────────
# 1) Routing accuracy test
# ──────────────────────────────────────────────────────────────────────────────

ROUTING_CASES = [
    # (question, expected_level)
    # ── L1 retrieval ──
    ("What is the revenue in 2025?", "L1"),
    ("How much was net profit?", "L1"),
    ("What is EPS for 2025?", "L1"),
    ("Total assets in 2024", "L1"),
    ("Tell me the cash balance", "L1"),
    ("dividend per share", "L1"),
    # ── L2 comparison ──
    ("Compare revenue in 2024 and 2025", "L2"),
    ("How did profit change year-over-year?", "L2"),
    ("Revenue vs last year", "L2"),
    ("Which increased more, revenue or net profit?", "L2"),
    # ── L3 ratios ──
    ("What is the gross profit margin?", "L3"),
    ("Calculate ROE", "L3"),
    ("Current ratio for 2025", "L3"),
    ("Debt to equity ratio", "L3"),
    # ── L4 analytical ──
    ("Why did revenue grow?", "L4"),
    ("Is the company becoming more leveraged?", "L4"),
    ("What are the financial risks?", "L4"),
    # ── L5 investor ──
    ("Is the company financially healthy?", "L5"),
    ("Would you invest in this company?", "L5"),
    ("Is it overvalued?", "L5"),
    # ── L6 strategic ──
    ("What is the long-term growth strategy?", "L6"),
    ("Predict next year revenue", "L6"),
    ("Executive briefing for the board", "L6"),
    # ── TEXT (qualitative) ──
    ("Who is the CEO?", "TEXT"),
    ("Who audited the report?", "TEXT"),
    ("Where is the registered office?", "TEXT"),
    # ── DETAIL ──
    ("Can you elaborate?", "DETAIL"),
    ("Tell me more", "DETAIL"),
    # ── OFF_TOPIC ──
    ("What's the weather like?", "OFF_TOPIC"),
    ("Tell me a joke", "OFF_TOPIC"),
    # ── ADVERSARIAL / TRICKY ──
    ("Why is the EPS so high?", "L4"),                  # has "why" → should be L4 (currently triggers L4 via word match — but also matches L1 "eps")
    ("What is ROCE?", "L3"),                            # ROCE isn't in formulas — should still route as L3? Tests gap
    ("Compare ROE between 2024 and 2025", "L3"),        # has "compare" AND "roe" — currently L6 ('long term')? no. Will pick first hit (L6→L5→L4→L3...) — should fall to L3
    ("Should management increase dividends?", "L6"),    # 'what should management' triggers L6
    ("Free cash flow in 2025", "L1"),                   # 'cash' L1
    ("What was the gross turnover?", "L1"),             # 'gross turnover' L1
    ("Explain depreciation policy", "L4"),              # has 'depreciation' (L1 word) and 'explain more' (DETAIL)? actually 'explain more' substring — 'explain' alone doesn't match DETAIL ('explain more' must be substring)
]

print("=" * 78)
print(" ROUTING ACCURACY TEST")
print("=" * 78)
correct = 0
wrong = []
for q, expected in ROUTING_CASES:
    got = ls.route_question(q)
    ok = got == expected
    if ok: correct += 1
    else:  wrong.append((q, expected, got))
    print(f"  [{'OK' if ok else '!! '}]  expected={expected:9s} got={got:9s}  '{q}'")

print(f"\nRouting accuracy: {correct}/{len(ROUTING_CASES)} = {correct/len(ROUTING_CASES)*100:.1f}%")
if wrong:
    print("\nMisrouted:")
    for q, e, g in wrong:
        print(f"  expected={e} got={g}  '{q}'")

# ──────────────────────────────────────────────────────────────────────────────
# 2) Formatter test
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(" FORMATTER TEST")
print("=" * 78)
FMT_CASES = [
    (29_362_000_000.0, "revenue", "£",  "£29.36bn"),
    (55_374_000_000.0, "revenue", "PKR ", "PKR 55.37bn"),
    (1_240_000.0,     "revenue", "$",  "$1.24m"),
    (8_500.0,         "revenue", "€",  "€8.50k"),
    (1.23,            "revenue", "£",  "£1.23"),
    (-2_500_000.0,    "operating_profit", "£", "-£2.50m"),
    (16.20,           "eps", "PKR ", "PKR 16.20 per share"),
    (5.00,            "dividend_per_share", "$", "$5.00 per share"),
]
for val, field, sym, expected in FMT_CASES:
    got = ls._fmt_value(val, field, sym=sym)
    ok = got == expected
    print(f"  [{'OK' if ok else '!! '}]  _fmt_value({val}, {field!r}, sym={sym!r}) → {got!r}  expected {expected!r}")

# ──────────────────────────────────────────────────────────────────────────────
# 3) Derived fields
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(" DERIVED FIELD TEST")
print("=" * 78)
src = {
    "current_assets": 23_000_000_000.0,
    "non_current_assets": 119_000_000_000.0,
    "current_liabilities": 22_000_000_000.0,
    "non_current_liabilities": 49_000_000_000.0,
    "revenue": 55_374_000_000.0,
    "cost_of_goods_sold": 35_504_000_000.0,
    "operating_expenses": 3_650_000_000.0,
    "profit_before_tax": 14_330_000_000.0,
    "tax_expense": 4_670_000_000.0,
}
enriched = ls.enrich_with_derived(src)
for k in ("total_assets", "total_liabilities", "gross_profit", "operating_profit", "net_profit"):
    print(f"  {k}: {enriched.get(k)!r}  (derived={enriched.get(f'_{k}_derived')})")

# Edge case: zero/falsy revenue — should now derive correctly after P0 fix
print("\n  Edge cases (post-fix should NOT return None):")
edge = ls.enrich_with_derived({"revenue": 0, "cost_of_goods_sold": 100})
gp = edge.get("gross_profit")
ok = gp == -100
print(f"    [{'OK' if ok else '!! '}]  gross_profit when revenue=0 cogs=100 → {gp!r} (want -100)")

edge2 = ls.enrich_with_derived({"profit_before_tax": -500, "tax_expense": 0})
np_ = edge2.get("net_profit")
ok = np_ == -500
print(f"    [{'OK' if ok else '!! '}]  net_profit when PBT=-500 tax=0 → {np_!r} (want -500)")

# New: total_equity derivation
edge3 = ls.enrich_with_derived({"total_assets": 100, "total_liabilities": 30})
te = edge3.get("total_equity")
ok = te == 70
print(f"    [{'OK' if ok else '!! '}]  total_equity when TA=100 TL=30 → {te!r} (want 70)")

# ──────────────────────────────────────────────────────────────────────────────
# 4) clean_sql test
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(" clean_sql TEST")
print("=" * 78)
SQL_CASES = [
    ("```sql\nSELECT revenue FROM financials WHERE company = 'Bestway Cement';\n```",
     "SELECT revenue FROM financials WHERE company = 'Bestway Cement';"),
    ("Here is the SQL:\nSELECT revenue FROM financials;",
     "SELECT revenue FROM financials;"),
    ("SELECT a FROM t WHERE name = 'O''Brien';",
     "SELECT a FROM t WHERE name = 'O''Brien';"),
    # injection-style: trailing DDL — clean_sql will return the first SELECT but leaves the question of whether it runs into _fresh_conn.
    ("SELECT 1; DROP TABLE financials;",
     "SELECT 1;"),
]
for raw, expected in SQL_CASES:
    got = ls.clean_sql(raw)
    ok = got == expected
    print(f"  [{'OK' if ok else '!! '}]  expected={expected!r}")
    print(f"           got     ={got!r}")

# ──────────────────────────────────────────────────────────────────────────────
# 5) Trace LLM call count per level (to find bottlenecks)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(" LLM CALL COUNT PER LEVEL")
print("=" * 78)
LLM_CALLS.clear()
ls.answer("What is revenue in 2025?", "Bestway Cement", "")
print(f"  L1: {len(LLM_CALLS)} call(s)")
for c in LLM_CALLS: print(f"     - prompt_len={c['prompt_len']} system_len={c['system_len']}")

LLM_CALLS.clear()
ls.answer("Compare revenue in 2024 and 2025", "Bestway Cement", "")
print(f"  L2: {len(LLM_CALLS)} call(s)")
for c in LLM_CALLS: print(f"     - prompt_len={c['prompt_len']} system_len={c['system_len']}")

LLM_CALLS.clear()
ls.answer("What is the ROE for 2025?", "Bestway Cement", "")
print(f"  L3: {len(LLM_CALLS)} call(s)")
for c in LLM_CALLS: print(f"     - prompt_len={c['prompt_len']} system_len={c['system_len']}")

LLM_CALLS.clear()
ls.answer("Why is the company more leveraged this year?", "Bestway Cement", "")
print(f"  L4: {len(LLM_CALLS)} call(s)")
for c in LLM_CALLS: print(f"     - prompt_len={c['prompt_len']} system_len={c['system_len']}")

LLM_CALLS.clear()
ls.answer("Long-term growth strategy for the next 12 months?", "Bestway Cement", "")
print(f"  L6: {len(LLM_CALLS)} call(s)")
for c in LLM_CALLS: print(f"     - prompt_len={c['prompt_len']} system_len={c['system_len']}")

# ──────────────────────────────────────────────────────────────────────────────
# 6) Connection-pressure simulation
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(" CONNECTION-PRESSURE SIMULATION")
print("=" * 78)
# Count Postgres connect() calls made by a single L2 question.
# (Currently each query in db_query uses _fresh_conn which opens a new socket.)
import finex.db_query as dq2
opened = 0
orig = dq2._fresh_conn
def _count_conn(*a, **kw):
    global opened
    opened += 1
    class _C:
        closed = 0
        def set_session(self, autocommit=True): pass
        def cursor(self): return _Cur()
        def close(self): pass
    return _C()
class _Cur:
    description = [("year",), ("period",), ("revenue",), ("gross_profit",),
                   ("operating_profit",), ("profit_before_tax",), ("net_profit",),
                   ("eps",), ("finance_cost",), ("depreciation",),
                   ("total_assets",), ("total_liabilities",), ("total_equity",),
                   ("cash_balance",), ("current_assets",), ("non_current_assets",),
                   ("current_liabilities",), ("non_current_liabilities",)]
    def execute(self, *a, **kw): pass
    def fetchall(self): return [tuple([2025, "FY2025"] + [1.0]*16)]
    def close(self): pass
dq2._fresh_conn = _count_conn

# Re-run a level that hits the DB a lot
opened = 0
ls.answer("Compare revenue in 2024 and 2025", "Bestway Cement", "")
print(f"  L2 opens {opened} fresh Postgres connection(s).  (Same data, no pooling.)")
opened = 0
ls.answer("What is the gross profit margin?", "Bestway Cement", "")
print(f"  L3 opens {opened} fresh Postgres connection(s).")
opened = 0
ls.answer("What is revenue?", "Bestway Cement", "")
print(f"  L1 opens {opened} fresh Postgres connection(s).")

dq2._fresh_conn = orig

print("\nDONE.")
