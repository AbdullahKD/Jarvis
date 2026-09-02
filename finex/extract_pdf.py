"""
extract_pdf.py — Grid-based financial statement extractor v3

Architecture
============
Phase 1 — Labeled grid (pure code, pdfplumber)
  Every financial-statement page is converted into a structured grid of
  labeled rows × indexed columns.  Each data row gets a letter label
  (A, B, C … Z, AA, AB …).  Column headers are scanned for 4-digit
  years (20XX) so every column is pinned to a specific year — this
  eliminates the "wrong year" bug completely.

Phase 2 — Semantic classification (dictionary-first, LLM fallback)
  Row labels are classified to standard field names (revenue,
  cost_of_goods_sold, …) using the existing alias dictionary first.
  Any row the dictionary cannot match is batched into a single Ollama
  call where the LLM receives ONLY the label text — no numbers — and
  must pick from the fixed field vocabulary.  This keeps LLM scope
  narrow and prevents hallucination.

Phase 3 — Deterministic value extraction
  Given (grid, field_map, year_col_pins) the extractor is a pure
  dictionary lookup: row[label].col_values[col_for_year].  No regex,
  no ambiguity, no implicit first-number-wins.

Phase 4 — Accounting identity rules
  Every known equation (GP = Rev − COGS, Assets = Liabilities + Equity,
  …) is applied in one systematic pass:
    • If result is missing and components are present → derive it.
    • If result disagrees with components beyond tolerance → auto-correct.
    • If one component is missing and result + others are present →
      back-derive the missing component.

Phase 5 — LLM gap-fill
  Only for critical fields still absent after Phases 1–4, targeted
  Ollama extraction is run (same as v2 Stage 3).

Validation (7 checks) runs last and reports pass/fail without blocking.

Typical extraction times
  Text PDF, no LLM needed      →  1–5 s
  Text PDF, LLM classifies     →  5–20 s  (one batch call)
  Text PDF, LLM gap-fill too   →  15–45 s
  Scanned PDF                  →  0 fields (OCR not included)
"""

from __future__ import annotations

import json
import os
import re
import time
import requests
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Set, Tuple

import pdfplumber


# ── Ollama ──────────────────────────────────────────────────────────────────────
# Every LLM call in this file goes through finex._llm_helper.chat_sync, which
# owns the endpoint, the model choice (FINEX_MODEL → OLLAMA_CHAT_MODEL) and the
# thinking-suppression needed by qwen3-family models. The URL and model
# constants that used to sit here were unreferenced and named llama3.2 long
# after the deployed model changed — a comment that lies is worse than none.

# Critical fields — if still absent after Phases 1-4, trigger LLM gap-fill
_CRITICAL_PNL = ["revenue", "profit_before_tax", "net_profit"]
_CRITICAL_BS  = ["total_assets", "total_liabilities", "total_equity"]
_CRITICAL_CF  = ["operating_cashflow"]


# ── Shared vocabulary ───────────────────────────────────────────────────────────
# The alias dictionaries, the field-behaviour sets and the cell parsers now live
# in finex/statement_fields.py so this module and finex/extract_v4.py read from
# ONE copy. Two copies of a 400-entry alias dictionary drift, and a drift shows
# up as a field that extracts on one code path and not the other — close to
# impossible to diagnose from the output alone.
from finex.statement_fields import (
    _ALWAYS_POSITIVE,
    _EPS_PAT,
    _NEG_TEXT,
    _NO_ACCUMULATE,
    _NO_SCALE,
    _PNL_TAX_PATTERN,
    _POS_TEXT,
    _SKIP,
    _YEAR_PAT,
    _best_match,
    _clean_label,
    _get_mapping,
    _is_likely_label,
    _is_note_ref,
    _parse_cell,
)



# ── Scale / currency detection ──────────────────────────────────────────────────

def _detect_scale_and_currency(text: str) -> Tuple[float, str, str]:
    """Return (scale_factor, currency_code, unit_label)."""
    t = text.lower()

    currency = "Unknown"
    if any(p in t for p in ["£", " gbp", "sterling", "pound sterling"]):
        currency = "GBP"
    elif any(p in t for p in ["€", " eur", "euro "]):
        currency = "EUR"
    elif any(p in t for p in ["rs.", "pkr", "rupees", " rupee"]):
        currency = "PKR"
    elif any(p in t for p in ["$", " usd", "dollar"]):
        currency = "USD"
    elif any(p in t for p in ["aed", "dirham"]):
        currency = "AED"
    elif any(p in t for p in ["sar", "riyal"]):
        currency = "SAR"

    sym = {"GBP": "£", "EUR": "€", "USD": "$", "PKR": "Rs.",
           "AED": "AED", "SAR": "SAR", "Unknown": ""}.get(currency, "")

    if any(p in t for p in ["in billions", "pkr billions", "billions"]):
        return 1_000_000_000.0, currency, f"{sym} billions"

    millions_kw = [
        "£ million", "£m", "$ million", "$m", "€ million", "€m",
        "in millions", "rs. millions", "pkr millions", "usd million",
        "gbp million", "eur million", "(£ millions)", "£ millions",
        "$ millions", "€ millions", "($ million", "(€ million",
        "millions", " mn",
    ]
    if any(p in t for p in millions_kw):
        return 1_000_000.0, currency, f"{sym} millions"

    thousands_kw = [
        "in thousands", "in '000", "'000s", "rs. '000", "(rupees in thousands)",
        "rs. in thousands", "amounts in thousands", "figures in '000",
        "amounts in rs. '000", "(rs. '000)", "rupees '000", "pkr thousands",
        "pkr '000", "000 omitted", "in rs. '000", "stated in thousands",
        "expressed in thousands", "thousand",
    ]
    if any(p in t for p in thousands_kw):
        return 1_000.0, currency, f"{sym} thousands"

    return 1_000_000.0, currency, f"{sym} millions (assumed)"


# ── Period detection ────────────────────────────────────────────────────────────

_MONTH = r'(?:january|february|march|april|may|june|july|august|september|october|november|december)'
_DATE  = re.compile(
    rf'\b(\d{{1,2}}\s+{_MONTH}\s+\d{{4}}|{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}|\d{{2}}/\d{{2}}/\d{{4}})',
    re.IGNORECASE,
)
_YEAR_BARE = re.compile(r'\b(20\d{2}|19\d{2})\b')


def _detect_periods(text: str) -> Tuple[Optional[str], Optional[str]]:
    found: List[str] = []
    for m in _DATE.finditer(text):
        p = m.group(1).strip()
        if p not in found:
            found.append(p)
        if len(found) >= 2:
            break
    if len(found) < 2:
        for m in _YEAR_BARE.finditer(text[:2000]):
            y = m.group(1)
            if y not in found:
                found.append(y)
            if len(found) >= 2:
                break
    return (found[0] if found else None, found[1] if len(found) > 1 else None)


def _parse_year(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = _YEAR_PAT.search(s)
    return int(m.group(1)) if m else None


# ── Page-type detection ─────────────────────────────────────────────────────────

_PAGE_SIGNATURES: Dict[str, List[str]] = {
    "pnl": [
        "income statement", "profit or loss", "profit and loss",
        "profit & loss", "statement of income", "statement of operations",
        "profit and loss account", "statement of profit",
        "condensed interim profit", "revenue and expenses",
        "income and expenditure", "statement of comprehensive income",
    ],
    "bs": [
        "balance sheet", "financial position", "statement of assets",
        "assets and liabilities", "assets & liabilities",
        "statement of net assets",
    ],
    "cf": [
        "cash flow statement", "cash flow", "statement of cash flows",
        "cashflow statement", "cash generated",
    ],
    "equity": [
        "changes in equity", "statement of equity",
        "changes in shareholders", "changes in owners",
    ],
}


def _page_type(text: str) -> str:
    t = text[:800].lower()
    for ptype, sigs in _PAGE_SIGNATURES.items():
        if any(s in t for s in sigs):
            return ptype
    return "narrative"




# ── Dataclasses ─────────────────────────────────────────────────────────────────

@dataclass
class LabeledRow:
    """One row of a financial statement grid, assigned a letter label."""
    label: str                              # "A", "B", "C" … "Z", "AA" …
    text: str                               # raw line-item text ("Revenue", …)
    col_values: Dict[int, float]            # column_index → parsed numeric value
    source: str = "table"                   # "table" or "text"


@dataclass
class PageGrid:
    """A financial statement page as a labeled row × column structure."""
    rows: List[LabeledRow]
    year_cols: Dict[int, int]   # col_index → year  (e.g. {1: 2025, 2: 2024})
    ptype: str                  # "pnl", "bs", "cf"
    page_n: int


# ── Label utilities ─────────────────────────────────────────────────────────────

def _label_from_index(i: int) -> str:
    """0→A, 1→B, …, 25→Z, 26→AA, 27→AB, …"""
    letters = []
    n = i
    while True:
        letters.append(chr(ord('A') + n % 26))
        n = n // 26 - 1
        if n < 0:
            break
    return ''.join(reversed(letters))


# ── Phase 1A: Build grid from pdfplumber table ──────────────────────────────────

_TABLE_STRATEGIES = [
    {"vertical_strategy": "lines",        "horizontal_strategy": "lines"},
    {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
    {"vertical_strategy": "text",         "horizontal_strategy": "text",
     "snap_tolerance": 3, "join_tolerance": 3},
    {},   # pdfplumber default
]


def _find_year_cols(table: List[List]) -> Tuple[Dict[int, int], int]:
    """
    Scan the first 8 rows for cells that contain a 4-digit year (20XX).
    Returns (year_cols dict, header_row_index).
    If multiple rows qualify, picks the one with the most year cells.
    """
    best_cols: Dict[int, int] = {}
    best_row = -1
    best_count = 0

    for row_idx, row in enumerate(table[:8]):
        cols: Dict[int, int] = {}
        for col_idx, cell in enumerate(row or []):
            if cell is None:
                continue
            m = _YEAR_PAT.search(str(cell).strip())
            if m:
                yr = int(m.group(1))
                if 2000 <= yr <= 2035:
                    cols[col_idx] = yr
        if len(cols) > best_count:
            best_cols, best_row, best_count = cols, row_idx, len(cols)

    return best_cols, best_row


def _build_grid_from_table(
    table: List[List], ptype: str, page_n: int
) -> Optional[PageGrid]:
    """Convert a raw pdfplumber table into a PageGrid with labeled rows."""
    if not table or len(table) < 2:
        return None

    year_cols, header_row_idx = _find_year_cols(table)

    labeled_rows: List[LabeledRow] = []
    label_idx = 0

    for row_idx, row in enumerate(table):
        if row_idx <= header_row_idx or not row:
            continue

        # Find the first cell that looks like a text label
        label_text: Optional[str] = None
        label_col = -1
        for col_idx, cell in enumerate(row):
            if cell and _is_likely_label(cell) and len(str(cell).strip()) >= 3:
                label_text = str(cell).strip()
                label_col = col_idx
                break

        if not label_text:
            continue

        clean = _clean_label(label_text)
        if len(clean) < 3 or any(s in clean for s in _SKIP):
            continue

        # Collect numeric values from columns after the label
        col_values: Dict[int, float] = {}
        nums_seen: List[float] = []

        for col_idx, cell in enumerate(row):
            if col_idx <= label_col:
                continue
            val = _parse_cell(cell)
            if val is None:
                continue
            if _is_note_ref(val, nums_seen):
                continue
            nums_seen.append(val)
            col_values[col_idx] = val

        if not col_values:
            continue

        lbl = _label_from_index(label_idx)
        labeled_rows.append(LabeledRow(
            label=lbl, text=label_text, col_values=col_values, source="table"
        ))
        label_idx += 1

    if not labeled_rows:
        return None

    return PageGrid(rows=labeled_rows, year_cols=year_cols, ptype=ptype, page_n=page_n)


# ── Phase 1B: Build grid from raw text (fallback) ──────────────────────────────

def _build_grid_from_text(text: str, ptype: str, page_n: int) -> Optional[PageGrid]:
    """
    Build a PageGrid from raw page text using character-position alignment.

    Strategy:
    1. Find the "year header" line — the first line containing 2+ years (20XX).
    2. Record the character position of each year → build col_index mapping.
    3. For every subsequent line with at least one number:
       • Everything left of the first number = label text.
       • Numbers are assigned to the column whose header position is nearest
         (within ±COL_TOLERANCE characters).
    """
    COL_TOLERANCE = 18  # characters

    lines = text.splitlines()

    # ── Find year header line ────────────────────────────────────────────────
    year_positions: Dict[int, int] = {}   # char_pos → year
    header_line_idx = -1

    for line_idx, line in enumerate(lines[:40]):
        yrs: List[Tuple[int, int]] = []
        for m in _YEAR_PAT.finditer(line):
            yr = int(m.group(1))
            if 2000 <= yr <= 2035:
                yrs.append((m.start(), yr))
        if len(yrs) >= 1:
            year_positions = {pos: yr for pos, yr in yrs}
            header_line_idx = line_idx
            break

    sorted_positions = sorted(year_positions.keys())
    pos_to_col = {pos: i + 1 for i, pos in enumerate(sorted_positions)}
    year_cols  = {pos_to_col[pos]: yr for pos, yr in year_positions.items()}

    def _find_col(char_pos: int) -> int:
        if not sorted_positions:
            return 1
        nearest = min(sorted_positions, key=lambda p: abs(p - char_pos))
        if abs(nearest - char_pos) <= COL_TOLERANCE:
            return pos_to_col[nearest]
        # Beyond known columns — assign a sequential extra index
        return len(pos_to_col) + 1

    # ── Build labeled rows ───────────────────────────────────────────────────
    labeled_rows: List[LabeledRow] = []
    label_idx = 0

    for line_idx, line in enumerate(lines):
        if line_idx <= header_line_idx:
            continue

        line_stripped = line.strip()
        if len(line_stripped) < 4:
            continue

        # Find all numbers and their character positions
        nums_in_line: List[Tuple[int, float]] = []

        bracket_spans = [(m.start(), m.end()) for m in _NEG_TEXT.finditer(line)]

        def _in_bracket(pos: int) -> bool:
            return any(s <= pos < e for s, e in bracket_spans)

        for m in _NEG_TEXT.finditer(line):
            nums_in_line.append((m.start(), -float(m.group(1).replace(',', ''))))

        for m in _POS_TEXT.finditer(line):
            if not _in_bracket(m.start()):
                nums_in_line.append((m.start(), float(m.group(1).replace(',', ''))))

        # EPS / DPS pre-scan: if the raw line looks like a per-share row, capture
        # small decimal values (e.g. "18.75", "5.00") that _POS_TEXT skips.
        # This runs BEFORE the early-exit so per-share rows with only small
        # decimals are not silently dropped.
        _line_lower = line_stripped.lower()
        _raw_is_per_share = any(kw in _line_lower for kw in (
            "earnings per share", "loss per share", "profit per share",
            "eps", "dividend per share", "dividends per share",
            "interim dividend", "final dividend",
        ))
        if _raw_is_per_share:
            pos_set = {p for p, _ in nums_in_line}
            for m in _EPS_PAT.finditer(line):
                if not _in_bracket(m.start()) and m.start() not in pos_set:
                    nums_in_line.append((m.start(), float(m.group(1))))

        nums_in_line.sort(key=lambda x: x[0])

        if not nums_in_line:
            continue

        first_num_pos = nums_in_line[0][0]
        label_text = line[:first_num_pos].strip()

        if not label_text or len(label_text) < 3:
            continue

        clean = _clean_label(label_text)
        if any(s in clean for s in _SKIP):
            continue
        if _parse_cell(label_text) is not None:
            continue   # label is a number — skip

        # Assign numbers to column indices.
        # Per-share rows bypass _is_note_ref — EPS/DPS values like 18.75 or 5.00
        # would otherwise be wrongly flagged as note references (small integers).
        _is_per_share_row = _raw_is_per_share or any(kw in clean for kw in (
            "earnings per share", "loss per share", "profit per share",
            "eps", "dividend per share", "dividends per share",
            "interim dividend", "final dividend",
        ))

        col_values: Dict[int, float] = {}
        nums_seen: List[float] = []

        for char_pos, val in nums_in_line:
            if not _is_per_share_row and _is_note_ref(val, nums_seen):
                continue
            nums_seen.append(val)
            col_idx = _find_col(char_pos)
            if col_idx not in col_values:   # first value per column wins
                col_values[col_idx] = val

        if not col_values:
            continue

        lbl = _label_from_index(label_idx)
        labeled_rows.append(LabeledRow(
            label=lbl, text=label_text, col_values=col_values, source="text"
        ))
        label_idx += 1

    if not labeled_rows:
        return None

    return PageGrid(rows=labeled_rows, year_cols=year_cols, ptype=ptype, page_n=page_n)


# ── Phase 1: Orchestrate grid building ─────────────────────────────────────────

def _build_page_grid(page, text: str, ptype: str, page_n: int) -> Optional[PageGrid]:
    """
    Try all table-extraction strategies.  Score each grid by:
      rows × 1  +  pinned year columns × 5  +  table bonus × 3
    Fall back to text-based grid if table extraction yields nothing useful.
    """
    best_grid: Optional[PageGrid] = None
    best_score = 0

    for strategy in _TABLE_STRATEGIES:
        try:
            tables = page.extract_tables(strategy) if strategy else page.extract_tables()
            if not tables:
                continue
            for tbl in tables:
                grid = _build_grid_from_table(tbl, ptype, page_n)
                if not grid:
                    continue
                score = len(grid.rows) + len(grid.year_cols) * 5 + 3  # table bonus
                if score > best_score:
                    best_grid, best_score = grid, score
        except Exception:
            continue

    # Text fallback
    text_grid = _build_grid_from_text(text, ptype, page_n)
    if text_grid:
        text_score = len(text_grid.rows) + len(text_grid.year_cols) * 5
        if text_score > best_score:
            best_grid = text_grid

    return best_grid


# ── Phase 2A: Dictionary classification ────────────────────────────────────────

def _dict_classify(text: str, ptype: str) -> Optional[str]:
    """Fast dictionary lookup for a single row label."""
    clean = _clean_label(text)
    if not clean or len(clean) < 3:
        return None
    if any(s in clean for s in _SKIP):
        return None
    # Tax filter: block standalone tax asset/liability rows that leak from BS pages.
    # BUT: "profit before tax / taxation" rows must be allowed through.
    if ptype == "pnl" and "tax" in clean:
        is_pbt = any(p in clean for p in [
            "before tax", "before taxation", "before income tax", "before wppf"
        ])
        if not is_pbt and not _PNL_TAX_PATTERN.search(text):
            return None
    return _best_match(clean, _get_mapping(ptype))


# ── Phase 2B: LLM semantic classification ──────────────────────────────────────

_FIELD_VOCAB: Dict[str, List[str]] = {
    "pnl": [
        "revenue", "cost_of_goods_sold", "gross_profit", "operating_expenses",
        "operating_profit", "finance_cost", "profit_before_tax", "tax_expense",
        "net_profit", "eps", "dividend_per_share", "depreciation",
    ],
    "bs": [
        "total_assets", "current_assets", "non_current_assets",
        "cash_balance", "trade_receivables", "inventory",
        "total_liabilities", "current_liabilities", "non_current_liabilities",
        "total_equity", "share_capital", "long_term_debt",
    ],
    "cf": [
        "operating_cashflow", "investing_cashflow", "financing_cashflow",
        "depreciation",
    ],
}

_STMT_NAMES = {
    "pnl": "Income Statement (P&L)",
    "bs":  "Balance Sheet",
    "cf":  "Cash Flow Statement",
}


_LLM_AVAILABLE_CACHE: Optional[bool] = None
_LLM_AVAILABLE_AT: float = 0.0
# Long enough that a single extraction run probes once; short enough that a
# server process recovers on its own after Ollama is restarted.
_LLM_PROBE_TTL_S = 30.0


def _llm_available() -> bool:
    """Probe the local Ollama server, cached for _LLM_PROBE_TTL_S.

    This is called per page grid in Phase 2 and again per statement type in
    Phase 5, so on a multi-page report the uncached version spent a full HTTP
    round trip per call answering a question that cannot change mid-extraction.

    Cached with a TTL rather than for the process lifetime, deliberately: this
    runs inside a long-lived FastAPI server, and a permanently cached False —
    from one upload that happened to land while Ollama was restarting — would
    silently disable LLM classification for every later upload with no error
    anywhere. A stale value in either direction is harmless for one run, since
    chat_sync degrades cleanly on its own.
    """
    global _LLM_AVAILABLE_CACHE, _LLM_AVAILABLE_AT
    now = time.monotonic()
    if _LLM_AVAILABLE_CACHE is None or (now - _LLM_AVAILABLE_AT) > _LLM_PROBE_TTL_S:
        try:
            from finex._llm_helper import health_check
            _LLM_AVAILABLE_CACHE = health_check()
        except Exception:
            _LLM_AVAILABLE_CACHE = False
        _LLM_AVAILABLE_AT = now
    return _LLM_AVAILABLE_CACHE


def reset_llm_availability() -> None:
    """Clear the cached probe. For tests, and after a known Ollama restart."""
    global _LLM_AVAILABLE_CACHE, _LLM_AVAILABLE_AT
    _LLM_AVAILABLE_CACHE = None
    _LLM_AVAILABLE_AT = 0.0


def _parse_json_object(content: str) -> Optional[dict]:
    r"""Pull the first parseable JSON object out of a model reply.

    Replaces `re.search(r'\{[^{}]+\}')`, which had two failure modes that both
    surface as silently missing fields rather than as errors: it cannot match an
    object containing a nested object, and `[^{}]+` will happily match a brace
    pair inside a reasoning preamble before the real answer. With json_mode the
    reply should be bare JSON, so this is mostly a safety net — but it is the
    net that decides whether a field lands in the database or not.

    Walks each `{` to its matching `}` respecting strings and escapes, and on a
    candidate that doesn't parse it moves to the next `{` rather than giving up
    — otherwise a stray `{x}` in a preamble discards the real object behind it.
    """
    if not content:
        return None

    search_from = 0
    while True:
        start = content.find("{", search_from)
        if start == -1:
            return None

        depth = 0
        in_str = False
        escaped = False
        close = -1
        for i in range(start, len(content)):
            ch = content[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    close = i
                    break

        if close == -1:
            return None  # unterminated — nothing usable after this point either

        try:
            parsed = json.loads(content[start:close + 1])
        except (ValueError, TypeError):
            search_from = start + 1
            continue

        if isinstance(parsed, dict):
            return parsed
        search_from = start + 1


def _llm_classify_rows(rows: List[LabeledRow], ptype: str) -> Dict[str, str]:
    """
    Ask the LLM to classify unmatched row labels to standard field names.

    Key design: ONLY label text is sent — no numbers, no scale, no context that
    could lead to hallucination.  The LLM must pick from a fixed vocabulary or
    return null.  This is a classification task, not open-ended generation.
    """
    if not rows:
        return {}

    vocab = _FIELD_VOCAB.get(ptype, [])
    if not vocab:
        return {}

    rows_text = "\n".join(f"{r.label} | {r.text}" for r in rows)
    vocab_str  = ", ".join(vocab)
    stmt_name  = _STMT_NAMES.get(ptype, ptype)

    prompt = f"""You are classifying {stmt_name} line items to standard field names.

Line items (Label | Description — NO numbers):
{rows_text}

Valid field names — choose ONLY from this list, or null if genuinely no match:
{vocab_str}

Rules:
- Only map if you are confident the description matches the field.
- Use null for sub-items, intermediary subtotals, or ambiguous rows.
- "Total non-current assets" → non_current_assets  ✓
- "Property, plant and equipment" → null  (it's a sub-item, not a total)
- EPS / earnings per share rows → eps

Return a single JSON object mapping every label above to a field name or null:
{{"A": "revenue", "B": null}}"""

    try:
        from finex._llm_helper import chat_sync
        content = chat_sync(
            [{"role": "user", "content": prompt}],
            max_tokens=500, temperature=0.0, timeout=40,
            json_mode=True,
        )
        if not content or content.startswith("[LLM"):
            return {}

        data = _parse_json_object(content)
        if data is None:
            return {}
        return {
            label: fn
            for label, fn in data.items()
            if fn and fn in vocab
        }
    except Exception as e:
        print(f"      ⚠ LLM classify error: {e}")
        return {}


# ── Phase 2: Classify all rows in a grid ───────────────────────────────────────

def _classify_grid(grid: PageGrid) -> Dict[str, str]:
    """
    Returns {row_label: field_name} for the grid.
    Uses dictionary first; any unmatched rows are batched into a single LLM call.
    """
    field_map: Dict[str, str] = {}
    unmatched: List[LabeledRow] = []

    for row in grid.rows:
        field = _dict_classify(row.text, grid.ptype)
        if field:
            field_map[row.label] = field
        else:
            unmatched.append(row)

    if unmatched and _llm_available():
        llm_result = _llm_classify_rows(unmatched, grid.ptype)
        if llm_result:
            print(f"      🤖 LLM mapped {len(llm_result)} rows: {list(llm_result.values())}")
        field_map.update(llm_result)

    return field_map


# ── Phase 3: Deterministic value extraction ─────────────────────────────────────

def _get_col_for_year(
    grid: PageGrid, target_year: Optional[int], col_offset: int = 0
) -> Optional[int]:
    """
    Return the column index that corresponds to target_year.
    Falls back to positional (col_offset=0 → leftmost value col,
    col_offset=1 → second) when no year pins are available.
    """
    # Try pinned year columns first
    if target_year:
        for col_idx, yr in grid.year_cols.items():
            if yr == target_year:
                return col_idx

    # Positional fallback — collect all column indices that appear in any row
    all_cols: Set[int] = set()
    for row in grid.rows:
        all_cols.update(row.col_values.keys())

    if not all_cols:
        return None

    sorted_cols = sorted(all_cols)
    return sorted_cols[col_offset] if col_offset < len(sorted_cols) else sorted_cols[-1]


def _extract_from_grid(
    grid: PageGrid,
    field_map: Dict[str, str],
    target_year: Optional[int],
    prior_year: Optional[int],
    scale: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Extract current and prior period values from a classified grid.
    Each field value is taken from the pinned year column (or positional
    fallback). Scale is applied; sign is normalised for _ALWAYS_POSITIVE fields.
    """
    cur_col  = _get_col_for_year(grid, target_year, col_offset=0)
    pri_col  = _get_col_for_year(grid, prior_year,  col_offset=1)

    row_by_label = {row.label: row for row in grid.rows}

    # Accumulate raw values per field before applying scale.
    # Multiple rows mapping to the same field (e.g. distribution costs +
    # admin expenses → operating_expenses) are SUMMED.
    cur_raw:  Dict[str, float] = {}
    pri_raw:  Dict[str, float] = {}
    # Track whether a field has been seen as a direct total row (longer match)
    # so we don't double-count when both a total row and its sub-rows appear.
    cur_is_total:  Dict[str, bool] = {}

    for label, field in field_map.items():
        row = row_by_label.get(label)
        if not row:
            continue

        # ── Current year ────────────────────────────────────────────────────
        raw = row.col_values.get(cur_col) if cur_col is not None else None
        if raw is None and row.col_values:
            raw = next(iter(sorted(row.col_values.items())))[1]

        if raw is not None:
            # Detect if this is a "total" row — totals override accumulated sub-items.
            is_total = _clean_label(row.text).startswith("total ")
            if field not in cur_raw:
                cur_raw[field]      = raw
                cur_is_total[field] = is_total
            elif is_total and not cur_is_total.get(field):
                cur_raw[field]      = raw
                cur_is_total[field] = True
            elif not cur_is_total.get(field) and field not in _NO_ACCUMULATE:
                # Accumulate sub-items (e.g. dist costs + admin → opex).
                # _NO_ACCUMULATE fields are never summed — first match wins.
                if field in _ALWAYS_POSITIVE:
                    cur_raw[field] = abs(cur_raw[field]) + abs(raw)
                else:
                    cur_raw[field] += raw

        # ── Prior year ──────────────────────────────────────────────────────
        if pri_col is not None and pri_col != cur_col:
            raw_p = row.col_values.get(pri_col)
            if raw_p is not None:
                if field not in pri_raw:
                    pri_raw[field] = raw_p
                elif (not _clean_label(row.text).startswith("total ")
                      and field not in _NO_ACCUMULATE):
                    if field in _ALWAYS_POSITIVE:
                        pri_raw[field] = abs(pri_raw[field]) + abs(raw_p)
                    else:
                        pri_raw[field] += raw_p

    # Apply scale and sign normalisation
    s_map = {f: (1.0 if f in _NO_SCALE else scale) for f in set(cur_raw) | set(pri_raw)}

    current: Dict[str, float] = {
        f: round(abs(v) * s_map[f] if f in _ALWAYS_POSITIVE else v * s_map[f], 2)
        for f, v in cur_raw.items()
    }
    prior: Dict[str, float] = {
        f: round(abs(v) * s_map[f] if f in _ALWAYS_POSITIVE else v * s_map[f], 2)
        for f, v in pri_raw.items()
    }

    return current, prior


# ── Phase 4: Accounting identity rules ─────────────────────────────────────────

# Each rule: (result_field, [addend_fields], [subtracted_fields], tolerance)
# result = Σ(addends) − Σ(subtracted)
_IDENTITY_RULES: List[Tuple[str, List[str], List[str], float]] = [
    # P&L cascade — tolerance loosens further down (adjustments accumulate)
    ("gross_profit",      ["revenue"],            ["cost_of_goods_sold"],              0.03),
    ("operating_profit",  ["gross_profit"],        ["operating_expenses"],              0.10),
    ("profit_before_tax", ["operating_profit"],    ["finance_cost"],                    0.15),
    ("net_profit",        ["profit_before_tax"],   ["tax_expense"],                     0.10),
    # Balance sheet — tight (accounting equations are exact)
    ("total_assets",      ["current_assets",    "non_current_assets"],     [],          0.02),
    ("total_liabilities", ["current_liabilities","non_current_liabilities"], [],        0.02),
    ("total_equity",      ["total_assets"],        ["total_liabilities"],               0.02),
]


def _apply_identity_rules(
    data: Dict[str, float],
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Three-way rule application:

    1. Derive  — result missing, all components present → compute result.
    2. Correct — result present but disagrees with components by > tol →
                 replace with computed value (components are more granular
                 and therefore more trustworthy).
    3. Back-derive — result + all-but-one component present → solve for
                     the missing component.
    """
    d = dict(data)

    for result_f, addends, subtracted, tol in _IDENTITY_RULES:
        components = addends + subtracted
        result_present = result_f in d

        # ── Can we compute the result? ───────────────────────────────────────
        if all(c in d for c in components):
            computed = sum(d[a] for a in addends) - sum(d[s] for s in subtracted)

            if result_present:
                stored = d[result_f]
                denom  = max(abs(computed), abs(stored), 1)
                diff   = abs(computed - stored) / denom
                if diff > tol:
                    if verbose:
                        print(f"   ⚠ Auto-correct {result_f}: "
                              f"{stored:,.0f} → {computed:,.0f}  ({diff:.1%} off)")
                    d[result_f] = round(computed, 2)
            else:
                if verbose:
                    print(f"   ℹ Derived {result_f}: {computed:,.0f}")
                d[result_f] = round(computed, 2)

        # ── Back-derive a missing component ─────────────────────────────────
        elif result_present:
            for missing_c in components:
                if missing_c in d:
                    continue   # not missing

                other_comps = [c for c in components if c != missing_c and c in d]
                if len(other_comps) != len(components) - 1:
                    continue   # more than one component missing — can't solve

                # Solve: result = Σ(addends) − Σ(subtracted)
                if missing_c in addends:
                    # missing_c is an addend:
                    #   missing_c = result − Σ(other addends) + Σ(subtracted)
                    derived = (d[result_f]
                               - sum(d[a] for a in addends if a != missing_c and a in d)
                               + sum(d[s] for s in subtracted if s in d))
                else:
                    # missing_c is subtracted:
                    #   missing_c = Σ(addends) − result − Σ(other subtracted)
                    derived = (sum(d[a] for a in addends if a in d)
                               - d[result_f]
                               - sum(d[s] for s in subtracted if s != missing_c and s in d))

                # Sanity: always-positive fields must come out positive
                if missing_c in _ALWAYS_POSITIVE and derived < 0:
                    continue

                if verbose:
                    print(f"   ℹ Back-derived {missing_c}: {derived:,.0f}")
                d[missing_c] = round(derived, 2)

    return d


# ── Phase 5: LLM gap-fill ───────────────────────────────────────────────────────

_FIELD_DESCRIPTIONS: Dict[str, str] = {
    "revenue":           "total revenue / turnover / net sales (top-line figure)",
    "gross_profit":      "gross profit or gross loss",
    "operating_profit":  "operating profit / EBIT / profit from operations",
    "profit_before_tax": "profit or loss before income tax / taxation",
    "net_profit":        "net profit / loss for the year after tax",
    "total_assets":      "total assets",
    "total_liabilities": "total liabilities",
    "total_equity":      "total equity / shareholders equity / net assets",
    "operating_cashflow":"net cash from / used in operating activities",
    "investing_cashflow":"net cash from / used in investing activities",
    "financing_cashflow":"net cash from / used in financing activities",
    "cost_of_goods_sold":"cost of sales / cost of goods sold",
    "finance_cost":      "finance costs / interest expense / net finance costs",
    "tax_expense":       "income tax expense / taxation charge",
    "cash_balance":      "cash and cash equivalents",
    "trade_receivables": "trade receivables / accounts receivable",
    "inventory":         "inventories / stock in trade",
}


def _llm_extract_fields(
    page_text: str,
    ptype: str,
    missing_fields: List[str],
    scale: float,
    currency: str,
    unit_label: str,
) -> Dict[str, float]:
    """Ask Ollama to extract specific missing fields from page text."""
    if not missing_fields or not page_text.strip():
        return {}

    fields_str = "\n".join(
        f'  "{f}": {_FIELD_DESCRIPTIONS.get(f, f)}'
        for f in missing_fields
    )

    prompt = f"""You are a financial data extraction assistant.
Currency: {currency}. Unit: {unit_label}.
Numbers in the text are already in {unit_label} — return them exactly as they appear (do NOT scale).
Return negative numbers for losses. If a field is not present, use null.
Return ONLY valid JSON, no explanation.

Fields to extract:
{{
{fields_str}
}}

Financial statement text:
{page_text[:3500]}

JSON:"""

    try:
        from finex._llm_helper import chat_sync
        content = chat_sync(
            [{"role": "user", "content": prompt}],
            max_tokens=300, temperature=0.0, timeout=45,
            json_mode=True,
        )
        if not content or content.startswith("[LLM"):
            return {}
        data = _parse_json_object(content)
        if data is None:
            return {}
        result: Dict[str, float] = {}
        for field, val in data.items():
            if field in missing_fields and val is not None:
                try:
                    fval = float(str(val).replace(',', ''))
                    s    = 1.0 if field in _NO_SCALE else scale
                    result[field] = round(
                        abs(fval) * s if field in _ALWAYS_POSITIVE else fval * s, 2
                    )
                except (ValueError, TypeError):
                    pass
        return result
    except Exception:
        return {}


def _stage_llm_gapfill(
    pages_typed: List[dict],
    scale: float,
    currency: str,
    unit_label: str,
    current: Dict[str, float],
    prior: Dict[str, float],
) -> None:
    """Run targeted LLM extraction for critical fields still missing."""
    if not _llm_available():
        return

    pnl_missing = [f for f in _CRITICAL_PNL if f not in current]
    bs_missing  = [f for f in _CRITICAL_BS  if f not in current]
    cf_missing  = [f for f in _CRITICAL_CF  if f not in current]

    for ptype, missing in [("pnl", pnl_missing), ("bs", bs_missing), ("cf", cf_missing)]:
        if not missing:
            continue
        best_page = next((p for p in pages_typed if p["ptype"] == ptype), None)
        if not best_page or not best_page["text"].strip():
            continue
        print(f"   🤖 LLM gap-fill for {ptype}: {missing}")
        extracted = _llm_extract_fields(
            best_page["text"], ptype, missing, scale, currency, unit_label
        )
        for field, val in extracted.items():
            if field not in current:
                current[field] = val
                print(f"      ✓ {field} = {val:,.2f}  (LLM gap-fill)")


# ── Validation (reporting only — corrections already done by identity rules) ────

def validate_financials(data: dict) -> dict:
    """
    7 checks across P&L, Balance Sheet and Cash Flow.
    Failures are reported but do NOT block storage.
    Auto-corrections are handled upstream by _apply_identity_rules().
    """
    passed, failed, warnings = [], [], []

    def chk(name: str, computed: float, stored: float, tol: float = 0.06) -> bool:
        denom = max(abs(computed), abs(stored), 1)
        diff  = abs(computed - stored) / denom
        if diff <= tol:
            passed.append(name)
        else:
            failed.append({
                "check":    name,
                "expected": round(computed, 2),
                "got":      round(stored,   2),
                "diff_pct": round(diff * 100, 2),
            })
        return diff <= tol

    d = data

    # 1. Gross Profit = Revenue − COGS
    if all(k in d for k in ["gross_profit", "revenue", "cost_of_goods_sold"]):
        chk("1. Gross Profit = Revenue − COGS",
            d["revenue"] - d["cost_of_goods_sold"], d["gross_profit"])

    # 2. Net Profit ≈ PBT − Tax
    if all(k in d for k in ["net_profit", "profit_before_tax", "tax_expense"]):
        chk("2. Net Profit ≈ PBT − Tax",
            d["profit_before_tax"] - d["tax_expense"], d["net_profit"], tol=0.10)

    # 3. Total Assets = Liabilities + Equity  (fundamental equation)
    if all(k in d for k in ["total_assets", "total_liabilities", "total_equity"]):
        chk("3. Assets = Liabilities + Equity",
            d["total_liabilities"] + d["total_equity"], d["total_assets"])

    # 4. Total Assets = Current + Non-Current
    if all(k in d for k in ["total_assets", "current_assets", "non_current_assets"]):
        chk("4. Total Assets = Current + Non-Current",
            d["current_assets"] + d["non_current_assets"], d["total_assets"])

    # 5. Total Liabilities = Current + Non-Current
    if all(k in d for k in ["total_liabilities", "current_liabilities", "non_current_liabilities"]):
        chk("5. Total Liabilities = Current + Non-Current",
            d["current_liabilities"] + d["non_current_liabilities"], d["total_liabilities"])

    # 6. Operating Profit ≈ Gross Profit − OpEx
    if all(k in d for k in ["operating_profit", "gross_profit", "operating_expenses"]):
        chk("6. Operating Profit ≈ Gross Profit − OpEx",
            d["gross_profit"] - d["operating_expenses"], d["operating_profit"], tol=0.12)

    # 7. Revenue ≥ Gross Profit (hard constraint)
    if all(k in d for k in ["revenue", "gross_profit"]):
        if d["revenue"] >= d["gross_profit"]:
            passed.append("7. Revenue ≥ Gross Profit")
        else:
            failed.append({
                "check":    "7. Revenue ≥ Gross Profit",
                "expected": f">= {d['gross_profit']:,.0f}",
                "got":      f"{d['revenue']:,.0f}",
                "diff_pct": None,
            })

    # ── Sanity warnings ──────────────────────────────────────────────────────
    if "revenue" in d and "net_profit" in d and d["revenue"] > 0:
        m = d["net_profit"] / d["revenue"]
        if m > 0.60:
            warnings.append(f"Net margin {m:.1%} — very high, verify unit scale")
        elif m < -0.50:
            warnings.append(f"Net margin {m:.1%} — very negative, verify extraction")

    if "revenue" in d and "operating_cashflow" in d and d["revenue"] > 0:
        ratio = d["operating_cashflow"] / d["revenue"]
        if abs(ratio) > 2.0:
            warnings.append(f"Operating CF / Revenue = {ratio:.1f}× — unusually high")

    if "eps" in d and d["eps"] > 10_000:
        warnings.append("EPS unusually high — may be wrong units")

    # ── Cross-year YoY sanity (catches column-swap errors) ───────────────────
    # (populated by caller if prior data is available)

    return {"passed": passed, "failed": failed, "warnings": warnings}


# ── Public helpers ──────────────────────────────────────────────────────────────

def extract_text_only(pdf_path: str) -> str:
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                chunks.append(t)
    return "\n".join(chunks)


def extract_pages(pdf_path: str) -> List[dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 40:
                pages.append({"page": i + 1, "text": text})
    return pages


# ── Master extraction function ──────────────────────────────────────────────────

def extract_financials_intelligent(pdf_path: str) -> dict:
    """Extract a set of accounts from one PDF. The entry point every caller uses.

    Which engine runs is controlled by ``FINEX_EXTRACTOR``:

      v4  (default)  finex/extract_v4.py — locates the one page that IS each
                     primary statement, reads years and scale from that
                     statement's own header, never substitutes a column,
                     carries provenance on every value, validates before
                     deriving, and writes NULL with a reason rather than a
                     figure it cannot stand behind.

      v3             the grid-harvesting extractor below. Kept for one release
                     so a PDF whose output changes can be diffed against the
                     old behaviour on the same machine. It is not the
                     supported path: measured against the ground-truth corpus
                     in tests/finex/fixtures it reads a document identifier as
                     Shell's revenue, dates Vodafone's comparative to 2027 and
                     reports "0 failed" on both.

    The return shape is identical either way — ``current``, ``prior``,
    ``metadata``, ``raw_text`` — so agents/finex_agent.py and server.py do not
    care which ran. v4 adds ``metadata["provenance"]``, ``metadata["nulls"]``
    and ``metadata["statements"]``; v3 simply omits them.
    """
    which = os.getenv("FINEX_EXTRACTOR", "v4").strip().lower() or "v4"
    if which == "v3":
        return _extract_financials_v3(pdf_path)
    if which != "v4":
        print(f"⚠ FINEX_EXTRACTOR={which!r} is not a known engine — using v4")
    from finex.extract_v4 import extract as _extract_v4
    return _extract_v4(pdf_path)


def _extract_financials_v3(pdf_path: str) -> dict:
    """
    Grid-based five-phase extractor (v3 — superseded, see above).

    Phase 1  Build labeled row × column grids from each financial page.
             Table extraction is tried with 4 strategies; text fallback
             uses character-position alignment.
    Phase 2  Classify row labels → field names via dictionary (fast path)
             then LLM for any rows the dictionary misses (one batch call
             per page, text only).
    Phase 3  Extract values from pinned year columns — the column index for
             each year is determined from the header row, so the current vs
             prior year cannot be swapped.
    Phase 4  Apply all accounting identity rules: derive missing fields,
             auto-correct totals that disagree with their sub-totals.
    Phase 5  LLM gap-fill for any critical fields still absent.
    """
    print(f"📄 Reading: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        all_pages = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if len(text.strip()) > 40:
                all_pages.append({"page_obj": page, "page_n": i + 1, "text": text})

    if not all_pages:
        return {
            "current": {}, "prior": {}, "raw_text": "",
            "metadata": {"error": "No readable text — PDF may be scanned or image-based"},
        }

    print(f"   {len(all_pages)} readable pages")

    full_text = "\n".join(p["text"] for p in all_pages)

    scale, currency, unit_label = _detect_scale_and_currency(full_text)
    print(f"   Scale: {unit_label}  |  Currency: {currency}  |  Factor: {scale:,.0f}")

    current_period, prior_period = _detect_periods(full_text)
    print(f"   Period: {current_period}  |  Prior: {prior_period}")

    target_year = _parse_year(current_period)
    prior_year  = _parse_year(prior_period)

    # If period detection returned the same year for both (e.g. the publication
    # date "20 May 2025" is picked up before the fiscal year-end "31 March 2025"),
    # try target_year-1 first, then target_year-2.  This avoids accidentally
    # picking up future strategy years like "Reimagine 2030".
    if target_year and prior_year and target_year == prior_year:
        corrected = False
        for candidate in [target_year - 1, target_year - 2]:
            if str(candidate) in full_text[:20_000]:
                prior_year = candidate
                print(f"   ⚠ Both periods = {target_year}; corrected prior_year → {prior_year}")
                corrected = True
                break
        if not corrected:
            # Broader scan but still restrict to realistic financial years
            for m in _YEAR_PAT.finditer(full_text[:20_000]):
                yr = int(m.group(1))
                if 2000 <= yr < target_year:
                    prior_year = yr
                    print(f"   ⚠ Both periods = {target_year}; corrected prior_year → {prior_year}")
                    break

    print(f"   Target year: {target_year}  |  Prior year: {prior_year}")

    # Tag every page with its financial statement type
    pages_typed: List[dict] = []
    type_counts: Dict[str, int] = {}
    for p in all_pages:
        ptype = _page_type(p["text"])
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
        pages_typed.append({
            "page_obj": p["page_obj"],
            "page_n":   p["page_n"],
            "text":     p["text"],
            "ptype":    ptype,
        })
    print(f"   Page types: {type_counts}")

    # ── Phases 1–3: Grid extraction ───────────────────────────────────────────
    current:    Dict[str, float] = {}
    prior:      Dict[str, float] = {}
    confidence: Dict[str, int]   = {}   # field → confidence level

    for page_info in pages_typed:
        if page_info["ptype"] == "narrative":
            continue

        ptype = page_info["ptype"]

        grid = _build_page_grid(
            page_info["page_obj"],
            page_info["text"],
            ptype,
            page_info["page_n"],
        )

        if not grid or not grid.rows:
            print(f"   Page {page_info['page_n']} [{ptype}]: no grid extracted")
            continue

        # Confidence scoring:
        #   base 3 if year columns are pinned, else 2 (positional fallback)
        #   +bonus for each early year column (col ≤ 3 is typical of a main
        #    financial statement; high indices suggest a notes/detail table)
        if grid.year_cols:
            min_yr_col = min(grid.year_cols.keys())
            col_bonus  = max(0, 4 - min_yr_col)   # col1→+3, col2→+2, col3→+1, col4+→0
            conf = 3 + col_bonus
        else:
            conf = 2

        yr_info = (
            f"year_cols={grid.year_cols} (conf={conf})" if grid.year_cols
            else "no year pins (positional fallback, conf=2)"
        )
        print(f"   Page {page_info['page_n']} [{ptype}]: "
              f"{len(grid.rows)} rows | {yr_info}")

        # Phase 2: classify rows
        field_map = _classify_grid(grid)
        if field_map:
            print(f"      → {len(field_map)} fields: {list(field_map.values())}")

        # Phase 3: extract values
        page_current, page_prior = _extract_from_grid(
            grid, field_map, target_year, prior_year, scale
        )

        # Merge — higher confidence wins; same confidence keeps first occurrence
        for field, val in page_current.items():
            old_conf = confidence.get(field, 0)
            if conf >= old_conf:
                current[field]    = val
                confidence[field] = conf

        for field, val in page_prior.items():
            if field not in prior:
                prior[field] = val

    print(f"   After grid phases: {len(current)} current fields, "
          f"{len(prior)} prior fields")

    # ── Post-extraction sanity sweep ──────────────────────────────────────────
    # Catch obviously wrong balance-sheet values (e.g. total_assets < current_assets)
    # that result from picking up subsidiary/notes tables instead of the group BS.
    def _bs_sanity(d: dict) -> dict:
        changes = []
        # total_assets must be ≥ current_assets and ≥ non_current_assets
        for sub in ("current_assets", "non_current_assets"):
            if sub in d and "total_assets" in d and d["total_assets"] < d[sub] * 0.9:
                changes.append(f"total_assets ({d['total_assets']:,.0f}) < {sub} ({d[sub]:,.0f})")
                del d["total_assets"]
                break
        # total_liabilities must be ≥ current_liabilities and non_current_liabilities
        for sub in ("current_liabilities", "non_current_liabilities"):
            if sub in d and "total_liabilities" in d and d["total_liabilities"] < d[sub] * 0.9:
                changes.append(f"total_liabilities ({d.get('total_liabilities',0):,.0f}) < {sub} ({d[sub]:,.0f})")
                d.pop("total_liabilities", None)
                break
        # total_equity must be < total_assets (if both known)
        if "total_equity" in d and "total_assets" in d and d["total_equity"] > d["total_assets"] * 1.05:
            changes.append(f"total_equity ({d['total_equity']:,.0f}) > total_assets ({d['total_assets']:,.0f})")
            del d["total_equity"]
        if changes:
            for c in changes:
                print(f"   ⚠ BS sanity removed implausible value: {c}")
        return d

    current = _bs_sanity(current)
    prior   = _bs_sanity(prior)

    # ── Phase 4: Accounting identity rules (two passes) ──────────────────────
    print("   Phase 4: Applying accounting identity rules...")
    current = _apply_identity_rules(current, verbose=True)
    prior   = _apply_identity_rules(prior,   verbose=False)
    # Second pass — rules may have derived new fields that enable further rules
    current = _apply_identity_rules(current, verbose=False)
    prior   = _apply_identity_rules(prior,   verbose=False)

    # ── Phase 5: LLM gap-fill ─────────────────────────────────────────────────
    print("   Phase 5: LLM gap-fill check...")
    _stage_llm_gapfill(pages_typed, scale, currency, unit_label, current, prior)

    # Final identity pass after gap-fill
    current = _apply_identity_rules(current, verbose=False)
    prior   = _apply_identity_rules(prior,   verbose=False)

    # ── Validation (reporting) ────────────────────────────────────────────────
    validation = validate_financials(current) if current else {}
    if validation:
        p = len(validation.get("passed",   []))
        f = len(validation.get("failed",   []))
        w = len(validation.get("warnings", []))
        print(f"   Validation: {p} passed  {f} failed  {w} warnings")
        for x in validation.get("failed", []):
            exp = x.get("expected")
            got = x.get("got")
            pct = x.get("diff_pct")
            exp_s = f"{exp:,.0f}" if isinstance(exp, (int, float)) else str(exp)
            got_s = f"{got:,.0f}" if isinstance(got, (int, float)) else str(got)
            print(f"   ❌ {x['check']}: expected {exp_s}  got {got_s}  ({pct}%)")
        for x in validation.get("warnings", []):
            print(f"   ⚠️  {x}")

    print(f"✅ {len(current)} current fields  {len(prior)} prior fields  [{unit_label}]")

    return {
        "current":  current,
        "prior":    prior,
        "metadata": {
            "period_current":  current_period,
            "period_prior":    prior_period,
            "target_year":     target_year,
            "prior_year":      prior_year,
            "currency":        currency,
            "unit_label":      unit_label,
            "scale_factor":    scale,
            "pages_processed": len(all_pages),
            "page_types":      type_counts,
            "validation":      validation,
        },
        "raw_text": full_text,
    }


# ── Backwards-compatible wrappers ───────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    return extract_text_only(pdf_path)


def extract_financials(text: str) -> dict:
    """Legacy stub — kept for import compatibility."""
    data: Dict[str, float] = {}
    NUM = r"([\d,]+(?:\.\d+)?)"
    patterns = {
        "revenue":           r"Net (?:turnover|revenue|sales)\s+" + NUM,
        "gross_profit":      r"Gross profit\s+" + NUM,
        "profit_before_tax": r"Profit before (?:tax|taxation)\s+" + NUM,
        "net_profit":        r"Profit (?:for the (?:period|year)|after tax)\s+" + NUM,
        "eps":               r"(?:Earnings|EPS) per share[^\d]+" + NUM,
        "total_assets":      r"Total assets\s+" + NUM,
        "total_liabilities": r"Total liabilities\s+" + NUM,
        "cash_balance":      r"Cash and (?:bank )?balances?\s+" + NUM,
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            data[key] = float(m.group(1).replace(",", ""))
    return data


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/Test PDF.pdf"

    result  = extract_financials_intelligent(pdf_path)
    print(f"   engine: {result.get('metadata', {}).get('extractor', 'v3')}")
    current = result["current"]
    prior   = result["prior"]
    meta    = result["metadata"]

    print(f"\n{'─'*55}")
    print(f"CURRENT PERIOD  [{meta.get('unit_label','?')}]  "
          f"(year: {meta.get('target_year','?')})")
    print(f"{'─'*55}")
    for k, v in sorted(current.items()):
        print(f"  {k:40} {v:>18,.2f}")

    print(f"\n{'─'*55}")
    print(f"PRIOR PERIOD  [{meta.get('unit_label','?')}]  "
          f"(year: {meta.get('prior_year','?')})")
    print(f"{'─'*55}")
    for k, v in sorted(prior.items()):
        print(f"  {k:40} {v:>18,.2f}")

    val = meta.get("validation", {})
    print(f"\n{'─'*55}")
    print(f"VALIDATION: {len(val.get('passed',[]))} passed  "
          f"{len(val.get('failed',[]))} failed  "
          f"{len(val.get('warnings',[]))} warnings")
    for x in val.get("failed", []):
        print(f"  ❌ {x['check']}")
    for x in val.get("warnings", []):
        print(f"  ⚠  {x}")
