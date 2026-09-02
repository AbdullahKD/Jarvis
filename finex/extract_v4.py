"""
finex/extract_v4.py — statement-scoped financial extraction.

Why this exists
===============
v3 (``extract_pdf.py``) treats an annual report as a bag of pages to harvest:
every page whose text mentions "cash flow" is parsed, every grid it can build
contributes values, document-wide guesses supply the scale and the years, and
whatever wins a confidence tie-break lands in the database. Measured against
the real reports in ``FinEx Data`` that produces silent, confident nonsense —
a revenue of 260 million read off a document identifier, a prior period dated
2027, a gross profit overwritten by an identity rule and then validated as
correct because the overwrite is what made it consistent.

v4 inverts the control flow. It does not ask "what can I find on this page?"
It asks "which single page IS the consolidated income statement?" — and then
reads that page, on that page's own terms.

The eight changes
-----------------
1. **Locate, don't harvest.** ``locate_statements`` scores every page against
   canonical statement titles plus the anchor line items that statement must
   contain, and returns at most ONE page per statement type. Notes pages,
   five-year summaries, segmental analyses and parent-company-only statements
   are actively excluded, not merely out-scored.

2. **Years come from the statement's own header.** Year columns are pinned
   from the located page's header row. There is no document-wide period
   detection, so a press-release cover date can no longer set the fiscal year
   and a strategy slogan ("Reimagine 2030") can no longer become a period.

3. **Scale and currency come from the statement's own header.** Resolved from
   the located page's header region first, the page body second, and only then
   from the document — and whichever it was is recorded in
   ``scale_source`` / ``currency_source`` so an assumption is never mistaken
   for a reading.

4. **Never substitute a column.** If the row has no value in the column pinned
   to the target year, the field is absent for that year. There is no
   positional fallback and no "first number wins". v3's two fallbacks are the
   direct cause of prior-year values being copies of current-year values.

5. **Geometry, not character positions.** The grid is built from
   ``page.extract_words()`` and columns are assigned by x-coordinate against
   the year anchors. A note-reference column is excluded structurally — it is
   not near a year anchor — rather than by guessing that small integers are
   note references.

6. **Provenance on every value.** Every number carries the page, the row label,
   the row's printed text, the column index, the year, the scale applied and
   how it was classified. Merging resolves by confidence, never by page order.

7. **Validate before correcting.** Identity rules may DERIVE a missing field.
   They may never overwrite a value that was read off the page. Validation runs
   against the read values first, so a failing check is visible rather than
   erased. A conflict marks its participants; it does not silently pick one.

8. **NULL with a reason.** A field that cannot be established is absent from
   ``current``/``prior`` and present in ``metadata["nulls"]`` with a reason
   code. The pipeline's job is to be honest about what it does not know.

The LLM
-------
Classification of row labels only: text in, field name out, no digits in the
prompt and no digits in the reply. There is no numeric gap-fill in v4. Every
number in the output came from pdfplumber geometry, which is what makes the
regression harness deterministic and therefore meaningful. Set
``FINEX_DISABLE_LLM=1`` to skip the classifier entirely.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pdfplumber

from finex.statement_fields import (
    _ALWAYS_POSITIVE,
    _NO_SCALE,
    _PNL_TAX_PATTERN,
    _SKIP,
    _clean_label,
    _get_mapping,
)

EXTRACTOR_VERSION = "v4"

# ── Field groups ────────────────────────────────────────────────────────────
# The eight lines every consolidated set of accounts has. These are the fields
# the refuse policy protects: below the confidence floor they are nulled with a
# reason rather than published.
PRIMARY_FIELDS: Tuple[str, ...] = (
    "revenue",
    "operating_profit",
    "profit_before_tax",
    "net_profit",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "operating_cashflow",
)

# Below this, a PRIMARY field is nulled instead of published.
CONFIDENCE_FLOOR = 0.45

# The only field a real statement legitimately splits across sibling rows:
# distribution costs + administrative expenses. v3 summed any repeated field
# not on a deny-list, which turned Shell's "Purchases" + "Production and
# manufacturing expenses" into a cost of goods sold that does not exist.
# An allowlist fails closed; the deny-list failed open.
_ACCUMULATE_OK: Set[str] = {"operating_expenses"}

# Fields a strict identity may solve for when everything else in that identity
# is known. Balance-sheet only — see the note in apply_identities.
_BACK_DERIVABLE: Set[str] = {
    "total_assets", "total_liabilities", "total_equity",
    "current_assets", "non_current_assets",
    "current_liabilities", "non_current_liabilities",
}

_STATEMENT_TYPES: Tuple[str, ...] = ("pnl", "bs", "cf")

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
# A column header year often carries a footnote marker: "2024(a)", "2023*",
# "2025†". Unilever's income statement is headed `Notes 2025 2024(a) 2023(a)`,
# so a strict full match found one year where there are three — and the
# two-year comparative was then read from whatever other line in the page
# happened to contain two bare years.
_YEAR_TOKEN_RE = re.compile(
    r"^(19\d{2}|20\d{2})(?:[\(\[]?[a-zA-Z0-9]{1,2}[\)\]]?|[*†‡§#¹²³])?$"
)
_MIN_YEAR, _MAX_YEAR = 1990, 2035


def _year_of_token(text: str) -> Optional[int]:
    """The year a header token names, tolerating a trailing footnote marker."""
    m = _YEAR_TOKEN_RE.match(text.strip())
    if not m:
        return None
    yr = int(m.group(1))
    return yr if _MIN_YEAR <= yr <= _MAX_YEAR else None


# ── Reason codes ────────────────────────────────────────────────────────────
class Reason:
    NO_STATEMENT_PAGE = "no_statement_page"
    NO_YEAR_COLUMN = "no_year_column"
    NO_VALUE_IN_YEAR_COLUMN = "no_value_in_year_column"
    ROW_NOT_CLASSIFIED = "row_not_classified"
    LOW_CONFIDENCE = "low_confidence"
    IDENTITY_CONFLICT = "identity_conflict"
    SCALE_UNKNOWN = "scale_unknown"
    NO_READABLE_TEXT = "no_readable_text"


# ── Provenance ──────────────────────────────────────────────────────────────
@dataclass
class Provenance:
    """Where a single number came from, and how much we trust it."""

    page: Optional[int] = None
    statement: Optional[str] = None      # pnl | bs | cf
    row_label: Optional[str] = None      # "A", "B", … within the page grid
    row_text: Optional[str] = None       # the line item exactly as printed
    column_index: Optional[int] = None
    year: Optional[int] = None
    raw: Optional[float] = None          # the number as printed, pre-scale
    scale: float = 1.0
    scale_source: str = "unknown"        # statement_header|page_body|document
    currency: str = "Unknown"
    currency_source: str = "unknown"
    classifier: str = "dictionary"       # dictionary|llm|identity
    origin: str = "read"                 # read|derived
    rule: Optional[str] = None           # identity rule, when origin == derived
    accumulated_from: int = 1            # rows summed into this value
    confidence: float = 1.0
    notes: List[str] = dc_field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "page": self.page,
            "statement": self.statement,
            "row_label": self.row_label,
            "row_text": self.row_text,
            "column_index": self.column_index,
            "year": self.year,
            "raw": self.raw,
            "scale": self.scale,
            "scale_source": self.scale_source,
            "currency": self.currency,
            "currency_source": self.currency_source,
            "classifier": self.classifier,
            "origin": self.origin,
            "confidence": round(self.confidence, 3),
        }
        if self.rule:
            d["rule"] = self.rule
        if self.accumulated_from > 1:
            d["accumulated_from"] = self.accumulated_from
        if self.notes:
            d["notes"] = list(self.notes)
        return d


@dataclass
class Cell:
    """A scaled value plus its provenance."""

    value: float
    prov: Provenance


# ── Grid primitives ─────────────────────────────────────────────────────────
@dataclass
class GridRow:
    label: str
    text: str
    values: Dict[int, float]             # column index → number as printed
    top: float


@dataclass
class StatementGrid:
    page_n: int
    ptype: str
    rows: List[GridRow]
    year_cols: Dict[int, int]            # column index → year
    scale: float
    scale_source: str
    currency: str
    currency_source: str
    unit_label: str
    header_top: float
    title: str


# ── Statement location ──────────────────────────────────────────────────────
# A page qualifies as a primary statement by title AND by containing the line
# items that statement is defined by. Title alone is not enough: "Notes to the
# consolidated income statement" and a five-year summary both mention the
# title. Anchors alone are not enough either: a segmental note lists revenue
# and operating profit without being the income statement.

_TITLE_PATTERNS: Dict[str, List[re.Pattern]] = {
    "pnl": [
        re.compile(r"\bincome statement\b", re.I),
        re.compile(r"\bstatement of income\b", re.I),
        re.compile(r"\bstatement of (profit or loss|profit and loss)\b", re.I),
        re.compile(r"\bprofit and loss account\b", re.I),
        re.compile(r"\bstatement of operations\b", re.I),
        re.compile(r"\bstatement of comprehensive income\b", re.I),
    ],
    "bs": [
        re.compile(r"\bstatement of financial position\b", re.I),
        re.compile(r"\bbalance sheet\b", re.I),
        re.compile(r"\bstatement of net assets\b", re.I),
    ],
    "cf": [
        re.compile(r"\bstatement of cash ?flows?\b", re.I),
        re.compile(r"\bcash ?flow statement\b", re.I),
    ],
}

# Anchor line items — the rows that define the statement.
_ANCHORS: Dict[str, List[re.Pattern]] = {
    "pnl": [
        re.compile(r"\b(revenue|turnover|net sales)\b", re.I),
        re.compile(r"\boperating (profit|loss|income)\b|\bprofit from operations\b", re.I),
        re.compile(r"\bbefore (tax|taxation|income tax)\b", re.I),
        re.compile(r"\b(profit|loss) for the (year|period)\b", re.I),
        re.compile(r"\bper share\b", re.I),
    ],
    "bs": [
        re.compile(r"\btotal assets\b", re.I),
        re.compile(r"\btotal (equity|shareholders)\b", re.I),
        re.compile(r"\btotal liabilities\b|\btotal equity and liabilities\b", re.I),
        re.compile(r"\bnon-?current assets\b", re.I),
        re.compile(r"\bcurrent liabilities\b", re.I),
    ],
    "cf": [
        re.compile(r"\boperating activities\b", re.I),
        re.compile(r"\binvesting activities\b", re.I),
        re.compile(r"\bfinancing activities\b", re.I),
        re.compile(r"\bcash and cash equivalents\b", re.I),
    ],
}

# Any of these near the top of a page disqualifies it outright.
_DISQUALIFY_TOP = (
    re.compile(r"\bnotes? to the\b", re.I),
    re.compile(r"\bfive[- ]year\b|\bten[- ]year\b|\bfinancial (summary|record|highlights)\b", re.I),
    re.compile(r"\bsegment(al)? (information|analysis|note)\b", re.I),
    re.compile(r"\bindex\b|\bcontents\b", re.I),
    re.compile(r"\bindependent auditor", re.I),
)

# Parent-company-only statements. Real statements, wrong entity: the database
# wants the group. Heavily penalised rather than disqualified, so that a
# company-only filing with no consolidated statements still extracts.
_PARENT_ONLY = re.compile(
    r"\b(company|parent|bank)\b.{0,25}\b(balance sheet|statement of financial position|"
    r"income statement|statement of cash ?flows?)\b|\bcompany only\b",
    re.I,
)
_CONSOLIDATED = re.compile(r"\b(consolidated|group)\b", re.I)

_LOCATE_THRESHOLD = 95

# A statement heading is a heading. "230 Consolidated Statement of Income" is a
# line in a table of contents — Shell's financial-statements contents page
# out-scored the real income statement on title match alone.
_LEADING_PAGE_NO = re.compile(r"^\s*\d{1,4}\s+\S")

# Comprehensive income is a different statement from the income statement and
# frequently carries only OCI lines. Rank it below, never above.
_COMPREHENSIVE = re.compile(r"\bcomprehensive income\b", re.I)


def _is_contents_page(text: str) -> bool:
    """True if the page lists several statement titles rather than being one.

    Counted across the whole page, not the heading region: a contents page is
    exactly the page on which every title appears at once.
    """
    hits = 0
    for pats in _TITLE_PATTERNS.values():
        if any(p.search(text) for p in pats):
            hits += 1
    if hits < 3:
        return False
    # Corroborate with the page-number gutter that a contents list always has.
    numbered = sum(
        1 for line in text.splitlines()
        if _LEADING_PAGE_NO.match(line) and any(
            p.search(line) for pats in _TITLE_PATTERNS.values() for p in pats
        )
    )
    return numbered >= 2


def _top_lines(text: str, n: int = 14) -> List[str]:
    out: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s:
            out.append(s)
        if len(out) >= n:
            break
    return out


def _score_page(text: str, ptype: str) -> Tuple[int, str]:
    """Score one page as a candidate primary statement of type `ptype`.

    Returns (score, matched_title). Score <= 0 means "not this statement".
    """
    heads = _top_lines(text)
    if not heads:
        return 0, ""

    head_blob = " \n ".join(heads[:6])
    for pat in _DISQUALIFY_TOP:
        if pat.search(head_blob):
            return 0, ""
    if _is_contents_page(text):
        return 0, ""

    best = 0
    title = ""
    for line_idx, line in enumerate(heads):
        # A statement title is a heading, not a sentence buried in prose.
        if len(line) > 120:
            continue
        for rank, pat in enumerate(_TITLE_PATTERNS[ptype]):
            if not pat.search(line):
                continue
            score = 100
            score -= rank * 6              # earlier pattern = more canonical
            score -= line_idx * 5          # earlier line = more like a heading
            if _CONSOLIDATED.search(line):
                score += 45
            if _PARENT_ONLY.search(line):
                score -= 90
            if _COMPREHENSIVE.search(line):
                score -= 25
            if _LEADING_PAGE_NO.match(line):
                score -= 80
            if score > best:
                best, title = score, line
    if best <= 0:
        return 0, ""

    hits = sum(1 for pat in _ANCHORS[ptype] if pat.search(text))
    best += hits * 14
    # A statement with fewer than two of its defining rows is not the statement.
    if hits < 2:
        return 0, ""

    return best, title


_MAX_CANDIDATES = 6
# A grid with fewer rows than this is not a financial statement, whatever the
# page is titled.
_MIN_GRID_ROWS = 4


def locate_statements(pages: Sequence[dict]) -> Dict[str, List[dict]]:
    """Return {ptype: [candidate pages, best first]}.

    Ranked rather than single-best because a title is evidence, not proof.
    JLR's annual report carries a narrative page headed "CONSOLIDATED INCOME
    STATEMENT" 74 pages before the statement itself; it out-scores nothing on
    structure — it simply has no year columns — so the caller walks the ranking
    until a page actually grids. A statement type absent from the result was
    not found; its fields are nulled with ``no_statement_page`` rather than
    scavenged from elsewhere.
    """
    scored: Dict[str, List[Tuple[int, dict, str]]] = {p: [] for p in _STATEMENT_TYPES}
    for rec in pages:
        for ptype in _STATEMENT_TYPES:
            score, title = _score_page(rec["text"], ptype)
            if score >= _LOCATE_THRESHOLD:
                scored[ptype].append((score, rec, title))

    out: Dict[str, List[dict]] = {}
    for ptype, cands in scored.items():
        if not cands:
            continue
        cands.sort(key=lambda c: (-c[0], c[1]["page_n"]))
        out[ptype] = [
            {**rec, "ptype": ptype, "score": score, "title": title}
            for score, rec, title in cands[:_MAX_CANDIDATES]
        ]
    return out


# ── Scale / currency, scoped to a region ────────────────────────────────────
_CURRENCY_MARKERS: List[Tuple[str, Tuple[str, ...]]] = [
    ("GBP", ("£",)),
    ("EUR", ("€",)),
    ("USD", ("$",)),
    ("PKR", ("rs.", "rs ", "pkr", "rupee")),
    ("AED", ("aed", "dirham")),
    ("SAR", ("sar", "riyal")),
    ("INR", ("inr", "₹")),
]
_CURRENCY_WORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("GBP", ("gbp", "sterling", "pound")),
    ("EUR", ("eur", "euro")),
    ("USD", ("usd", "dollar")),
]

_SYMBOL = {
    "GBP": "£", "EUR": "€", "USD": "$", "PKR": "Rs.",
    "AED": "AED", "SAR": "SAR", "INR": "₹", "Unknown": "",
}

_BILLION_RE = re.compile(r"\bbn\b|\bbillions?\b|['’]?000,?000\b", re.I)
_MILLION_RE = re.compile(r"\bm\b|\bmn\b|\bmillions?\b|[€£$]\s?m\b", re.I)
_THOUSAND_RE = re.compile(r"\bthousands?\b|['’]000\b|\b000s\b|\bk\b", re.I)


def _detect_currency(region: str) -> Optional[str]:
    """Currency from a region of text. Symbol counts beat word mentions."""
    counts: Dict[str, int] = {}
    low = region.lower()
    for code, marks in _CURRENCY_MARKERS:
        c = sum(low.count(m) for m in marks)
        if c:
            counts[code] = counts.get(code, 0) + c
    if counts:
        return max(counts.items(), key=lambda kv: kv[1])[0]
    for code, words in _CURRENCY_WORDS:
        if any(w in low for w in words):
            return code
    return None


def _detect_scale(region: str) -> Optional[float]:
    """Scale factor from a region of text, or None if the region says nothing."""
    if _BILLION_RE.search(region):
        return 1_000_000_000.0
    if _MILLION_RE.search(region):
        return 1_000_000.0
    if _THOUSAND_RE.search(region):
        return 1_000.0
    return None


def resolve_units(
    header_region: str, page_text: str, doc_text: str
) -> Tuple[float, str, str, str, str]:
    """(scale, scale_source, currency, currency_source, unit_label).

    Tried narrowest-first. The source string is part of the contract: a caller
    that sees ``document`` knows the number was scaled by an assumption made
    somewhere else in the report, not by this statement's own header.
    """
    scale = _detect_scale(header_region)
    scale_source = "statement_header"
    if scale is None:
        scale = _detect_scale(page_text)
        scale_source = "page_body"
    if scale is None:
        scale = _detect_scale(doc_text[:40_000])
        scale_source = "document"
    if scale is None:
        scale, scale_source = 1.0, "unscaled_default"

    currency = _detect_currency(header_region)
    currency_source = "statement_header"
    if currency is None:
        currency = _detect_currency(page_text)
        currency_source = "page_body"
    if currency is None:
        currency = _detect_currency(doc_text[:40_000])
        currency_source = "document"
    if currency is None:
        currency, currency_source = "Unknown", "not_found"

    word = {
        1_000_000_000.0: "billions",
        1_000_000.0: "millions",
        1_000.0: "thousands",
        1.0: "units",
    }.get(scale, "units")
    sym = _SYMBOL.get(currency, "")
    unit_label = f"{sym} {word}".strip()
    if scale_source != "statement_header":
        unit_label += f" ({scale_source})"
    return scale, scale_source, currency, currency_source, unit_label


# ── Word-geometry grid ──────────────────────────────────────────────────────
_NUM_TOKEN = re.compile(
    r"^\(?[-−–]?[£€$]?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?$|^\(?[-−–]?[£€$]?\s?\d+(?:\.\d+)?\)?$"
)


def _token_to_number(tok: str) -> Optional[float]:
    """Parse one printed token. Brackets and unicode minus mean negative."""
    s = tok.strip()
    if not s:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    s = s.replace("£", "").replace("€", "").replace("$", "").replace(",", "").strip()
    if s.startswith(("−", "–", "-")):
        neg, s = True, s[1:].strip()
    if not s or not re.fullmatch(r"\d+(?:\.\d+)?", s):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _cluster_lines(words: List[dict], tol: float = 3.0) -> List[List[dict]]:
    """Group extracted words into visual lines by their `top` coordinate."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines: List[List[dict]] = []
    current: List[dict] = [ordered[0]]
    for w in ordered[1:]:
        if abs(w["top"] - current[-1]["top"]) <= tol:
            current.append(w)
        else:
            lines.append(sorted(current, key=lambda x: x["x0"]))
            current = [w]
    lines.append(sorted(current, key=lambda x: x["x0"]))
    return lines


def _merge_bracket_tokens(line: List[dict]) -> List[dict]:
    """pdfplumber sometimes splits "(1,234)" into separate word boxes."""
    out: List[dict] = []
    i = 0
    while i < len(line):
        w = dict(line[i])
        if w["text"] == "(" and i + 1 < len(line):
            j = i + 1
            merged = "("
            while j < len(line) and not merged.endswith(")"):
                merged += line[j]["text"]
                j += 1
            if merged.endswith(")"):
                w["text"] = merged
                w["x1"] = line[j - 1]["x1"]
                out.append(w)
                i = j
                continue
        out.append(w)
        i += 1
    return out


def _find_year_header(lines: List[List[dict]]) -> Tuple[Dict[int, int], Dict[int, Tuple[float, float]], float]:
    """Find the header line carrying the year columns.

    Returns (col_index → year, col_index → (x_centre, x_right), header_top).
    Scans the whole page rather than a fixed prefix: on many reports the
    statement title sits under a large section banner, so the header row is
    well below the top of the page.
    """
    best_anchors: List[Tuple[float, float, int]] = []
    best_top = 0.0
    best_idx = -1
    for li, line in enumerate(lines):
        anchors: List[Tuple[float, float, int]] = []
        for w in line:
            yr = _year_of_token(w["text"])
            if yr is not None:
                anchors.append(((w["x0"] + w["x1"]) / 2.0, w["x1"], yr))
        # Most years wins; ties keep the earliest such line, which is the
        # column header rather than a comparative mentioned further down.
        if len(anchors) > len(best_anchors):
            best_anchors = anchors
            best_top = line[0]["top"]
            best_idx = li

    if not best_anchors:
        return {}, {}, 0.0

    best_anchors.sort(key=lambda a: a[0])

    # ── Sub-columns under a year group ──────────────────────────────────────
    # Tesco's income statement heads each year with three columns — "Before
    # adjusting items", "Adjusting items" and "Total" — so the year anchor sits
    # above a group, not above a column. Matching a value to the nearest year
    # anchor then picks whichever of the three happens to be closest, which is
    # not the number anyone means. When a sub-header row carries exactly one
    # "Total" per year, re-anchor each year onto its own Total column.
    totals: List[Tuple[float, float]] = []
    for line in lines[best_idx + 1: best_idx + 4]:
        found = [
            ((w["x0"] + w["x1"]) / 2.0, w["x1"])
            for w in line if w["text"].strip().lower() == "total"
        ]
        if len(found) == len(best_anchors) and len(found) > 1:
            totals = sorted(found)
            best_top = max(best_top, line[0]["top"])
            break
    if totals:
        best_anchors = [
            (tx, tx1, yr) for (tx, tx1), (_, _, yr) in zip(totals, best_anchors)
        ]

    year_cols: Dict[int, int] = {}
    col_x: Dict[int, Tuple[float, float]] = {}
    for idx, (xc, x1, yr) in enumerate(best_anchors):
        year_cols[idx] = yr
        col_x[idx] = (xc, x1)
    return year_cols, col_x, best_top


def _split_panels(
    col_x: Dict[int, Tuple[float, float]], year_cols: Dict[int, int]
) -> List[Tuple[float, float, List[int]]]:
    """Split a page into side-by-side statement panels.

    Tesco prints its balance sheet as two panels on one page — assets on the
    left, liabilities and equity on the right — each with its own 2025 and 2024
    columns. Read as a single grid, every line pairs the left panel's label
    with the right panel's numbers, so "Goodwill and other intangible assets"
    comes back holding the trade payables figure. A repeated year in the header
    is the signal: it cannot happen within one panel.

    Returns [(x_left, x_right, [column indices])] in reading order. A
    single-panel page returns one band spanning the width of the page.
    """
    ordered = sorted(col_x.items(), key=lambda kv: kv[1][0])
    panels: List[List[int]] = []
    seen: Set[int] = set()
    for ci, _ in ordered:
        yr = year_cols[ci]
        if not panels or yr in seen:
            panels.append([ci])
            seen = {yr}
        else:
            panels[-1].append(ci)
            seen.add(yr)

    bands: List[Tuple[float, float, List[int]]] = []
    for k, cols in enumerate(panels):
        left = 0.0 if k == 0 else max(col_x[c][1] for c in panels[k - 1]) + 2.0
        right = (float("inf") if k == len(panels) - 1
                 else max(col_x[c][1] for c in cols) + 2.0)
        bands.append((left, right, cols))
    return bands


def build_grid(
    page, text: str, ptype: str, page_n: int, doc_text: str
) -> Optional[StatementGrid]:
    """Build a labelled grid for one located statement page.

    Columns are pinned to years by x-geometry. A page whose header carries no
    year is not gridded at all — v4 has no positional fallback, by design.
    """
    try:
        words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    except Exception:
        return None
    if not words:
        return None

    lines = [_merge_bracket_tokens(l) for l in _cluster_lines(words)]
    year_cols, col_x, header_top = _find_year_header(lines)
    if not year_cols:
        return None

    # Column matching tolerance: half the narrowest gap between adjacent year
    # anchors, capped. Narrow columns get a strict tolerance automatically.
    xs = [col_x[i][0] for i in sorted(col_x)]
    if len(xs) > 1:
        min_gap = min(b - a for a, b in zip(xs, xs[1:]))
        tol = max(8.0, min(38.0, min_gap * 0.45))
    else:
        tol = 38.0

    header_region = "\n".join(
        " ".join(w["text"] for w in line)
        for line in lines
        if line and line[0]["top"] <= header_top + 4
    )
    scale, scale_src, currency, cur_src, unit_label = resolve_units(
        header_region, text, doc_text
    )

    bands = _split_panels(col_x, year_cols)

    rows: List[GridRow] = []
    label_idx = 0
    for band_left, band_right, band_cols in bands:
        for line in lines:
            if not line or line[0]["top"] <= header_top + 2:
                continue
            cells = [w for w in line if band_left <= w["x0"] < band_right]
            if not cells:
                continue

            numeric: List[Tuple[dict, float]] = []
            for w in cells:
                v = _token_to_number(w["text"])
                if v is not None:
                    numeric.append((w, v))
            if not numeric:
                continue

            first_num_x0 = numeric[0][0]["x0"]
            label_text = " ".join(
                w["text"] for w in cells if w["x1"] <= first_num_x0 + 0.5
            ).strip()
            if len(label_text) < 3:
                continue
            clean = _clean_label(label_text)
            if not clean or any(s in clean for s in _SKIP):
                continue

            values: Dict[int, float] = {}
            for w, v in numeric:
                xc = (w["x0"] + w["x1"]) / 2.0
                x1 = w["x1"]
                best_col, best_dist = None, None
                for ci in band_cols:
                    a_xc, a_x1 = col_x[ci]
                    d = min(abs(xc - a_xc), abs(x1 - a_x1))
                    if best_dist is None or d < best_dist:
                        best_col, best_dist = ci, d
                # Not near any year anchor → not a value of this statement.
                # This is what excludes the "Notes" reference column structurally.
                if best_col is None or best_dist > tol:
                    continue
                values.setdefault(best_col, v)

            if not values:
                continue

            rows.append(GridRow(
                label=_label_from_index(label_idx),
                text=label_text,
                values=values,
                top=line[0]["top"],
            ))
            label_idx += 1

    if not rows:
        return None

    return StatementGrid(
        page_n=page_n,
        ptype=ptype,
        rows=rows,
        year_cols=year_cols,
        scale=scale,
        scale_source=scale_src,
        currency=currency,
        currency_source=cur_src,
        unit_label=unit_label,
        header_top=header_top,
        title="",
    )


def _label_from_index(i: int) -> str:
    letters: List[str] = []
    n = i
    while True:
        letters.append(chr(ord("A") + n % 26))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(letters))


# ── Classification ──────────────────────────────────────────────────────────
# Rows whose printed label contains a field's name without BEING that field.
# "Total revenue and other income" is revenue plus share of JV profit plus
# interest income — mapping it to `revenue` overstates Shell's top line by
# $6.8bn. "Income attributable to shareholders" is the post-minority
# allocation, not profit for the period.
_REJECT_ROW: Dict[str, Tuple[re.Pattern, ...]] = {
    "pnl": (
        re.compile(r"\band other income\b", re.I),
        re.compile(r"\battributable to\b", re.I),
        re.compile(r"\bdiscontinued\b", re.I),
        re.compile(r"\bother comprehensive\b", re.I),
        # eps means basic EPS. Shell prints basic (3.03) then diluted (3.00),
        # and "diluted earnings per share" is the longer alias, so without this
        # the diluted figure wins on specificity.
        re.compile(r"\bdiluted\b", re.I),
    ),
    "bs": (
        re.compile(r"\battributable to\b", re.I),
        # "Net current liabilities" is assets less liabilities, not liabilities.
        re.compile(r"\bnet current\b", re.I),
        # Held-for-sale groupings are line items inside a section, not the
        # section total they textually resemble.
        re.compile(r"\bdisposal group\b|\bheld for sale\b", re.I),
    ),
    "cf": (
        # IAS 7 defines "cash generated from operations" as a line WITHIN the
        # operating section — before interest and tax paid. The statement's
        # operating total is the "net cash from operating activities" line
        # below it. JLR prints 4,909 then 4,598; only the second is the figure
        # the accounts report.
        re.compile(r"^cash (generated from|from|flows? from) operations\b", re.I),
    ),
}


_PAREN_ALT = re.compile(r"\([^)]*\)")
_SLASH = re.compile(r"\s*/\s*")

# A row whose label opens with a dash or bullet is a component of the line
# above it, not a line in its own right. HSBC's income statement reads
#     Net interest income        34,794
#     – interest income          97,872
#     – interest expense        (63,078)
# and matching "interest expense" as the group's finance cost picks a bank's
# gross funding cost. The same shape hides "– insurance service revenue" under
# an insurance result and offers it as group revenue.
_COMPONENT_PREFIX = re.compile(r"^\s*[–—\-•‒·]\s+")

# Fields that only ever name a subtotal. A label may claim one of these only if
# it reads as a subtotal: "Total current liabilities", or the bare section name.
# Unilever's balance sheet prints no subtotals at all, and matching the alias
# inside a component line gave current_liabilities = "Trade payables and other
# current liabilities" (16,939) — which then contradicted the printed total
# liabilities of 52,884 and caused a correctly-read primary to be withheld.
_SUBTOTAL_ONLY: Set[str] = {
    "total_assets", "total_liabilities", "total_equity",
    "current_assets", "non_current_assets",
    "current_liabilities", "non_current_liabilities",
}


def normalise_label(text: str) -> str:
    """Strip the "/(alternative)" wording statements use for sign pairs.

    Real labels read "Profit/(loss) before tax", "Net cash generated
    from/(used in) operating activities", "Operating (loss)/profit". The alias
    dictionary is written in the plain form, and substring matching cannot see
    through the inserted alternative — Tesco's profit before tax and operating
    cash flow both failed to classify for exactly this reason. Removing the
    parenthesised alternative and flattening the slash recovers the plain form.
    Labels only: values are parsed from their own tokens, so bracketed
    negatives are untouched by this.
    """
    flat = _PAREN_ALT.sub(" ", _clean_label(text))
    flat = _SLASH.sub(" ", flat)
    return re.sub(r"\s+", " ", flat).strip()


def _match_alias(label: str, mapping: Dict[str, str]) -> Optional[Tuple[str, int, bool]]:
    """(field, alias length, exact) for the most specific alias in `label`."""
    best: Optional[Tuple[str, int, bool]] = None
    for pattern, field in mapping.items():
        if pattern not in label:
            continue
        cand = (field, len(pattern), label == pattern)
        if best is None or (cand[2], cand[1]) > (best[2], best[1]):
            best = cand
    return best


def _row_strength(label: str, alias_len: int, exact: bool) -> Tuple[int, int, int]:
    """How canonically this row's label names its field. Bigger is better.

    Exactness first, then how few extra words the label carries beyond the
    alias, then brevity. Tesco's income statement prints "Revenue from sale of
    goods and services" (69,191) two rows above "Revenue" (69,916) and both are
    exact aliases; the bare one is the top line.
    """
    return (int(exact), -(len(label) - alias_len), -len(label))


def _dict_classify(text: str, ptype: str) -> Optional[Tuple[str, int, bool]]:
    """(field, alias length, exact) or None."""
    clean = _clean_label(text)
    if not clean or len(clean) < 3:
        return None
    if any(s in clean for s in _SKIP):
        return None
    if _COMPONENT_PREFIX.match(text):
        return None
    flat = normalise_label(text)
    rejects = _REJECT_ROW.get(ptype, ())
    # Tested against both forms: Tesco writes "Cash generated from/(used in)
    # operations", which only reads as the IAS 7 sub-total once the "/(used
    # in)" alternative is flattened away.
    if any(pat.search(clean) or pat.search(flat) for pat in rejects):
        return None
    if ptype == "pnl" and "tax" in clean:
        is_pbt = any(p in clean for p in (
            "before tax", "before taxation", "before income tax", "before wppf",
        ))
        if not is_pbt and not _PNL_TAX_PATTERN.search(text):
            return None

    mapping = _get_mapping(ptype)
    best: Optional[Tuple[str, int, bool]] = None
    best_label = clean
    for candidate in ((clean, flat) if flat != clean else (clean,)):
        m = _match_alias(candidate, mapping)
        if m and (best is None
                  or _row_strength(candidate, m[1], m[2])
                  > _row_strength(best_label, best[1], best[2])):
            best, best_label = m, candidate
    if best is None:
        return None
    if best[0] in _SUBTOTAL_ONLY:
        alias = next(
            (a for a, f in mapping.items()
             if f == best[0] and len(a) == best[1] and a in best_label),
            None,
        )
        if not (best_label.startswith("total ")
                or (alias and best_label.startswith(alias))):
            return None
    return best


def llm_disabled() -> bool:
    return os.getenv("FINEX_DISABLE_LLM", "").strip().lower() in ("1", "true", "yes")


def classify_rows(grid: StatementGrid) -> Dict[str, Tuple[str, str, Tuple[int, int, int]]]:
    """{row_label: (field, classifier, strength)} — dictionary first, LLM after.

    ``strength`` is (exact, alias length): how specifically the row's printed
    label names the field. It decides which row wins when several map to the
    same field — HSBC's income statement offers four candidates for
    ``operating_profit`` and only one of them is the line called "Operating
    profit".

    The LLM sees line-item TEXT only. No digits go into the prompt and no
    digits come back; it returns a field name from a fixed vocabulary or null.
    """
    out: Dict[str, Tuple[str, str, Tuple[int, int, int]]] = {}
    unmatched: List[GridRow] = []
    for row in grid.rows:
        m = _dict_classify(row.text, grid.ptype)
        if m:
            field, alias_len, exact = m
            label = normalise_label(row.text) if exact else _clean_label(row.text)
            out[row.label] = (
                field, "dictionary", _row_strength(label, alias_len, exact),
            )
        else:
            unmatched.append(row)

    if not unmatched or llm_disabled():
        return out

    try:
        from finex.extract_pdf import LabeledRow, _llm_available, _llm_classify_rows
        if not _llm_available():
            return out
        shim = [
            LabeledRow(label=r.label, text=r.text, col_values={}, source="v4")
            for r in unmatched
        ]
        for label, field in (_llm_classify_rows(shim, grid.ptype) or {}).items():
            out.setdefault(label, (field, "llm", (0, -999, -999)))
    except Exception as exc:  # classification is best-effort by construction
        print(f"      ⚠ LLM classification unavailable: {exc}")
    return out


# ── Value extraction ────────────────────────────────────────────────────────
# A line qualified by segment or adjustment is not the statutory line, even
# when it maps to the same field. Tesco prints "Profit for the year from
# continuing operations" (1,604) above "Profit for the year" (1,630); the
# second is the number the accounts report. Unqualified always wins.
_QUALIFIED = re.compile(
    r"\b(continuing|discontinued)\b|\bbefore exceptional\b|\badjusted\b|"
    r"\bunderlying\b|\bexcluding\b|\bbefore adjusting\b",
    re.I,
)

def _confidence(classifier: str, row_text: str, accumulated: int) -> float:
    conf = 1.0
    if classifier == "llm":
        conf -= 0.20
    clean = _clean_label(row_text)
    if clean.startswith("total ") or clean.startswith("net "):
        conf += 0.05
    if accumulated > 1:
        conf -= 0.15
    return max(0.0, min(1.0, conf))


def extract_year(
    grid: StatementGrid,
    field_map: Dict[str, Tuple[str, str, Tuple[int, int, int]]],
    year: int,
) -> Dict[str, Cell]:
    """Read one year's column off a grid. No column substitution.

    If the pinned column for `year` holds no value on a row, that row simply
    contributes nothing. v3 fell back to the first available number here, which
    is how prior-year figures ended up as copies of current-year figures.
    """
    # A year can own several columns. Tesco's balance sheet is printed as two
    # side-by-side panels — assets on the left, liabilities and equity on the
    # right — each with its own 2025 and 2024 columns. Taking only the first
    # matching column silently discards half the balance sheet.
    cols = [ci for ci, yr in sorted(grid.year_cols.items()) if yr == year]
    if not cols:
        return {}

    rows = {r.label: r for r in grid.rows}
    raw: Dict[str, float] = {}
    src_row: Dict[str, GridRow] = {}
    src_cls: Dict[str, str] = {}
    is_total: Dict[str, bool] = {}
    qualified: Dict[str, bool] = {}
    counted: Dict[str, int] = {}

    strength: Dict[str, Tuple[int, int, int]] = {}
    for label, (field, classifier, row_strength) in field_map.items():
        row = rows.get(label)
        if row is None:
            continue
        val = None
        col = None
        for ci in cols:
            if ci in row.values:
                val, col = row.values[ci], ci
                break
        if val is None:
            continue

        total_row = _clean_label(row.text).startswith("total ")
        qual_row = bool(_QUALIFIED.search(row.text))

        def _take() -> None:
            raw[field] = val
            src_row[field] = row
            src_cls[field] = classifier
            is_total[field] = total_row
            qualified[field] = qual_row
            strength[field] = row_strength
            counted[field] = 1

        # Precedence when several rows claim one field:
        #   1. statutory beats a segment/adjusted variant
        #   2. a printed total beats a component line — HSBC's "Total operating
        #      expenses" (36,428) over "General and administrative expenses"
        #      (11,959), which carries the longer alias but is one line inside it
        #   3. the more specific label wins — HSBC offers four candidates for
        #      operating_profit and only "Operating profit" is exact
        #   4. first occurrence
        if field not in raw:
            _take()
        elif qualified.get(field) and not qual_row:
            _take()
        elif qual_row and not qualified.get(field):
            continue
        elif total_row and not is_total.get(field):
            _take()
        elif is_total.get(field) and not total_row:
            continue
        elif row_strength > strength.get(field, (0, -999, -999)):
            _take()
        elif row_strength < strength.get(field, (0, -999, -999)):
            continue
        elif not is_total.get(field) and field in _ACCUMULATE_OK:
            if field in _ALWAYS_POSITIVE:
                raw[field] = abs(raw[field]) + abs(val)
            else:
                raw[field] += val
            counted[field] += 1

    out: Dict[str, Cell] = {}
    for field, val in raw.items():
        scale = 1.0 if field in _NO_SCALE else grid.scale
        scaled = abs(val) * scale if field in _ALWAYS_POSITIVE else val * scale
        row = src_row[field]
        classifier = src_cls[field]
        prov = Provenance(
            page=grid.page_n,
            statement=grid.ptype,
            row_label=row.label,
            row_text=row.text,
            column_index=col,
            year=year,
            raw=val,
            scale=scale,
            scale_source=grid.scale_source,
            currency=grid.currency,
            currency_source=grid.currency_source,
            classifier=classifier,
            origin="read",
            accumulated_from=counted[field],
            confidence=_confidence(classifier, row.text, counted[field]),
        )
        if grid.scale_source != "statement_header" and field not in _NO_SCALE:
            prov.notes.append(f"scale taken from {grid.scale_source}")
            prov.confidence -= 0.10
        out[field] = Cell(value=round(scaled, 2), prov=prov)
    return out


def merge_cells(dst: Dict[str, Cell], src: Dict[str, Cell]) -> None:
    """Merge by confidence, never by page order.

    Ties are broken toward the incumbent so the result does not depend on the
    order pages happen to be visited.
    """
    for field, cell in src.items():
        cur = dst.get(field)
        if cur is None or cell.prov.confidence > cur.prov.confidence:
            dst[field] = cell


# ── Identities: derive only, never overwrite ────────────────────────────────
# (result, addends, subtracted, tolerance)
#
# STRICT identities are definitional and hold on any set of accounts. Assets
# ARE liabilities plus equity, and each side ARE its current and non-current
# halves. A disagreement means something was misread, so it lowers confidence
# in every participant, and a missing field may be derived from the others.
#
# Note what is NOT here. The whole P&L cascade was strict until it was measured
# against real statements, and every rung of it broke on correctly-extracted
# data — see ADVISORY_IDENTITIES below. Only the balance sheet is an equation.
STRICT_IDENTITIES: List[Tuple[str, List[str], List[str], float]] = [
    ("total_assets",      ["current_assets", "non_current_assets"],            [],                    0.02),
    ("total_liabilities", ["current_liabilities", "non_current_liabilities"],  [],                    0.02),
    ("total_equity",      ["total_assets"],                                   ["total_liabilities"],  0.02),
]

# ADVISORY identities are approximations of the P&L cascade. None of them is an
# accounting identity on an IFRS income statement, which carries other
# operating income, impairments, net credit losses, finance income, share of
# associates and discontinued operations between these lines. Measured on the
# regression corpus, with every figure verified correct against the printed
# statements:
#
#   gross profit = revenue - cost of sales      Tesco, out by 12% (insurance
#                                               service expenses sit between)
#   operating profit = gross profit - opex      Vodafone, out by 110%
#   profit before tax = op profit - finance     HSBC, out by 48%
#   net profit = PBT - tax                      Tesco and JLR, out by 32%
#                                               (discontinued operations)
#
# So these never derive and never correct. They emit an advisory, and nothing
# downstream loses confidence because of them. v3 treated the whole cascade as
# strict, which is how a correctly-read operating loss of (411) became a
# computed 4,138 — and, worse, how a correctly-read profit before tax got
# withheld for disagreeing with a rule that was never true.
ADVISORY_IDENTITIES: List[Tuple[str, List[str], List[str], float]] = [
    ("gross_profit",      ["revenue"],           ["cost_of_goods_sold"], 0.03),
    ("operating_profit",  ["gross_profit"],      ["operating_expenses"], 0.10),
    ("profit_before_tax", ["operating_profit"],  ["finance_cost"],       0.15),
    ("net_profit",        ["profit_before_tax"], ["tax_expense"],        0.10),
]


def _rule_text(result_f: str, addends: List[str], subtracted: List[str]) -> str:
    txt = f"{result_f} = {' + '.join(addends)}"
    if subtracted:
        txt += f" − {' − '.join(subtracted)}"
    return txt


def apply_identities(
    cells: Dict[str, Cell], verbose: bool = False
) -> Tuple[List[dict], List[dict]]:
    """Fill genuine gaps from strict identities. Never touch a read value.

    Returns (conflicts, advisories). v3 auto-corrected: when a printed total
    disagreed with its components it replaced the printed total with the
    computed one, then ran validation and reported a pass. That is why the
    validator could not fail. Here a disagreement is recorded and both sides
    keep their values; deciding which side is wrong needs evidence the
    extractor does not have.
    """
    conflicts: Dict[str, dict] = {}
    advisories: Dict[str, dict] = {}

    for _ in range(2):  # a derived field can enable a later rule
        for result_f, addends, subtracted, tol in STRICT_IDENTITIES:
            comps = addends + subtracted
            if not all(c in cells for c in comps):
                continue
            computed = (sum(cells[a].value for a in addends)
                        - sum(cells[s].value for s in subtracted))
            rule = _rule_text(result_f, addends, subtracted)

            if result_f in cells:
                stored = cells[result_f].value
                denom = max(abs(computed), abs(stored), 1.0)
                diff = abs(computed - stored) / denom
                if diff <= tol or rule in conflicts:
                    continue
                conflicts[rule] = {
                    "rule": rule,
                    "field": result_f,
                    "read": round(stored, 2),
                    "computed": round(computed, 2),
                    "diff_pct": round(diff * 100, 2),
                    "components": comps,
                }
                for f in [result_f] + comps:
                    cell = cells.get(f)
                    if cell and Reason.IDENTITY_CONFLICT not in cell.prov.notes:
                        cell.prov.notes.append(Reason.IDENTITY_CONFLICT)
                        cell.prov.confidence *= 0.4
                if verbose:
                    print(f"   ⚠ Identity conflict on {result_f}: read {stored:,.0f} "
                          f"vs computed {computed:,.0f} ({diff:.1%}) — "
                          f"both kept, neither corrected")
            else:
                base = min(cells[c].prov.confidence for c in comps)
                src = cells[comps[0]].prov
                cells[result_f] = Cell(
                    value=round(computed, 2),
                    prov=Provenance(
                        origin="derived", classifier="identity", rule=rule,
                        year=src.year, statement=src.statement,
                        currency=src.currency, currency_source=src.currency_source,
                        scale=src.scale, scale_source=src.scale_source,
                        confidence=round(base * 0.75, 3),
                    ),
                )
                if verbose:
                    print(f"   ℹ Derived {result_f} = {computed:,.0f}  [{rule}]")

        # Back-derive a single missing component of a strict identity.
        for result_f, addends, subtracted, tol in STRICT_IDENTITIES:
            comps = addends + subtracted
            if result_f not in cells:
                continue
            missing = [c for c in comps if c not in cells]
            if len(missing) != 1:
                continue
            m = missing[0]
            # Back-derivation is only sound where the identity is an equation
            # over a closed set. The balance sheet is; the P&L is not. Tesco's
            # income statement, with revenue unread, back-derived revenue = 0
            # from gross profit and cost of sales, and a profit before tax of
            # (215)m from a net profit line that was not the group total.
            if m not in _BACK_DERIVABLE:
                continue
            if m in addends:
                derived = (cells[result_f].value
                           - sum(cells[a].value for a in addends if a != m)
                           + sum(cells[s].value for s in subtracted))
            else:
                derived = (sum(cells[a].value for a in addends)
                           - cells[result_f].value
                           - sum(cells[s].value for s in subtracted if s != m))
            if m in _ALWAYS_POSITIVE and derived < 0:
                continue
            known = [result_f] + [c for c in comps if c in cells]
            base = min(cells[k].prov.confidence for k in known)
            src = cells[result_f].prov
            rule = f"{m} back-derived from {_rule_text(result_f, addends, subtracted)}"
            cells[m] = Cell(
                value=round(derived, 2),
                prov=Provenance(
                    origin="derived", classifier="identity", rule=rule,
                    year=src.year, statement=src.statement,
                    currency=src.currency, currency_source=src.currency_source,
                    scale=src.scale, scale_source=src.scale_source,
                    confidence=round(base * 0.6, 3),
                ),
            )
            if verbose:
                print(f"   ℹ Back-derived {m} = {derived:,.0f}  [{rule}]")

    # Advisories: reported, never acted on.
    for result_f, addends, subtracted, tol in ADVISORY_IDENTITIES:
        comps = addends + subtracted
        if result_f not in cells or not all(c in cells for c in comps):
            continue
        computed = (sum(cells[a].value for a in addends)
                    - sum(cells[s].value for s in subtracted))
        stored = cells[result_f].value
        denom = max(abs(computed), abs(stored), 1.0)
        diff = abs(computed - stored) / denom
        if diff <= tol:
            continue
        rule = _rule_text(result_f, addends, subtracted)
        advisories[rule] = {
            "rule": rule,
            "field": result_f,
            "read": round(stored, 2),
            "computed": round(computed, 2),
            "diff_pct": round(diff * 100, 2),
            "note": "cascade approximation, not an identity — no action taken",
        }
        if verbose:
            print(f"   ℹ Advisory {rule}: read {stored:,.0f} vs "
                  f"cascade {computed:,.0f} ({diff:.1%}) — not an identity, ignored")

    return list(conflicts.values()), list(advisories.values())


# ── Validation ──────────────────────────────────────────────────────────────
def validate(
    values: Dict[str, float], derived: Optional[Set[str]] = None
) -> dict:
    """Consistency checks over a plain {field: number} mapping.

    Deliberately unable to mutate its input. In v3 the corrections happened
    first and this ran second, so it could only ever agree with itself.

    A check whose inputs include a DERIVED field is tautological — the value
    was computed from the identity the check then verifies — so it is reported
    separately and never counted as a pass. Without this the pass count grows
    every time the extractor fills a gap, which reads as the extraction getting
    better while it is only getting more self-referential.
    """
    derived = derived or set()
    passed: List[str] = []
    failed: List[dict] = []
    advisory: List[dict] = []
    tautological: List[str] = []
    warnings: List[str] = []

    def chk(name: str, computed: float, stored: float, tol: float = 0.06,
            fields: Sequence[str] = (), strict: bool = True) -> None:
        if any(f in derived for f in fields):
            tautological.append(name)
            return
        denom = max(abs(computed), abs(stored), 1.0)
        diff = abs(computed - stored) / denom
        entry = {
            "check": name,
            "expected": round(computed, 2),
            "got": round(stored, 2),
            "diff_pct": round(diff * 100, 2),
        }
        if diff <= tol:
            passed.append(name)
        elif strict:
            failed.append(entry)
        else:
            # The P&L cascade is not an equation on a real IFRS statement, so a
            # disagreement here is information, not a defect. Counting it as a
            # failure trains the reader to ignore failures.
            advisory.append(entry)

    d = values
    if all(k in d for k in ("gross_profit", "revenue", "cost_of_goods_sold")):
        chk("1. Gross Profit = Revenue − COGS",
            d["revenue"] - d["cost_of_goods_sold"], d["gross_profit"],
            fields=("revenue", "cost_of_goods_sold", "gross_profit"), strict=False)
    if all(k in d for k in ("net_profit", "profit_before_tax", "tax_expense")):
        chk("2. Net Profit ≈ PBT − Tax",
            d["profit_before_tax"] - d["tax_expense"], d["net_profit"], tol=0.10,
            fields=("profit_before_tax", "tax_expense", "net_profit"), strict=False)
    if all(k in d for k in ("total_assets", "total_liabilities", "total_equity")):
        chk("3. Assets = Liabilities + Equity",
            d["total_liabilities"] + d["total_equity"], d["total_assets"],
            fields=("total_liabilities", "total_equity", "total_assets"))
    if all(k in d for k in ("total_assets", "current_assets", "non_current_assets")):
        chk("4. Total Assets = Current + Non-Current",
            d["current_assets"] + d["non_current_assets"], d["total_assets"],
            fields=("current_assets", "non_current_assets", "total_assets"))
    if all(k in d for k in ("total_liabilities", "current_liabilities", "non_current_liabilities")):
        chk("5. Total Liabilities = Current + Non-Current",
            d["current_liabilities"] + d["non_current_liabilities"], d["total_liabilities"],
            fields=("current_liabilities", "non_current_liabilities", "total_liabilities"))
    if all(k in d for k in ("operating_profit", "gross_profit", "operating_expenses")):
        chk("6. Operating Profit ≈ Gross Profit − OpEx",
            d["gross_profit"] - d["operating_expenses"], d["operating_profit"], tol=0.12,
            fields=("gross_profit", "operating_expenses", "operating_profit"), strict=False)
    if all(k in d for k in ("revenue", "gross_profit")):
        if d["revenue"] >= d["gross_profit"]:
            passed.append("7. Revenue ≥ Gross Profit")
        else:
            failed.append({
                "check": "7. Revenue ≥ Gross Profit",
                "expected": f">= {d['gross_profit']:,.0f}",
                "got": f"{d['revenue']:,.0f}",
                "diff_pct": None,
            })

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

    return {
        "passed": passed,
        "failed": failed,
        "advisory": advisory,
        "tautological": tautological,
        "warnings": warnings,
    }


# ── Refuse policy ───────────────────────────────────────────────────────────
def apply_refuse_policy(
    cells: Dict[str, Cell], nulls: Dict[str, str]
) -> Dict[str, Cell]:
    """Drop PRIMARY fields we do not trust, recording why.

    Non-primary fields are left alone: their confidence travels with them in
    the provenance and a consumer can filter on it. The primaries are the ones
    a person reads off a dashboard and acts on, so for those the pipeline
    would rather say nothing than say something wrong.
    """
    kept: Dict[str, Cell] = {}
    for field, cell in cells.items():
        below = cell.prov.confidence < CONFIDENCE_FLOOR
        # Primaries are withheld on low confidence. A DERIVED value is withheld
        # on low confidence whatever the field: it was never on the page, so a
        # weak derivation is pure invention.
        if below and (field in PRIMARY_FIELDS or cell.prov.origin == "derived"):
            reason = (Reason.IDENTITY_CONFLICT
                      if Reason.IDENTITY_CONFLICT in cell.prov.notes
                      else Reason.LOW_CONFIDENCE)
            nulls[field] = (
                f"{reason} (confidence {cell.prov.confidence:.2f} < "
                f"{CONFIDENCE_FLOOR:.2f}; value withheld: {cell.value:,.2f})"
            )
            continue
        kept[field] = cell
    return kept


def _statement_of(field: str) -> str:
    if field in ("total_assets", "total_liabilities", "total_equity"):
        return "bs"
    if field == "operating_cashflow":
        return "cf"
    return "pnl"


# ── Entry point ─────────────────────────────────────────────────────────────
def extract(pdf_path: str, verbose: bool = True) -> dict:
    """Extract a consolidated set of accounts from one PDF.

    Returns the v3-compatible shape — ``current``, ``prior``, ``metadata``,
    ``raw_text`` — extended with ``metadata["provenance"]``,
    ``metadata["nulls"]`` and ``metadata["statements"]``.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"📄 [v4] Reading: {pdf_path}")

    pages: List[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if len(text.strip()) > 40:
                pages.append({"page_obj": page, "page_n": i + 1, "text": text})

        if not pages:
            return {
                "current": {}, "prior": {}, "raw_text": "",
                "metadata": {
                    "extractor": EXTRACTOR_VERSION,
                    "error": "No readable text — PDF may be scanned or image-based",
                    "nulls": {f: Reason.NO_READABLE_TEXT for f in PRIMARY_FIELDS},
                },
            }

        doc_text = "\n".join(p["text"] for p in pages)
        log(f"   {len(pages)} readable pages")

        candidates = locate_statements(pages)

        # Grids are built while the pdfplumber document is still open.
        grids: Dict[str, StatementGrid] = {}
        chosen: Dict[str, dict] = {}
        rejected: Dict[str, List[str]] = {}
        for ptype in _STATEMENT_TYPES:
            cands = candidates.get(ptype, [])
            if not cands:
                log(f"   {ptype.upper():3} → NOT FOUND")
                continue
            for rec in cands:
                grid = build_grid(
                    rec["page_obj"], rec["text"], ptype, rec["page_n"], doc_text
                )
                why = None
                if grid is None:
                    why = "no year columns in header"
                elif len(grid.rows) < _MIN_GRID_ROWS:
                    why = f"only {len(grid.rows)} data rows"
                if why:
                    rejected.setdefault(ptype, []).append(
                        f"page {rec['page_n']} ({rec['score']}): {why}"
                    )
                    log(f"   {ptype.upper():3} ✗ page {rec['page_n']:>4} "
                        f"score {rec['score']:>4}: {why}")
                    continue
                grid.title = rec["title"]
                grids[ptype] = grid
                chosen[ptype] = rec
                log(f"   {ptype.upper():3} → page {grid.page_n:>4}  score {rec['score']:>4}  "
                    f"“{rec['title'][:56]}”")
                log(f"       {len(grid.rows)} rows | "
                    f"years {sorted(set(grid.year_cols.values()), reverse=True)} | "
                    f"{grid.unit_label} [{grid.currency}/{grid.currency_source}]")
                break
            else:
                log(f"   {ptype.upper():3} → {len(cands)} candidate page(s), none gridded "
                    f"— refusing (v4 has no positional fallback)")
        located = chosen
        no_year = [p for p in _STATEMENT_TYPES
                   if p in candidates and p not in chosen]

    # Years: from the located statements' own headers only.
    years: Set[int] = set()
    for grid in grids.values():
        years.update(grid.year_cols.values())
    ordered_years = sorted(years, reverse=True)
    target_year = ordered_years[0] if ordered_years else None
    prior_year = ordered_years[1] if len(ordered_years) > 1 else None
    log(f"   Years from statement headers: current {target_year}  prior {prior_year}")

    nulls: Dict[str, str] = {}
    for field in PRIMARY_FIELDS:
        st = _statement_of(field)
        if st not in located:
            nulls[field] = f"{Reason.NO_STATEMENT_PAGE} ({st})"
        elif st in no_year:
            tried = "; ".join(rejected.get(st, [])) or "no candidate gridded"
            nulls[field] = f"{Reason.NO_YEAR_COLUMN} — {tried}"

    current_cells: Dict[str, Cell] = {}
    prior_cells: Dict[str, Cell] = {}

    for ptype, grid in grids.items():
        field_map = classify_rows(grid)
        if verbose and field_map:
            log(f"      {ptype}: {len(field_map)} rows classified "
                f"({sum(1 for v in field_map.values() if v[1] == 'llm')} by LLM)")
        if target_year is not None:
            merge_cells(current_cells, extract_year(grid, field_map, target_year))
        if prior_year is not None:
            merge_cells(prior_cells, extract_year(grid, field_map, prior_year))

    read_current = {f: c.value for f, c in current_cells.items()}

    # ── Validate the READ values, before any derivation touches them ────────
    validation_read = validate(read_current)
    log(f"   Validation (read values only): {len(validation_read['passed'])} passed  "
        f"{len(validation_read['failed'])} failed")
    for x in validation_read["failed"]:
        log(f"   ❌ {x['check']}: expected {x['expected']}  got {x['got']}")

    # ── Derive-only identity pass ───────────────────────────────────────────
    conflicts, advisories = apply_identities(current_cells, verbose=verbose)
    conflicts_prior, advisories_prior = apply_identities(prior_cells, verbose=False)

    current_cells = apply_refuse_policy(current_cells, nulls)
    prior_nulls: Dict[str, str] = {}
    prior_cells = apply_refuse_policy(prior_cells, prior_nulls)

    current = {f: c.value for f, c in current_cells.items()}
    prior = {f: c.value for f, c in prior_cells.items()}

    for field in PRIMARY_FIELDS:
        if field not in current and field not in nulls:
            st = _statement_of(field)
            nulls[field] = (
                f"{Reason.ROW_NOT_CLASSIFIED} (statement located on page "
                f"{located[st]['page_n']}, no row matched)"
                if st in located else f"{Reason.NO_STATEMENT_PAGE} ({st})"
            )

    derived_fields = {f for f, c in current_cells.items() if c.prov.origin == "derived"}
    validation_final = validate(current, derived_fields)

    # Currency and units are reported from the income statement when we have
    # one — it is the statement most likely to state them explicitly.
    unit_grid = grids.get("pnl") or grids.get("bs") or grids.get("cf")
    currency = unit_grid.currency if unit_grid else "Unknown"
    unit_label = unit_grid.unit_label if unit_grid else "unknown"
    scale = unit_grid.scale if unit_grid else 1.0
    scale_source = unit_grid.scale_source if unit_grid else "unknown"

    log(f"✅ [v4] {len(current)} current, {len(prior)} prior, "
        f"{len(nulls)} primary field(s) NULL  [{unit_label}]")
    for f, why in sorted(nulls.items()):
        log(f"   ∅ {f}: {why}")

    return {
        "current": current,
        "prior": prior,
        "metadata": {
            "extractor": EXTRACTOR_VERSION,
            "period_current": str(target_year) if target_year else None,
            "period_prior": str(prior_year) if prior_year else None,
            "target_year": target_year,
            "prior_year": prior_year,
            "currency": currency,
            "unit_label": unit_label,
            "scale_factor": scale,
            "scale_source": scale_source,
            "pages_processed": len(pages),
            "statements": {
                p: {"page": r["page_n"], "score": r["score"], "title": r["title"]}
                for p, r in located.items()
            },
            "statements_rejected": rejected,
            "nulls": nulls,
            "prior_nulls": prior_nulls,
            "identity_conflicts": conflicts,
            "identity_conflicts_prior": conflicts_prior,
            "identity_advisories": advisories,
            "identity_advisories_prior": advisories_prior,
            "derived_fields": sorted(derived_fields),
            "validation": validation_final,
            "validation_read": validation_read,
            "provenance": {f: c.prov.to_dict() for f, c in current_cells.items()},
            "provenance_prior": {f: c.prov.to_dict() for f, c in prior_cells.items()},
        },
        "raw_text": doc_text,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/Test PDF.pdf"
    res = extract(path)
    meta = res["metadata"]

    print(f"\n{'─' * 78}")
    print(f"CURRENT {meta.get('target_year')}   [{meta.get('unit_label')}]")
    print(f"{'─' * 78}")
    prov = meta.get("provenance", {})
    for k, v in sorted(res["current"].items()):
        p = prov.get(k, {})
        origin = p.get("origin", "?")
        src = (f"p{p.get('page')} {p.get('row_text', '')[:34]}"
               if origin == "read" else f"derived: {p.get('rule', '')}")
        print(f"  {k:26} {v:>20,.2f}  conf {p.get('confidence', 0):.2f}  {src}")

    print(f"\n{'─' * 78}")
    print(f"PRIOR {meta.get('prior_year')}")
    print(f"{'─' * 78}")
    for k, v in sorted(res["prior"].items()):
        print(f"  {k:26} {v:>20,.2f}")

    val = meta.get("validation", {})
    print(f"\nVALIDATION  {len(val.get('passed', []))} passed  "
          f"{len(val.get('failed', []))} failed  "
          f"{len(val.get('tautological', []))} tautological  "
          f"{len(val.get('warnings', []))} warnings")
    for x in val.get("tautological", []):
        print(f"  ∼  {x}  (inputs include a derived value — not evidence)")
    for x in val.get("failed", []):
        print(f"  ❌ {x['check']}")
    for x in meta.get("identity_conflicts", []):
        print(f"  ⚠  conflict {x['rule']}: read {x['read']:,.0f} vs "
              f"computed {x['computed']:,.0f} ({x['diff_pct']}%)")
    for x in meta.get("identity_advisories", []):
        print(f"  ℹ  advisory {x['rule']}: read {x['read']:,.0f} vs "
              f"cascade {x['computed']:,.0f} ({x['diff_pct']}%)")
    for f, why in sorted(meta.get("nulls", {}).items()):
        print(f"  ∅  {f}: {why}")
