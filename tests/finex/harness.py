"""
tests/finex/harness.py — regression harness for the FinEx PDF extractor.

Purpose
=======
"Perfect extraction" has no definition without a fixed set of documents and a
hand-checked expected value for every figure. This module supplies both: it
loads the YAML ground-truth fixtures in ``fixtures/``, runs the extractor over
the matching PDF, and reports every field as correct, wrong, missing or
correctly refused.

Determinism
-----------
The harness runs with ``FINEX_DISABLE_LLM=1``. Every number the v4 extractor
produces comes from pdfplumber geometry, so with the label classifier switched
off the run is fully reproducible: the same PDF gives the same answer on any
machine, with no model loaded. That is the whole reason the LLM is confined to
label classification. A harness whose expected values depend on a model version
measures the model, not the extractor.

Caching
-------
Extraction over six annual reports is minutes of work, so results are cached
under ``.cache/`` keyed by the PDF's size and mtime plus the extractor version.
Delete the directory, or pass ``refresh=True``, to force a re-run.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

FIXTURE_DIR = Path(__file__).parent / "fixtures"
CACHE_DIR = Path(__file__).parent / ".cache"

# Where the PDFs live. Overridable so the suite can run against a copy.
DEFAULT_PDF_DIR = Path(
    os.getenv("FINEX_TEST_PDF_DIR", str(Path.home() / "Desktop" / "Jarvis" / "FinEx Data"))
)

# Relative tolerance on a matched value. Values are read, not computed, so this
# is tight on purpose: it exists for float round-tripping, not for slack.
REL_TOL = 0.005


@dataclass
class FieldResult:
    field: str
    period: str                  # "current" | "prior"
    expected: Optional[float]
    actual: Optional[float]
    status: str                  # ok | wrong | missing | refused_ok | leaked | derived_ok
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status in ("wrong", "missing", "leaked")


@dataclass
class DocResult:
    name: str
    company: str
    pdf: Path
    fields: List[FieldResult]
    currency_ok: bool
    currency_detail: str
    statements_ok: bool
    statements_detail: str
    seconds: float
    nulls: Dict[str, str]

    @property
    def failures(self) -> List[FieldResult]:
        return [f for f in self.fields if f.failed]

    @property
    def ok(self) -> bool:
        return not self.failures and self.currency_ok and self.statements_ok


def load_fixtures() -> List[dict]:
    out = []
    for path in sorted(FIXTURE_DIR.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text())
        spec["_name"] = path.stem
        out.append(spec)
    return out


def pdf_path(spec: dict, pdf_dir: Optional[Path] = None) -> Path:
    return (pdf_dir or DEFAULT_PDF_DIR) / spec["pdf"]


def _cache_key(path: Path) -> Path:
    from finex.extract_v4 import EXTRACTOR_VERSION

    st = path.stat()
    stamp = f"{EXTRACTOR_VERSION}-{st.st_size}-{int(st.st_mtime)}"
    return CACHE_DIR / f"{path.stem}.{stamp}.json"


def run_extraction(path: Path, refresh: bool = False) -> dict:
    """Extract one PDF with the LLM disabled, memoised on disk."""
    cache = _cache_key(path)
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())

    os.environ["FINEX_DISABLE_LLM"] = "1"
    from finex.extract_v4 import extract

    started = time.time()
    result = extract(str(path), verbose=False)
    result.pop("raw_text", None)          # megabytes, never asserted on
    result["_seconds"] = round(time.time() - started, 1)

    CACHE_DIR.mkdir(exist_ok=True)
    for stale in CACHE_DIR.glob(f"{path.stem}.*.json"):
        stale.unlink()
    cache.write_text(json.dumps(result, indent=1, default=str))
    return result


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(abs(b) * REL_TOL, 0.01)


def check(spec: dict, result: dict) -> DocResult:
    """Compare one extraction against its ground truth."""
    meta = result.get("metadata", {})
    scale = float(spec.get("scale", 1))
    fields: List[FieldResult] = []

    def expected_value(entry: dict, period: str) -> Optional[float]:
        if period not in entry:
            return None
        raw = entry[period]
        return float(raw) if entry.get("no_scale") else float(raw) * scale

    for period, key in (("current", "current"), ("prior", "prior")):
        got = result.get(key, {})

        for field, entry in (spec.get("values") or {}).items():
            exp = expected_value(entry, period)
            if exp is None:
                continue
            act = got.get(field)
            if act is None:
                fields.append(FieldResult(
                    field, period, exp, None, "missing",
                    f"expected from p{entry.get('page')} “{entry.get('label')}”",
                ))
            elif _close(float(act), exp):
                fields.append(FieldResult(field, period, exp, float(act), "ok"))
            else:
                fields.append(FieldResult(
                    field, period, exp, float(act), "wrong",
                    f"p{entry.get('page')} “{entry.get('label')}”",
                ))

        for field, entry in (spec.get("derived_ok") or {}).items():
            exp = expected_value(entry, period)
            if exp is None:
                continue
            act = got.get(field)
            if act is None:
                fields.append(FieldResult(
                    field, period, exp, None, "missing", entry.get("note", ""),
                ))
            elif _close(float(act), exp):
                fields.append(FieldResult(
                    field, period, exp, float(act), "derived_ok", entry.get("note", ""),
                ))
            else:
                fields.append(FieldResult(
                    field, period, exp, float(act), "wrong", entry.get("note", ""),
                ))

    # A field the accounts do not report must be absent AND explained. Silence
    # is not a refusal: the reason code is the deliverable.
    nulls = meta.get("nulls", {})
    for field, why in (spec.get("null_expected") or {}).items():
        act = result.get("current", {}).get(field)
        if act is not None:
            fields.append(FieldResult(
                field, "current", None, float(act), "leaked",
                f"should be NULL — {why}",
            ))
        elif field not in nulls:
            fields.append(FieldResult(
                field, "current", None, None, "leaked",
                f"absent but carries no reason code — {why}",
            ))
        else:
            fields.append(FieldResult(
                field, "current", None, None, "refused_ok", nulls[field],
            ))

    exp_cur = spec.get("currency")
    got_cur = meta.get("currency")
    currency_ok = (exp_cur is None) or (got_cur == exp_cur)

    exp_stmts = spec.get("statements") or {}
    got_stmts = {k: v.get("page") for k, v in (meta.get("statements") or {}).items()}
    wrong_pages = {
        k: (v, got_stmts.get(k)) for k, v in exp_stmts.items() if got_stmts.get(k) != v
    }

    return DocResult(
        name=spec["_name"],
        company=spec.get("company", spec["_name"]),
        pdf=Path(spec["pdf"]),
        fields=fields,
        currency_ok=currency_ok,
        currency_detail=f"expected {exp_cur}, got {got_cur}",
        statements_ok=not wrong_pages,
        statements_detail="; ".join(
            f"{k}: expected p{e}, got p{g}" for k, (e, g) in wrong_pages.items()
        ),
        seconds=result.get("_seconds", 0.0),
        nulls=nulls,
    )
