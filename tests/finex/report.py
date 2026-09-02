"""
tests/finex/report.py — coverage and accuracy report over the ground-truth corpus.

    python -m tests.finex.report            # all fixtures
    python -m tests.finex.report --refresh  # ignore the extraction cache
    python -m tests.finex.report tesco      # one fixture by name

Prints, per document, every ground-truth field as ok / wrong / missing /
refused, then a corpus total. This is the number that means "how good is the
extractor" — pytest tells you whether it regressed, this tells you where it
stands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tests.finex.harness import (
    DEFAULT_PDF_DIR,
    check,
    load_fixtures,
    pdf_path,
    run_extraction,
)

_MARK = {
    "ok": "✓",
    "derived_ok": "≈",
    "refused_ok": "∅",
    "wrong": "✗",
    "missing": "·",
    "leaked": "!",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="?", default="", help="substring of a fixture name")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    ap.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR))
    args = ap.parse_args(argv)

    pdf_dir = Path(args.pdf_dir)
    specs = [s for s in load_fixtures() if args.filter in s["_name"]]
    if not specs:
        print("no fixtures matched", file=sys.stderr)
        return 2

    totals = {k: 0 for k in _MARK}
    docs_ok = 0

    for spec in specs:
        path = pdf_path(spec, pdf_dir)
        if not path.exists():
            print(f"── {spec['company']}: PDF not found at {path}")
            continue

        res = check(spec, run_extraction(path, refresh=args.refresh))
        head = "PASS" if res.ok else "FAIL"
        print(f"\n── {res.company}  [{head}]  {res.pdf.name}  ({res.seconds}s)")
        if not res.statements_ok:
            print(f"   statement location: {res.statements_detail}")
        if not res.currency_ok:
            print(f"   currency: {res.currency_detail}")

        for f in res.fields:
            totals[f.status] += 1
            if f.period == "prior" and f.status in ("ok", "derived_ok"):
                continue  # keep the table readable; prior-year hits are silent
            mark = _MARK[f.status]
            exp = "—" if f.expected is None else f"{f.expected:,.0f}"
            act = "—" if f.actual is None else f"{f.actual:,.0f}"
            line = f"   {mark} {f.field:24} {f.period:7} expected {exp:>18}  got {act:>18}"
            if f.status not in ("ok", "derived_ok"):
                line += f"   {f.detail}"
            print(line)
        docs_ok += int(res.ok)

    checked = sum(totals.values())
    good = totals["ok"] + totals["derived_ok"] + totals["refused_ok"]
    print(f"\n{'─' * 78}")
    print(f"documents  {docs_ok}/{len(specs)} clean")
    print(f"fields     {good}/{checked} correct  "
          f"({totals['ok']} read, {totals['derived_ok']} derived, "
          f"{totals['refused_ok']} correctly refused)")
    print(f"           {totals['wrong']} wrong, {totals['missing']} missing, "
          f"{totals['leaked']} refusal failures")
    return 0 if totals["wrong"] == totals["missing"] == totals["leaked"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
