"""
Regression tests for the FinEx PDF extractor.

These are the tests that give "perfect extraction" a definition. Each fixture in
``tests/finex/fixtures/`` names one annual report and every figure that should
come out of it, hand-read from the printed statements with the page and line
item recorded alongside. A change to the extractor either keeps all of them
green or it does not ship.

Running
-------
    pytest tests/test_finex_extraction.py

The PDFs are not in the repository. Point the suite at them with

    FINEX_TEST_PDF_DIR="/path/to/FinEx Data" pytest tests/test_finex_extraction.py

and the whole module skips cleanly when they are not there, so a checkout
without the corpus still runs the rest of the suite.

The first run takes several minutes — six annual reports, one of them 370
pages. Results are cached under ``tests/finex/.cache/`` keyed by the PDF's
mtime and the extractor version, so later runs are instant until either
changes.

The LLM is disabled for the whole module. v4 takes every number from pdfplumber
geometry and uses the model only to classify row labels, so with it off the run
is deterministic — which is the only condition under which these assertions
mean anything.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("FINEX_DISABLE_LLM", "1")

from tests.finex.harness import (  # noqa: E402
    DEFAULT_PDF_DIR,
    check,
    load_fixtures,
    pdf_path,
    run_extraction,
)

_FIXTURES = load_fixtures()
_IDS = [f["_name"] for f in _FIXTURES]

pytestmark = pytest.mark.skipif(
    not DEFAULT_PDF_DIR.exists(),
    reason=(
        f"FinEx PDF corpus not found at {DEFAULT_PDF_DIR}; "
        "set FINEX_TEST_PDF_DIR to run the extraction regression suite"
    ),
)


@pytest.fixture(scope="session")
def extracted():
    """One extraction per PDF for the whole session, memoised on disk."""
    cache = {}

    def _get(spec):
        name = spec["_name"]
        if name not in cache:
            path = pdf_path(spec)
            if not path.exists():
                pytest.skip(f"missing PDF: {path}")
            cache[name] = run_extraction(path)
        return cache[name]

    return _get


@pytest.fixture(scope="session")
def results(extracted):
    return {spec["_name"]: check(spec, extracted(spec)) for spec in _FIXTURES}


@pytest.mark.parametrize("spec", _FIXTURES, ids=_IDS)
def test_values_match_the_printed_statements(spec, results):
    """Every figure in the fixture comes out of the extractor unchanged."""
    res = results[spec["_name"]]
    bad = [f for f in res.fields if f.status in ("wrong", "missing")]
    if bad:
        lines = [
            f"  {f.field} ({f.period}): expected {f.expected:,.2f}, "
            f"got {'—' if f.actual is None else format(f.actual, ',.2f')}"
            f"  [{f.status}] {f.detail}"
            for f in bad
        ]
        pytest.fail(f"{res.company}: {len(bad)} field(s) off\n" + "\n".join(lines))


@pytest.mark.parametrize("spec", _FIXTURES, ids=_IDS)
def test_unreported_lines_are_refused_with_a_reason(spec, results):
    """A line the accounts do not print must be NULL and must say why.

    This is the half of correctness v3 had no way to express. Shell reports no
    operating profit and HSBC reports no revenue; inventing either is worse
    than leaving the cell empty, and leaving it empty without a reason code is
    indistinguishable from a bug.
    """
    res = results[spec["_name"]]
    leaked = [f for f in res.fields if f.status == "leaked"]
    if leaked:
        lines = [
            f"  {f.field}: {'value ' + format(f.actual, ',.2f') if f.actual is not None else 'no reason code'}"
            f" — {f.detail}"
            for f in leaked
        ]
        pytest.fail(f"{res.company}: refusal not honoured\n" + "\n".join(lines))


@pytest.mark.parametrize("spec", _FIXTURES, ids=_IDS)
def test_currency_comes_from_the_statement(spec, results):
    """Vodafone reports in euro. v3 read GBP off a stray £ elsewhere in the PDF."""
    res = results[spec["_name"]]
    assert res.currency_ok, f"{res.company}: {res.currency_detail}"


@pytest.mark.parametrize("spec", _FIXTURES, ids=_IDS)
def test_locates_the_primary_statements(spec, results):
    """The right page, not a contents list, a narrative page or a note."""
    res = results[spec["_name"]]
    assert res.statements_ok, f"{res.company}: {res.statements_detail}"


@pytest.mark.parametrize("spec", _FIXTURES, ids=_IDS)
def test_years_come_from_the_statement_header(spec, extracted):
    """No document-wide period guessing: a cover date is not a fiscal year."""
    meta = extracted(spec)["metadata"]
    assert meta.get("target_year") == spec["fiscal_year"], (
        f"{spec['company']}: current year {meta.get('target_year')} "
        f"!= {spec['fiscal_year']}"
    )
    assert meta.get("prior_year") == spec["prior_year"], (
        f"{spec['company']}: prior year {meta.get('prior_year')} "
        f"!= {spec['prior_year']}"
    )


@pytest.mark.parametrize("spec", _FIXTURES, ids=_IDS)
def test_no_read_value_was_overwritten(spec, extracted):
    """Identity rules may fill a gap. They may never replace a printed figure.

    v3 auto-corrected a read gross profit to match revenue less cost of sales,
    then validated the corrected data and reported a pass. If any provenance
    entry ever comes back with origin ``corrected``, that behaviour is back.
    """
    meta = extracted(spec)["metadata"]
    for field, prov in (meta.get("provenance") or {}).items():
        assert prov.get("origin") in ("read", "derived"), (
            f"{field}: unexpected origin {prov.get('origin')!r}"
        )


@pytest.mark.parametrize("spec", _FIXTURES, ids=_IDS)
def test_every_value_carries_provenance(spec, extracted):
    """No number reaches the caller without a page or a rule behind it."""
    res = extracted(spec)
    prov = res["metadata"].get("provenance") or {}
    for field in res["current"]:
        assert field in prov, f"{field} has no provenance"
        p = prov[field]
        if p.get("origin") == "read":
            assert p.get("page") and p.get("row_text"), f"{field}: incomplete provenance"
        else:
            assert p.get("rule"), f"{field}: derived without a rule"


@pytest.mark.parametrize("spec", _FIXTURES, ids=_IDS)
def test_validation_does_not_count_tautologies(spec, extracted):
    """A check whose inputs were derived is not evidence of anything."""
    meta = extracted(spec)["metadata"]
    val = meta.get("validation") or {}
    derived = set(meta.get("derived_fields") or [])
    assert "tautological" in val, "validation must separate tautological checks"
    if derived:
        # Any check listed as passed must be free of derived inputs.
        assert not (set(val.get("passed") or []) & set(val.get("tautological") or []))


def test_the_corpus_is_not_empty():
    assert _FIXTURES, "no ground-truth fixtures found"
