"""The hud-data cache: one Google Sheets read serves many callers.

One hud-data run reads three sheets against a 60-reads-per-minute Google
quota. Two UI pollers at 6s intervals exceeded it and turned a working board
into "Quota exceeded", so the contract these tests pin down is: reading more
often must not read *Google* more often.
"""

from __future__ import annotations

import asyncio

import pytest

import server


@pytest.fixture
def cache(monkeypatch):
    c = server._JamsHudCache()
    monkeypatch.setattr(server, "_jams_hud_cache", c)
    return c


def _stub_reads(monkeypatch, results):
    """Feed _jams_read a scripted sequence; count how often it's called."""
    calls = {"n": 0}
    seq = list(results)

    async def fake(path, total=8.0):
        calls["n"] += 1
        return seq[min(calls["n"] - 1, len(seq) - 1)]

    monkeypatch.setattr(server, "_jams_read", fake)
    return calls


def test_a_second_read_inside_the_ttl_costs_no_upstream_call(cache, monkeypatch):
    calls = _stub_reads(monkeypatch, [({"jobs": [1]}, None)])

    async def go():
        await cache.get()
        return await cache.get()

    data, error, meta = asyncio.run(go())
    assert calls["n"] == 1
    assert data == {"jobs": [1]} and error is None
    assert meta["cached"] is True


def test_concurrent_callers_share_one_upstream_call(cache, monkeypatch):
    """The actual failure mode: N pollers arriving together must not become N
    Sheets reads. Without single-flight this is where the quota went."""
    calls = {"n": 0}

    async def slow(path, total=8.0):
        calls["n"] += 1
        await asyncio.sleep(0.05)          # long enough for the others to pile up
        return {"jobs": []}, None

    monkeypatch.setattr(server, "_jams_read", slow)

    async def go():
        return await asyncio.gather(*[cache.get() for _ in range(10)])

    results = asyncio.run(go())
    assert calls["n"] == 1
    assert all(r[0] is not None for r in results)


def test_an_expired_ttl_does_fetch_again(cache, monkeypatch):
    calls = _stub_reads(monkeypatch, [({"jobs": [1]}, None), ({"jobs": [2]}, None)])

    async def go():
        await cache.get(ttl=0)
        return await cache.get(ttl=0)

    data, _e, _m = asyncio.run(go())
    assert calls["n"] == 2 and data == {"jobs": [2]}


def test_a_quota_error_serves_the_last_good_board(cache, monkeypatch):
    """A rate-limit blip must not blank a board that was fine a moment ago —
    that would trade one misleading empty board for another."""
    _stub_reads(monkeypatch, [
        ({"jobs": ["a"]}, None),
        (None, "n8n workflow failed: Quota exceeded for quota metric "
               "'Read requests' of service 'sheets.googleapis.com'"),
    ])

    async def go():
        await cache.get(ttl=0)
        return await cache.get(ttl=0)

    data, error, meta = asyncio.run(go())
    assert data == {"jobs": ["a"]}
    assert error is None
    assert meta["stale"] is True
    assert "Quota exceeded" in meta["stale_reason"]


def test_a_credential_error_is_not_papered_over_with_stale_data(cache, monkeypatch):
    """Quota errors pass on their own; a revoked credential does not. Serving
    stale rows through one would hide the thing that needs a human."""
    _stub_reads(monkeypatch, [
        ({"jobs": ["a"]}, None),
        (None, 'The credential "Google Sheets account" needs to be reconnected.'),
    ])

    async def go():
        await cache.get(ttl=0)
        return await cache.get(ttl=0)

    data, error, meta = asyncio.run(go())
    assert data is None
    assert "needs to be reconnected" in error
    assert meta["stale"] is False


def test_stale_data_expires_rather_than_being_shown_forever(cache, monkeypatch):
    _stub_reads(monkeypatch, [
        ({"jobs": ["a"]}, None),
        (None, "Quota exceeded"),
    ])
    monkeypatch.setattr(server, "_JAMS_STALE_CEILING", 0.0)

    async def go():
        await cache.get(ttl=0)
        return await cache.get(ttl=0)

    data, error, _m = asyncio.run(go())
    assert data is None and "Quota exceeded" in error


def test_a_failure_with_nothing_cached_reports_the_error(cache, monkeypatch):
    _stub_reads(monkeypatch, [(None, "Quota exceeded")])
    data, error, meta = asyncio.run(cache.get())
    assert data is None and error == "Quota exceeded"
    assert meta["stale"] is False


@pytest.mark.parametrize("msg", [
    "Quota exceeded for quota metric 'Read requests'",
    "The service is receiving too many requests from you",
    "RATE LIMIT hit",
    "n8n did not respond within 8s.",
])
def test_transient_failures_are_recognised(msg):
    assert server._is_transient(msg) is True


@pytest.mark.parametrize("msg", [
    'The credential "Google Sheets account" needs to be reconnected.',
    "Can't reach n8n at http://localhost:5678 (ClientConnectorError).",
    "n8n returned HTTP 404 for 'hud-data'.",
    "",
])
def test_permanent_failures_are_not_treated_as_transient(msg):
    assert server._is_transient(msg) is False


def test_invalidate_forces_the_next_read_to_go_upstream(cache, monkeypatch):
    """Discovery rewrites the Jobs sheet. A poller served the pre-run copy
    would conclude the run changed nothing, which is why /jams/trigger drops
    the cache before the UI starts polling."""
    calls = _stub_reads(monkeypatch, [({"jobs": []}, None), ({"jobs": ["new"]}, None)])

    async def go():
        await cache.get()
        cache.invalidate()
        return await cache.get()

    data, _e, meta = asyncio.run(go())
    assert calls["n"] == 2
    assert data == {"jobs": ["new"]}
    assert meta["cached"] is False


def test_invalidate_also_drops_the_stale_fallback(cache, monkeypatch):
    """Otherwise an invalidated cache could still resurrect old rows through
    the transient-error path."""
    _stub_reads(monkeypatch, [({"jobs": ["old"]}, None), (None, "Quota exceeded")])

    async def go():
        await cache.get()
        cache.invalidate()
        return await cache.get()

    data, error, _m = asyncio.run(go())
    assert data is None and "Quota exceeded" in error
