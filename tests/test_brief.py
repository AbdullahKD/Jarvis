"""The morning brief's data contract.

Every test here corresponds to something the previous brief got wrong on a
real screenshot:

  * an automated "We've sent your application to …" from a reed.co.uk
    no-reply, rendered with a green OFFER badge
  * CV-writing marketing sitting in "Interviews & follow-ups"
  * a tile reading "0 INTERVIEWS" directly above a list of interview invites
  * a summary claiming "five applications in the pipeline" when the tile
    beside it said 51
  * every index showing +0.00% because a missing change field was coerced to 0

The brief is the first thing read each morning, so a wrong number in it is
worse than a missing one. These lock that in.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import server


def _jams(jobs=None, pipe=None, inbox=None):
    """Stand in for the n8n hud-data webhook."""
    payload = json.dumps({"jobs": jobs or [], "pipe": pipe or [], "inbox": inbox or []})

    async def fake_get(path, total=6.0):
        return 200, payload

    return fake_get


@pytest.fixture
def client(monkeypatch):
    # Calendar/Gmail/weather/news/markets all come back empty via the harness's
    # fakes; JAMS is the one we drive per-test.
    monkeypatch.setattr(server, "_jams_get", _jams())
    server._jams_hud_cache.invalidate()
    return TestClient(server.app)


def brief(client, monkeypatch, **jams):
    monkeypatch.setattr(server, "_jams_get", _jams(**jams))
    # The hud-data cache is a process-wide singleton with a 10s TTL, and the
    # whole suite runs well inside that — without this, every test after the
    # first is served the previous test's inbox.
    server._jams_hud_cache.invalidate()
    r = client.get("/brief/data")
    assert r.status_code == 200
    return r.json()


# ── the OFFER bug ───────────────────────────────────────────────────────────

REED_ACK = {
    "subject": "We've sent your application to Opus Recruitment Solutions Ltd",
    "from": '"reed.co.uk" <no-reply@jobs.reed.co.uk>',
    "category": "offer",          # what the local model wrongly decided
    "is_job": "yes", "link": "https://mail.google.com/x",
}


def test_an_automated_application_confirmation_is_not_an_offer(client, monkeypatch):
    d = brief(client, monkeypatch, inbox=[REED_ACK])
    tags = [i.get("tag") for i in d["needs_you"]]
    assert "offer" not in tags
    assert d["jobs"]["counts"]["offer"] == 0


def test_a_real_offer_still_counts(client, monkeypatch):
    real = {"subject": "We are delighted to offer you the position",
            "from": "hannah.reed@acme.com", "category": "offer", "is_job": "yes"}
    d = brief(client, monkeypatch, inbox=[real])
    assert d["jobs"]["counts"]["offer"] == 1
    assert [i["tag"] for i in d["needs_you"]] == ["offer"]


def test_an_offer_from_a_no_reply_address_is_demoted(client, monkeypatch):
    """Real offers come from a person. This one claims 'offer' but is
    automated, which in practice has always been a misclassification."""
    row = {"subject": "Update on your application", "from": "noreply@jobs.example.com",
           "category": "offer", "is_job": "yes"}
    d = brief(client, monkeypatch, inbox=[row])
    assert d["jobs"]["counts"]["offer"] == 0


# ── noise ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("row", [
    {"subject": "Pro CVs are 3x more likely to get interviewed",
     "from": "sara larson from topcv <sara@topcv.co.uk>", "category": "recruiter", "is_job": "yes"},
    {"subject": "I'd love for you to meet Jip — he took the LSE Career Accelerator",
     "from": "dani van rijswijck <lseonline@fourthrev.com>", "category": "recruiter", "is_job": "yes"},
])
def test_marketing_never_reaches_needs_you(client, monkeypatch, row):
    """JAMS's filter is deliberately generous — a missed interview invite costs
    more than a stray newsletter. The brief needs the stricter bar, because
    anything it shows is claiming to need a decision."""
    d = brief(client, monkeypatch, inbox=[row])
    assert d["needs_you"] == []


def test_rows_marked_not_a_job_are_ignored(client, monkeypatch):
    row = {"subject": "Dinner on Friday?", "from": "a.friend@gmail.com",
           "category": "personal", "is_job": "no"}
    assert brief(client, monkeypatch, inbox=[row])["needs_you"] == []


# ── the tiles must agree with the lists ─────────────────────────────────────


def test_interview_tile_counts_the_interviews_that_are_shown(client, monkeypatch):
    """The screenshot showed "0 INTERVIEWS" above a list of interview invites,
    because the tile read reply_status while the list read the inbox."""
    inbox = [{"subject": "Invitation to take test", "from": "elaine@4cassociates.com",
              "category": "interview", "is_job": "yes"}]
    d = brief(client, monkeypatch, inbox=inbox)
    shown = [i for i in d["needs_you"] if i["tag"] == "interview"]
    assert d["jobs"]["counts"]["interview"] == len(shown) == 1


def test_summary_never_states_a_number_the_tiles_disagree_with(client, monkeypatch):
    pipe = [{"dedupe_key": f"k{i}", "status": "submitted", "applied_date": "2026-07-01"}
            for i in range(51)]
    d = brief(client, monkeypatch, pipe=pipe)
    summary = d["summary"]
    # The old LLM summary said "five applications in the pipeline" against a
    # tile reading 51. Whatever the wording, no figure may contradict the data.
    import re
    for n in re.findall(r"\b(\d+)\b", summary):
        n = int(n)
        assert n in {d["jobs"]["total_pipeline"], d["jobs"]["counts"]["applied"],
                     d["jobs"]["counts"]["interview"], d["jobs"]["counts"]["offer"],
                     len(d["needs_you"]), len(d["today"]["events"]),
                     len(d["jobs"]["new_high_fit"]), len(d["jobs"]["follow_ups"])}, \
            f"summary cites {n}, which matches no counter: {summary!r}"


def test_summary_is_composed_not_generated(client, monkeypatch):
    """No model call means no latency and no drift. If the summary ever starts
    coming from the LLM again, this catches it: the fake LLM would be asked."""
    calls = []
    original = server.jarvis.llm.chat

    async def spy(*a, **k):
        calls.append(a)
        return await original(*a, **k)

    monkeypatch.setattr(server.jarvis.llm, "chat", spy)
    brief(client, monkeypatch)
    assert not calls, "brief_data called the LLM"


def test_quiet_day_says_so_plainly(client, monkeypatch):
    d = brief(client, monkeypatch)
    assert d["needs_you"] == []
    assert "nothing needs a decision" in d["summary"].lower()


# ── digest ──────────────────────────────────────────────────────────────────


def test_missing_market_change_stays_none(client, monkeypatch):
    """`change_pct or 0` turned "no data" into a green +0.00%, which reads as a
    flat market rather than a missing field."""
    async def markets():
        return {"prices": [{"name": "Dow Jones", "symbol": "^DJI", "price": 51711.65},
                           {"name": "S&P 500", "symbol": "^GSPC", "price": 7408.3,
                            "change_pct": 0.42}]}

    monkeypatch.setattr(server.jarvis.markets, "get_all", markets)
    d = brief(client, monkeypatch)
    by_name = {m["name"]: m["change_pct"] for m in d["digest"]["markets"]}
    assert by_name["Dow Jones"] is None
    assert by_name["S&P 500"] == 0.42


def test_digest_does_not_stack_one_outlet(client, monkeypatch):
    """Four TechCrunch headlines is not a digest."""
    async def news(**kwargs):
        return {"stories": [{"title": f"Story {i}", "sources": ["TechCrunch"], "url": "#"}
                            for i in range(6)]
                + [{"title": "Elsewhere", "sources": ["The Verge"], "url": "#"}]}

    monkeypatch.setattr(server.jarvis.news, "get_headlines", news)
    d = brief(client, monkeypatch)
    sources = [n["source"] for n in d["digest"]["news"]]
    assert sources.count("TechCrunch") <= 2
    assert "The Verge" in sources


# ── ranking ─────────────────────────────────────────────────────────────────


def test_needs_you_is_ranked_by_consequence(client, monkeypatch):
    inbox = [
        {"subject": "Quick chat about a role?", "from": "rec@agency.com",
         "category": "recruiter", "is_job": "yes"},
        {"subject": "We are pleased to offer you the position", "from": "head@acme.com",
         "category": "offer", "is_job": "yes"},
        {"subject": "Invitation to interview", "from": "talent@acme.com",
         "category": "interview", "is_job": "yes"},
    ]
    d = brief(client, monkeypatch, inbox=inbox)
    assert [i["tag"] for i in d["needs_you"]] == ["offer", "interview", "recruiter"]


def test_gmail_thread_links_are_passed_through(client, monkeypatch):
    """The rows carry a Gmail deep link that the old UI silently dropped."""
    row = {"subject": "Invitation to interview", "from": "talent@acme.com",
           "category": "interview", "is_job": "yes",
           "link": "https://mail.google.com/mail/u/0/#inbox/abc123"}
    d = brief(client, monkeypatch, inbox=[row])
    assert d["needs_you"][0]["href"].endswith("abc123")


# ── prayer ──────────────────────────────────────────────────────────────────


def test_prayer_schedule_is_labelled_and_marks_what_is_next(client, monkeypatch):
    async def times():
        return {"fajr": "03:14", "dhuhr": "13:10", "asr": "17:23",
                "maghrib": "21:00", "isha": "23:04"}

    monkeypatch.setattr(server.jarvis.prayer, "get_times", times)
    sched = brief(client, monkeypatch)["prayer"]["schedule"]
    assert [p["name"] for p in sched] == ["Fajr", "Dhuhr", "Asr", "Maghrib", "Isha"]
    assert sum(1 for p in sched if p["next"]) <= 1
    for p in sched:
        assert p["time"], "every entry carries its own time, not a bare run of numbers"
