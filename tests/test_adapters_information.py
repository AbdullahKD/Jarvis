"""Tests for the information-tool adapters.

The doubles below mimic the *real* signatures and return shapes of
``tools/weather.py``, ``tools/prayer_times.py``, ``tools/markets.py``,
``tools/news.py``, ``tools/sports.py`` and ``tools/web_search.py`` — including
their habit of flattening every failure into ``{"success": False, "error": str}``.
What's under test is the adapter: does it classify that failure correctly, does
it keep data and prose separate, and does it fail cleanly when the underlying
tool misbehaves.

No network, no API keys.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from core.adapters.information import (
    MarketsAdapter,
    NewsAdapter,
    PrayerTimesAdapter,
    SportsAdapter,
    WeatherAdapter,
    WebSearchAdapter,
)
from core.tool import ErrorType, HealthStatus


# ── Doubles ─────────────────────────────────────────────────────────────────


class FakeWeather:
    def __init__(self, payload: Optional[Dict[str, Any]] = None):
        self.payload = payload if payload is not None else {
            "success": True, "location": "High Wycombe",
            "temperature": 14.2, "conditions": "Overcast",
        }
        self.seen: list = []

    async def get_current(self, **kw):
        self.seen.append(("get_current", kw))
        return self.payload

    async def get_current_for_location(self, location):
        self.seen.append(("get_current_for_location", location))
        return self.payload

    async def get_forecast(self, days=7, **kw):
        self.seen.append(("get_forecast", days))
        return {"success": True, "location": "High Wycombe",
                "forecast": [{"day": i} for i in range(days)]}

    async def get_forecast_for_location(self, location, days=7):
        self.seen.append(("get_forecast_for_location", location, days))
        return {"success": True, "location": location,
                "forecast": [{"day": i} for i in range(days)]}

    def format_current(self, d):
        return f"{d.get('conditions')}, {d.get('temperature')}°C in {d.get('location')}."

    def format_forecast(self, d):
        return f"{len(d.get('forecast', []))}-day forecast for {d.get('location')}."


class FakePrayer:
    def __init__(self, payload=None):
        self.payload = payload if payload is not None else {
            "success": True, "timings": {"Fajr": "04:12", "Maghrib": "20:41"},
        }

    async def get_times(self, **kw):
        return self.payload

    def format_times(self, d):
        return ", ".join(f"{k} {v}" for k, v in d.get("timings", {}).items())

    def get_next_prayer(self, d):
        return {"name": "Maghrib", "time": "20:41"}


class FakeMarkets:
    def __init__(self, price=None, all_payload=None):
        self.price = price if price is not None else {
            "success": True, "symbol": "AAPL", "price": 213.4, "change_pct": 1.2,
        }
        self.all_payload = all_payload if all_payload is not None else {
            "success": True, "prices": [{"symbol": "AAPL", "price": 213.4}],
        }

    async def get_price(self, symbol):
        return self.price

    async def get_all(self, symbols=None):
        return self.all_payload

    def format_prices(self, d):
        return " | ".join(f"{p.get('symbol')} {p.get('price')}"
                          for p in d.get("prices", []))


class FakeNews:
    def __init__(self, payload=None):
        self.payload = payload if payload is not None else {
            "success": True, "articles": [{"title": "Something happened"}],
        }
        self.seen: list = []

    async def get_headlines(self, **kw):
        self.seen.append(kw)
        return self.payload

    def format_headlines(self, d, detailed=False):
        return "; ".join(a["title"] for a in d.get("articles", []))

    def list_sources(self):
        return ["bbc", "guardian", "techcrunch"]


class FakeSports:
    def __init__(self, payload=None, detect=None):
        self.payload = payload if payload is not None else {
            "success": True, "games": [{"home": "Man Utd", "away": "Arsenal"}],
        }
        self._detect = detect
        self.seen: list = []

    def detect_league(self, q):
        return self._detect

    async def get_scores(self, league_key, limit=10):
        self.seen.append(("scores", league_key, limit))
        return self.payload

    async def get_standings(self, league_key):
        self.seen.append(("standings", league_key))
        return self.payload

    async def search_team(self, query, league_key=None):
        self.seen.append(("team", query, league_key))
        return self.payload

    def format_scores(self, d):
        return f"{len(d.get('games', []))} games"

    def format_standings(self, d):
        return "table"

    def list_leagues(self):
        return ["premier_league", "nba", "cricket"]


class FakeSearch:
    def __init__(self, payload=None, scrape_payload=None):
        self.payload = payload if payload is not None else {
            "success": True, "results": [{"title": "R1", "url": "https://x"}],
        }
        self.scrape_payload = scrape_payload

    async def search(self, raw_query, max_results=5):
        return self.payload

    async def scrape(self, url, max_chars=5000):
        if self.scrape_payload is not None:
            return self.scrape_payload
        return {"success": True, "text": "page body " * 100}

    def format_results(self, d):
        return f"{len(d.get('results', []))} results"


# ── Weather ─────────────────────────────────────────────────────────────────


async def test_weather_current_splits_data_and_prose():
    t = FakeWeather()
    r = await WeatherAdapter(t).execute("get_current")
    assert r.success
    assert r.data["temperature"] == 14.2
    assert r.message == "Overcast, 14.2°C in High Wycombe."


async def test_weather_uses_location_variant_when_given():
    t = FakeWeather()
    await WeatherAdapter(t).execute("get_current", {"location": "Lahore"})
    assert t.seen[-1] == ("get_current_for_location", "Lahore")


async def test_weather_unknown_location_is_not_found_not_upstream():
    t = FakeWeather({"success": False, "error": "Could not find location: Atlantis"})
    r = await WeatherAdapter(t).execute("get_current", {"location": "Atlantis"})
    assert r.success is False
    assert r.error_type is ErrorType.NOT_FOUND


async def test_weather_api_failure_is_upstream_and_retryable():
    t = FakeWeather({"success": False, "error": "HTTP 502 from open-meteo"})
    r = await WeatherAdapter(t).execute("get_current")
    assert r.error_type is ErrorType.UPSTREAM
    assert r.retryable is True


async def test_weather_timeout_text_is_classified_as_timeout():
    t = FakeWeather({"success": False, "error": "request timed out"})
    r = await WeatherAdapter(t).execute("get_current")
    assert r.error_type is ErrorType.TIMEOUT
    assert r.retryable is True


async def test_weather_forecast_days_bounds_enforced_by_schema():
    a = WeatherAdapter(FakeWeather())
    assert (await a.execute("get_forecast", {"days": 3})).success
    bad = await a.execute("get_forecast", {"days": 99})
    assert bad.success is False
    assert bad.error_type is ErrorType.INPUT


async def test_weather_health_probe():
    assert (await WeatherAdapter(FakeWeather()).health_check()).status is HealthStatus.OK
    bad = await WeatherAdapter(FakeWeather({"success": False, "error": "down"})).health_check()
    assert bad.status is HealthStatus.ERROR
    assert "down" in bad.detail


# ── Prayer times ────────────────────────────────────────────────────────────


async def test_prayer_times_and_next_prayer():
    a = PrayerTimesAdapter(FakePrayer())
    times = await a.execute("get_times")
    assert times.success
    assert "Fajr 04:12" in times.message

    nxt = await a.execute("get_next_prayer")
    assert nxt.success
    assert nxt.message == "Next prayer: Maghrib at 20:41."


async def test_prayer_failure_is_typed():
    a = PrayerTimesAdapter(FakePrayer({"success": False, "error": "HTTP 500"}))
    r = await a.execute("get_times")
    assert r.error_type is ErrorType.UPSTREAM


# ── Markets ─────────────────────────────────────────────────────────────────


async def test_markets_price_uppercases_symbol():
    a = MarketsAdapter(FakeMarkets())
    r = await a.execute("get_price", {"symbol": " aapl "})
    assert r.success
    assert "AAPL 213.4" in r.message


async def test_markets_missing_symbol_is_input_error():
    r = await MarketsAdapter(FakeMarkets()).execute("get_price", {})
    assert r.error_type is ErrorType.INPUT


async def test_markets_partial_watchlist_is_degraded_not_failed():
    """get_all reports success=False when a provider fell over, but partial
    prices are still worth showing. Losing them was the old behaviour."""
    partial = {"success": False, "prices": [{"symbol": "AAPL", "price": 213.4}]}
    r = await MarketsAdapter(FakeMarkets(all_payload=partial)).execute("get_all")
    assert r.success is True
    assert r.degraded is True
    assert "AAPL" in r.message


async def test_markets_empty_watchlist_is_a_real_failure():
    empty = {"success": False, "prices": [], "error": "all endpoints failed"}
    r = await MarketsAdapter(FakeMarkets(all_payload=empty)).execute("get_all")
    assert r.success is False
    assert r.error_type is ErrorType.UPSTREAM


async def test_markets_health_degrades_rather_than_erroring():
    a = MarketsAdapter(FakeMarkets(price={"success": False, "error": "rate limited"}))
    h = await a.health_check()
    assert h.status is HealthStatus.DEGRADED
    assert h.healthy is True


# ── News ────────────────────────────────────────────────────────────────────


async def test_news_headlines_passes_filters_through():
    t = FakeNews()
    r = await NewsAdapter(t).execute("get_headlines",
                                     {"source": "guardian", "max_items": 3})
    assert r.success
    assert t.seen[-1]["source"] == "guardian"
    assert t.seen[-1]["max_items"] == 3


async def test_news_list_sources():
    r = await NewsAdapter(FakeNews()).execute("list_sources")
    assert r.data["sources"] == ["bbc", "guardian", "techcrunch"]


async def test_news_max_items_upper_bound():
    r = await NewsAdapter(FakeNews()).execute("get_headlines", {"max_items": 500})
    assert r.error_type is ErrorType.INPUT


# ── Sports ──────────────────────────────────────────────────────────────────


async def test_sports_uses_explicit_league_key():
    t = FakeSports()
    await SportsAdapter(t).execute("get_scores", {"league_key": "nba"})
    assert t.seen[-1] == ("scores", "nba", 10)


async def test_sports_infers_league_from_query():
    t = FakeSports(detect="premier_league")
    await SportsAdapter(t).execute("get_scores", {"query": "how did united do"})
    assert t.seen[-1][1] == "premier_league"


async def test_sports_undetectable_league_is_input_error_with_guidance():
    """The old dispatcher had no branch for sports at all, so this path
    returned 'Unknown agent/action'. It should say what's actually missing."""
    t = FakeSports(detect=None)
    r = await SportsAdapter(t).execute("get_scores", {"query": "what happened"})
    assert r.success is False
    assert r.error_type is ErrorType.INPUT
    assert "league_key" in r.error


async def test_sports_search_team_requires_query():
    r = await SportsAdapter(FakeSports()).execute("search_team", {})
    assert r.error_type is ErrorType.INPUT


async def test_sports_health_degrades_on_probe_failure():
    a = SportsAdapter(FakeSports(payload={"success": False, "error": "espn 503"}))
    assert (await a.health_check()).status is HealthStatus.DEGRADED


# ── Web search ──────────────────────────────────────────────────────────────


async def test_search_returns_results():
    r = await WebSearchAdapter(FakeSearch()).execute("search", {"query": "python"})
    assert r.success
    assert r.message == "1 results"


async def test_search_rejects_non_http_url():
    r = await WebSearchAdapter(FakeSearch()).execute(
        "scrape", {"url": "file:///etc/passwd"})
    assert r.success is False
    assert r.error_type is ErrorType.INPUT


async def test_scrape_truncates_to_max_chars():
    r = await WebSearchAdapter(FakeSearch()).execute(
        "scrape", {"url": "https://example.com", "max_chars": 250})
    assert r.success
    assert len(r.message) <= 250


async def test_scrape_tolerates_a_plain_string_return():
    """web_search.scrape's return type isn't consistent in the codebase."""
    a = WebSearchAdapter(FakeSearch(scrape_payload="just a string"))
    r = await a.execute("scrape", {"url": "https://example.com"})
    assert r.success
    assert r.message == "just a string"


async def test_search_all_backends_down_is_degraded_health():
    a = WebSearchAdapter(FakeSearch(payload={"success": False, "error": "no backend"}))
    assert (await a.health_check()).status is HealthStatus.DEGRADED


# ── Cross-cutting ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("adapter,double", [
    (WeatherAdapter, FakeWeather), (PrayerTimesAdapter, FakePrayer),
    (MarketsAdapter, FakeMarkets), (NewsAdapter, FakeNews),
    (SportsAdapter, FakeSports), (WebSearchAdapter, FakeSearch),
])
def test_every_adapter_declares_a_name_description_and_actions(adapter, double):
    a = adapter(double())
    assert a.name and " " not in a.name and a.name.islower()
    assert a.description
    assert a.actions
    for act in a.actions.values():
        assert act.description, f"{a.name}.{act.name} has no description"
        assert act.input_schema["type"] == "object"


@pytest.mark.parametrize("adapter,double", [
    (WeatherAdapter, FakeWeather), (PrayerTimesAdapter, FakePrayer),
    (MarketsAdapter, FakeMarkets), (NewsAdapter, FakeNews),
    (SportsAdapter, FakeSports), (WebSearchAdapter, FakeSearch),
])
async def test_a_raising_underlying_tool_never_escapes_the_adapter(adapter, double):
    """The point of the interface: a tool that explodes produces a typed
    failure, not an exception that unwinds the whole DAG."""
    a = adapter(double())
    for name in list(a.actions):
        for meth in ("get_current", "get_times", "get_price", "get_all",
                     "get_headlines", "get_scores", "get_standings",
                     "search_team", "search", "scrape"):
            if hasattr(a._t, meth):
                async def _boom(*args, **kw):
                    raise RuntimeError("underlying tool exploded")
                setattr(a._t, meth, _boom)

        result = await a.execute(name, _minimal_params(a, name))
        assert isinstance(result.success, bool)
        if not result.success:
            assert result.error_type is not None


def _minimal_params(adapter, action_name):
    """Build the smallest schema-valid params for an action."""
    schema = adapter.actions[action_name].input_schema
    out = {}
    for req in schema.get("required", []):
        spec = schema["properties"][req]
        if spec.get("type") == "string":
            out[req] = "https://example.com" if spec.get("pattern") else "x"
        elif spec.get("type") == "integer":
            out[req] = spec.get("minimum", 1)
    return out
