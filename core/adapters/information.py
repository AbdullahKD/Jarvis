"""
Adapters — information tools (weather, prayer times, markets, news, sports,
web search).

These wrap the existing tool classes rather than rewriting them. The classes in
``tools/`` keep their current behaviour and their current tests; the adapter is
the only thing that knows about the common interface. That's deliberate: it
means Phase 2 can't regress any tool's actual logic, and the adapter is a small
enough surface to review line by line.

Two things every adapter does that the raw tools don't:

* **Turn ``{"success": False, "error": "..."}`` into a typed failure.** The raw
  tools flatten every failure into one shape, so a rate limit, a bad postcode
  and a dead endpoint are indistinguishable. Classifying them here is what lets
  the Phase 3 circuit breaker retry an UPSTREAM blip and not retry a bad param.

* **Separate data from prose.** ``data`` is the structured payload for
  dependent subtasks, ``message`` the rendered text — currently the tools'
  ``format_*`` helpers are called ad hoc at ~40 call sites in the orchestrator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.tool import (
    Action,
    BaseTool,
    HealthReport,
    HealthStatus,
    ToolInputError,
    ToolUpstreamError,
)


def _unwrap(payload: Dict[str, Any], *, what: str) -> Dict[str, Any]:
    """Raise a typed error if a legacy ``{"success": ...}`` dict failed.

    Classification is by error text because that's all the raw tools give us.
    It's a heuristic, and it's contained to this one function so it can be
    replaced once the underlying tools raise properly.
    """
    if payload.get("success"):
        return payload

    err = str(payload.get("error") or f"{what} failed")
    low = err.lower()
    if "could not find" in low or "not found" in low or "no results" in low:
        from core.tool import ToolNotFoundError
        raise ToolNotFoundError(err)
    if "timeout" in low or "timed out" in low:
        from core.tool import ToolTimeoutError
        raise ToolTimeoutError(err)
    if "api key" in low or "unauthorized" in low or "401" in low or "403" in low:
        from core.tool import ToolAuthError
        raise ToolAuthError(err)
    raise ToolUpstreamError(err)


# ── Weather ─────────────────────────────────────────────────────────────────


class WeatherAdapter(BaseTool):
    _name = "weather"
    _description = "Current conditions and multi-day forecasts for any location."

    def __init__(self, tool: Any) -> None:
        self._t = tool
        super().__init__()

    def _register_actions(self) -> None:
        location = {
            "type": "string",
            "description": "Place name, e.g. 'High Wycombe'. Omit for the configured default.",
        }
        self.add_action(Action(
            name="get_current",
            description="Current weather. Returns temperature, conditions, wind and humidity.",
            input_schema={"properties": {"location": location}},
            handler=self._current, timeout=15.0,
        ))
        self.add_action(Action(
            name="get_forecast",
            description="Daily forecast for the next N days (default 7, max 16).",
            input_schema={"properties": {
                "location": location,
                "days": {"type": "integer", "minimum": 1, "maximum": 16, "default": 7},
            }},
            handler=self._forecast, timeout=15.0,
        ))

    async def _current(self, location: Optional[str] = None):
        data = (await self._t.get_current_for_location(location)
                if location else await self._t.get_current())
        return _unwrap(data, what="weather"), self._t.format_current(data)

    async def _forecast(self, location: Optional[str] = None, days: int = 7):
        data = (await self._t.get_forecast_for_location(location, days)
                if location else await self._t.get_forecast(days=days))
        return _unwrap(data, what="forecast"), self._t.format_forecast(data)

    async def _check_health(self) -> HealthReport:
        # Open-Meteo needs no key, so a real call is the honest probe. Cheap
        # (one small JSON response) and it exercises the same path a user hits.
        try:
            data = await self._t.get_current()
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name, f"{type(exc).__name__}: {exc}")
        if data.get("success"):
            return HealthReport(HealthStatus.OK, self.name, "open-meteo reachable")
        return HealthReport(HealthStatus.ERROR, self.name, str(data.get("error", "unknown")))


# ── Prayer times ────────────────────────────────────────────────────────────


class PrayerTimesAdapter(BaseTool):
    _name = "prayer"
    _description = "Islamic prayer times for a location, plus the next prayer due."

    def __init__(self, tool: Any) -> None:
        self._t = tool
        super().__init__()

    def _register_actions(self) -> None:
        schema = {"properties": {
            "location": {"type": "string", "description": "City name. Defaults to the configured location."},
            "method": {"type": "integer", "description": "Calculation method id (AlAdhan convention)."},
        }}
        self.add_action(Action(
            name="get_times", description="All five prayer times for today.",
            input_schema=schema, handler=self._times, timeout=15.0,
        ))
        self.add_action(Action(
            name="get_next_prayer",
            description="Which prayer is next and how long until it.",
            input_schema=schema, handler=self._next, timeout=15.0,
        ))

    async def _times(self, location: Optional[str] = None, method: Optional[int] = None):
        kw: Dict[str, Any] = {}
        if location:
            kw["location"] = location
        if method is not None:
            kw["method"] = method
        data = _unwrap(await self._t.get_times(**kw), what="prayer times")
        return data, self._t.format_times(data)

    async def _next(self, location: Optional[str] = None, method: Optional[int] = None):
        kw: Dict[str, Any] = {}
        if location:
            kw["location"] = location
        if method is not None:
            kw["method"] = method
        data = _unwrap(await self._t.get_times(**kw), what="prayer times")
        nxt = self._t.get_next_prayer(data)
        if not nxt:
            from core.tool import ToolNotFoundError
            raise ToolNotFoundError("no upcoming prayer found in today's times")
        name = nxt.get("name") if isinstance(nxt, dict) else nxt
        time_ = nxt.get("time", "") if isinstance(nxt, dict) else ""
        return nxt, f"Next prayer: {name}{f' at {time_}' if time_ else ''}."

    async def _check_health(self) -> HealthReport:
        try:
            data = await self._t.get_times()
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name, f"{type(exc).__name__}: {exc}")
        ok = bool(data.get("success"))
        return HealthReport(
            HealthStatus.OK if ok else HealthStatus.ERROR, self.name,
            "aladhan api reachable" if ok else str(data.get("error", "unknown")),
        )


# ── Markets ─────────────────────────────────────────────────────────────────


class MarketsAdapter(BaseTool):
    _name = "markets"
    _description = "Live stock, index, crypto and commodity prices."

    def __init__(self, tool: Any) -> None:
        self._t = tool
        super().__init__()

    def _register_actions(self) -> None:
        self.add_action(Action(
            name="get_price",
            description="Price and daily change for one ticker, e.g. AAPL or BTC-USD.",
            input_schema={
                "properties": {"symbol": {"type": "string", "minLength": 1,
                                          "description": "Ticker symbol."}},
                "required": ["symbol"],
            },
            handler=self._price, timeout=20.0,
        ))
        self.add_action(Action(
            name="get_all",
            description="Prices for the default watchlist (indices, crypto, commodities, tech).",
            input_schema={"properties": {}},
            handler=self._all, timeout=30.0,
        ))

    async def _price(self, symbol: str):
        sym = symbol.strip().upper()
        if not sym:
            raise ToolInputError("symbol must not be empty")
        data = _unwrap(await self._t.get_price(sym), what=f"price for {sym}")
        return data, self._t.format_prices({"success": True, "prices": [data]})

    async def _all(self):
        data = await self._t.get_all()
        # get_all reports success=False when *every* provider failed, but a
        # partial result is still useful — flag it degraded rather than losing it.
        prices = data.get("prices") or []
        if not prices:
            _unwrap(data, what="market data")
        from core.tool import ToolResult
        return ToolResult.ok(
            self.name, "get_all", data=data,
            message=self._t.format_prices(data),
            degraded=not data.get("success"),
        )

    async def _check_health(self) -> HealthReport:
        try:
            data = await self._t.get_price("AAPL")
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name, f"{type(exc).__name__}: {exc}")
        if data.get("success"):
            return HealthReport(HealthStatus.OK, self.name, "quote provider reachable")
        return HealthReport(HealthStatus.DEGRADED, self.name,
                            f"AAPL probe failed: {data.get('error', 'unknown')}")


# ── News ────────────────────────────────────────────────────────────────────


class NewsAdapter(BaseTool):
    _name = "news"
    _description = "Headlines from BBC, Guardian, TechCrunch, Hacker News and others."

    def __init__(self, tool: Any) -> None:
        self._t = tool
        super().__init__()

    def _register_actions(self) -> None:
        self.add_action(Action(
            name="get_headlines",
            description="Recent headlines, optionally filtered by source, category or topic.",
            input_schema={"properties": {
                "query": {"type": "string", "description": "Free-text topic; source/category are inferred from it."},
                "source": {"type": "string", "description": "Feed key, e.g. bbc, guardian, techcrunch, hackernews."},
                "category": {"type": "string", "description": "e.g. world, tech, business, sport."},
                "topic": {"type": "string", "description": "Narrow the results to a subject."},
                "max_items": {"type": "integer", "minimum": 1, "maximum": 30, "default": 5},
            }},
            handler=self._headlines, timeout=25.0,
        ))
        self.add_action(Action(
            name="list_sources", description="Available feed keys.",
            input_schema={"properties": {}}, handler=self._sources, timeout=5.0,
        ))

    async def _headlines(self, query: Optional[str] = None, source: Optional[str] = None,
                         category: Optional[str] = None, topic: Optional[str] = None,
                         max_items: int = 5):
        data = await self._t.get_headlines(
            query=query, source=source, category=category,
            topic=topic, max_items=max_items,
        )
        _unwrap(data, what="news")
        return data, self._t.format_headlines(data)

    async def _sources(self):
        srcs = self._t.list_sources()
        return {"sources": srcs}, "Available sources: " + ", ".join(map(str, srcs))

    async def _check_health(self) -> HealthReport:
        try:
            data = await self._t.get_headlines(source="bbc", max_items=1)
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name, f"{type(exc).__name__}: {exc}")
        if data.get("success"):
            return HealthReport(HealthStatus.OK, self.name, "bbc feed parsed")
        # Worth knowing: the shipped `reuters` feed URL was retired years ago
        # and fails silently. The audit flagged it; this probe uses bbc.
        return HealthReport(HealthStatus.ERROR, self.name,
                            f"bbc feed failed: {data.get('error', 'unknown')}")


# ── Sports ──────────────────────────────────────────────────────────────────


class SportsAdapter(BaseTool):
    _name = "sports"
    _description = "Live scores, fixtures and league tables across football, cricket, F1, tennis and more."

    def __init__(self, tool: Any) -> None:
        self._t = tool
        super().__init__()

    def _register_actions(self) -> None:
        league = {"type": "string",
                  "description": "League key, e.g. premier_league, nba, cricket. Inferred from the query if omitted."}
        self.add_action(Action(
            name="get_scores",
            description="Recent and live scores for a league.",
            input_schema={"properties": {
                "league_key": league,
                "query": {"type": "string", "description": "Natural-language request; used to detect the league."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            }},
            handler=self._scores, timeout=30.0,
        ))
        self.add_action(Action(
            name="get_standings", description="Current league table.",
            input_schema={"properties": {"league_key": league,
                                         "query": {"type": "string"}}},
            handler=self._standings, timeout=30.0,
        ))
        self.add_action(Action(
            name="search_team",
            description="Recent and upcoming fixtures for a named team.",
            input_schema={
                "properties": {"query": {"type": "string", "minLength": 1,
                                         "description": "Team name, e.g. 'Manchester United'."},
                               "league_key": league},
                "required": ["query"],
            },
            handler=self._team, timeout=40.0,
        ))
        self.add_action(Action(
            name="list_leagues", description="Supported league keys.",
            input_schema={"properties": {}}, handler=self._leagues, timeout=5.0,
        ))

    def _league_from(self, league_key: Optional[str], query: Optional[str]) -> str:
        if league_key:
            return league_key
        if query:
            detected = self._t.detect_league(query)
            if detected:
                return detected
        raise ToolInputError(
            "could not determine which league — pass league_key, or a query naming the sport or team"
        )

    async def _scores(self, league_key: Optional[str] = None,
                      query: Optional[str] = None, limit: int = 10):
        key = self._league_from(league_key, query)
        data = await self._t.get_scores(key, limit=limit)
        _unwrap(data, what=f"scores for {key}")
        return data, self._t.format_scores(data)

    async def _standings(self, league_key: Optional[str] = None,
                         query: Optional[str] = None):
        key = self._league_from(league_key, query)
        data = await self._t.get_standings(key)
        _unwrap(data, what=f"standings for {key}")
        return data, self._t.format_standings(data)

    async def _team(self, query: str, league_key: Optional[str] = None):
        data = await self._t.search_team(query, league_key=league_key)
        _unwrap(data, what=f"fixtures for {query}")
        return data, self._t.format_scores(data)

    async def _leagues(self):
        leagues = self._t.list_leagues()
        return {"leagues": leagues}, "Supported leagues: " + ", ".join(map(str, leagues))

    async def _check_health(self) -> HealthReport:
        try:
            data = await self._t.get_scores("premier_league", limit=1)
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name, f"{type(exc).__name__}: {exc}")
        ok = bool(data.get("success"))
        return HealthReport(
            HealthStatus.OK if ok else HealthStatus.DEGRADED, self.name,
            "espn api reachable" if ok else f"probe failed: {data.get('error', 'unknown')}",
        )


# ── Web search ──────────────────────────────────────────────────────────────


class WebSearchAdapter(BaseTool):
    _name = "websearch"
    _description = "Search the web and read pages. Falls back across DuckDuckGo and Wikipedia."

    def __init__(self, tool: Any) -> None:
        self._t = tool
        super().__init__()

    def _register_actions(self) -> None:
        self.add_action(Action(
            name="search",
            description="Search the web. Returns ranked results with snippets.",
            input_schema={
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
                "required": ["query"],
            },
            handler=self._search, timeout=30.0,
        ))
        self.add_action(Action(
            name="scrape",
            description="Fetch a URL and return its readable text.",
            input_schema={
                "properties": {
                    "url": {"type": "string", "pattern": "^https?://"},
                    "max_chars": {"type": "integer", "minimum": 200, "maximum": 50000, "default": 5000},
                },
                "required": ["url"],
            },
            handler=self._scrape, timeout=30.0,
        ))

    async def _search(self, query: str, max_results: int = 5):
        if not query.strip():
            raise ToolInputError("query must not be empty")
        data = await self._t.search(query, max_results=max_results)
        _unwrap(data, what="web search")
        return data, self._t.format_results(data)

    async def _scrape(self, url: str, max_chars: int = 5000):
        data = await self._t.scrape(url, max_chars=max_chars)
        if isinstance(data, dict):
            _unwrap(data, what="page fetch")
            text = data.get("text") or data.get("content") or ""
            return data, str(text)[:max_chars]
        return {"text": data}, str(data)[:max_chars]

    async def _check_health(self) -> HealthReport:
        try:
            data = await self._t.search("jarvis health probe", max_results=1)
        except Exception as exc:  # noqa: BLE001
            return HealthReport(HealthStatus.ERROR, self.name, f"{type(exc).__name__}: {exc}")
        if data.get("success"):
            return HealthReport(HealthStatus.OK, self.name, "search backend reachable")
        return HealthReport(HealthStatus.DEGRADED, self.name,
                            "all search backends failed the probe")


__all__ = [
    "WeatherAdapter", "PrayerTimesAdapter", "MarketsAdapter",
    "NewsAdapter", "SportsAdapter", "WebSearchAdapter",
]
