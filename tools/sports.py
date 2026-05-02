"""
Sports Tool
Uses ESPN's public API endpoints for real-time scores,
fixtures, standings and match summaries. No API key required.
Covers: Premier League, Champions League, La Liga, Serie A,
        Bundesliga, NFL, NBA, MLB, NHL, Cricket, Tennis, F1
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

TIMEOUT  = aiohttp.ClientTimeout(total=15)
HEADERS  = {"User-Agent": "Jarvis/1.0 (BNU Dissertation Project) Python/aiohttp"}
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# ── League directory ────────────────────────────────────────────────────────
LEAGUES: Dict[str, Dict] = {
    # Football / Soccer
    "premier_league":     {"sport": "soccer",       "league": "eng.1",        "name": "Premier League"},
    "champions_league":   {"sport": "soccer",       "league": "UEFA.CHAMPIONS","name": "Champions League"},
    "la_liga":            {"sport": "soccer",       "league": "esp.1",        "name": "La Liga"},
    "serie_a":            {"sport": "soccer",       "league": "ita.1",        "name": "Serie A"},
    "bundesliga":         {"sport": "soccer",       "league": "ger.1",        "name": "Bundesliga"},
    "ligue_1":            {"sport": "soccer",       "league": "fra.1",        "name": "Ligue 1"},
    "mls":                {"sport": "soccer",       "league": "usa.1",        "name": "MLS"},
    "efl_championship":   {"sport": "soccer",       "league": "eng.2",        "name": "EFL Championship"},

    # American Sports
    "nfl":                {"sport": "football",     "league": "nfl",          "name": "NFL"},
    "nba":                {"sport": "basketball",   "league": "nba",          "name": "NBA"},
    "mlb":                {"sport": "baseball",     "league": "mlb",          "name": "MLB"},
    "nhl":                {"sport": "hockey",       "league": "nhl",          "name": "NHL"},

    # Other
    "f1":                 {"sport": "racing",       "league": "f1",           "name": "Formula 1"},
    "tennis":             {"sport": "tennis",       "league": "atp",          "name": "ATP Tennis"},
    "cricket_test":       {"sport": "cricket",      "league": "test",         "name": "Test Cricket"},
    "cricket_odi":        {"sport": "cricket",      "league": "odi",          "name": "ODI Cricket"},
    "cricket_t20":        {"sport": "cricket",      "league": "t20",          "name": "T20 Cricket"},
}

# Natural language aliases
LEAGUE_ALIASES = {
    "premier_league":   ["premier league", "epl", "pl", "english football", "premiership"],
    "champions_league": ["champions league", "ucl", "cl", "european football", "europe"],
    "la_liga":          ["la liga", "spain", "spanish football", "laliga"],
    "serie_a":          ["serie a", "italy", "italian football", "seria"],
    "bundesliga":       ["bundesliga", "germany", "german football", "german league"],
    "ligue_1":          ["ligue 1", "france", "french football", "ligue1"],
    "efl_championship": ["championship", "efl", "efl championship"],
    "nfl":              ["nfl", "american football", "nfl football"],
    "nba":              ["nba", "basketball", "nba basketball"],
    "mlb":              ["mlb", "baseball", "mlb baseball"],
    "nhl":              ["nhl", "hockey", "ice hockey"],
    "f1":               ["f1", "formula 1", "formula one", "grand prix", "racing"],
    "tennis":           ["tennis", "atp", "wimbledon", "us open", "french open"],
    "cricket_test":     ["test cricket", "test match", "test series"],
    "cricket_odi":      ["odi", "one day", "one day international"],
    "cricket_t20":      ["t20", "twenty20", "t20i", "pakistan cricket", "cricket"],
}


class SportsTool:
    """
    ESPN-powered sports data tool.
    Gets scores, fixtures, standings for major leagues worldwide.
    """

    # ── Public API ──────────────────────────────────────────────────────────

    async def get_scores(
        self,
        league_key: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Get recent and upcoming scores/fixtures for a league.
        Returns finished results + live games + upcoming fixtures.
        """
        league = LEAGUES.get(league_key)
        if not league:
            return {"success": False, "error": f"Unknown league: {league_key}"}

        url = f"{ESPN_BASE}/{league['sport']}/{league['league']}/scoreboard"

        try:
            async with aiohttp.ClientSession(
                timeout=TIMEOUT, headers=HEADERS
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return {"success": False, "error": f"HTTP {resp.status}"}
                    data = await resp.json()

            events = data.get("events", [])
            games = []

            for event in events[:limit]:
                game = self._parse_event(event)
                if game:
                    games.append(game)

            return {
                "success": True,
                "league": league["name"],
                "league_key": league_key,
                "games": games,
                "count": len(games),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_standings(self, league_key: str) -> Dict[str, Any]:
        """Get current league standings/table."""
        league = LEAGUES.get(league_key)
        if not league:
            return {"success": False, "error": f"Unknown league: {league_key}"}

        url = f"{ESPN_BASE}/{league['sport']}/{league['league']}/standings"

        try:
            async with aiohttp.ClientSession(
                timeout=TIMEOUT, headers=HEADERS
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return {"success": False, "error": f"HTTP {resp.status}"}
                    data = await resp.json()

            standings = self._parse_standings(data)
            return {
                "success": True,
                "league": league["name"],
                "standings": standings,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def search_team(
        self,
        query: str,
        league_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for a specific team's recent results.
        Filters scores by team name.
        """
        # Try to detect league from query
        if not league_key:
            league_key = self.detect_league(query)
        if not league_key:
            league_key = "premier_league"  # default

        scores = await self.get_scores(league_key, limit=20)
        if not scores.get("success"):
            return scores

        query_lower = query.lower()
        # Extract just the team name (remove league/sport keywords)
        for alias_list in LEAGUE_ALIASES.values():
            for alias in alias_list:
                query_lower = query_lower.replace(alias, "").strip()

        team_games = [
            g for g in scores["games"]
            if query_lower in g.get("home_team", "").lower()
            or query_lower in g.get("away_team", "").lower()
        ]

        return {
            "success": True,
            "league": scores["league"],
            "team_query": query_lower,
            "games": team_games,
        }

    def detect_league(self, query: str) -> Optional[str]:
        """Detect league from natural language."""
        q = query.lower()
        for league_key, aliases in LEAGUE_ALIASES.items():
            if any(alias in q for alias in aliases):
                return league_key
        return None

    # ── Formatters ──────────────────────────────────────────────────────────

    def format_scores(self, data: Dict[str, Any], show_upcoming: bool = True) -> str:
        """Format scores for display."""
        if not data.get("success"):
            return f"Could not get scores: {data.get('error', 'unknown error')}"

        games = data.get("games", [])
        if not games:
            return f"No recent games found for {data.get('league', 'this league')}."

        league = data.get("league", "")
        lines = [f"{league} Results & Fixtures:", ""]

        finished = [g for g in games if g["status"] == "final"]
        live     = [g for g in games if g["status"] == "live"]
        upcoming = [g for g in games if g["status"] == "upcoming"]

        if live:
            lines.append("LIVE:")
            for g in live:
                lines.append(f"  {g['home_team']} {g['home_score']} - {g['away_score']} {g['away_team']}  [{g['clock']}]")
            lines.append("")

        if finished:
            lines.append("Recent Results:")
            for g in finished[-6:]:
                result = self._result_indicator(g)
                lines.append(f"  {result} {g['home_team']} {g['home_score']} - {g['away_score']} {g['away_team']}")
            lines.append("")

        if show_upcoming and upcoming:
            lines.append("Upcoming:")
            for g in upcoming[:4]:
                lines.append(f"  {g['home_team']} vs {g['away_team']}  {g.get('date_str', '')}")
            lines.append("")

        return "\n".join(lines).strip()

    def format_standings(self, data: Dict[str, Any], top_n: int = 10) -> str:
        """Format league table."""
        if not data.get("success"):
            return f"Could not get standings: {data.get('error', 'unknown error')}"

        standings = data.get("standings", [])
        if not standings:
            return "No standings available."

        league = data.get("league", "")
        lines = [f"{league} Table:", ""]
        lines.append(f"{'Pos':<4} {'Team':<25} {'P':<4} {'W':<4} {'D':<4} {'L':<4} {'GD':<6} {'Pts':<4}")
        lines.append("-" * 55)

        for entry in standings[:top_n]:
            pos  = str(entry.get("position", "")).ljust(4)
            team = entry.get("team", "")[:24].ljust(25)
            p    = str(entry.get("played", "")).ljust(4)
            w    = str(entry.get("wins", "")).ljust(4)
            d    = str(entry.get("draws", "")).ljust(4)
            l    = str(entry.get("losses", "")).ljust(4)
            gd   = str(entry.get("goal_diff", "")).ljust(6)
            pts  = str(entry.get("points", "")).ljust(4)
            lines.append(f"{pos} {team} {p} {w} {d} {l} {gd} {pts}")

        return "\n".join(lines)

    def list_leagues(self) -> str:
        lines = ["Available sports leagues:\n"]
        by_sport: Dict[str, List[str]] = {}
        for key, info in LEAGUES.items():
            sport = info["sport"].title()
            by_sport.setdefault(sport, []).append(info["name"])
        for sport, names in sorted(by_sport.items()):
            lines.append(f"  {sport}: {', '.join(names)}")
        return "\n".join(lines)

    # ── Parsers ─────────────────────────────────────────────────────────────

    def _parse_event(self, event: Dict) -> Optional[Dict]:
        """Parse an ESPN event into a clean game dict."""
        try:
            competitions = event.get("competitions", [])
            if not competitions:
                return None

            comp = competitions[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                return None

            # Find home and away
            home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

            status_obj = event.get("status", {})
            status_type = status_obj.get("type", {})
            state = status_type.get("state", "pre")  # pre, in, post
            completed = status_type.get("completed", False)

            if state == "post" or completed:
                status = "final"
            elif state == "in":
                status = "live"
            else:
                status = "upcoming"

            # Date
            date_str = event.get("date", "")
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                date_fmt = dt.strftime("%a %d %b, %H:%M")
            except Exception:
                date_fmt = date_str[:10]

            # Clock/period for live games
            clock = ""
            if status == "live":
                clock = status_type.get("shortDetail", "")

            return {
                "home_team": home.get("team", {}).get("shortDisplayName", "Home"),
                "away_team": away.get("team", {}).get("shortDisplayName", "Away"),
                "home_score": home.get("score", "-"),
                "away_score": away.get("score", "-"),
                "status": status,
                "date_str": date_fmt,
                "clock": clock,
                "venue": comp.get("venue", {}).get("fullName", ""),
            }
        except Exception:
            return None

    def _parse_standings(self, data: Dict) -> List[Dict]:
        """Parse ESPN standings response."""
        entries = []
        try:
            for group in data.get("standings", {}).get("entries", []):
                team_name = group.get("team", {}).get("displayName", "Unknown")
                stats = {s["name"]: s.get("value", 0) for s in group.get("stats", [])}

                entries.append({
                    "position":  int(stats.get("rank", 0)),
                    "team":      team_name,
                    "played":    int(stats.get("gamesPlayed", stats.get("playoffSeed", 0))),
                    "wins":      int(stats.get("wins", 0)),
                    "draws":     int(stats.get("ties", stats.get("draws", 0))),
                    "losses":    int(stats.get("losses", 0)),
                    "goal_diff": int(stats.get("pointDifferential", stats.get("goalDifferential", 0))),
                    "points":    int(stats.get("points", stats.get("totalPoints", 0))),
                })
            entries.sort(key=lambda x: x["position"])
        except Exception:
            pass
        return entries

    def _result_indicator(self, game: Dict) -> str:
        """Return W/D/L or neutral indicator."""
        try:
            hs = int(game.get("home_score", 0))
            as_ = int(game.get("away_score", 0))
            if hs > as_:
                return "W"
            elif hs < as_:
                return "W"
            else:
                return "D"
        except Exception:
            return " "