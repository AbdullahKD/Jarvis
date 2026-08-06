"""
Sports Tool
Uses ESPN's public API endpoints for real-time scores,
fixtures, standings and match summaries. No API key required.
Covers: Premier League, Champions League, La Liga, Serie A,
        Bundesliga, NFL, NBA, MLB, NHL, Cricket, Tennis, F1
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from config.logging_config import get_logger

# Module logger. High-frequency polling lines (cricket fetch counts, etc.)
# log at DEBUG so they stay hidden at the default INFO level — they fire on
# every /live-tick (20s) and /sidebar (60s) poll and otherwise flood the
# terminal. Set JARVIS_LOG_LEVEL=DEBUG to see them again.
logger = get_logger("sports")

TIMEOUT  = aiohttp.ClientTimeout(total=15)
HEADERS  = {"User-Agent": "Jarvis/1.0 Python/aiohttp"}
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
# ESPNcricinfo consumer API — lists ALL current cricket matches (live / recent /
# upcoming) across every series, so we don't need fragile per-series league IDs.
CRICINFO_CURRENT = "https://hs-consumer-api.espncricinfo.com/v1/pages/matches/current?lang=en"

# ── Data-accuracy guards ─────────────────────────────────────────────────────
# ESPN sometimes leaves abandoned / suspended / long-finished events with
# state="in", which used to render as LIVE in the UI for days ("live" tennis
# matches from last week's tournament, etc.). A game may only be LIVE if it
# actually started recently — anything older is demoted to final, anything
# in the future to upcoming. Per-sport windows (hours):
LIVE_MAX_AGE_HOURS = {
    "soccer": 4, "basketball": 6, "football": 6, "hockey": 5, "baseball": 7,
    # Cricket: Test matches are legitimately "live" for up to 5 days from
    # their start date, so the window is much wider — it only catches feeds
    # stuck on "live" for over a week.
    "tennis": 10, "mma": 12, "boxing": 12, "racing": 6, "cricket": 130,
}
# Finished results older than this stop being interesting on a "scores" rail;
# we keep the single most-recent final (with its date) so a team/league card
# still shows the last result during the off-season.
FINISHED_MAX_AGE_DAYS = 14


def _parse_iso(iso: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat((iso or "").replace("Z", "+00:00"))
    except Exception:
        return None


def sanitize_game_status(g: Dict, sport: str = "") -> Dict:
    """Demote impossible LIVE states using the event's own start time."""
    if g.get("status") != "live":
        return g
    start = _parse_iso(g.get("date_iso", ""))
    if start is None:
        return g
    age_h = (datetime.now(timezone.utc) - start).total_seconds() / 3600
    max_h = LIVE_MAX_AGE_HOURS.get(sport, 8)
    if age_h > max_h:
        return {**g, "status": "final"}      # stale "live" — long over
    if age_h < -0.25:
        return {**g, "status": "upcoming"}   # hasn't started yet
    return g


def cap_finished_age(finished: List[Dict], max_days: int = FINISHED_MAX_AGE_DAYS) -> List[Dict]:
    """Keep finals from the last `max_days`; always keep the most recent one
    (dated) so off-season cards aren't empty. Input must be date-ascending."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_days)
    recent = []
    for g in finished:
        dt = _parse_iso(g.get("date_iso", ""))
        if dt is not None and dt >= cutoff:
            recent.append(g)
    if not recent and finished:
        recent = finished[-1:]
    return recent

# Sports whose ESPN scoreboard is NOT team-vs-team and needs individual parsing.
INDIVIDUAL_SPORTS = {"racing", "tennis", "mma", "boxing"}

# Mapping from API sport name → ESPN CDN logo subfolder
_CDN_SPORT: Dict[str, str] = {
    "basketball": "nba",
    "soccer":     "soccer",
    "cricket":    "cricket",
    "football":   "nfl",
    "baseball":   "mlb",
    "hockey":     "nhl",
    "racing":     "f1",
    "tennis":     "tennis",
}

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
    # Pre-season / summer-tour matches live in their own ESPN "league" — the
    # PL/La Liga scoreboards never contain them, which is why July showed
    # "no fixtures" for clubs mid-tour.
    "club_friendlies":    {"sport": "soccer",       "league": "club.friendly","name": "Club Friendlies"},

    # American Sports
    "nfl":                {"sport": "football",     "league": "nfl",          "name": "NFL"},
    "nba":                {"sport": "basketball",   "league": "nba",          "name": "NBA"},
    "mlb":                {"sport": "baseball",     "league": "mlb",          "name": "MLB"},
    "nhl":                {"sport": "hockey",       "league": "nhl",          "name": "NHL"},

    # Other
    "f1":                 {"sport": "racing",       "league": "f1",           "name": "Formula 1"},
    "tennis":             {"sport": "tennis",       "league": "atp",          "name": "ATP Tennis"},
    "tennis_wta":         {"sport": "tennis",       "league": "wta",          "name": "WTA Tennis"},
    "ufc":                {"sport": "mma",          "league": "ufc",          "name": "UFC"},
    "pfl":                {"sport": "mma",          "league": "pfl",          "name": "PFL"},
    "boxing":             {"sport": "boxing",       "league": "boxing",       "name": "Boxing"},
    "cricket_test":       {"sport": "cricket",      "league": "test",         "name": "Test Cricket",     "format": "Test"},
    "cricket_odi":        {"sport": "cricket",      "league": "odi",          "name": "ODI Cricket",      "format": "ODI"},
    "cricket_t20":        {"sport": "cricket",      "league": "t20",          "name": "T20 Cricket",      "format": "T20I"},
    "cricket_ipl":        {"sport": "cricket",      "league": "8677",         "name": "IPL",              "format": "IPL"},
    "cricket_psl":        {"sport": "cricket",      "league": "8810",         "name": "PSL",              "format": "PSL"},
    "cricket_bbl":        {"sport": "cricket",      "league": "5181",         "name": "Big Bash",         "format": "BBL"},
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
    "cricket_ipl":      ["ipl", "indian premier league", "iplt20"],
    "cricket_psl":      ["psl", "pakistan super league"],
    "cricket_bbl":      ["bbl", "big bash"],
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

        base_url   = f"{ESPN_BASE}/{league['sport']}/{league['league']}/scoreboard"
        sport_name = league.get("sport", "")

        async def _fetch(url: str) -> List[Dict]:
            """Fetch one scoreboard URL and parse its events. Never raises."""
            try:
                async with aiohttp.ClientSession(
                    timeout=TIMEOUT, headers=HEADERS
                ) as session:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            return []
                        data = await resp.json()
                out = []
                for event in data.get("events", []):
                    if sport_name in INDIVIDUAL_SPORTS:
                        g = self._parse_individual_event(event, sport_name)
                    else:
                        g = self._parse_event(event, sport=sport_name)
                    if not g:
                        continue
                    # Tennis returns a LIST of matches per tournament event
                    # (matches are nested under groupings) — flatten it.
                    if isinstance(g, list):
                        out.extend(g)
                    else:
                        out.append(g)
                return out
            except Exception:
                return []

        try:
            # 1) Current window — recent results + anything live right now.
            games = await _fetch(base_url)

            # 2) Forward window — explicitly request the next ~45 days so
            #    UPCOMING fixtures appear even when ESPN's default scoreboard
            #    only returns the current/most-recent matchday (e.g. end of
            #    season). Guarded: if it fails we simply keep the default set.
            has_upcoming = any(g.get("status") == "upcoming" for g in games)
            today = datetime.now(timezone.utc)
            if not has_upcoming:
                end = today + timedelta(days=45)
                games += await _fetch(f"{base_url}?dates={today:%Y%m%d}-{end:%Y%m%d}")

            # 2b) Off-season reach — if 45 days still found nothing upcoming
            #     (July asking about the NBA, say), look up to ~5 months out so
            #     preseason / friendlies / season openers appear instead of an
            #     empty card.
            if not any(g.get("status") == "upcoming" for g in games):
                start = today + timedelta(days=45)
                end   = today + timedelta(days=150)
                games += await _fetch(f"{base_url}?dates={start:%Y%m%d}-{end:%Y%m%d}")

            # De-duplicate by (date, home, away) and order the result so the UI
            # shows live → recent results → upcoming fixtures.
            seen: set = set()
            deduped: List[Dict] = []
            for g in games:
                key = (g.get("date_iso", "")[:10], g.get("home_team", ""), g.get("away_team", ""))
                if key in seen:
                    continue
                seen.add(key)
                # Demote impossible "live" states (stale/abandoned events that
                # ESPN never flipped to post — they used to show LIVE for days).
                deduped.append(sanitize_game_status(g, sport_name))

            live     = [g for g in deduped if g.get("status") == "live"]
            finished = sorted(
                [g for g in deduped if g.get("status") == "final"],
                key=lambda x: x.get("date_iso", ""),
            )
            # Drop finals older than 2 weeks (keeping the single most recent,
            # dated, so off-season cards still show the last result).
            finished = cap_finished_age(finished)
            upcoming = sorted(
                [g for g in deduped if g.get("status") == "upcoming"],
                key=lambda x: x.get("date_iso", ""),
            )
            ordered = live + finished[-limit:] + upcoming[:limit]

            return {
                "success": True,
                "league": league["name"],
                "league_key": league_key,
                "games": ordered[: max(limit * 2, limit)],
                "count": len(ordered),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Cricket (ESPNcricinfo current matches) ────────────────────────────────
    async def get_cricket_current(self, limit: int = 16) -> Dict[str, Any]:
        """
        Fetch ALL current cricket matches (live / recent / upcoming) from the
        ESPNcricinfo consumer API. This avoids the brittle per-series league IDs
        the ESPN site API needs — it returns every ongoing match across formats,
        so bilateral series (e.g. Pakistan v Australia) are captured reliably.
        """
        # ESPNcricinfo's consumer host blocks bot-style User-Agents, so send
        # browser-like headers here (the ESPN *site* API is more lenient).
        cric_headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.espncricinfo.com/",
            "Origin": "https://www.espncricinfo.com",
        }
        try:
            data = None
            async with aiohttp.ClientSession(timeout=TIMEOUT, headers=cric_headers) as session:
                async with session.get(CRICINFO_CURRENT) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                    else:
                        # Surface WHY cricket is empty — this used to fail
                        # silently and the UI just showed nothing for cricket.
                        logger.warning(f"🏏 cricket: cricinfo HTTP {resp.status} — "
                                       f"falling back to ESPN site API")

            # Matches may live under a few shapes depending on endpoint version.
            matches = []
            if isinstance(data, dict):
                matches = (data.get("matches")
                           or (data.get("content", {}) or {}).get("matches")
                           or data.get("items")
                           or [])
            games = []
            for m in (matches or []):
                g = self._parse_cricinfo_match(m)
                if g:
                    games.append(g)

            # Fallback chain — cricinfo's consumer API now 403s bot traffic,
            # so try progressively simpler sources until one yields matches:
            #   1. ESPN homepage "header" API (all current cricket, no league
            #      IDs needed — same data that powers espn.com's score strip)
            #   2. ESPN site API across known competition IDs
            #   3. static.cricinfo.com RSS live scores (ancient, but has
            #      survived every cricinfo redesign since 2005)
            if not games:
                games = await self._cricket_via_espn_header()
                if games:
                    logger.info(f"🏏 cricket: ESPN header API → {len(games)} matches")
            if not games:
                games = await self._cricket_via_espn()
            if not games:
                games = await self._cricket_via_rss()
                if games:
                    logger.info(f"🏏 cricket: cricinfo RSS → {len(games)} live matches")

            logger.debug(f"🏏 cricket: cricinfo matches={len(matches or [])} parsed={len(games)}")
            if not games:
                return {"success": False, "error": "no cricket matches"}

            # Order: live → upcoming (soonest) → recent finished.
            # (Multi-day Tests are legitimately "live" for days, so the
            # cricket liveness window is deliberately wide — see
            # LIVE_MAX_AGE_HOURS.)
            games    = [sanitize_game_status(g, "cricket") for g in games]
            live     = [g for g in games if g["status"] == "live"]
            upcoming = sorted([g for g in games if g["status"] == "upcoming"], key=lambda x: x.get("date_iso", ""))
            finished = sorted([g for g in games if g["status"] == "final"], key=lambda x: x.get("date_iso", ""), reverse=True)
            finished = cap_finished_age(finished[::-1])[::-1]
            ordered  = (live + upcoming + finished)[:limit]
            return {"success": True, "league": "Cricket", "league_key": "cricket",
                    "games": ordered, "count": len(ordered)}
        except Exception as e:
            logger.warning(f"cricket fetch error: {e}")
            return {"success": False, "error": str(e)}

    async def _cricket_via_espn_header(self) -> List[Dict]:
        """
        Cricket via ESPN's homepage 'header' scoreboard API. Unlike the site
        API it needs NO league/competition IDs — it returns whatever cricket
        is current right now across all series, which is exactly what the
        sidebar wants. Shape: sports[0].leagues[*].events[*].
        """
        url = ("https://site.web.api.espn.com/apis/personalized/v2/"
               "scoreboard/header?sport=cricket")
        try:
            async with aiohttp.ClientSession(timeout=TIMEOUT, headers=HEADERS) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.debug(f"🏏 cricket header API HTTP {resp.status}")
                        return []
                    data = await resp.json(content_type=None)
        except Exception as exc:
            logger.debug(f"🏏 cricket header API failed: {exc}")
            return []

        out: List[Dict] = []
        for sport in (data.get("sports") or []):
            for lg in (sport.get("leagues") or []):
                series = lg.get("name") or lg.get("abbreviation") or ""
                for ev in (lg.get("events") or []):
                    try:
                        # Status lives either in fullStatus.type or flat.
                        fst = ((ev.get("fullStatus") or {}).get("type") or {})
                        state = (fst.get("state")
                                 or ev.get("status") or "pre").lower()
                        completed = fst.get("completed", False)
                        if state == "post" or completed or state in ("final",):
                            status = "final"
                        elif state == "in":
                            status = "live"
                        else:
                            status = "upcoming"
                        comps = ev.get("competitors") or []
                        if len(comps) < 2:
                            continue
                        h, a = comps[0], comps[1]
                        date_iso = ev.get("date", "")
                        try:
                            dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
                            date_str = dt.strftime("%a %d %b")
                        except Exception:
                            date_str = date_iso[:10]
                        out.append({
                            "home_team":  h.get("displayName") or h.get("name", ""),
                            "away_team":  a.get("displayName") or a.get("name", ""),
                            "home_score": str(h.get("score", "") or ""),
                            "away_score": str(a.get("score", "") or ""),
                            "status":     status,
                            "date_iso":   date_iso,
                            "date_str":   date_str,
                            "clock":      fst.get("shortDetail", "") or "",
                            "home_logo":  h.get("logo", "") or h.get("logoDark", ""),
                            "away_logo":  a.get("logo", "") or a.get("logoDark", ""),
                            "home_innings": [], "away_innings": [],
                            "note":   ev.get("summary", "") or fst.get("detail", ""),
                            "format": series,
                        })
                    except Exception:
                        continue
        return out

    async def _cricket_via_rss(self) -> List[Dict]:
        """
        Last-resort cricket source: static.cricinfo.com's plain-XML live
        scores RSS. Only carries LIVE matches, as one-line titles like
        'Pakistan 234/5 v Australia 230 *' — thin, but never blocked and
        means the widget degrades to something rather than nothing.
        """
        url = "https://static.cricinfo.com/rss/livescores.xml"
        try:
            async with aiohttp.ClientSession(timeout=TIMEOUT, headers=HEADERS) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return []
                    body = await resp.text()
        except Exception:
            return []

        import re as _re
        out: List[Dict] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        for m in _re.finditer(r"<item>.*?<title>(.*?)</title>", body, _re.S):
            title = m.group(1).strip()
            if not title or " v " not in title:
                continue
            left, _, right = title.partition(" v ")

            def _split_team(side: str) -> Tuple[str, str]:
                side = side.replace("*", "").strip()
                # "Pakistan 234/5 & 180" → name + score tail
                sm = _re.match(r"^(.*?)(\d[\d/&\s-]*)$", side)
                if sm and sm.group(1).strip():
                    return sm.group(1).strip(), sm.group(2).strip()
                return side, ""

            h_name, h_score = _split_team(left)
            a_name, a_score = _split_team(right)
            out.append({
                "home_team": h_name, "away_team": a_name,
                "home_score": h_score, "away_score": a_score,
                "status": "live", "date_iso": now_iso, "date_str": "Today",
                "clock": "", "home_logo": "", "away_logo": "",
                "home_innings": [], "away_innings": [],
                "note": title, "format": "",
            })
        return out

    async def _cricket_via_espn(self) -> List[Dict]:
        """
        Fallback cricket source via the ESPN site API. Tries a set of cricket
        competition IDs (the site API needs an ID; these cover the leagues plus
        common international class IDs) and merges any events that come back.
        """
        candidate_ids = [
            ("8048", ""), ("8047", ""),      # international class aggregates (best-effort)
            ("8677", "IPL"), ("8810", "PSL"), ("5181", "BBL"),
        ]
        seen: set = set()
        out: List[Dict] = []
        for lg_id, fmt in candidate_ids:
            url = f"{ESPN_BASE}/cricket/{lg_id}/scoreboard"
            try:
                async with aiohttp.ClientSession(timeout=TIMEOUT, headers=HEADERS) as session:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
            except Exception:
                continue
            for event in data.get("events", []):
                g = self._parse_event(event, sport="cricket")
                if not g:
                    continue
                key = (g.get("date_iso", "")[:10], g.get("home_team", ""), g.get("away_team", ""))
                if key in seen:
                    continue
                seen.add(key)
                if fmt:
                    g["format"] = fmt
                out.append(g)
        return out

    def _parse_cricinfo_match(self, m: dict) -> Optional[Dict]:
        """Parse one ESPNcricinfo match object into our cricket game dict."""
        try:
            teams = m.get("teams", []) or []
            if len(teams) < 2:
                return None

            def _tname(t: dict) -> str:
                tt = t.get("team", {}) or {}
                return tt.get("name") or tt.get("longName") or tt.get("abbreviation") or "TBD"

            def _tlogo(t: dict) -> str:
                tt = t.get("team", {}) or {}
                img = tt.get("image", {}) or {}
                return img.get("url", "") or ""

            def _tscore(t: dict) -> str:
                s = t.get("score")
                if isinstance(s, str) and s.strip():
                    return s.strip()
                if isinstance(s, dict):
                    return (s.get("displayValue") or s.get("text") or "").strip()
                # Build from scoreInfo if present
                si = t.get("scoreInfo") or {}
                if isinstance(si, dict) and si.get("runs") is not None:
                    runs, wkts = si.get("runs"), si.get("wickets")
                    overs = si.get("overs")
                    base = f"{runs}/{wkts}" if wkts not in (None, 10) else f"{runs}"
                    return f"{base} ({overs})" if overs else base
                return ""

            home, away = teams[0], teams[1]
            state = (m.get("state") or m.get("status") or "").upper()
            status_text = m.get("statusText") or m.get("statusEnum") or m.get("status") or ""

            if state in ("LIVE",) or m.get("isLive"):
                status = "live"
            elif state in ("POST", "RESULT", "COMPLETE", "COMPLETED", "FINISHED") or m.get("isComplete"):
                status = "final"
            elif state in ("PRE", "UPCOMING", "SCHEDULED"):
                status = "upcoming"
            else:
                status = "upcoming"

            fmt_raw = str(m.get("format") or m.get("internationalClassName") or m.get("generalClassCard") or "").upper()
            if "TEST" in fmt_raw:        fmt = "Test"
            elif "ODI" in fmt_raw or fmt_raw == "ODM":  fmt = "ODI"
            elif "T20" in fmt_raw:       fmt = "T20I"
            elif "IPL" in fmt_raw:       fmt = "IPL"
            elif "PSL" in fmt_raw:       fmt = "PSL"
            elif fmt_raw:                fmt = fmt_raw.title()
            else:                        fmt = ""

            start = m.get("startDate") or m.get("startTime") or ""
            try:
                dt = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
                date_str = dt.strftime("%a %d %b, %H:%M UTC")
                date_iso = dt.isoformat()
            except Exception:
                date_str = str(start)[:10]
                date_iso = str(start)

            ground = m.get("ground", {}) or {}
            hs, as_ = _tscore(home), _tscore(away)
            return {
                "home_team":   _tname(home), "away_team":  _tname(away),
                "home_score":  hs or "-",    "away_score": as_ or "-",
                "status":      status,
                "date_str":    date_str,     "date_iso":   date_iso,
                "clock":       "",
                "venue":       ground.get("name", "") if isinstance(ground, dict) else "",
                "home_color":  "",           "away_color": "",
                "home_logo":   _tlogo(home), "away_logo":  _tlogo(away),
                "home_innings": [{"display": hs}] if hs else [],
                "away_innings": [{"display": as_}] if as_ else [],
                "note":        status_text,
                "format":      fmt,
            }
        except Exception:
            return None

    # ── Individual / event sports (F1, tennis, UFC/MMA, boxing) ────────────────
    def _parse_individual_event(self, event: Dict, sport: str) -> Optional[Dict]:
        """
        Parse ESPN events for sports that are NOT team-vs-team.
        Returns a typed dict (type=f1|fight|tennis) plus home_team/away_team
        (the two participants) so generic UI paths still work.
        """
        try:
            comps = event.get("competitions", []) or []
            comp  = comps[0] if comps else {}
            status_obj = event.get("status") or comp.get("status", {}) or {}
            st = status_obj.get("type", {}) or {}
            state     = st.get("state", "pre")
            completed = st.get("completed", False)
            detail    = st.get("shortDetail", "") or st.get("detail", "") or st.get("description", "")

            if state == "post" or completed:
                status = "final"
            elif state == "in":
                status = "live"
            else:
                status = "upcoming"

            date_iso = event.get("date", "")
            try:
                dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
                date_str = dt.strftime("%a %d %b, %H:%M UTC")
            except Exception:
                date_str = date_iso[:10] if date_iso else ""

            ev_name = event.get("name") or event.get("shortName") or ""

            # ── Formula 1 — a race weekend ──────────────────────────────────
            if sport == "racing":
                venue = (comp.get("venue", {}) or {}).get("fullName", "")
                drivers = comp.get("competitors", []) or []
                results = []
                if status == "final":
                    ranked = sorted(drivers, key=lambda c: c.get("order", 999))[:3]
                    for c in ranked:
                        ath = c.get("athlete", {}) or {}
                        results.append({
                            "pos":    c.get("order", ""),
                            "driver": ath.get("shortName") or ath.get("displayName", ""),
                            "team":   (c.get("team", {}) or {}).get("displayName", ""),
                        })
                return {
                    "type": "f1", "status": status, "date_iso": date_iso, "date_str": date_str,
                    "name": ev_name or "Grand Prix", "circuit": venue, "detail": detail,
                    "results": results,
                    "home_team": ev_name or "Grand Prix", "away_team": "",
                    "home_score": "", "away_score": "",
                }

            # ── UFC / MMA / Boxing — a fight card ───────────────────────────
            # An event is a whole CARD; event["competitions"] is the list of
            # bouts. ESPN lists prelims first and the headliner LAST, so the
            # old code (which read competitions[0]) showed a random prelim
            # instead of the main event. Surface the headline bout per card.
            if sport in ("mma", "boxing"):
                return self._parse_fight_card(event, sport, ev_name)

            # ── Tennis — a tournament containing many matches ───────────────
            # IMPORTANT: tennis events nest their matches under
            # event["groupings"][i]["competitions"], NOT event["competitions"].
            # Reading the top-level competitions (as the other sports do)
            # returns nothing, which is why every player showed as "TBD".
            if sport == "tennis":
                return self._parse_tennis_matches(event, ev_name, date_iso, date_str)
            return None
        except Exception:
            return None

    # ── Fight-card extraction (MMA / Boxing) ───────────────────────────────────
    @staticmethod
    def _fighter_name(c: Dict) -> str:
        at = c.get("athlete", {}) or {}
        return (at.get("shortName") or at.get("displayName")
                or (c.get("team", {}) or {}).get("displayName", "") or "TBD")

    @staticmethod
    def _fighter_logo(c: Dict) -> str:
        at = c.get("athlete", {}) or {}
        for key in ("headshot", "flag"):
            v = at.get(key)
            if isinstance(v, dict) and v.get("href"):
                return v["href"]
        return ""

    @staticmethod
    def _fighter_record(c: Dict) -> str:
        for r in (c.get("records") or []):
            if r.get("summary"):
                return r["summary"]
        return ""

    def _parse_fight_card(self, event: Dict, sport: str, ev_name: str) -> Optional[Dict]:
        """
        Build one row per fight CARD, surfacing the headline (main-event) bout.

        ESPN orders bouts prelims-first, headliner-last, so the previous
        competitions[0] read showed a prelim. We pick the main event as the
        latest-starting bout (tie-break: last in the list) so the card row
        shows the fight people actually recognise, and include the card's bout
        count in the detail line.
        """
        comps = event.get("competitions", []) or []
        if not comps:
            return None

        # Main event = latest start time, then last index as the tie-break.
        def _start(c):
            return c.get("date") or c.get("startDate") or ""
        main = max(
            enumerate(comps),
            key=lambda pair: (_start(pair[1]), pair[0]),
        )[1]

        cs = main.get("competitors", []) or []
        a, b = (cs[0] if cs else {}), (cs[1] if len(cs) > 1 else {})

        st = (main.get("status") or {}).get("type", {}) or {}
        state = st.get("state", "pre")
        completed = st.get("completed", False)
        detail = st.get("shortDetail") or st.get("detail") or st.get("description", "")
        if state == "post" or completed:
            status = "final"
        elif state == "in":
            status = "live"
        else:
            status = "upcoming"

        date_iso = main.get("date") or event.get("date", "")
        try:
            dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
            date_str = dt.strftime("%a %d %b, %H:%M UTC")
        except Exception:
            date_str = date_iso[:10] if date_iso else ""

        weight = (main.get("type", {}) or {}).get("text", "") \
            or (main.get("type", {}) or {}).get("abbreviation", "") \
            or main.get("note", "") or ""

        bout_count = len(comps)
        card_note = f"Main event · {bout_count} bouts" if bout_count > 1 else "Main event"
        rec1, rec2 = self._fighter_record(a), self._fighter_record(b)

        return {
            "type": "fight", "status": status, "date_iso": date_iso, "date_str": date_str,
            "event_name": ev_name or ("UFC" if sport == "mma" else "Boxing"),
            "fighter1": self._fighter_name(a), "fighter2": self._fighter_name(b),
            "fighter1_logo": self._fighter_logo(a), "fighter2_logo": self._fighter_logo(b),
            "fighter1_record": rec1, "fighter2_record": rec2,
            "weight": weight, "detail": detail or card_note, "card_note": card_note,
            "home_team": self._fighter_name(a), "away_team": self._fighter_name(b),
            "home_score": rec1, "away_score": rec2,
        }

    # ── Tennis match extraction (groupings-aware) ──────────────────────────────
    @staticmethod
    def _tennis_name(c: Dict) -> str:
        """Resolve a tennis competitor's display name across feed shapes.

        Singles put the player under `athlete`; doubles put a pair under
        `athletes`; some feeds only carry a `team`. We try each in turn so a
        determined match never renders as TBD.
        """
        at = c.get("athlete") or {}
        name = at.get("shortName") or at.get("displayName") or at.get("fullName")
        if name:
            return name
        ats = c.get("athletes") or []
        pair = [(x.get("shortName") or x.get("displayName") or "").strip() for x in ats]
        pair = [p for p in pair if p]
        if pair:
            return "/".join(pair)
        team = c.get("team") or {}
        return team.get("displayName") or team.get("abbreviation") or "TBD"

    @staticmethod
    def _tennis_score(c: Dict) -> str:
        ls = c.get("linescores", []) or []
        return " ".join(
            str(x.get("value", "")) for x in ls if x.get("value") is not None
        ).strip()

    def _tennis_match_from_competition(
        self, c: Dict, tournament: str, fallback_date_iso: str
    ) -> Optional[Dict]:
        cs = c.get("competitors", []) or []
        if len(cs) < 2:
            return None
        a, b = cs[0], cs[1]
        p1, p2 = self._tennis_name(a), self._tennis_name(b)
        # Skip undetermined matches (future draws) — they're just TBD vs TBD.
        if p1 == "TBD" and p2 == "TBD":
            return None

        st = (c.get("status") or {}).get("type", {}) or {}
        state = st.get("state", "pre")
        completed = st.get("completed", False)
        detail = st.get("shortDetail") or st.get("detail") or st.get("description", "")
        if state == "post" or completed:
            status = "final"
        elif state == "in":
            status = "live"
        else:
            status = "upcoming"

        date_iso = c.get("date") or fallback_date_iso or ""
        try:
            dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
            date_str = dt.strftime("%a %d %b, %H:%M UTC")
        except Exception:
            date_str = date_iso[:10] if date_iso else ""

        # Round / draw label (e.g. "Men's Singles - 2nd Round")
        round_label = (c.get("type", {}) or {}).get("text", "") or c.get("note", "")
        return {
            "type": "tennis", "status": status, "date_iso": date_iso, "date_str": date_str,
            "tournament": tournament, "round": round_label,
            "player1": p1, "player2": p2,
            "score1": self._tennis_score(a), "score2": self._tennis_score(b),
            "detail": detail,
            "home_team": p1, "away_team": p2,
            "home_score": "", "away_score": "",
        }

    def _parse_tennis_matches(
        self, event: Dict, ev_name: str, date_iso: str, date_str: str
    ) -> List[Dict]:
        """
        Extract individual tennis matches from a tournament event.

        Matches live under event["groupings"][*]["competitions"]; some feeds
        also expose event["competitions"] directly, so we read both. If a
        tournament has no determined matches yet (draw not made), we emit a
        single placeholder so the upcoming tournament still appears on the
        schedule — without spamming bogus "TBD vs TBD" rows for every slot.
        """
        tour = ev_name or event.get("name") or event.get("shortName") or "Tennis"
        comps: List[Dict] = []
        for grp in (event.get("groupings") or []):
            comps.extend(grp.get("competitions") or [])
        comps.extend(event.get("competitions") or [])

        matches: List[Dict] = []
        for c in comps:
            m = self._tennis_match_from_competition(c, tour, date_iso)
            if m:
                matches.append(m)

        if matches:
            return matches

        # Placeholder for an upcoming tournament with no determined matches.
        return [{
            "type": "tennis", "status": "upcoming",
            "date_iso": date_iso, "date_str": date_str,
            "tournament": tour, "round": "",
            "player1": "", "player2": "", "score1": "", "score2": "",
            "detail": "Draw not yet available",
            "home_team": tour, "away_team": "",
            "home_score": "", "away_score": "",
        }]

    async def get_standings(self, league_key: str) -> Dict[str, Any]:
        """
        Get current league standings/table.

        Strategy (in order):
        1. ESPN widget API  (site.web.api.espn.com/apis/v2) — most reliable
        2. ESPN base API    (site.api.espn.com/apis/site/v2) — original endpoint
        3. Scoreboard records — build standings from team records embedded in
           the working scoreboard response (always available)
        """
        league = LEAGUES.get(league_key)
        if not league:
            return {"success": False, "error": f"Unknown league: {league_key}"}

        sport, lg = league["sport"], league["league"]

        # ── Attempt 1 & 2: dedicated standings endpoints ──────────────────
        urls = [
            f"https://site.web.api.espn.com/apis/v2/sports/{sport}/{lg}/standings",
            f"{ESPN_BASE}/{sport}/{lg}/standings",
        ]
        for url in urls:
            try:
                async with aiohttp.ClientSession(
                    timeout=TIMEOUT, headers=HEADERS
                ) as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            standings, conferences = self._parse_standings(data)
                            if standings:
                                print(f"✅ Standings from: {url}")
                                return {
                                    "success":     True,
                                    "league":      league["name"],
                                    "sport":       sport,
                                    "standings":   standings,
                                    "conferences": conferences,
                                }
            except Exception:
                pass

        # ── Attempt 3: derive from scoreboard team records ────────────────
        print(f"⚠️  Standings endpoints failed for {league_key} — building from scoreboard")
        try:
            standings = await self._standings_from_scoreboard(league_key)
            if standings:
                return {
                    "success":     True,
                    "league":      league["name"],
                    "sport":       sport,
                    "standings":   standings,
                    "conferences": [],
                    "source":      "derived",
                }
        except Exception as e:
            print(f"⚠️  Scoreboard standings fallback error: {e}")

        return {
            "success": True, "league": league["name"],
            "sport": sport, "standings": [], "conferences": [],
        }

    async def _standings_from_scoreboard(self, league_key: str) -> List[Dict]:
        """
        Build standings from team season records embedded in the scoreboard API.
        ESPN includes each team's full season record (W/D/L, pts, rank, GD)
        in every scoreboard event — this endpoint always works.
        """
        league = LEAGUES[league_key]
        url = f"{ESPN_BASE}/{league['sport']}/{league['league']}/scoreboard"

        async with aiohttp.ClientSession(
            timeout=TIMEOUT, headers=HEADERS
        ) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

        teams: dict = {}
        for event in data.get("events", []):
            for comp in event.get("competitions", []):
                for competitor in comp.get("competitors", []):
                    team_name = competitor.get("team", {}).get("displayName", "")
                    if not team_name or team_name in teams:
                        continue

                    records = competitor.get("records", [])
                    # Prefer "overall" record; fall back to first available
                    overall = next(
                        (r for r in records if r.get("name") == "overall"),
                        records[0] if records else None,
                    )
                    if not overall:
                        continue

                    stats = {
                        s["name"]: s.get("value", 0)
                        for s in overall.get("stats", [])
                    }
                    # summary is "W-D-L" (soccer) or "W-L" (basketball)
                    summary = overall.get("summary", "0-0")
                    parts = [int(x) for x in summary.split("-") if x.isdigit()]

                    wins   = int(stats.get("wins",   parts[0] if len(parts) > 0 else 0))
                    draws  = int(stats.get("ties",   stats.get("draws",
                                          parts[1] if len(parts) > 2 else 0)))
                    losses = int(stats.get("losses", parts[-1] if parts else 0))
                    points = int(stats.get("points", stats.get("totalPoints",
                                          wins * 3 + draws)))  # calc if missing
                    rank   = int(stats.get("rank",   stats.get("playoffSeed", 0)))
                    gd     = int(stats.get("goalDifferential",
                                          stats.get("pointDifferential", 0)))
                    played = int(stats.get("gamesPlayed", wins + draws + losses))

                    teams[team_name] = {
                        "position":  rank,
                        "team":      team_name,
                        "played":    played,
                        "wins":      wins,
                        "draws":     draws,
                        "losses":    losses,
                        "goal_diff": gd,
                        "points":    points,
                    }

        if not teams:
            return []

        result = list(teams.values())
        # Sort by explicit rank if available, otherwise by points then GD
        if any(t["position"] > 0 for t in result):
            result.sort(key=lambda x: (x["position"] if x["position"] > 0 else 999))
        else:
            result.sort(key=lambda x: (-x["points"], -x["goal_diff"], x["team"]))
            for i, entry in enumerate(result):
                entry["position"] = i + 1

        return result

    # Domestic leagues whose clubs may compete in UEFA competitions.
    # ANY team found in one of these leagues automatically gets a CL lookup —
    # no whitelist of individual teams needed.
    _EUROPEAN_DOMESTIC_LEAGUES = {
        "premier_league", "la_liga", "serie_a", "bundesliga", "ligue_1",
        "efl_championship",   # some clubs reach Europa League qualifiers
    }

    async def search_team(
        self,
        query: str,
        league_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for a specific team's results.
        Strategy:
          1. Resolve team ID via ESPN teams list → fetch full season schedule
          2. For any European domestic league, also fetch CL games and merge
          3. If domestic lookup fails entirely, try CL/EL rosters directly
          4. Fallback: filter current scoreboard (legacy, narrow window)
        """
        if not league_key:
            league_key = self.detect_league(query)
        if not league_key:
            league_key = "premier_league"

        league = LEAGUES.get(league_key)
        if not league:
            return {"success": False, "error": f"Unknown league: {league_key}"}

        team_tokens = self._extract_team_tokens(query)

        # ── Strategy 1: full season schedule via team ID ──────────────────────
        team_id, team_display = await self._find_team_id(league, team_tokens)
        if team_id:
            games = await self._fetch_team_schedule(league, team_id)
            if games:
                # For ANY European domestic league, also fetch CL games.
                # _fetch_cl_games_for_team uses the team name (not a whitelist)
                # and just returns [] if the team isn't in the CL roster.
                cl_games: List[Dict] = []
                if league_key in self._EUROPEAN_DOMESTIC_LEAGUES:
                    cl_games = await self._fetch_cl_games_for_team(
                        team_tokens, team_display
                    )

                if cl_games:
                    merged = self._merge_team_games(games, cl_games)
                    return {
                        "success":    True,
                        "league":     f"{league['name']} + Champions League",
                        "league_key": league_key,
                        "team_query": team_display or team_tokens.title(),
                        "team_raw":   team_tokens,   # original user tokens for W/L/D matching
                        "games":      merged,
                    }
                return {
                    "success":    True,
                    "league":     league["name"],
                    "league_key": league_key,
                    "team_query": team_display or team_tokens.title(),
                    "team_raw":   team_tokens,
                    "games":      games,
                }

        # ── Strategy 1b: domestic lookup failed — try CL/EL rosters directly ─
        if league_key in self._EUROPEAN_DOMESTIC_LEAGUES:
            for european_key in ("champions_league",):
                eu_league = LEAGUES[european_key]
                eu_team_id, eu_team_display = await self._find_team_id(
                    eu_league, team_tokens
                )
                if eu_team_id:
                    games = await self._fetch_team_schedule(eu_league, eu_team_id)
                    if games:
                        return {
                            "success":    True,
                            "league":     eu_league["name"],
                            "league_key": european_key,
                            "team_query": eu_team_display or team_tokens.title(),
                            "team_raw":   team_tokens,
                            "games":      games,
                        }

        # ── Strategy 2: fallback — filter current scoreboard ─────────────────
        scores = await self.get_scores(league_key, limit=30)
        if not scores.get("success"):
            return {"success": True, "league": league["name"],
                    "team_query": team_tokens, "team_raw": team_tokens, "games": []}

        def _matches(name: str) -> bool:
            n = name.lower()
            if team_tokens and team_tokens in n:
                return True
            for word in n.split():
                if len(word) >= 4 and word in team_tokens:
                    return True
            return False

        team_games = [
            g for g in scores["games"]
            if _matches(g.get("home_team", "")) or _matches(g.get("away_team", ""))
        ]
        return {
            "success":    True,
            "league":     scores["league"],
            "league_key": league_key,
            "team_query": team_tokens,
            "team_raw":   team_tokens,
            "games":      team_games,
        }

    async def _fetch_cl_games_for_team(
        self, team_tokens: str, team_display: Optional[str]
    ) -> List[Dict]:
        """
        Look up the team in the Champions League roster and fetch their schedule.
        Returns [] if the team isn't in the CL this season (no whitelist check).
        Filters the returned games to only those the team actually participated in
        (ESPN's CL bracket endpoint can include full-bracket games).
        """
        cl_league = LEAGUES["champions_league"]
        try:
            cl_team_id, cl_team_display = await self._find_team_id(
                cl_league, team_display or team_tokens
            )
            if not cl_team_id:
                return []

            games = await self._fetch_team_schedule(cl_league, cl_team_id)
            if not games:
                return []

            # Build a set of search terms to verify each game involves this team.
            # We use shortDisplayName in game dicts so "PSG" must match even when
            # team_display is "Paris Saint-Germain".
            raw_terms: List[str] = []
            for t in filter(None, [team_tokens, team_display, cl_team_display]):
                raw_terms.append(t.lower())

            def _team_in_game(g: Dict) -> bool:
                home = g.get("home_team", "").lower()
                away = g.get("away_team", "").lower()
                for term in raw_terms:
                    words = [w for w in term.split() if len(w) >= 3]
                    for name in (home, away):
                        if term in name or name in term:
                            return True
                        if any(w in name for w in words):
                            return True
                return False

            return [g for g in games if _team_in_game(g)]
        except Exception:
            return []

    def _merge_team_games(
        self, domestic: List[Dict], cl: List[Dict]
    ) -> List[Dict]:
        """
        Merge domestic + CL games, de-duplicate by (date, home, away),
        keep chronological order with most recent finished last.
        """
        seen: set = set()
        merged: List[Dict] = []
        for g in domestic + cl:
            key = (g.get("date_iso", ""), g.get("home_team", ""), g.get("away_team", ""))
            if key not in seen:
                seen.add(key)
                merged.append(sanitize_game_status(g, "soccer"))

        finished = sorted(
            [g for g in merged if g["status"] == "final"],
            key=lambda x: x.get("date_iso", x["date_str"]),
        )
        # Off-season guard: cap stale finals but keep the most recent (dated)
        # result so the team card shows "last match" rather than months of
        # history presented as if current.
        finished = cap_finished_age(finished)
        live     = [g for g in merged if g["status"] == "live"]
        upcoming = sorted(
            [g for g in merged if g["status"] == "upcoming"],
            key=lambda x: x.get("date_iso", x["date_str"]),
        )[:6]
        return live + finished[-12:] + upcoming

    # ── Team lookup helpers ─────────────────────────────────────────────────

    def _extract_team_tokens(self, query: str) -> str:
        """Strip noise words from a query to isolate the team name."""
        import re as _re
        _strip = [
            "results", "scores", "score", "fixtures", "fixture", "latest",
            "recent", "match", "matches", "game", "games", "how did",
            "how are", "what are", "what were", "did", "play", "played",
            "win", "won", "lose", "lost", "draw", "drew", "today",
            "yesterday", "this week", "last week", "tonight", "show me",
            "tell me", "the", "vs", "fc", "premier league", "champions league",
            "la liga", "serie a", "bundesliga", "ligue 1", "epl", "ucl",
            "nba", "nfl", "nhl", "mlb",
        ]
        t = query.lower()
        for sw in _strip:
            t = t.replace(sw, " ")
        t = _re.sub(r"[^a-z0-9\s]", " ", t)
        return " ".join(t.split()).strip()

    async def _find_team_id(
        self, league: Dict, team_tokens: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve ESPN team ID for a query string.
        Returns (team_id, display_name) or (None, None).
        """
        url = f"{ESPN_BASE}/{league['sport']}/{league['league']}/teams"
        try:
            async with aiohttp.ClientSession(
                timeout=TIMEOUT, headers=HEADERS
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return None, None
                    data = await resp.json()

            # ESPN wraps teams in sports[0].leagues[0].teams
            raw_teams = (
                data.get("sports", [{}])[0]
                    .get("leagues", [{}])[0]
                    .get("teams", [])
            )
            tq = team_tokens.lower()
            tq_words = [w for w in tq.split() if len(w) >= 4]

            # Pass 1 — exact / substring match on any name field
            for obj in raw_teams:
                t = obj.get("team", {})
                candidates = [
                    t.get("displayName", "").lower(),
                    t.get("shortDisplayName", "").lower(),
                    t.get("name", "").lower(),
                    t.get("abbreviation", "").lower(),
                ]
                if any(tq in c or c in tq for c in candidates if c):
                    return t.get("id"), t.get("displayName")

            # Pass 2 — token overlap (e.g. "Bayern" matches "FC Bayern München")
            for obj in raw_teams:
                t = obj.get("team", {})
                full = t.get("displayName", "").lower()
                if any(w in full for w in tq_words):
                    return t.get("id"), t.get("displayName")

        except Exception as e:
            print(f"⚠️  _find_team_id error: {e}")
        return None, None

    async def _fetch_team_schedule(
        self, league: Dict, team_id: str
    ) -> List[Dict]:
        """
        Fetch a team's full season schedule from ESPN.
        Returns finished games (most recent first) + live + upcoming (next 4).
        """
        url = (
            f"{ESPN_BASE}/{league['sport']}/{league['league']}"
            f"/teams/{team_id}/schedule"
        )
        try:
            async with aiohttp.ClientSession(
                timeout=TIMEOUT, headers=HEADERS
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()

            games = []
            sport_name = league.get("sport", "")
            for event in data.get("events", []):
                g = self._parse_event(event, sport=sport_name)
                if g:
                    games.append(g)

            # Off-season reach: US leagues' /schedule defaults to the regular
            # season, which is empty in the summer. If nothing upcoming came
            # back, also pull PRESEASON (seasontype=1) so October preseason
            # fixtures show in July instead of an empty card.
            if not any(g["status"] == "upcoming" for g in games) \
                    and sport_name in ("basketball", "football", "hockey", "baseball"):
                try:
                    async with aiohttp.ClientSession(
                        timeout=TIMEOUT, headers=HEADERS
                    ) as session:
                        async with session.get(f"{url}?seasontype=1") as resp:
                            if resp.status == 200:
                                pre = await resp.json()
                                for event in pre.get("events", []):
                                    g = self._parse_event(event, sport=sport_name)
                                    if g:
                                        games.append(g)
                except Exception:
                    pass

            games = [sanitize_game_status(g, sport_name) for g in games]

            # Sort by ISO timestamp for correct chronological order
            finished = sorted(
                [g for g in games if g["status"] == "final"],
                key=lambda x: x.get("date_iso", x["date_str"]),
            )
            finished = cap_finished_age(finished)
            live     = [g for g in games if g["status"] == "live"]
            upcoming = sorted(
                [g for g in games if g["status"] == "upcoming"],
                key=lambda x: x.get("date_iso", x["date_str"]),
            )[:6]

            # Return recent finished games + any live + next 6 upcoming
            return live + finished[-10:] + upcoming
        except Exception as e:
            print(f"⚠️  _fetch_team_schedule error: {e}")
            return []

    def detect_league(self, query: str) -> Optional[str]:
        """Detect league from natural language."""
        q = query.lower()
        for league_key, aliases in LEAGUE_ALIASES.items():
            if any(alias in q for alias in aliases):
                return league_key
        return None

    # ── Formatters ──────────────────────────────────────────────────────────

    def format_scores(self, data: Dict[str, Any]) -> str:
        """
        Format scores using the UI's native rendering conventions:
        - ALL CAPS lines  → .sec-hd purple header
        - * item          → .bi bullet list row
        - [text]          → .otag styled tag

        query_type in data drives what section is emphasised:
          "upcoming" → upcoming fixtures first; if none, says so explicitly
          "results"  → recent results only (no upcoming)
          None       → results then upcoming (default)
        """
        if not data.get("success"):
            return f"Could not get scores: {data.get('error', 'unknown error')}"

        games = data.get("games", [])
        if not games:
            team_q = data.get("team_query", "")
            league = data.get("league", "this league")
            if team_q:
                return f"No recent games found for '{team_q}' in {league}."
            return f"No recent games found for {league}."

        league      = data.get("league", "")
        team_q      = data.get("team_query", "")
        team_raw    = data.get("team_raw", "")
        query_type  = data.get("query_type")   # "upcoming" | "results" | None

        header = f"{team_q.upper()} — {league.upper()}" if team_q else f"{league.upper()} RESULTS & FIXTURES"
        lines  = [header, ""]

        finished = [g for g in games if g["status"] == "final"]
        live     = [g for g in games if g["status"] == "live"]
        upcoming = [g for g in games if g["status"] == "upcoming"]

        # ── Live (always shown first regardless of intent) ────────────────────
        if live:
            lines.append("LIVE NOW:")
            for g in live:
                lines.append(
                    f"* {g['home_team']} {g['home_score']} — {g['away_score']} {g['away_team']}  [{g['clock']}]"
                )
            lines.append("")

        def _render_results(limit: int = 12) -> None:
            if not finished:
                return
            lines.append("RECENT RESULTS:")
            for g in finished[-limit:]:
                indicator = self._result_indicator(g, team_q, team_raw)
                prefix    = f"**{indicator}** " if indicator else ""
                date      = g.get("date_str", "")
                date_tag  = f"  [{date}]" if date else ""
                lines.append(
                    f"* {prefix}{g['home_team']} {g['home_score']} — {g['away_score']} {g['away_team']}{date_tag}"
                )
            lines.append("")

        def _render_upcoming(limit: int = 4) -> None:
            lines.append("UPCOMING FIXTURES:")
            if upcoming:
                for g in upcoming[:limit]:
                    date = g.get("date_str", "")
                    lines.append(f"* {g['home_team']} vs {g['away_team']}  [{date}]")
            else:
                ctx = f" in {league}" if league else ""
                lines.append(
                    f"  No upcoming fixtures scheduled{ctx}."
                    + (" The season may be on a break or has ended." if team_q else "")
                )
            lines.append("")

        # ── Render sections according to intent ───────────────────────────────
        if query_type == "upcoming":
            # Lead with upcoming; show last 3 results as context below
            _render_upcoming(limit=4)
            if finished:
                _render_results(limit=3)

        elif query_type == "results":
            # Results only — skip upcoming section
            display_count = 12 if team_q else 6
            _render_results(limit=display_count)

        else:
            # Default: results then upcoming
            display_count = 12 if team_q else 6
            _render_results(limit=display_count)
            if upcoming:
                _render_upcoming(limit=4)

        return "\n".join(lines).strip()

    def format_standings(self, data: Dict[str, Any]) -> str:
        """
        Format league table as a Markdown pipe table.
        The UI's fmt() converts | col | col | blocks into styled <table> elements.

        Always shows ALL teams (no cap).
        - Soccer       → # | Team | GP | W | D | L | GD | Pts
        - NBA/NHL/NFL  → Two tables, one per conference (East / West)
        - Other        → # | Team | GP | W | L
        """
        if not data.get("success"):
            return f"Could not get standings: {data.get('error', 'unknown error')}"

        standings = data.get("standings", [])
        if not standings:
            return "No standings available."

        league      = data.get("league", "")
        sport       = data.get("sport", "")
        conferences = data.get("conferences", [])

        # Soccer uses W-D-L + points; all others (basketball, hockey, etc.) use W-L
        is_soccer = (sport == "soccer") or any(
            e.get("draws", 0) > 0 for e in standings
        )

        def _table_rows(entries: List[Dict], soccer: bool) -> List[str]:
            if soccer:
                rows = [
                    "| # | Team | GP | W | D | L | GD | Pts |",
                    "|---|------|----|---|---|---|----|----|",
                ]
                for e in entries:
                    gd = e.get("goal_diff", 0)
                    gd_str = f"+{gd}" if gd >= 0 else str(gd)
                    rows.append(
                        f"| {e.get('position','')} | {e.get('team','?')} "
                        f"| {e.get('played',0)} | {e.get('wins',0)} "
                        f"| {e.get('draws',0)} | {e.get('losses',0)} "
                        f"| {gd_str} | {e.get('points',0)} |"
                    )
            else:
                rows = [
                    "| # | Team | GP | W | L |",
                    "|---|------|----|---|---|",
                ]
                for e in entries:
                    rows.append(
                        f"| {e.get('position','')} | {e.get('team','?')} "
                        f"| {e.get('played',0)} | {e.get('wins',0)} "
                        f"| {e.get('losses',0)} |"
                    )
            return rows

        # ── Multi-conference leagues (NBA, NFL, NHL) ──────────────────────────
        if conferences:
            title_line = f"{league.upper()} STANDINGS:"
            output = [title_line, ""]
            for conf in conferences:
                conf_name = conf.get("name", "")
                conf_teams = conf.get("teams", [])
                if not conf_teams:
                    continue
                output.append(f"{conf_name.upper()}:")
                output.extend(_table_rows(conf_teams, is_soccer))
                output.append("")
            return "\n".join(output).strip()

        # ── Single-group (soccer / flat league) ──────────────────────────────
        title_line = f"{league.upper()} TABLE:" if is_soccer else f"{league.upper()} STANDINGS:"
        output = [title_line, ""]
        output.extend(_table_rows(standings, is_soccer))
        return "\n".join(output)

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

    def _parse_event(self, event: Dict, sport: str = "") -> Optional[Dict]:
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

            def _score(val) -> str:
                """ESPN score can be a plain string/number or a reference dict."""
                if isinstance(val, dict):
                    return str(val.get("displayValue") or int(val.get("value", 0)))
                return str(val) if val not in (None, "", "null") else "-"

            # ── Status detection ──────────────────────────────────────────────
            # ESPN places the status at the event level for scoreboards but
            # at competition level for team schedule endpoints — try both.
            status_obj  = event.get("status") or comp.get("status", {})
            status_type = status_obj.get("type", {})
            type_id     = str(status_type.get("id", ""))   # "1"=pre "2"=live "3"=post
            state       = status_type.get("state", "pre")  # pre | in | post
            completed   = status_type.get("completed", False)
            type_name   = status_type.get("name", "").upper()
            short_detail = status_type.get("shortDetail", "")

            if (state == "post" or completed or type_id == "3"
                    or "FINAL" in type_name or "POST" in type_name
                    or "HALFTIME" in type_name):
                status = "final"
            elif (state == "in" or type_id == "2"
                    or "PROGRESS" in type_name or type_name in ("IN", "LIVE")):
                status = "live"
            else:
                status = "upcoming"

            # ── Date ─────────────────────────────────────────────────────────
            date_iso = event.get("date", "")
            try:
                dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
                date_fmt = dt.strftime("%a %d %b, %H:%M UTC")
            except Exception:
                date_fmt = date_iso[:10] if date_iso else ""

            # ── Clock / match detail ──────────────────────────────────────────
            clock = short_detail if status == "live" else ""

            # ── Team logos ───────────────────────────────────────────────────
            cdn_sport = _CDN_SPORT.get(sport, sport)

            def _logo(competitor: dict) -> str:
                team = competitor.get("team", {})
                # 1. Try logos array in the response
                logos = team.get("logos") or []
                for logo in logos:
                    if isinstance(logo, dict) and logo.get("href"):
                        return logo["href"]
                # 2. Fall back to ESPN CDN using team ID
                team_id = team.get("id", "")
                if team_id and cdn_sport:
                    return f"https://a.espncdn.com/i/teamlogos/{cdn_sport}/500/{team_id}.png"
                return ""

            def _color(competitor: dict) -> str:
                return competitor.get("team", {}).get("color") or ""

            game = {
                "home_team":  home.get("team", {}).get("shortDisplayName", "Home"),
                "away_team":  away.get("team", {}).get("shortDisplayName", "Away"),
                "home_score": _score(home.get("score")),
                "away_score": _score(away.get("score")),
                "status":     status,
                "date_str":   date_fmt,
                "date_iso":   date_iso,
                "clock":      clock,
                "venue":      comp.get("venue", {}).get("fullName", ""),
                "home_color": _color(home),
                "away_color": _color(away),
                "home_logo":  _logo(home),
                "away_logo":  _logo(away),
            }

            # ── Cricket-specific enrichment ───────────────────────────────────
            # Cricket scores like "334 & 267/5d" don't fit the football-style
            # "X – Y" template. Capture innings totals per side, the result
            # text ("India won by 295 runs"), and the human-readable status
            # blurb so the UI can render it properly.
            if sport == "cricket":
                # ESPN cricket linescores: per-innings under each competitor
                def _innings(competitor: dict) -> list:
                    linescores = competitor.get("linescores") or []
                    out = []
                    for ls in linescores:
                        runs = ls.get("value") if isinstance(ls.get("value"), (int, float)) else None
                        # displayValue is often the canonical "267/5d" string
                        display = ls.get("displayValue") or (
                            f"{int(runs)}" if runs is not None else ""
                        )
                        wickets = ls.get("wickets")
                        overs = ls.get("overs") or ls.get("oversBowled")
                        out.append({
                            "display": display,
                            "runs": int(runs) if runs is not None else None,
                            "wickets": wickets,
                            "overs": overs,
                            "declared": "d" in (display or "").lower(),
                        })
                    return out

                game["home_innings"] = _innings(home)
                game["away_innings"] = _innings(away)
                # The shortDetail is usually the most readable status:
                #   "India won by 295 runs"          (final)
                #   "Day 4, Tea: 121 & 143-7"        (live)
                #   "Match starts in 2 hours"        (upcoming)
                game["note"] = short_detail or status_type.get("detail", "")

            return game
        except Exception:
            return None

    def _collect_conferences(self, data: Dict) -> List[Dict]:
        """
        Return conference-aware groupings from an ESPN standings payload.
        For soccer (no children): single group with name="".
        For NBA/NFL/NHL:          one group per conference.
        For NFL divisions:        entries aggregated per conference.
        Each item: {"name": <conference_name>, "entries": [raw_espn_entry, ...]}
        """
        children = data.get("children", [])
        if not children:
            # No sub-groups — flat league (soccer)
            entries = data.get("standings", {}).get("entries", [])
            return [{"name": "", "entries": entries}] if entries else []

        groups: List[Dict] = []
        for child in children:
            conf_name = child.get("name", "")
            # Try direct entries
            direct = child.get("standings", {}).get("entries", [])
            if direct:
                groups.append({"name": conf_name, "entries": direct})
                continue
            # One more level deep (e.g. NFL conference → division → entries)
            combined: List = []
            for sub in child.get("children", []):
                combined.extend(sub.get("standings", {}).get("entries", []))
            if combined:
                groups.append({"name": conf_name, "entries": combined})

        # Fallback: collect all entries flat if nothing resolved above
        if not groups:
            flat: List = []
            for child in children:
                flat.extend(child.get("standings", {}).get("entries", []))
            if flat:
                groups.append({"name": "", "entries": flat})
        return groups

    # Keep old name as alias so existing callers don't break
    def _collect_entries(self, node: Dict, depth: int = 0) -> List[Dict]:
        groups = self._collect_conferences(node)
        flat: List[Dict] = []
        for g in groups:
            flat.extend(g["entries"])
        return flat

    def _entries_to_teams(self, raw_entries: List[Dict]) -> List[Dict]:
        """Convert raw ESPN entry list to clean team dicts."""
        teams = []
        for i, group in enumerate(raw_entries):
            team_name = group.get("team", {}).get("displayName", "Unknown")
            stats = {s["name"]: s.get("value", 0) for s in group.get("stats", [])}
            teams.append({
                "position":  int(stats.get("rank", stats.get("playoffSeed", i + 1))),
                "team":      team_name,
                "played":    int(stats.get("gamesPlayed", 0)),
                "wins":      int(stats.get("wins", 0)),
                "draws":     int(stats.get("ties", stats.get("draws", 0))),
                "losses":    int(stats.get("losses", 0)),
                "goal_diff": int(stats.get("pointDifferential",
                                           stats.get("goalDifferential", 0))),
                "points":    int(stats.get("points", stats.get("totalPoints", 0))),
            })
        teams.sort(key=lambda x: x["position"])
        return teams

    def _parse_standings(self, data: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        Parse ESPN standings response.
        Returns:
            (flat_standings, conferences)
            - flat_standings : all teams in one list (sorted by position)
            - conferences    : [{"name": str, "teams": [...]}, ...] — empty list
                               for single-conference / soccer leagues
        """
        groups = self._collect_conferences(data)
        if not groups:
            print(f"⚠️  Standings parser: no entries. Keys: {list(data.keys())}")
            return [], []

        try:
            # Multi-conference (NBA, NFL, NHL)
            if len(groups) > 1:
                conferences = []
                all_teams: List[Dict] = []
                for g in groups:
                    teams = self._entries_to_teams(g["entries"])
                    conferences.append({"name": g["name"], "teams": teams})
                    all_teams.extend(teams)
                return all_teams, conferences

            # Single group (soccer, MLS, etc.)
            teams = self._entries_to_teams(groups[0]["entries"])
            return teams, []

        except Exception as e:
            print(f"⚠️  Standings parse error: {e}")
            return [], []

    def _result_indicator(
        self, game: Dict, team_query: str = "", team_raw: str = ""
    ) -> str:
        """
        Return perspective-aware W/D/L only when we know which team the
        user asked about; otherwise return an empty string.

        Tries both team_query (ESPN display name, e.g. "Paris Saint-Germain")
        AND team_raw (original user tokens, e.g. "psg") so that abbreviated
        names in ESPN game data ("PSG") are caught by the raw tokens fallback.

        Matching is bidirectional + word-overlap so that a long ESPN display
        name ("FC Bayern München") still matches the short game label ("Bayern")
        and vice-versa.
        """
        if not team_query and not team_raw:
            return ""
        try:
            home_name = game.get("home_team", "").lower()
            away_name = game.get("away_team", "").lower()

            # Build a list of search terms to try in order:
            # 1. ESPN display name  (e.g. "paris saint-germain")
            # 2. Raw user tokens    (e.g. "psg") — catches abbreviated game labels
            search_terms: List[str] = []
            if team_query:
                search_terms.append(team_query.lower())
            if team_raw and team_raw.lower() != team_query.lower():
                search_terms.append(team_raw.lower())

            def _matches(name: str) -> bool:
                for tq in search_terms:
                    if tq in name or name in tq:
                        return True
                    tq_words = [w for w in tq.split() if len(w) >= 3]
                    if any(w in name for w in tq_words):
                        return True
                return False

            hs  = int(game.get("home_score", 0))
            as_ = int(game.get("away_score", 0))

            if _matches(home_name):
                return "W" if hs > as_ else ("D" if hs == as_ else "L")
            elif _matches(away_name):
                return "W" if as_ > hs else ("D" if as_ == hs else "L")
        except Exception:
            pass
        return ""