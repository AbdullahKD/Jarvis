#!/usr/bin/env python3
"""
Sports data probe — run this ON YOUR MAC to see exactly what each sports
feed returns, so we can tell whether an empty/stale widget is a Jarvis
parsing bug or the upstream API misbehaving.

Usage (from the Jarvis folder, venv active):
    python tests/sports_probe.py            # probe everything
    python tests/sports_probe.py cricket    # just cricket
    python tests/sports_probe.py nba tennis # specific feeds

Prints per feed: HTTP status, event count, and each event's name, start
date, and raw ESPN state (pre/in/post) BEFORE Jarvis's sanitization —
followed by what Jarvis's parser turns it into.
"""
import asyncio
import json
import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")

from tools.sports import (  # noqa: E402
    SportsTool, LEAGUES, ESPN_BASE, CRICINFO_CURRENT, HEADERS,
)
import aiohttp  # noqa: E402

FEEDS = ["premier_league", "champions_league", "nba", "tennis", "tennis_wta",
         "ufc", "boxing", "f1"]


async def probe_espn(league_key: str) -> None:
    league = LEAGUES.get(league_key)
    if not league:
        print(f"  ?? unknown league {league_key}")
        return
    url = f"{ESPN_BASE}/{league['sport']}/{league['league']}/scoreboard"
    print(f"\n== {league_key} ==\n   GET {url}")
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15), headers=HEADERS
        ) as s:
            async with s.get(url) as resp:
                print(f"   HTTP {resp.status}")
                if resp.status != 200:
                    return
                data = await resp.json()
    except Exception as exc:
        print(f"   FETCH FAILED: {type(exc).__name__}: {exc}")
        return
    events = data.get("events", [])
    print(f"   {len(events)} events")
    for e in events[:8]:
        st = (e.get("status") or {}).get("type", {})
        print(f"   - {e.get('date','?')}  {e.get('name','?')[:60]}"
              f"  [state={st.get('state','?')} completed={st.get('completed')}"
              f" detail={st.get('shortDetail','')!r}]")

    tool = SportsTool()
    parsed = await tool.get_scores(league_key, limit=8)
    if parsed.get("success"):
        print(f"   → Jarvis parse: {len(parsed.get('games', []))} games after guards:")
        for g in parsed.get("games", [])[:8]:
            print(f"     [{g.get('status','?'):8}] {g.get('date_str',''):24}"
                  f" {g.get('home_team','')} vs {g.get('away_team','')}")
    else:
        print(f"   → Jarvis parse FAILED: {parsed.get('error')}")


async def probe_cricket() -> None:
    print(f"\n== cricket ==\n   GET {CRICINFO_CURRENT}")
    cric_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.espncricinfo.com/",
        "Origin": "https://www.espncricinfo.com",
    }
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15), headers=cric_headers
        ) as s:
            async with s.get(CRICINFO_CURRENT) as resp:
                print(f"   HTTP {resp.status}")
                body = await resp.text()
    except Exception as exc:
        print(f"   FETCH FAILED: {type(exc).__name__}: {exc}")
        body = ""
    if body:
        try:
            data = json.loads(body)
            matches = (data.get("matches")
                       or (data.get("content", {}) or {}).get("matches")
                       or data.get("items") or [])
            print(f"   {len(matches)} raw matches; top-level keys: {list(data)[:10]}")
            for m in (matches or [])[:6]:
                print(f"   - state={m.get('state') or m.get('status')}"
                      f" | {str(m.get('title') or m.get('slug') or '')[:70]}")
        except json.JSONDecodeError:
            print(f"   NOT JSON — first 200 chars: {body[:200]!r}")

    tool = SportsTool()
    parsed = await tool.get_cricket_current()
    if parsed.get("success"):
        print(f"   → Jarvis parse: {len(parsed.get('games', []))} games:")
        for g in parsed.get("games", [])[:8]:
            print(f"     [{g.get('status','?'):8}] {g.get('date_str',''):24}"
                  f" {g.get('home_team','')} vs {g.get('away_team','')}"
                  f"  {g.get('note','')[:40]}")
    else:
        print(f"   → Jarvis parse FAILED: {parsed.get('error')}"
              f"  (check the warning lines above for the HTTP status)")


async def main() -> None:
    wanted = [a.lower() for a in sys.argv[1:]] or FEEDS + ["cricket"]
    print(f"sports probe — {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    for key in wanted:
        if key == "cricket":
            await probe_cricket()
        else:
            await probe_espn(key)
    print("\nDone. If a feed shows HTTP 200 with events but Jarvis parses 0, "
          "it's a parsing bug — send this output back to Claude.")


if __name__ == "__main__":
    asyncio.run(main())
