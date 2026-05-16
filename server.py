"""
J.A.R.V.I.S Web Server
FastAPI backend with:
- WebSocket for real-time streaming chat
- HTTP /chat endpoint as fallback
- HTTP /sidebar endpoint for widget data
- Static file serving for the UI
"""

from __future__ import annotations

import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

import asyncio
import json
import time
from pathlib import Path

import aiohttp

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, Form
from fastapi.responses import JSONResponse
import json as _json

class SafeJSONResponse(JSONResponse):
    """JSONResponse that converts sets to lists automatically."""
    def render(self, content) -> bytes:
        def make_serializable(obj):
            if isinstance(obj, set):
                return list(obj)
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [make_serializable(i) for i in obj]
            return obj
        return _json.dumps(
            make_serializable(content),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from orchestrator import JarvisOrchestrator
from agents.finex_agent import FinExAgent
from tools.reminders import ReminderScheduler

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="J.A.R.V.I.S", version="1.0.0")


@app.on_event("startup")
async def startup():
    scheduler = ReminderScheduler(jarvis.reminders)
    scheduler.start()

    # ── Warm the Ollama model so the first user query isn't cold ──
    # We expose the future on `app.state` so the request path can await
    # it before the first chat call. This avoids a user query and the
    # warmup competing for the cold model — instead, the first request
    # blocks until warmup completes (~30–60s once, then never again
    # while keep_alive holds the model resident).
    async def _warmup():
        # Use chat_stream and discard the chunks. This goes through the
        # idle-only timeout profile, so the cold model load can take as
        # long as it needs without tripping a wall-clock cap.
        try:
            t0 = time.time()
            async for _ in jarvis.llm.chat_stream(
                [{"role": "user", "content": "ping"}],
                max_tokens=4,
            ):
                pass
            print(f"🔥 LLM warmup complete in {time.time()-t0:.2f}s")
        except Exception as exc:
            print(f"⚠️  LLM warmup failed: {type(exc).__name__}: {exc}")

    app.state.warmup_task = asyncio.ensure_future(_warmup())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared orchestrator instance
jarvis = JarvisOrchestrator()
finex  = FinExAgent()

UI_DIR = Path(__file__).parent / "ui"


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    """Serve the main UI."""
    return FileResponse(UI_DIR / "index.html")


class ChatRequest(BaseModel):
    message: str
    # The UI tracks conversation history in-memory and sends it with each
    # request so the HTTP /chat endpoint has the same context the WebSocket
    # path gets. Without this, follow-ups like "elaborate" / "give me more
    # detail" have no prior exchange to expand on and the orchestrator
    # has to ask "what would you like me to elaborate on?" — or worse,
    # routes the bare word through the LLM and gets a capability list.
    history: list = []


@app.post("/chat")
async def chat(req: ChatRequest):
    """HTTP chat endpoint — fallback when WebSocket unavailable."""
    response = await jarvis.handle(
        req.message,
        conversation_history=req.history,
    )
    return {
        "success": response.success,
        "message": response.message,
        "latency_ms": response.latency_ms,
    }


@app.get("/sidebar")
async def sidebar():
    """
    Return all sidebar widget data in one call.
    Called on page load and every 60 seconds.
    """
    # Fetch everything in parallel
    async def safe(coro):
        try:
            return await coro
        except Exception as e:
            return {}

    (weather, markets, calendar, spotify, news, tech_news, sports_news,
     world_news, politics_news, science_news, entertainment_news,
     s_pl, s_ucl, s_nba,
     s_cric_test, s_cric_odi, s_cric_t20, s_cric_ipl, s_cric_psl, s_cric_bbl,
     s_rm, prayer, gmail) = await asyncio.gather(
        safe(jarvis.weather.get_current()),
        safe(jarvis.markets.get_all()),
        safe(jarvis.calendar.search_events()),
        safe(jarvis.spotify.get_now_playing()),
        safe(jarvis.news.get_headlines(max_stories=8)),
        safe(jarvis.news.get_headlines(category="technology", max_stories=8, query="artificial intelligence machine learning tech")),
        safe(jarvis.news.get_headlines(category="sports", max_stories=8)),
        safe(jarvis.news.get_headlines(category="world", max_stories=10)),
        safe(jarvis.news.get_headlines(category="world", max_stories=10, query="politics election government parliament minister policy vote")),
        safe(jarvis.news.get_headlines(category="science", max_stories=10)),
        safe(jarvis.news.get_headlines(category="entertainment", max_stories=10)),
        safe(jarvis.sports.get_scores("premier_league", limit=8)),
        safe(jarvis.sports.get_scores("champions_league", limit=6)),
        safe(jarvis.sports.get_scores("nba", limit=6)),
        # Cricket: pulled per-format and merged below into one consolidated
        # `sports_cricket` payload so the UI can render a single section.
        safe(jarvis.sports.get_scores("cricket_test", limit=4)),
        safe(jarvis.sports.get_scores("cricket_odi",  limit=4)),
        safe(jarvis.sports.get_scores("cricket_t20",  limit=4)),
        safe(jarvis.sports.get_scores("cricket_ipl",  limit=6)),
        safe(jarvis.sports.get_scores("cricket_psl",  limit=6)),
        safe(jarvis.sports.get_scores("cricket_bbl",  limit=4)),
        safe(jarvis.sports.search_team("Real Madrid", "la_liga")),
        safe(jarvis.prayer.get_times()),
        safe(jarvis.gmail.get_inbox(max_results=8, query="is:inbox")),
    )

    result = {}
    if weather.get("success"): result["weather"] = weather
    if markets.get("success"): result["markets"] = markets
    if calendar.get("success"): result["calendar"] = {"events": calendar.get("events", []), "connected": not jarvis.calendar.is_mock}
    if spotify.get("success"): result["spotify"] = {
        "track": spotify.get("track",""),
        "artist": spotify.get("artist",""),
        "playing": spotify.get("playing", False),
        "image_url": spotify.get("image_url",""),
        "progress_pct": spotify.get("progress_pct", 0),
    }
    if news.get("success"):
        result["news"] = {"stories": [{"title":s.get("title",""),"description":s.get("description",""),"sources":list(s.get("sources",[]) if not isinstance(s.get("sources",[]),set) else s.get("sources",set())),"url":s.get("url","")} for s in news.get("stories",[])]}
    if tech_news.get("success"):
        result["tech_news"] = {"stories": [{"title":s.get("title",""),"description":s.get("description",""),"sources":list(s.get("sources",[]) if not isinstance(s.get("sources",[]),set) else s.get("sources",set())),"url":s.get("url","")} for s in tech_news.get("stories",[])]}
    if sports_news.get("success"):
        result["sports_news"] = {"stories": [{"title":s.get("title",""),"description":s.get("description",""),"sources":list(s.get("sources",[]) if not isinstance(s.get("sources",[]),set) else s.get("sources",set())),"url":s.get("url","")} for s in sports_news.get("stories",[])]}
    if world_news.get("success"):
        result["world_news"] = {"stories": [{"title":s.get("title",""),"description":s.get("description",""),"sources":list(s.get("sources",[]) if not isinstance(s.get("sources",[]),set) else s.get("sources",set())),"url":s.get("url","")} for s in world_news.get("stories",[])]}
    if politics_news.get("success"):
        result["politics_news"] = {"stories": [{"title":s.get("title",""),"description":s.get("description",""),"sources":list(s.get("sources",[]) if not isinstance(s.get("sources",[]),set) else s.get("sources",set())),"url":s.get("url","")} for s in politics_news.get("stories",[])]}
    if science_news.get("success"):
        result["science_news"] = {"stories": [{"title":s.get("title",""),"description":s.get("description",""),"sources":list(s.get("sources",[]) if not isinstance(s.get("sources",[]),set) else s.get("sources",set())),"url":s.get("url","")} for s in science_news.get("stories",[])]}
    if entertainment_news.get("success"):
        result["entertainment_news"] = {"stories": [{"title":s.get("title",""),"description":s.get("description",""),"sources":list(s.get("sources",[]) if not isinstance(s.get("sources",[]),set) else s.get("sources",set())),"url":s.get("url","")} for s in entertainment_news.get("stories",[])]}
    def _ser_games(raw_list):
        """Serialise a list of parsed game dicts, including logo/color/date fields."""
        out = []
        for g in raw_list:
            out.append({
                "home_team":  g.get("home_team", ""),
                "away_team":  g.get("away_team", ""),
                "home_score": g.get("home_score", ""),
                "away_score": g.get("away_score", ""),
                "status":     g.get("status", ""),
                "date_str":   g.get("date_str", ""),
                "date_iso":   g.get("date_iso", ""),
                "clock":      g.get("clock", ""),
                "home_color": g.get("home_color", ""),
                "away_color": g.get("away_color", ""),
                "home_logo":  g.get("home_logo", ""),
                "away_logo":  g.get("away_logo", ""),
            })
        return out

    if s_pl.get("success"):
        result["sports_pl"]  = {"success":True,"league":s_pl.get("league",""),"league_key":s_pl.get("league_key",""),"games":_ser_games(s_pl.get("games",[]))}
    if s_ucl.get("success"):
        result["sports_ucl"] = {"success":True,"league":s_ucl.get("league",""),"league_key":s_ucl.get("league_key",""),"games":_ser_games(s_ucl.get("games",[]))}
    if s_nba.get("success"):
        result["sports_nba"] = {"success":True,"league":s_nba.get("league",""),"league_key":s_nba.get("league_key",""),"games":_ser_games(s_nba.get("games",[]))}
    # ── Consolidated cricket payload ────────────────────────────────────────
    # All formats and league competitions go into ONE sports_cricket section.
    # Each game gets a `format` tag (Test/ODI/T20I/IPL/PSL/BBL) the UI can
    # render as a small badge. We dedupe by date+teams in case ESPN returns
    # the same fixture under multiple sub-leagues.
    def _ser_cricket_games(raw_list, fmt_label):
        out = []
        for g in raw_list:
            # _ser_games loses the cricket-specific fields — copy them
            # forward by hand. Same shape as football games plus
            # home_innings / away_innings / note / format.
            out.append({
                **{
                    "home_team":  g.get("home_team", ""),
                    "away_team":  g.get("away_team", ""),
                    "home_score": g.get("home_score", ""),
                    "away_score": g.get("away_score", ""),
                    "status":     g.get("status", ""),
                    "date_str":   g.get("date_str", ""),
                    "date_iso":   g.get("date_iso", ""),
                    "clock":      g.get("clock", ""),
                    "home_color": g.get("home_color", ""),
                    "away_color": g.get("away_color", ""),
                    "home_logo":  g.get("home_logo", ""),
                    "away_logo":  g.get("away_logo", ""),
                },
                "home_innings": g.get("home_innings", []),
                "away_innings": g.get("away_innings", []),
                "note":         g.get("note", ""),
                "format":       fmt_label,
            })
        return out

    _cricket_sources = [
        (s_cric_test, "Test"),
        (s_cric_odi,  "ODI"),
        (s_cric_t20,  "T20I"),
        (s_cric_ipl,  "IPL"),
        (s_cric_psl,  "PSL"),
        (s_cric_bbl,  "BBL"),
    ]
    _all_cricket_games = []
    _seen_fixtures = set()  # (date_iso[:10], home, away) → dedupe
    for src, fmt in _cricket_sources:
        if not src or not src.get("success"):
            continue
        for g in _ser_cricket_games(src.get("games", []), fmt):
            key = (g.get("date_iso", "")[:10], g.get("home_team", ""), g.get("away_team", ""))
            if key in _seen_fixtures:
                continue
            _seen_fixtures.add(key)
            _all_cricket_games.append(g)

    if _all_cricket_games:
        result["sports_cricket"] = {
            "success":    True,
            "league":     "Cricket",
            "league_key": "cricket",
            "games":      _all_cricket_games,
        }
    if s_rm.get("success"):
        result["sports_rm"] = {"success":True,"league":s_rm.get("league","La Liga"),"league_key":"la_liga","team":"Real Madrid","games":_ser_games(s_rm.get("games",[]))}
    if prayer.get("success"): result["prayer"] = prayer
    if gmail.get("success"):
        result["emails"] = gmail.get("emails", [])

    # Connection diagnostics — the widgets render different empty states
    # depending on whether the agent is genuinely unconfigured, the token
    # expired, or a real auth error occurred.
    result["google"] = {
        "gmail_connected": not jarvis.gmail.is_mock,
        "calendar_connected": not jarvis.calendar.is_mock,
        "gmail_error": getattr(jarvis.gmail, "auth_error", None),
        "calendar_error": getattr(jarvis.calendar, "auth_error", None),
    }

    return SafeJSONResponse(result)


@app.get("/live-tick")
async def live_tick():
    """
    Lightweight endpoint for the UI's live-refresh poll.

    Returns only the fast-changing widgets — markets + all sports payloads —
    so the UI can keep prices and scores up to date every 20-30s without
    re-fetching news, weather, gmail, calendar, prayer times etc. that
    change much less often (those stay on the 60s /sidebar tick).
    """
    async def safe(coro):
        try:
            return await coro
        except Exception:
            return {}

    (markets, s_pl, s_ucl, s_nba,
     s_cric_test, s_cric_odi, s_cric_t20,
     s_cric_ipl, s_cric_psl, s_cric_bbl,
     s_rm) = await asyncio.gather(
        safe(jarvis.markets.get_all()),
        safe(jarvis.sports.get_scores("premier_league", limit=8)),
        safe(jarvis.sports.get_scores("champions_league", limit=6)),
        safe(jarvis.sports.get_scores("nba", limit=6)),
        safe(jarvis.sports.get_scores("cricket_test", limit=4)),
        safe(jarvis.sports.get_scores("cricket_odi",  limit=4)),
        safe(jarvis.sports.get_scores("cricket_t20",  limit=4)),
        safe(jarvis.sports.get_scores("cricket_ipl",  limit=6)),
        safe(jarvis.sports.get_scores("cricket_psl",  limit=6)),
        safe(jarvis.sports.get_scores("cricket_bbl",  limit=4)),
        safe(jarvis.sports.search_team("Real Madrid", "la_liga")),
    )

    out = {}
    if markets.get("success"): out["markets"] = markets

    def _ser_games(raw_list):
        return [{
            "home_team":  g.get("home_team", ""),
            "away_team":  g.get("away_team", ""),
            "home_score": g.get("home_score", ""),
            "away_score": g.get("away_score", ""),
            "status":     g.get("status", ""),
            "date_str":   g.get("date_str", ""),
            "date_iso":   g.get("date_iso", ""),
            "clock":      g.get("clock", ""),
            "home_color": g.get("home_color", ""),
            "away_color": g.get("away_color", ""),
            "home_logo":  g.get("home_logo", ""),
            "away_logo":  g.get("away_logo", ""),
        } for g in raw_list]

    if s_pl.get("success"):
        out["sports_pl"]  = {"success":True,"league":s_pl.get("league",""),"league_key":"premier_league","games":_ser_games(s_pl.get("games",[]))}
    if s_ucl.get("success"):
        out["sports_ucl"] = {"success":True,"league":s_ucl.get("league",""),"league_key":"champions_league","games":_ser_games(s_ucl.get("games",[]))}
    if s_nba.get("success"):
        out["sports_nba"] = {"success":True,"league":s_nba.get("league",""),"league_key":"nba","games":_ser_games(s_nba.get("games",[]))}
    if s_rm.get("success"):
        out["sports_rm"]  = {"success":True,"league":s_rm.get("league","La Liga"),"league_key":"la_liga","team":"Real Madrid","games":_ser_games(s_rm.get("games",[]))}

    # Consolidated cricket — same dedupe logic as /sidebar
    def _ser_cricket(raw_list, fmt_label):
        out = []
        for g in raw_list:
            out.append({
                **{k: g.get(k, "") for k in (
                    "home_team", "away_team", "home_score", "away_score",
                    "status", "date_str", "date_iso", "clock",
                    "home_color", "away_color", "home_logo", "away_logo",
                )},
                "home_innings": g.get("home_innings", []),
                "away_innings": g.get("away_innings", []),
                "note":         g.get("note", ""),
                "format":       fmt_label,
            })
        return out

    _cricket_sources = [
        (s_cric_test, "Test"),
        (s_cric_odi,  "ODI"),
        (s_cric_t20,  "T20I"),
        (s_cric_ipl,  "IPL"),
        (s_cric_psl,  "PSL"),
        (s_cric_bbl,  "BBL"),
    ]
    _games = []
    _seen = set()
    for src, fmt in _cricket_sources:
        if not src or not src.get("success"):
            continue
        for g in _ser_cricket(src.get("games", []), fmt):
            key = (g.get("date_iso", "")[:10], g.get("home_team", ""), g.get("away_team", ""))
            if key in _seen:
                continue
            _seen.add(key)
            _games.append(g)
    if _games:
        out["sports_cricket"] = {
            "success": True, "league": "Cricket", "league_key": "cricket", "games": _games,
        }

    return SafeJSONResponse(out)


@app.get("/spotify/now-playing")
async def spotify_now_playing():
    """Lightweight endpoint polled by UI every 10s for now playing widget."""
    try:
        data = await jarvis.spotify.get_now_playing()
        return SafeJSONResponse(data)
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.post("/spotify/play")
async def spotify_play():
    try:
        data = await jarvis.spotify.play()
        return SafeJSONResponse(data)
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.post("/spotify/pause")
async def spotify_pause():
    try:
        data = await jarvis.spotify.pause()
        return SafeJSONResponse(data)
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.post("/spotify/next")
async def spotify_next():
    try:
        data = await jarvis.spotify.skip()
        return SafeJSONResponse(data)
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.post("/spotify/previous")
async def spotify_previous():
    try:
        data = await jarvis.spotify.previous()
        return SafeJSONResponse(data)
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.post("/spotify/volume")
async def spotify_volume(level: int = 50):
    try:
        data = await jarvis.spotify.set_volume(level)
        return SafeJSONResponse(data)
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.get("/hardware")
async def hardware():
    """Return system hardware info — CPU, memory, network, thermal, battery, wifi, disk."""
    import subprocess, asyncio, re, time
    loop = asyncio.get_event_loop()

    async def run(cmd):
        try:
            result = await loop.run_in_executor(None,
                lambda: subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5))
            return result.stdout.strip()
        except Exception:
            return ""

    # ── Battery ────────────────────────────────────────────────────────────
    bat_raw = await run("pmset -g batt | grep -Eo '[0-9]+%' | head -1")
    battery = bat_raw.replace('%','') if bat_raw else None

    # ── Wi-Fi ──────────────────────────────────────────────────────────────
    iface_raw = await run("networksetup -listallhardwareports | awk '/Wi-Fi/{getline; print $2}'")
    wifi_iface = iface_raw.strip() or "en0"
    wifi_raw = await run(f"networksetup -getairportnetwork {wifi_iface}")
    if ":" in wifi_raw:
        wifi_name = wifi_raw.split(":", 1)[-1].strip()
        wifi = wifi_name if wifi_name else "Connected"
    else:
        wifi = "Connected"

    # ── Disk ───────────────────────────────────────────────────────────────
    disk_raw = await run("df -h / | tail -1 | awk '{print $3, $4, $5}'")
    parts = disk_raw.split() if disk_raw else []
    disk_used = parts[0] if parts else None
    disk_free = parts[1] if len(parts) > 1 else None
    disk_pct  = parts[2].rstrip('%') if len(parts) > 2 else None

    # ── CPU usage (single sample of top) ───────────────────────────────────
    cpu_raw = await run("top -l 1 -n 0 | awk '/CPU usage/ {print $3, $5}'")
    # e.g. "2.41% 5.10%" → user + sys
    cpu_pct = None
    try:
        nums = re.findall(r"([\d.]+)%", cpu_raw)
        if nums:
            cpu_pct = round(sum(float(n) for n in nums), 1)
    except Exception:
        cpu_pct = None

    # ── Memory ─────────────────────────────────────────────────────────────
    # Total RAM in bytes
    mem_total_raw = await run("sysctl -n hw.memsize")
    try:
        mem_total_bytes = int(mem_total_raw) if mem_total_raw else 0
    except Exception:
        mem_total_bytes = 0
    mem_total_gb = round(mem_total_bytes / (1024 ** 3), 1) if mem_total_bytes else None

    # Active + wired memory from vm_stat
    vm_raw = await run("vm_stat | head -20")
    page_size = 4096
    pm = re.search(r"page size of (\d+) bytes", vm_raw)
    if pm:
        page_size = int(pm.group(1))
    def _vm(field):
        m = re.search(rf"{field}:\s+([\d]+)\.", vm_raw)
        return int(m.group(1)) if m else 0
    used_pages = _vm("Pages active") + _vm("Pages wired down") + _vm("Pages occupied by compressor")
    mem_used_bytes = used_pages * page_size
    mem_used_gb = round(mem_used_bytes / (1024 ** 3), 1) if mem_used_bytes else None
    mem_pct = round(mem_used_bytes / mem_total_bytes * 100, 1) if mem_total_bytes else None

    # ── Network throughput (delta over 1s) ─────────────────────────────────
    async def _netbytes():
        out = await run(f"netstat -ibn | awk '$1==\"{wifi_iface}\" {{ib+=$7; ob+=$10}} END {{print ib, ob}}'")
        try:
            ib, ob = out.split()
            return int(ib), int(ob)
        except Exception:
            return 0, 0
    n1_in, n1_out = await _netbytes()
    await asyncio.sleep(1.0)
    n2_in, n2_out = await _netbytes()
    rx_bps = max(0, n2_in - n1_in)   # bytes/sec
    tx_bps = max(0, n2_out - n1_out)

    def _fmt_bps(n):
        if n < 1024: return f"{n} B/s"
        if n < 1024 ** 2: return f"{n/1024:.1f} KB/s"
        if n < 1024 ** 3: return f"{n/(1024**2):.1f} MB/s"
        return f"{n/(1024**3):.2f} GB/s"

    # ── Thermal (CPU temp) ─────────────────────────────────────────────────
    # Apple Silicon doesn't expose temps without sudo/extra tools; we try a
    # few options and fall back gracefully.
    therm_c = None
    therm_raw = await run("osx-cpu-temp 2>/dev/null | head -1")
    if therm_raw:
        m = re.search(r"([\d.]+)", therm_raw)
        if m:
            therm_c = round(float(m.group(1)), 1)
    if therm_c is None:
        # iStats / smc fallback (not always installed)
        alt = await run("istats cpu temp --value-only 2>/dev/null")
        m = re.search(r"([\d.]+)", alt or "")
        if m:
            therm_c = round(float(m.group(1)), 1)
    if therm_c is None:
        # Synthetic fallback: rough mapping from CPU% so the gauge moves
        therm_c = round(38 + (cpu_pct or 0) * 0.4, 1) if cpu_pct is not None else None

    return SafeJSONResponse({
        "battery": int(battery) if battery and battery.isdigit() else None,
        "wifi": wifi,
        "disk_used": disk_used,
        "disk_free": disk_free,
        "disk_pct": int(disk_pct) if disk_pct and disk_pct.isdigit() else None,
        "cpu_pct": cpu_pct,
        "mem_used_gb": mem_used_gb,
        "mem_total_gb": mem_total_gb,
        "mem_pct": mem_pct,
        "net_rx_bps": rx_bps,
        "net_tx_bps": tx_bps,
        "net_rx_human": _fmt_bps(rx_bps),
        "net_tx_human": _fmt_bps(tx_bps),
        "thermal_c": therm_c,
    })


# ── Google OAuth diagnostics ──────────────────────────────────────────────────

@app.get("/google/status")
async def google_status():
    """
    Return the connection state of the Gmail + Calendar agents.

    Used by the UI's Inbox / Calendar widgets to show a meaningful message
    when something's broken — e.g. "token refresh failed: invalid_grant"
    instead of a generic "Connect Gmail via .env".
    """
    return SafeJSONResponse({
        "gmail": {
            "connected": not jarvis.gmail.is_mock,
            "error": getattr(jarvis.gmail, "auth_error", None),
        },
        "calendar": {
            "connected": not jarvis.calendar.is_mock,
            "error": getattr(jarvis.calendar, "auth_error", None),
        },
    })


@app.post("/google/reauth")
async def google_reauth():
    """
    Force a fresh OAuth flow for both Gmail and Calendar.

    Deletes the saved token.json so the next agent initialisation triggers
    the interactive `flow.run_local_server` flow — your browser will open
    to the Google consent screen. Use this when token refresh fails.
    """
    try:
        from config.settings import GOOGLE_TOKEN_PATH
        if GOOGLE_TOKEN_PATH.exists():
            GOOGLE_TOKEN_PATH.unlink()
        # Re-initialise both agents in place so the next /sidebar call
        # picks up the new service. This will block until the user
        # completes the browser-based OAuth consent.
        from agents.calendar_agent import CalendarAgent
        from agents.gmail_agent import GmailAgent
        jarvis.calendar = CalendarAgent()
        jarvis.gmail = GmailAgent()
        return SafeJSONResponse({
            "success": True,
            "gmail_connected": not jarvis.gmail.is_mock,
            "calendar_connected": not jarvis.calendar.is_mock,
            "gmail_error": getattr(jarvis.gmail, "auth_error", None),
            "calendar_error": getattr(jarvis.calendar, "auth_error", None),
        })
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


# ── Reminders ─────────────────────────────────────────────────────────────────
# Backed by the same ReminderStore the orchestrator uses, so anything created
# via voice/chat ("remind me to call mum in 10 minutes") shows up in the UI
# widget on the next refresh, and vice versa.

class ReminderCreate(BaseModel):
    title: str
    body: str = ""
    due_at: str | None = None          # ISO datetime; mutually exclusive with offset_minutes
    offset_minutes: int | None = None  # "in N minutes from now"
    recurring_minutes: int | None = None


@app.get("/reminders")
async def list_reminders():
    """Return all pending reminders (uncompleted), oldest-due first."""
    try:
        pending = jarvis.reminders.list_pending()
        # The DB returns timestamps as raw ISO strings — the UI wants the
        # field names it already uses. Map them once here so the front-end
        # stays simple.
        return SafeJSONResponse({
            "success": True,
            "reminders": [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "body": r.get("body", ""),
                    "due_at": r["due_at"],
                    "recurring_minutes": r.get("recurring_minutes"),
                    "completed": bool(r.get("completed", 0)),
                    "created_at": r.get("created_at", ""),
                }
                for r in pending
            ],
        })
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e), "reminders": []})


@app.post("/reminders")
async def create_reminder(req: ReminderCreate):
    """Create a new reminder. Returns the new reminder ID."""
    try:
        rid = jarvis.reminders.add(
            title=req.title,
            body=req.body or "",
            due_at=req.due_at,
            offset_minutes=req.offset_minutes,
            recurring_minutes=req.recurring_minutes,
        )
        return SafeJSONResponse({"success": True, "id": rid})
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.delete("/reminders/{reminder_id}")
async def delete_reminder(reminder_id: str):
    """Permanently delete a reminder by ID."""
    try:
        ok = jarvis.reminders.delete(reminder_id)
        if not ok:
            return SafeJSONResponse({"success": False, "error": "Reminder not found"})
        return SafeJSONResponse({"success": True})
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.post("/reminders/{reminder_id}/complete")
async def complete_reminder(reminder_id: str):
    """Mark a reminder as complete (it'll stop being returned by /reminders)."""
    try:
        ok = jarvis.reminders.complete(reminder_id)
        if not ok:
            return SafeJSONResponse({"success": False, "error": "Reminder not found"})
        return SafeJSONResponse({"success": True})
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.post("/upload")
async def upload_document(file: UploadFile):
    """Accept document upload and store for analysis."""
    import tempfile, os
    try:
        content_bytes = await file.read()
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content_bytes)
            tmp_path = tmp.name
        # Store path for Jarvis to use
        jarvis._last_uploaded_doc = tmp_path
        jarvis._last_uploaded_name = file.filename
        return SafeJSONResponse({"success": True, "filename": file.filename, "path": tmp_path})
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


# ── FinEx UI ──────────────────────────────────────────────────────────────────

@app.get("/finex")
async def finex_ui():
    """Serve the FinEx financial analysis dashboard."""
    return FileResponse(UI_DIR / "finex.html")


# ── FinEx Financial Statement Endpoints ───────────────────────────────────────

class FinExChatRequest(BaseModel):
    question: str
    company: str = "Bestway Cement"
    history: list = []


@app.post("/finex/chat")
async def finex_chat(req: FinExChatRequest):
    """Financial statement Q&A — powered by the FinEx engine (6 reasoning levels)."""
    result = await finex.chat(req.question, req.company, req.history)
    return SafeJSONResponse(result)


@app.post("/finex/upload")
async def finex_upload(
    file: UploadFile,
    company: str = Form("Bestway Cement"),
):
    """Upload a PDF financial statement, extract data, and store in Postgres."""
    import tempfile, os
    tmp_path = None
    try:
        content = await file.read()
        if not content:
            return SafeJSONResponse({"success": False, "error": "Uploaded file is empty."})
        suffix = os.path.splitext(file.filename or "upload")[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        print(f"💹 FinEx upload: company={company!r}  file={file.filename!r}  tmp={tmp_path}")
        result = await finex.upload_pdf(tmp_path, company)
        print(f"💹 FinEx result: success={result.get('success')}  error={result.get('error','—')}")
        return SafeJSONResponse(result)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"💹 FinEx upload exception:\n{tb}")
        return SafeJSONResponse({"success": False, "error": str(e), "traceback": tb})
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.get("/finex/companies")
async def finex_companies():
    """List all companies and periods stored in the FinEx database."""
    result = await finex.list_companies()
    return SafeJSONResponse(result)


# Market symbol groups for the FinEx financial dashboard
_INDICES = {
    "^GSPC":  "S&P 500",
    "^FTSE":  "FTSE 100",
    "^DJI":   "Dow Jones",
    "^IXIC":  "Nasdaq",
    "^N225":  "Nikkei 225",
}
_CRYPTO = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "BNB-USD": "BNB",
    "XRP-USD": "XRP",
    "SOL-USD": "Solana",
}
_COMMODITIES = {
    "GLD":      "Gold ETF",
    "USO":      "Oil ETF",
    "GBPUSD=X": "GBP/USD",
    "EURUSD=X": "EUR/USD",
    "JPYUSD=X": "JPY/USD",
}
_TECH = {
    "AAPL":  "Apple",
    "NVDA":  "NVIDIA",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet",
    "META":  "Meta",
}
_FINANCE_STOCKS = {
    "JPM": "JPMorgan",
    "GS":  "Goldman Sachs",
    "BAC": "Bank of America",
    "V":   "Visa",
    "MA":  "Mastercard",
}
_ENERGY_HEALTH = {
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer",
    "UNH": "UnitedHealth",
}

# Finance-specific RSS feeds for the FinEx news panel
_FINANCE_FEEDS = {
    "reuters_biz":  ("Reuters Business",  "https://feeds.reuters.com/reuters/businessNews"),
    "cnbc":         ("CNBC",              "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    "yahoo_fin":    ("Yahoo Finance",     "https://finance.yahoo.com/news/rssindex"),
    "marketwatch":  ("MarketWatch",       "https://feeds.marketwatch.com/marketwatch/topstories"),
    "ft":           ("Financial Times",   "https://www.ft.com/rss/home"),
    "investopedia": ("Investopedia",      "https://www.investopedia.com/feeds/news.xml"),
}


async def _fetch_finance_news() -> dict:
    """Fetch headlines from finance-specific RSS feeds."""
    import xml.etree.ElementTree as ET
    TIMEOUT = aiohttp.ClientTimeout(total=8)
    HEADERS = {"User-Agent": "Jarvis/1.0 Python/aiohttp"}
    stories = []

    async def _one(key, name, url):
        try:
            async with aiohttp.ClientSession(timeout=TIMEOUT, headers=HEADERS) as s:
                async with s.get(url) as r:
                    if r.status != 200:
                        return []
                    text = await r.text()
            root = ET.fromstring(text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//item") or root.findall(".//atom:entry", ns)
            out = []
            for item in items[:6]:
                title = (item.findtext("title") or item.findtext("atom:title", namespaces=ns) or "").strip()
                desc  = (item.findtext("description") or item.findtext("atom:summary", namespaces=ns) or "").strip()
                link  = (item.findtext("link") or item.findtext("atom:link", namespaces=ns) or "").strip()
                if title:
                    out.append({"title": title, "description": desc[:200], "source": name, "url": link})
            return out
        except Exception:
            return []

    tasks = [_one(k, n, u) for k, (n, u) in _FINANCE_FEEDS.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, list):
            stories.extend(r)
    return {"success": True, "stories": stories[:30]}


@app.get("/finex/sidebar")
async def finex_sidebar():
    """
    All financial widget data for the FinEx dashboard — fetched in parallel.
    Returns: indices, crypto, commodities, tech stocks, finance stocks,
             energy/health stocks, financial news, and company list.
    """
    async def safe(coro):
        try:
            return await coro
        except Exception:
            return {"success": False, "prices": []}

    (indices, crypto, commodities, tech, fin_stocks, energy_health,
     fin_news, companies) = await asyncio.gather(
        safe(jarvis.markets.get_all(_INDICES)),
        safe(jarvis.markets.get_all(_CRYPTO)),
        safe(jarvis.markets.get_all(_COMMODITIES)),
        safe(jarvis.markets.get_all(_TECH)),
        safe(jarvis.markets.get_all(_FINANCE_STOCKS)),
        safe(jarvis.markets.get_all(_ENERGY_HEALTH)),
        safe(_fetch_finance_news()),
        safe(finex.list_companies()),
    )

    return SafeJSONResponse({
        "indices":       indices,
        "crypto":        crypto,
        "commodities":   commodities,
        "tech":          tech,
        "fin_stocks":    fin_stocks,
        "energy_health": energy_health,
        "fin_news":      fin_news,
        "companies":     companies,
    })


@app.get("/finex/markets-tick")
async def finex_markets_tick():
    """
    Lightweight live-refresh endpoint for the FinEx market widgets.

    Returns only the six price baskets — indices, crypto, commodities, tech,
    finance stocks, energy/health. Skips finance news and the company list
    which change much less often and add unnecessary latency to a poll
    that runs every 20 seconds.

    The FinEx UI calls this aggressively to keep tickers live while the
    full /finex/sidebar payload is fetched once on page load and then
    only when news/companies actually need to refresh.
    """
    async def safe(coro):
        try:
            return await coro
        except Exception:
            return {"success": False, "prices": []}

    indices, crypto, commodities, tech, fin_stocks, energy_health = await asyncio.gather(
        safe(jarvis.markets.get_all(_INDICES)),
        safe(jarvis.markets.get_all(_CRYPTO)),
        safe(jarvis.markets.get_all(_COMMODITIES)),
        safe(jarvis.markets.get_all(_TECH)),
        safe(jarvis.markets.get_all(_FINANCE_STOCKS)),
        safe(jarvis.markets.get_all(_ENERGY_HEALTH)),
    )

    return SafeJSONResponse({
        "indices":       indices,
        "crypto":        crypto,
        "commodities":   commodities,
        "tech":          tech,
        "fin_stocks":    fin_stocks,
        "energy_health": energy_health,
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat.
    Streams responses as they come in.
    """
    await websocket.accept()

    # Push sidebar data in the background — don't block the receive loop
    async def _push_sidebar():
        try:
            sidebar_data = await sidebar()
            await websocket.send_text(json.dumps({
                "type": "sidebar",
                "data": json.loads(sidebar_data.body),
            }))
        except Exception:
            pass

    asyncio.ensure_future(_push_sidebar())

    # Per-session conversation history (survives across turns in this connection)
    conversation_history: list = []

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message", "")

            if not message:
                continue

            # Signal typing immediately
            await websocket.send_text(json.dumps({"type": "typing"}))

            # ── Wait on warmup if it's still in flight ─────────────────────
            # If the user fires a query in the first ~30–60s of server life,
            # we'd otherwise have warmup and the user request fighting for a
            # cold model. Block here briefly so warmup wins.
            warmup_task = getattr(app.state, "warmup_task", None)
            if warmup_task is not None and not warmup_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(warmup_task), timeout=90.0)
                except asyncio.TimeoutError:
                    print("⚠️  Warmup still running after 90s — proceeding anyway")
                except Exception as exc:
                    print(f"⚠️  Warmup task error: {exc}")

            # ── Stream response via handle_stream() ────────────────────────
            try:
                last_response = None
                async def _stream_with_timeout():
                    async for event in jarvis.handle_stream(
                        message,
                        conversation_history=list(conversation_history),
                    ):
                        yield event

                async def _run():
                    nonlocal last_response
                    async for event in _stream_with_timeout():
                        await websocket.send_text(json.dumps(event))
                        if event.get("type") == "response":
                            last_response = event

                try:
                    # 180s gives cold-loads (~60–90s) + tier-2 streaming
                    # (~30–60s) comfortable headroom. Once warmup completes
                    # and keep_alive holds the model resident, real queries
                    # should land in 1–5s anyway.
                    await asyncio.wait_for(_run(), timeout=180.0)
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "message": "Request timed out after 3 minutes. The model may be overloaded — please try again.",
                        "success": False,
                        "timeout": True,
                    }))
                    continue

                # Store this exchange in conversation history (keep last 10 turns)
                if last_response and last_response.get("message"):
                    conversation_history.append({"role": "user", "content": message})
                    conversation_history.append({"role": "assistant", "content": last_response["message"]})
                    if len(conversation_history) > 20:  # cap at 10 exchanges
                        conversation_history[:] = conversation_history[-20:]

                # Push updated sidebar data after state-changing actions
                keywords = ["schedule", "email", "spotify", "play", "pause", "volume"]
                if any(kw in message.lower() for kw in keywords):
                    try:
                        updated = await sidebar()
                        body = json.loads(updated.body)
                        for key, val in body.items():
                            spayload = {"type": key}
                            if isinstance(val, dict):
                                spayload.update(val)
                            else:
                                spayload["data"] = val
                            await websocket.send_text(json.dumps(spayload))
                    except Exception:
                        pass

            except Exception as e:
                try:
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "message": f"Error: {str(e)}",
                        "success": False,
                    }))
                except Exception:
                    break  # Connection gone — exit the loop cleanly

    except (WebSocketDisconnect, RuntimeError):
        pass  # Client disconnected — normal, not an error


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="warning",
    )