"""
J.A.R.V.I.S Web Server
FastAPI backend with:
- WebSocket for real-time streaming chat
- HTTP /chat endpoint as fallback
- HTTP /sidebar endpoint for widget data
- Static file serving for the UI
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import aiohttp

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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


@app.post("/chat")
async def chat(req: ChatRequest):
    """HTTP chat endpoint — fallback when WebSocket unavailable."""
    response = await jarvis.handle(req.message)
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

    weather, markets, calendar, spotify, news, tech_news, sports_news, s_pl, s_ucl, s_nba, s_cric, prayer = await asyncio.gather(
        safe(jarvis.weather.get_current()),
        safe(jarvis.markets.get_all()),
        safe(jarvis.calendar.search_events()),
        safe(jarvis.spotify.get_now_playing()),
        safe(jarvis.news.get_headlines(max_stories=8)),
        safe(jarvis.news.get_headlines(category="technology", max_stories=8, query="artificial intelligence machine learning tech")),
        safe(jarvis.news.get_headlines(category="sports", max_stories=8)),
        safe(jarvis.sports.get_scores("premier_league", limit=8)),
        safe(jarvis.sports.get_scores("champions_league", limit=6)),
        safe(jarvis.sports.get_scores("nba", limit=6)),
        safe(jarvis.sports.get_scores("cricket_t20", limit=6)),
        safe(jarvis.prayer.get_times()),
    )

    result = {}
    if weather.get("success"): result["weather"] = weather
    if markets.get("success"): result["markets"] = markets
    if calendar.get("success"): result["calendar"] = {"events": calendar.get("events", [])}
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
    if s_pl.get("success"):
        games = [{"home_team":g.get("home_team",""),"away_team":g.get("away_team",""),"home_score":g.get("home_score",""),"away_score":g.get("away_score",""),"status":g.get("status",""),"date_str":g.get("date_str",""),"clock":g.get("clock","")} for g in s_pl.get("games",[])]
        result["sports_pl"] = {"success":True,"league":s_pl.get("league",""),"league_key":s_pl.get("league_key",""),"games":games}
    if s_ucl.get("success"):
        games = [{"home_team":g.get("home_team",""),"away_team":g.get("away_team",""),"home_score":g.get("home_score",""),"away_score":g.get("away_score",""),"status":g.get("status",""),"date_str":g.get("date_str",""),"clock":g.get("clock","")} for g in s_ucl.get("games",[])]
        result["sports_ucl"] = {"success":True,"league":s_ucl.get("league",""),"league_key":s_ucl.get("league_key",""),"games":games}
    if s_nba.get("success"):
        games = [{"home_team":g.get("home_team",""),"away_team":g.get("away_team",""),"home_score":g.get("home_score",""),"away_score":g.get("away_score",""),"status":g.get("status",""),"date_str":g.get("date_str",""),"clock":g.get("clock","")} for g in s_nba.get("games",[])]
        result["sports_nba"] = {"success":True,"league":s_nba.get("league",""),"league_key":s_nba.get("league_key",""),"games":games}
    if s_cric.get("success"):
        games = [{"home_team":g.get("home_team",""),"away_team":g.get("away_team",""),"home_score":g.get("home_score",""),"away_score":g.get("away_score",""),"status":g.get("status",""),"date_str":g.get("date_str",""),"clock":g.get("clock","")} for g in s_cric.get("games",[])]
        result["sports_cricket"] = {"success":True,"league":s_cric.get("league",""),"league_key":s_cric.get("league_key",""),"games":games}
    if prayer.get("success"): result["prayer"] = prayer

    return SafeJSONResponse(result)


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
    """Return system hardware info."""
    import subprocess, asyncio
    loop = asyncio.get_event_loop()

    async def run(cmd):
        try:
            result = await loop.run_in_executor(None,
                lambda: subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5))
            return result.stdout.strip()
        except Exception:
            return ""

    bat_raw = await run("pmset -g batt | grep -Eo '[0-9]+%' | head -1")
    battery = bat_raw.replace('%','') if bat_raw else None

    # Dynamically detect the Wi-Fi interface (en0 on most Macs, en1 on some)
    iface_raw = await run("networksetup -listallhardwareports | awk '/Wi-Fi/{getline; print $2}'")
    wifi_iface = iface_raw.strip() or "en0"
    wifi_raw = await run(f"networksetup -getairportnetwork {wifi_iface}")
    if ":" in wifi_raw:
        wifi_name = wifi_raw.split(":", 1)[-1].strip()
        wifi = wifi_name if wifi_name else "Not Connected"
    else:
        # "You are not associated with an AirPort network." → show disconnect state
        wifi = "Not Connected"

    disk_raw = await run("df -h / | tail -1 | awk '{print $3, $4}'")
    parts = disk_raw.split() if disk_raw else []
    disk_used = parts[0] if parts else None
    disk_free = parts[1] if len(parts)>1 else None

    return SafeJSONResponse({
        "battery": int(battery) if battery and battery.isdigit() else None,
        "wifi": wifi,
        "disk_used": disk_used,
        "disk_free": disk_free,
    })


@app.post("/upload")
async def upload_document(file: "UploadFile"):
    """Accept document upload and store for analysis."""
    from fastapi import UploadFile
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
async def finex_upload(file: "UploadFile", company: str = "Bestway Cement"):
    """Upload a PDF financial statement, extract data, and store in Postgres."""
    from fastapi import UploadFile
    import tempfile, os
    try:
        content = await file.read()
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        result = await finex.upload_pdf(tmp_path, company)
        return SafeJSONResponse(result)
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})
    finally:
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat.
    Streams responses as they come in.
    """
    await websocket.accept()

    # Send initial sidebar data
    try:
        sidebar_data = await sidebar()
        await websocket.send_text(json.dumps({
            "type": "sidebar",
            "data": json.loads(sidebar_data.body),
        }))
    except Exception:
        pass

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message", "")

            if not message:
                continue

            # Signal typing immediately
            await websocket.send_text(json.dumps({"type": "typing"}))

            # ── Stream response via handle_stream() ────────────────────────
            try:
                last_response = None
                async def _stream_with_timeout():
                    async for event in jarvis.handle_stream(message):
                        yield event

                async def _run():
                    nonlocal last_response
                    async for event in _stream_with_timeout():
                        await websocket.send_text(json.dumps(event))
                        if event.get("type") == "response":
                            last_response = event

                try:
                    await asyncio.wait_for(_run(), timeout=120.0)
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "message": "Request timed out after 2 minutes. The model may be overloaded — please try again.",
                        "success": False,
                        "timeout": True,
                    }))
                    continue

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
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "message": f"Error: {str(e)}",
                    "success": False,
                }))

    except WebSocketDisconnect:
        pass


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