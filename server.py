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

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="J.A.R.V.I.S", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared orchestrator instance
jarvis = JarvisOrchestrator()

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

    wifi_raw = await run("networksetup -getairportnetwork en0")
    wifi = wifi_raw.split(":",1)[-1].strip() if ":" in wifi_raw else "Unknown"

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

            # Signal typing
            await websocket.send_text(json.dumps({"type": "typing"}))

            # Process with Jarvis — with timeout to detect crashes
            try:
                try:
                    response = await asyncio.wait_for(
                        jarvis.handle(message),
                        timeout=120.0  # 2 minute max
                    )
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "message": "Request timed out after 2 minutes. The model may be overloaded — please try again.",
                        "success": False,
                        "timeout": True,
                    }))
                    continue
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "message": response.message,
                    "success": response.success,
                    "latency_ms": response.latency_ms,
                }))

                # Push updated sidebar data after actions that might change it
                keywords = ["schedule", "email", "spotify", "play", "pause", "volume"]
                if any(kw in message.lower() for kw in keywords):
                    try:
                        updated = await sidebar()
                        body = json.loads(updated.body)
                        for key, val in body.items():
                            payload = {"type": key}
                            if isinstance(val, dict):
                                payload.update(val)
                            else:
                                payload["data"] = val
                            await websocket.send_text(json.dumps(payload))
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