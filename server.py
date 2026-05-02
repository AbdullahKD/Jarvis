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
    weather_task  = jarvis.weather.get_current()
    markets_task  = jarvis.markets.get_all()
    calendar_task = jarvis.calendar.search_events()
    spotify_task  = jarvis.spotify.get_now_playing()

    weather, markets, calendar, spotify = await asyncio.gather(
        weather_task, markets_task, calendar_task, spotify_task,
        return_exceptions=True
    )

    result = {}

    if isinstance(weather, dict) and weather.get("success"):
        result["weather"] = weather

    if isinstance(markets, dict) and markets.get("success"):
        result["markets"] = markets

    if isinstance(calendar, dict) and calendar.get("success"):
        result["calendar"] = {
            "events": calendar.get("events", [])
        }

    if isinstance(spotify, dict) and spotify.get("success"):
        result["spotify"] = {
            "track": spotify.get("track", ""),
            "artist": spotify.get("artist", ""),
            "playing": spotify.get("playing", False),
        }

    return JSONResponse(result)


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