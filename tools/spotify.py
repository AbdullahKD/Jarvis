"""
Spotify Tool
Controls Spotify playback via the Spotify Web API.
Requires a free Spotify developer app (client ID + secret).
Setup: https://developer.spotify.com/dashboard
"""

from __future__ import annotations

import base64
import time
from typing import Any, Dict, List, Optional

import aiohttp

from config.settings import (
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    SPOTIFY_REDIRECT_URI,
)

TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE  = "https://api.spotify.com/v1"


class SpotifyTool:
    """
    Spotify Web API client with automatic token refresh.

    For the dissertation, this demonstrates secure OAuth2 integration
    with a consumer API — complementing the Google OAuth2 implementation.

    Fallback: If credentials not set, all methods return mock responses
    so the rest of Jarvis can be tested without Spotify credentials.
    """

    def __init__(self):
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._mock = not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET)
        if self._mock:
            print("🎵 SpotifyTool running in mock mode (set SPOTIFY_CLIENT_ID/SECRET to enable)")
        else:
            print("🎵 SpotifyTool ready")

    # ── Auth ───────────────────────────────────────────────────────────────

    async def _get_token(self) -> str:
        """Get or refresh access token using client credentials flow."""
        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token

        credentials = base64.b64encode(
            f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()
        ).decode()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                TOKEN_URL,
                headers={"Authorization": f"Basic {credentials}"},
                data={"grant_type": "client_credentials"},
            ) as resp:
                data = await resp.json()

        self._access_token = data["access_token"]
        self._token_expiry = time.time() + data["expires_in"]
        return self._access_token

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Dict[str, Any]:
        if self._mock:
            return {"success": True, "mock": True, "message": "Spotify mock response"}

        token = await self._get_token()
        headers = {"Authorization": f"Bearer {token}"}

        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                f"{API_BASE}{endpoint}",
                headers=headers,
                **kwargs,
            ) as resp:
                if resp.status == 204:
                    return {"success": True}
                if resp.status >= 400:
                    error = await resp.json()
                    return {"success": False, "error": error.get("error", {}).get("message", str(resp.status))}
                return {"success": True, **(await resp.json())}

    # ── Search ─────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        search_type: str = "track",
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Search Spotify for tracks, artists, albums, or playlists."""
        if self._mock:
            return {
                "success": True, "mock": True,
                "tracks": [{"name": f"Mock: {query}", "artist": "Mock Artist", "uri": "spotify:track:mock"}]
            }
        data = await self._request("GET", "/search", params={
            "q": query, "type": search_type, "limit": limit
        })
        if not data.get("success"):
            return data

        items = data.get(f"{search_type}s", {}).get("items", [])
        results = []
        for item in items:
            r = {"name": item.get("name"), "uri": item.get("uri")}
            if search_type == "track":
                r["artist"] = ", ".join(a["name"] for a in item.get("artists", []))
                r["album"] = item.get("album", {}).get("name")
                r["duration_ms"] = item.get("duration_ms")
            results.append(r)

        return {"success": True, "results": results, "type": search_type}

    # ── Playback (requires Premium + active device) ─────────────────────

    async def play(self, uri: Optional[str] = None) -> Dict[str, Any]:
        """Play a track/playlist by URI, or resume current playback."""
        body = {"uris": [uri]} if uri and "track" in uri else {}
        return await self._request("PUT", "/me/player/play", json=body or None)

    async def pause(self) -> Dict[str, Any]:
        return await self._request("PUT", "/me/player/pause")

    async def skip(self) -> Dict[str, Any]:
        return await self._request("POST", "/me/player/next")

    async def previous(self) -> Dict[str, Any]:
        return await self._request("POST", "/me/player/previous")

    async def set_volume(self, level: int) -> Dict[str, Any]:
        """Set Spotify volume (0–100). Requires Premium."""
        return await self._request(
            "PUT", "/me/player/volume",
            params={"volume_percent": max(0, min(100, level))}
        )

    async def get_now_playing(self) -> Dict[str, Any]:
        """Get the currently playing track."""
        data = await self._request("GET", "/me/player/currently-playing")
        if not data.get("success"):
            return data

        item = data.get("item", {})
        if not item:
            return {"success": True, "playing": False}

        return {
            "success": True,
            "playing": data.get("is_playing", False),
            "track": item.get("name"),
            "artist": ", ".join(a["name"] for a in item.get("artists", [])),
            "album": item.get("album", {}).get("name"),
            "progress_ms": data.get("progress_ms"),
            "duration_ms": item.get("duration_ms"),
        }

    # ── Formatting ─────────────────────────────────────────────────────────

    def format_now_playing(self, data: Dict[str, Any]) -> str:
        if not data.get("success"):
            return "Could not get Spotify status."
        if not data.get("playing"):
            return "Nothing currently playing on Spotify."
        return (
            f"Now playing: {data['track']} by {data['artist']} "
            f"({data['album']})"
        )
