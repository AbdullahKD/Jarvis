"""
Spotify Tool — Full Implementation
Uses Authorization Code flow for complete playback control.

Prerequisites:
1. Run: python3 spotify_auth.py  (one time only)
2. Tokens are stored in ~/.jarvis/spotify_token.json
3. All token refreshing is automatic from here on

Supports:
- Play / pause / skip / previous
- Search tracks, artists, albums, playlists
- Play by name (searches then plays top result)
- Volume control
- Now playing (track, artist, album, progress)
- Get user's playlists
- Queue a track
"""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from config.settings import (
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    SPOTIFY_REDIRECT_URI,
)

TOKEN_URL  = "https://accounts.spotify.com/api/token"
API_BASE   = "https://api.spotify.com/v1"
TOKEN_PATH = Path.home() / ".jarvis" / "spotify_token.json"


class SpotifyTool:
    """
    Full Spotify Web API client using Authorization Code flow.
    Requires one-time auth via spotify_auth.py.
    Falls back to mock mode if not authenticated.
    """

    def __init__(self):
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._mock = False
        self._load_tokens()

    def _load_tokens(self):
        """Load saved tokens from disk."""
        if not TOKEN_PATH.exists():
            if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
                print("🎵 Spotify: run python3 spotify_auth.py to authenticate")
            else:
                print("🎵 SpotifyTool running in mock mode (credentials not set)")
            self._mock = True
            return

        try:
            data = json.loads(TOKEN_PATH.read_text())
            self._access_token  = data.get("access_token")
            self._refresh_token = data.get("refresh_token")
            self._token_expiry  = data.get("expires_at", 0)
            print("🎵 SpotifyTool ready — authenticated")
        except Exception as e:
            print(f"🎵 Spotify token load error: {e}")
            self._mock = True

    def _save_tokens(self):
        """Persist tokens to disk."""
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_PATH.write_text(json.dumps({
            "access_token":  self._access_token,
            "refresh_token": self._refresh_token,
            "expires_at":    self._token_expiry,
        }, indent=2))

    # ── Auth ───────────────────────────────────────────────────────────────

    async def _get_token(self) -> Optional[str]:
        """Return valid access token, refreshing if needed."""
        if self._mock:
            return None

        # Still valid
        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token

        # Refresh
        if not self._refresh_token:
            self._mock = True
            return None

        credentials = base64.b64encode(
            f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()
        ).decode()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    TOKEN_URL,
                    headers={
                        "Authorization": f"Basic {credentials}",
                        "Content-Type":  "application/x-www-form-urlencoded",
                    },
                    data={
                        "grant_type":    "refresh_token",
                        "refresh_token": self._refresh_token,
                    },
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        print(f"🎵 Token refresh failed: {text}")
                        return None
                    data = await resp.json()

            self._access_token = data["access_token"]
            self._token_expiry = time.time() + data.get("expires_in", 3600)
            # Spotify may issue a new refresh token
            if "refresh_token" in data:
                self._refresh_token = data["refresh_token"]
            self._save_tokens()
            return self._access_token

        except Exception as e:
            print(f"🎵 Token refresh error: {e}")
            return None

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_body=None,
        params=None,
        expect_empty: bool = False,
    ) -> Dict[str, Any]:
        """Make an authenticated API request."""
        if self._mock:
            return {"success": True, "mock": True,
                    "message": "Spotify not authenticated — run python3 spotify_auth.py"}

        token = await self._get_token()
        if not token:
            return {"success": False, "error": "Not authenticated. Run python3 spotify_auth.py"}

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method,
                    f"{API_BASE}{endpoint}",
                    headers=headers,
                    json=json_body,
                    params=params,
                ) as resp:
                    if resp.status == 204 or expect_empty:
                        return {"success": True}
                    if resp.status == 401:
                        # Force token refresh next call
                        self._token_expiry = 0
                        return {"success": False, "error": "Token expired — will refresh on next request"}
                    if resp.status == 403:
                        return {"success": False, "error": "Spotify Premium required for this action"}
                    if resp.status == 404:
                        return {"success": False, "error": "No active Spotify device found. Open Spotify on any device first."}
                    if resp.status >= 400:
                        try:
                            err = await resp.json()
                            msg = err.get("error", {}).get("message", f"HTTP {resp.status}")
                        except Exception:
                            msg = f"HTTP {resp.status}"
                        return {"success": False, "error": msg}
                    if resp.content_length == 0:
                        return {"success": True}
                    return {"success": True, **(await resp.json())}

        except aiohttp.ClientError as e:
            return {"success": False, "error": str(e)}

    # ── Search ─────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        search_type: str = "track",
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Search Spotify for tracks, artists, albums, or playlists."""
        data = await self._request("GET", "/search", params={
            "q": query, "type": search_type, "limit": limit,
            "market": "GB",
        })
        if not data.get("success"):
            return data

        items = data.get(f"{search_type}s", {}).get("items", [])
        results = []
        for item in items:
            if not item:
                continue
            r = {"name": item.get("name"), "uri": item.get("uri"), "id": item.get("id")}
            if search_type == "track":
                r["artist"] = ", ".join(a["name"] for a in item.get("artists", []))
                r["album"]  = item.get("album", {}).get("name", "")
                r["duration_ms"] = item.get("duration_ms", 0)
                r["image"] = (item.get("album", {}).get("images") or [{}])[0].get("url", "")
            elif search_type == "artist":
                r["genres"] = item.get("genres", [])
                r["followers"] = item.get("followers", {}).get("total", 0)
                r["image"] = (item.get("images") or [{}])[0].get("url", "")
            elif search_type == "playlist":
                r["owner"] = item.get("owner", {}).get("display_name", "")
                r["tracks"] = item.get("tracks", {}).get("total", 0)
            results.append(r)

        return {"success": True, "results": results, "type": search_type, "query": query}

    # ── Smart play — search then play ──────────────────────────────────────

    async def _normalise_song_query(self, raw_query: str) -> tuple[str, str]:
        """
        Use a tiny LLM call to extract clean track name and artist from
        natural language. Returns (track, artist) tuple.
        Costs ~1s but massively improves search accuracy.
        """
        from config.llm_client import OllamaClient
        _llm = OllamaClient()
        prompt = (
            f'Extract the song title and artist from this request: "{raw_query}"\n'
            f'Reply with ONLY two lines:\n'
            f'track: <song title>\n'
            f'artist: <artist name>\n'
            f'If no artist is mentioned, write "artist: unknown".'
        )
        try:
            resp = await _llm.chat(
                [{"role": "user", "content": prompt}],
                inject_system=False,
                max_tokens=40,
            )
            lines = resp.strip().lower().splitlines()
            track = ""
            artist = ""
            for line in lines:
                if line.startswith("track:"):
                    track = line.split(":", 1)[1].strip()
                elif line.startswith("artist:"):
                    artist = line.split(":", 1)[1].strip()
                    if artist == "unknown":
                        artist = ""
            return track or raw_query, artist
        except Exception:
            return raw_query, ""

    async def play_by_name(self, query: str) -> Dict[str, Any]:
        """
        Search for a track/artist/playlist and play the best result.
        Uses LLM normalisation for accurate song/artist extraction.
        """
        # Check for active device first
        devices_data = await self.get_devices()
        devices = devices_data.get("devices", [])
        if not devices:
            return {
                "success": False,
                "error": "No active Spotify device found. Open the Spotify app on your Mac and play any song first, then try again."
            }
        active = next((d for d in devices if d.get("active")), devices[0])
        device_id = active.get("id")

        # Detect intent type first
        q_low = query.lower()
        is_playlist = any(w in q_low for w in ["playlist", "mix", "my playlist"])
        is_artist   = any(w in q_low for w in ["songs by", "music by", "discography", "artist"])

        if is_playlist:
            search_type = "playlist"
            norm_query  = query
        elif is_artist:
            search_type = "artist"
            norm_query  = re.sub(r'songs by|music by|discography|artist', '', query, flags=re.IGNORECASE).strip()
        else:
            # Normalise with LLM for track search
            search_type = "track"
            track, artist = await self._normalise_song_query(query)
            # Build Spotify search query: "track:X artist:Y" format is most accurate
            if artist:
                norm_query = f"track:{track} artist:{artist}"
            else:
                norm_query = track

        search_data = await self.search(norm_query, search_type=search_type, limit=3)

        # Fallback: if no results with formatted query, try plain text
        if not search_data.get("results") and "track:" in norm_query:
            plain = re.sub(r'track:|artist:', '', norm_query).strip()
            search_data = await self.search(plain, search_type="track", limit=3)

        if not search_data.get("success") or not search_data.get("results"):
            return {"success": False, "error": f"Nothing found for: {query}"}

        top = search_data["results"][0]
        uri = top.get("uri")

        if search_type == "track":
            result = await self._request(
                "PUT", "/me/player/play",
                json_body={"uris": [uri]},
                params={"device_id": device_id},
            )
            result["track"]  = top.get("name")
            result["artist"] = top.get("artist", "")
        else:
            result = await self._request(
                "PUT", "/me/player/play",
                json_body={"context_uri": uri},
                params={"device_id": device_id},
            )
            result["playlist"] = top.get("name")
            result["artist"]   = top.get("name") if search_type == "artist" else ""

        return {**result, "uri": uri, "type": search_type, "query": query}

    # ── Playback controls ──────────────────────────────────────────────────

    async def play(self, uri: Optional[str] = None) -> Dict[str, Any]:
        """Play a specific track URI, or resume current playback."""
        # Get active device
        devices_data = await self.get_devices()
        devices = devices_data.get("devices", [])
        device_id = None
        if devices:
            active = next((d for d in devices if d.get("active")), devices[0])
            device_id = active.get("id")

        if uri and "track" in uri:
            body = {"uris": [uri]}
        elif uri:
            body = {"context_uri": uri}
        else:
            body = None

        params = {"device_id": device_id} if device_id else {}
        return await self._request("PUT", "/me/player/play", json_body=body, params=params, expect_empty=True)

    async def pause(self) -> Dict[str, Any]:
        return await self._request("PUT", "/me/player/pause", expect_empty=True)

    async def skip(self) -> Dict[str, Any]:
        return await self._request("POST", "/me/player/next", expect_empty=True)

    async def previous(self) -> Dict[str, Any]:
        return await self._request("POST", "/me/player/previous", expect_empty=True)

    async def set_volume(self, level: int) -> Dict[str, Any]:
        """Set Spotify volume 0–100. Requires Premium."""
        level = max(0, min(100, level))
        return await self._request(
            "PUT", "/me/player/volume",
            params={"volume_percent": level},
            expect_empty=True,
        )

    async def queue_track(self, uri: str) -> Dict[str, Any]:
        """Add a track to the queue."""
        return await self._request(
            "POST", "/me/player/queue",
            params={"uri": uri},
            expect_empty=True,
        )

    async def shuffle(self, state: bool = True) -> Dict[str, Any]:
        return await self._request(
            "PUT", "/me/player/shuffle",
            params={"state": str(state).lower()},
            expect_empty=True,
        )

    async def repeat(self, mode: str = "track") -> Dict[str, Any]:
        """mode: 'track', 'context', or 'off'"""
        return await self._request(
            "PUT", "/me/player/repeat",
            params={"state": mode},
            expect_empty=True,
        )

    # ── Status ─────────────────────────────────────────────────────────────

    async def get_now_playing(self) -> Dict[str, Any]:
        """Get the currently playing track with full details."""
        data = await self._request("GET", "/me/player/currently-playing")
        if not data.get("success"):
            return data

        item = data.get("item")
        if not item:
            return {"success": True, "playing": False, "track": None}

        progress  = data.get("progress_ms", 0)
        duration  = item.get("duration_ms", 1)
        pct       = int((progress / duration) * 100) if duration else 0
        artists   = ", ".join(a["name"] for a in item.get("artists", []))
        images    = item.get("album", {}).get("images", [])
        image_url = images[0]["url"] if images else ""

        return {
            "success":     True,
            "playing":     data.get("is_playing", False),
            "track":       item.get("name"),
            "artist":      artists,
            "album":       item.get("album", {}).get("name", ""),
            "uri":         item.get("uri"),
            "progress_ms": progress,
            "duration_ms": duration,
            "progress_pct": pct,
            "image_url":   image_url,
            "shuffle":     data.get("shuffle_state", False),
            "repeat":      data.get("repeat_state", "off"),
        }

    async def get_devices(self) -> Dict[str, Any]:
        """Get available Spotify devices."""
        data = await self._request("GET", "/me/player/devices")
        if not data.get("success"):
            return data
        devices = data.get("devices", [])
        return {
            "success": True,
            "devices": [
                {
                    "id":     d.get("id"),
                    "name":   d.get("name"),
                    "type":   d.get("type"),
                    "active": d.get("is_active"),
                    "volume": d.get("volume_percent"),
                }
                for d in devices
            ]
        }

    async def get_playlists(self, limit: int = 10) -> Dict[str, Any]:
        """Get user's playlists."""
        data = await self._request("GET", "/me/playlists", params={"limit": limit})
        if not data.get("success"):
            return data
        items = data.get("items", [])
        return {
            "success": True,
            "playlists": [
                {
                    "name":   p.get("name"),
                    "uri":    p.get("uri"),
                    "tracks": p.get("tracks", {}).get("total", 0),
                    "owner":  p.get("owner", {}).get("display_name", ""),
                }
                for p in items if p
            ]
        }

    # ── Formatting ─────────────────────────────────────────────────────────

    def format_now_playing(self, data: Dict[str, Any]) -> str:
        if not data.get("success"):
            return data.get("error", "Could not get Spotify status.")
        if data.get("mock"):
            return "Spotify not authenticated — run python3 spotify_auth.py"
        if not data.get("playing") or not data.get("track"):
            return "Nothing currently playing on Spotify."
        mins, secs = divmod(data.get("progress_ms", 0) // 1000, 60)
        total_mins, total_secs = divmod(data.get("duration_ms", 0) // 1000, 60)
        bar = self._progress_bar(data.get("progress_pct", 0))
        return (
            f"Now playing: {data['track']} by {data['artist']}\n"
            f"Album: {data['album']}\n"
            f"{bar} {mins}:{secs:02d} / {total_mins}:{total_secs:02d}"
        )

    def format_search(self, data: Dict[str, Any]) -> str:
        if not data.get("success"):
            return data.get("error", "Search failed.")
        results = data.get("results", [])
        if not results:
            return f"No results found for \"{data.get('query', '')}\""
        lines = [f"Spotify results for \"{data.get('query', '')}\":\n"]
        for i, r in enumerate(results, 1):
            if data.get("type") == "track":
                dur = self._fmt_duration(r.get("duration_ms", 0))
                lines.append(f"{i}. {r['name']} — {r.get('artist', '')} ({dur})")
            elif data.get("type") == "artist":
                lines.append(f"{i}. {r['name']}")
            elif data.get("type") == "playlist":
                lines.append(f"{i}. {r['name']} by {r.get('owner','')} ({r.get('tracks',0)} tracks)")
        return "\n".join(lines)

    def format_play_result(self, data: Dict[str, Any]) -> str:
        if not data.get("success"):
            return data.get("error", "Could not play.")
        if data.get("mock"):
            return "Spotify not authenticated — run python3 spotify_auth.py"
        track = data.get("track")
        artist = data.get("artist")
        playlist = data.get("playlist")
        if track and artist:
            return f"Playing: {track} by {artist}"
        elif track:
            return f"Playing: {track}"
        elif playlist:
            return f"Playing playlist: {playlist}"
        elif data.get("artist"):
            return f"Playing music by {data['artist']}"
        return "Playback started."

    def _progress_bar(self, pct: int, width: int = 20) -> str:
        filled = int(width * pct / 100)
        return "▓" * filled + "░" * (width - filled)

    def _fmt_duration(self, ms: int) -> str:
        secs = ms // 1000
        return f"{secs // 60}:{secs % 60:02d}"