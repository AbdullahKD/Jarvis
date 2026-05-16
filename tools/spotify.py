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
from config.llm_client import OllamaClient

TOKEN_URL  = "https://accounts.spotify.com/api/token"
API_BASE   = "https://api.spotify.com/v1"
TOKEN_PATH = Path.home() / ".jarvis" / "spotify_token.json"


class SpotifyTool:
    """
    Full Spotify Web API client using Authorization Code flow.
    Requires one-time auth via spotify_auth.py.
    Falls back to mock mode if not authenticated.
    """

    def __init__(self, llm: Optional["OllamaClient"] = None):
        self._llm = llm
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
        """
        Make an authenticated API request.

        Retries once on 502/503/504 with a short delay because Spotify's
        Web API returns transient gateway errors during device wake-up
        even when the action *did* succeed (e.g. PUT /me/player/play
        starts playback then 502s on the way back).
        """
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

        # Up to 2 attempts: original + 1 retry on transient 5xx errors.
        # Spotify's API is famously flaky during playback start.
        import asyncio as _asyncio
        for attempt in range(2):
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
                        # Transient gateway errors — retry once after a brief pause.
                        # Spotify returns these when the device is still waking
                        # up; the action itself often succeeded on the server.
                        if resp.status in (502, 503, 504) and attempt == 0:
                            await _asyncio.sleep(0.4)
                            continue
                        if resp.status >= 400:
                            try:
                                err = await resp.json()
                                msg = err.get("error", {}).get("message", f"HTTP {resp.status}")
                            except Exception:
                                msg = f"HTTP {resp.status}"
                            return {"success": False, "error": msg, "http_status": resp.status}
                        if resp.content_length == 0:
                            return {"success": True}
                        return {"success": True, **(await resp.json())}

            except aiohttp.ClientError as e:
                # Network error — retry once
                if attempt == 0:
                    await _asyncio.sleep(0.4)
                    continue
                return {"success": False, "error": str(e)}

        # Should never get here, but keep the type stable
        return {"success": False, "error": "Request failed after retries"}

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

    # Filler words that aren't part of any track/artist name and only
    # confuse Spotify's search ranking.
    _FILLER_RE = re.compile(
        r'\b(?:please|for me|just|kinda|some|the song|the track|that song|'
        r'that one|you know|i wanna|i want to|can you|could you|would you|'
        r'put on|put me on|throw on|chuck on|stick on|spin|cue up|fire up)\b',
        re.IGNORECASE,
    )

    # "song X by Y" / "X by Y" — high-confidence pattern that doesn't need an LLM.
    _BY_PATTERN = re.compile(
        r'^(.+?)\s+by\s+(.+?)\s*$',
        re.IGNORECASE,
    )

    def _clean_query(self, raw: str) -> str:
        """Strip filler words / trailing 'on spotify' / quotes from a query."""
        q = raw.strip().strip('"\'')
        q = re.sub(r'\s+on spotify\s*$', '', q, flags=re.IGNORECASE)
        q = self._FILLER_RE.sub(' ', q)
        return re.sub(r'\s+', ' ', q).strip(' ,.?!')

    async def _normalise_song_query(self, raw_query: str) -> tuple[str, str]:
        """
        Extract (track, artist) from a natural language request.

        Strategy (fast path first, LLM only as fallback):
          1. Cleaned plain query — strip filler words
          2. "X by Y" regex — captures most explicit cases without an LLM
          3. LLM extraction — only if nothing else worked
        """
        cleaned = self._clean_query(raw_query)

        # Fast path 1: "X by Y"
        m = self._BY_PATTERN.match(cleaned)
        if m:
            track = m.group(1).strip(' ,.?!"\'')
            artist = m.group(2).strip(' ,.?!"\'')
            # Don't be fooled by phrases like "songs by Drake" — that's an
            # artist-only intent, handled separately by the caller.
            if track and artist and track.lower() not in ("song", "songs", "music", "track", "tracks"):
                return track, artist

        # Fast path 2: nothing fancy in the query — just use the cleaned form
        # as a plain text Spotify search. Modern Spotify search handles
        # "blinding lights weeknd" as well as it handles "blinding lights".
        # Skip the LLM call entirely — it adds ~1s and often hurts accuracy.
        return cleaned or raw_query, ""

    async def play_by_name(self, query: str) -> Dict[str, Any]:
        """
        Search for a track/artist/playlist and play the best result.

        Search strategy (broad → narrow, falls back gracefully):
          1. Try the plain cleaned query (e.g. "blinding lights the weeknd")
          2. If nothing, try field-typed query ("track:X artist:Y") when we
             have a "by" split
          3. Pick the highest-popularity match from the first page of results
        """
        import asyncio as _asyncio
        import subprocess as _subprocess

        # Check for active device first; auto-open Spotify if none found
        devices_data = await self.get_devices()
        devices = devices_data.get("devices", [])
        if not devices:
            # Open Spotify silently and wait for it to register a playback device
            try:
                _subprocess.Popen(["open", "-a", "Spotify"])
                await _asyncio.sleep(4)
                devices_data = await self.get_devices()
                devices = devices_data.get("devices", [])
            except Exception:
                pass

        if not devices:
            return {
                "success": False,
                "error": "Spotify opened but no playback device registered yet. Please try again in a moment.",
            }
        active = next((d for d in devices if d.get("active")), devices[0])
        device_id = active.get("id")

        # Detect intent type first
        q_low = query.lower()
        is_playlist = any(w in q_low for w in ["playlist", "mix", "my playlist"])
        is_artist   = any(w in q_low for w in ["songs by", "music by", "discography",
                                                  "play artist", "the artist"])

        if is_playlist:
            search_type = "playlist"
            norm_query  = self._clean_query(re.sub(r'\bplaylist\b', '', query, flags=re.IGNORECASE))
            search_data = await self.search(norm_query, search_type=search_type, limit=5)
        elif is_artist:
            search_type = "artist"
            norm_query  = re.sub(
                r'songs by|music by|discography|play artist|the artist|artist',
                '', query, flags=re.IGNORECASE,
            ).strip()
            norm_query = self._clean_query(norm_query)
            search_data = await self.search(norm_query, search_type=search_type, limit=5)
        else:
            # Track search — try multiple strategies in order
            search_type = "track"
            track, artist = await self._normalise_song_query(query)

            # Strategy 1: plain cleaned text — Spotify's relevance ranking
            # handles "song title artist" naturally. limit=10 lets us pick
            # by popularity rather than blindly taking position 0.
            plain_query = f"{track} {artist}".strip() if artist else track
            search_data = await self.search(plain_query, search_type="track", limit=10)

            # Strategy 2: field-typed query when artist is known and plain
            # search came back empty (common for obscure tracks).
            if not search_data.get("results") and artist:
                typed = f'track:"{track}" artist:"{artist}"'
                search_data = await self.search(typed, search_type="track", limit=10)

            # Strategy 3: last-ditch — strip everything to just the track name
            if not search_data.get("results"):
                search_data = await self.search(track, search_type="track", limit=10)

        if not search_data.get("success") or not search_data.get("results"):
            return {"success": False, "error": f"Nothing found for: {query}"}

        # Pick the best match. Spotify's API ranks by relevance + popularity,
        # but on ambiguous queries the first result is sometimes a cover or
        # karaoke version. Filter those out before picking.
        results = search_data["results"]
        if search_type == "track":
            top = self._best_track_match(results, query)
        else:
            top = results[0]
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

        # Verify against the actual now-playing state — Spotify's PUT
        # /me/player/play endpoint returns 5xx during device wake-up even
        # when playback DID start. If the song the user requested is
        # actually playing, override the failure flag so we report the
        # truth instead of "Bad gateway".
        if not result.get("success"):
            import asyncio as _asyncio
            await _asyncio.sleep(0.5)
            now = await self.get_now_playing()
            if now.get("success") and now.get("playing") and now.get("uri") == uri:
                result["success"] = True
                # Clear the stale error so the formatter doesn't print it
                result.pop("error", None)
                result.pop("http_status", None)

        return {**result, "uri": uri, "type": search_type, "query": query}

    # Markers that indicate a track is a cover / karaoke / instrumental /
    # remake rather than the original recording the user almost certainly
    # wanted. Push these to the back of the candidate list.
    _COVER_MARKERS = (
        "karaoke", "instrumental", "cover", "tribute", "remake",
        "in the style of", "made famous by", "originally performed",
        "8d audio", "slowed", "sped up", "nightcore", "lo-fi",
    )

    def _best_track_match(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Pick the best track from a candidate list.

        Drops covers/karaoke/instrumentals first, then falls back to
        Spotify's own ranking (already roughly popularity-ordered).
        """
        if not results:
            return {}

        def is_cover(r: Dict[str, Any]) -> bool:
            name = (r.get("name") or "").lower()
            album = (r.get("album") or "").lower()
            artist = (r.get("artist") or "").lower()
            blob = f"{name} {album} {artist}"
            return any(m in blob for m in self._COVER_MARKERS)

        # If the user's query itself mentions karaoke/instrumental/etc., they
        # actually want that version — don't filter.
        q_low = query.lower()
        wants_variant = any(m in q_low for m in self._COVER_MARKERS)

        if not wants_variant:
            originals = [r for r in results if not is_cover(r)]
            if originals:
                return originals[0]
        return results[0]

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