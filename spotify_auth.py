"""
Spotify One-Time Auth Script
Run this ONCE from terminal to get your refresh token:

    python3 spotify_auth.py

It will:
1. Open Spotify login in your browser
2. Start a local server to catch the callback
3. Exchange the code for tokens
4. Save refresh_token to ~/.jarvis/spotify_token.json

After this, spotify.py handles all token refreshing automatically.
"""

import asyncio
import base64
import json
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests

# Load from .env
from dotenv import load_dotenv
import os

load_dotenv()

CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
TOKEN_PATH    = Path.home() / ".jarvis" / "spotify_token.json"

SCOPES = " ".join([
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "playlist-read-private",
    "user-library-read",
    "user-top-read",
])

auth_code = None


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global auth_code
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        if "code" in params:
            auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html><body style="font-family:sans-serif;text-align:center;padding:60px;background:#060410;color:#a78bfa">
                <h1>Jarvis connected to Spotify</h1>
                <p>You can close this tab and return to the terminal.</p>
                </body></html>
            """)
        else:
            self.send_response(400)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress server logs


def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        print("❌ SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET not set in .env")
        return

    # Build auth URL
    params = {
        "client_id":     CLIENT_ID,
        "response_type": "code",
        "redirect_uri":  REDIRECT_URI,
        "scope":         SCOPES,
        "show_dialog":   "true",
    }
    auth_url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(params)

    print("=" * 60)
    print("  Jarvis — Spotify Authentication")
    print("=" * 60)
    print("\nOpening Spotify login in your browser...")
    print("If it doesn't open, visit this URL manually:\n")
    print(f"  {auth_url}\n")
    webbrowser.open(auth_url)

    # Start local server to catch callback
    port = int(REDIRECT_URI.split(":")[-1].split("/")[0])
    server = HTTPServer(("localhost", port), CallbackHandler)
    print(f"Waiting for Spotify callback on port {port}...")
    server.handle_request()  # Handle exactly one request

    if not auth_code:
        print("❌ No auth code received. Did you approve access?")
        return

    # Exchange code for tokens
    credentials = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type":  "application/x-www-form-urlencoded",
        },
        data={
            "grant_type":   "authorization_code",
            "code":         auth_code,
            "redirect_uri": REDIRECT_URI,
        },
    )

    if resp.status_code != 200:
        print(f"❌ Token exchange failed: {resp.text}")
        return

    tokens = resp.json()

    # Save tokens
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps({
        "access_token":  tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "expires_at":    0,  # Force refresh on first use
    }, indent=2))

    print(f"\n✅ Spotify authenticated successfully!")
    print(f"   Tokens saved to: {TOKEN_PATH}")
    print(f"\n   You can now restart Jarvis — Spotify is ready.")


if __name__ == "__main__":
    main()