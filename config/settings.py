"""
Jarvis Configuration
All settings loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level up from config/)
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Google API serialisation ────────────────────────────────────────────────
# The Google API client's default transport (httplib2) is NOT thread-safe: a
# `service` object shares one connection / SSL socket. Calling `.execute()` from
# multiple threads at once corrupts OpenSSL's BIO state and segfaults the whole
# process (EXC_BAD_ACCESS in libcrypto). We therefore route EVERY Gmail +
# Calendar call through one shared single-worker executor so they run strictly
# one at a time. Latency cost is negligible (a handful of calls per refresh);
# the alternative is a hard crash once Gmail/Calendar are live.
import functools as _functools
import asyncio as _asyncio
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor

GOOGLE_API_EXECUTOR = _ThreadPoolExecutor(max_workers=1, thread_name_prefix="google-api")


def google_call(fn, *args):
    """Run a blocking Google API call on the shared single-thread executor.

    Returns an awaitable. Use in place of ``asyncio.to_thread(...)`` for any
    Gmail/Calendar service call so concurrent SSL access can never happen.
    """
    loop = _asyncio.get_running_loop()
    target = _functools.partial(fn, *args) if args else fn
    return loop.run_in_executor(GOOGLE_API_EXECUTOR, target)


# ── Paths ──────────────────────────────────────────────────────────────────
# JARVIS_DATA_DIR relocates chroma + logs + notes as a group — useful for
# pointing them at an external disk. Unset, everything lives in ./data
# alongside the source.
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv("JARVIS_DATA_DIR", str(BASE_DIR / "data")))
LOGS_DIR = Path(os.getenv("JARVIS_LOGS_DIR", str(DATA_DIR / "logs"))) \
    if os.getenv("JARVIS_DATA_DIR") else BASE_DIR / "logs"
NOTES_DIR = Path(os.getenv("JARVIS_NOTES_DIR", str(DATA_DIR / "notes")))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(DATA_DIR / "chroma")))
SQLITE_PATH = Path(os.getenv("JARVIS_SQLITE_PATH", str(DATA_DIR / "jarvis.db")))

for _dir in [LOGS_DIR, DATA_DIR, NOTES_DIR, CHROMA_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Ollama / LLM ───────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
OLLAMA_ROUTER_MODEL = os.getenv("OLLAMA_ROUTER_MODEL", "llama3.2:1b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Models available for benchmarking comparison
BENCHMARK_MODELS = os.getenv(
    "BENCHMARK_MODELS", "llama3,mistral"
).split(",")

# ── LLM generation settings ────────────────────────────────────────────────
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
# WS layer caps total work at 120s — keep per-call budget well under that
# so retries can produce a real error rather than a misleading "timed out".
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
# Default context window for chat calls. Ollama's own default is 4096;
# 8192 is comfortable on the M5/24GB machine (per-call override still wins).
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))
# keep_alive sent with every Ollama request. -1 pins models in RAM forever
# (right call on 24GB); "30m" was the old 8GB-friendly setting.
# NOTE: Ollama's JSON API accepts a duration STRING ("30m", "24h") or a
# NUMBER (seconds; -1 = forever). A bare "-1" from the env arrives as a
# string and fails Go duration parsing ('missing unit in duration "-1"'),
# so coerce purely-numeric values to int before they hit the payload.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
try:
    OLLAMA_KEEP_ALIVE = int(OLLAMA_KEEP_ALIVE)
except ValueError:
    pass  # duration string like "30m" — pass through as-is
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "0.5"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "2.0"))

# ── Memory ─────────────────────────────────────────────────────────────────
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "5"))
MEMORY_SIMILARITY_THRESHOLD = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.3"))
MEMORY_COLLECTION_NAME = os.getenv("MEMORY_COLLECTION_NAME", "jarvis_memory")

# ── Google APIs ────────────────────────────────────────────────────────────
# Resolve Google OAuth file locations robustly. Order of precedence:
#   1. Explicit env override (GOOGLE_CREDENTIALS_PATH / GOOGLE_TOKEN_PATH)
#   2. ~/.jarvis/<file>            (the documented default location)
#   3. <repo root>/<file>         (files kept next to the source, as this repo has)
# This stops Gmail + Calendar from silently dropping to mock mode just because
# credentials.json lives in the repo root instead of ~/.jarvis. A freshly minted
# token is written next to whichever credentials.json we resolved, so re-auth
# lands somewhere the next startup will actually read.
def _resolve_google_path(env_key: str, filename: str) -> Path:
    override = os.getenv(env_key)
    if override:
        return Path(override).expanduser()
    for cand in (Path.home() / ".jarvis" / filename, BASE_DIR / filename):
        if cand.exists():
            return cand
    # Nothing on disk yet — default to ~/.jarvis (created on demand by OAuth flow).
    return Path.home() / ".jarvis" / filename

GOOGLE_CREDENTIALS_PATH = _resolve_google_path("GOOGLE_CREDENTIALS_PATH", "credentials.json")

_token_override = os.getenv("GOOGLE_TOKEN_PATH")
if _token_override:
    GOOGLE_TOKEN_PATH = Path(_token_override).expanduser()
else:
    _existing_token = next(
        (c for c in (Path.home() / ".jarvis" / "token.json", BASE_DIR / "token.json") if c.exists()),
        None,
    )
    # Honour an existing token; otherwise write the fresh one beside the creds.
    GOOGLE_TOKEN_PATH = _existing_token or (GOOGLE_CREDENTIALS_PATH.parent / "token.json")
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.modify",
]

# ── Spotify ────────────────────────────────────────────────────────────────
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

# ── Voice ──────────────────────────────────────────────────────────────────
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
USE_LOCAL_TTS = os.getenv("USE_LOCAL_TTS", "true").lower() == "true"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny/base/small/medium

# ── Web UI ─────────────────────────────────────────────────────────────────
UI_HOST = os.getenv("UI_HOST", "0.0.0.0")
UI_PORT = int(os.getenv("UI_PORT", "8000"))

# ── Evaluator thresholds ───────────────────────────────────────────────────
EVALUATOR_MIN_SCORE = float(os.getenv("EVALUATOR_MIN_SCORE", "0.6"))
CRITIC_REPLAN_THRESHOLD = float(os.getenv("CRITIC_REPLAN_THRESHOLD", "0.5"))

# ── News RSS feeds ─────────────────────────────────────────────────────────
RSS_FEEDS = {
    "bbc":       "http://feeds.bbci.co.uk/news/rss.xml",
    # feeds.reuters.com was retired by Reuters and has returned nothing for
    # years. tools/news.py catches ET.ParseError with a bare `pass`, so the
    # source simply never appeared in results and nothing said why.
    "reuters":   "https://news.google.com/rss/search?q=when:24h+allinurl:reuters.com&hl=en-GB&gl=GB&ceid=GB:en",
    "hackernews":"https://hnrss.org/frontpage",
    "techcrunch":"https://techcrunch.com/feed/",
    "guardian":  "https://www.theguardian.com/world/rss",
}

# ── User preferences ──────────────────────────────────────────────────────
FAVOURITE_TEAMS = os.getenv("FAVOURITE_TEAMS", "Manchester United,Real Madrid,Golden State Warriors,Pakistan Cricket").split(",")
FAVOURITE_FOOTBALL_LEAGUE = os.getenv("FAVOURITE_FOOTBALL_LEAGUE", "premier_league")
FAVOURITE_BASKETBALL_LEAGUE = os.getenv("FAVOURITE_BASKETBALL_LEAGUE", "nba")

# ── Weather ────────────────────────────────────────────────────────────────
DEFAULT_LATITUDE = float(os.getenv("DEFAULT_LATITUDE", "51.5074"))
DEFAULT_LONGITUDE = float(os.getenv("DEFAULT_LONGITUDE", "-0.1278"))
DEFAULT_LOCATION_NAME = os.getenv("DEFAULT_LOCATION_NAME", "High Wycombe")