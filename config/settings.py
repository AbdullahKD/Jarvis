"""
Jarvis Configuration
All settings loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level up from config/)
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────
# In cloud deployments (Fly.io), JARVIS_DATA_DIR points at a mounted volume
# (e.g. /data) so chroma + logs + notes survive deploys/restarts. Locally
# we fall back to ./data alongside the source.
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
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "0.5"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "2.0"))

# ── Memory ─────────────────────────────────────────────────────────────────
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "5"))
MEMORY_SIMILARITY_THRESHOLD = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.3"))
MEMORY_COLLECTION_NAME = os.getenv("MEMORY_COLLECTION_NAME", "jarvis_memory")

# ── Google APIs ────────────────────────────────────────────────────────────
GOOGLE_CREDENTIALS_PATH = Path(
    os.getenv("GOOGLE_CREDENTIALS_PATH", str(Path.home() / ".jarvis" / "credentials.json"))
)
GOOGLE_TOKEN_PATH = Path(
    os.getenv("GOOGLE_TOKEN_PATH", str(Path.home() / ".jarvis" / "token.json"))
)
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
    "reuters":   "https://feeds.reuters.com/reuters/topNews",
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