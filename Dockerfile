# syntax=docker/dockerfile:1.6
#
# Jarvis — cloud deployment image (Fly.io / Render / Railway).
#
# Stages:
#   1. base    — system libs (ffmpeg, libsndfile, portaudio for voice imports)
#   2. builder — install Python deps, pre-download Whisper + embedding models
#   3. runtime — slim final image, non-root user, models copied in
#
# Build:    docker build -t jarvis:cloud .
# Run:      docker run --rm -p 8000:8000 --env-file .env jarvis:cloud
# Deploy:   fly deploy

# ── Stage 1: base ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# System packages:
#   - ffmpeg, libsndfile1   → audio decoding for whisper / TTS
#   - libportaudio2         → satisfies sounddevice import (no actual mic in cloud)
#   - libpq5, libpq-dev     → psycopg2 → Postgres for FinEx
#   - curl                  → healthcheck + debugging
#   - build-essential, gcc  → build wheels that have no cp311 binary
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libportaudio2 \
        libpq5 \
        libpq-dev \
        curl \
        ca-certificates \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: builder ───────────────────────────────────────────────────────
FROM base AS builder

WORKDIR /build
COPY requirements.txt voice_requirements.txt ./

# Install Python deps. We add `python-dotenv`, `sentence-transformers`, and
# `groq`-related libs explicitly. The voice file references PortAudio-bound
# packages which the base image satisfies.
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt \
 && pip install -r voice_requirements.txt \
 && pip install sentence-transformers python-dotenv

# Pre-download models so cold start is fast on Fly:
#   - sentence-transformers/all-MiniLM-L6-v2 (~80 MB) for embeddings
#   - faster-whisper small.en (~250 MB) for STT (when added via upload endpoint)
# These are cached under /opt/models and copied to the runtime image.
ENV HF_HOME=/opt/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence-transformers \
    XDG_CACHE_HOME=/opt/models/cache

RUN python - <<'PY'
import os, sys
print("→ pre-downloading sentence-transformers model …")
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("→ pre-downloading faster-whisper small.en …")
try:
    from faster_whisper import WhisperModel
    WhisperModel("small.en", device="cpu", compute_type="int8")
except Exception as e:
    print(f"   (faster-whisper preload skipped: {e})", file=sys.stderr)
print("✅ models cached")
PY

# ── Stage 3: runtime ───────────────────────────────────────────────────────
FROM base AS runtime

# Copy installed Python packages and pre-downloaded models from builder.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /opt/models /opt/models

# Non-root user for runtime safety.
RUN useradd -m -u 1001 jarvis
WORKDIR /app

# Copy the app source. .dockerignore should exclude venv/, __pycache__/, .git/, data/, logs/
COPY --chown=jarvis:jarvis . /app

# Persistent data dir (mounted as a Fly volume in fly.toml).
RUN mkdir -p /data/chroma /data/notes /data/logs \
 && chown -R jarvis:jarvis /data /app

USER jarvis

# Tell the app where to find data / models / runtime mode.
ENV LLM_BACKEND=groq \
    RUNNING_IN_DOCKER=1 \
    JARVIS_DATA_DIR=/data \
    CHROMA_DIR=/data/chroma \
    HF_HOME=/opt/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence-transformers \
    XDG_CACHE_HOME=/opt/models/cache \
    UI_HOST=0.0.0.0 \
    UI_PORT=8000 \
    PORT=8000

EXPOSE 8000

# Healthcheck — hit the root path. Server.py serves the UI from /.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# Use uvicorn directly so we can tune workers/timeouts.
# Single worker is fine — model singletons need shared state.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "75"]
