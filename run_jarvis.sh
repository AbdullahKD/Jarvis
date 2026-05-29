#!/usr/bin/env bash
# Convenience launcher for Jarvis.
#  - activates the venv
#  - makes sure Ollama is up (and starts it if not)
#  - runs the server with VISIBLE uvicorn logs so you can see startup
#
# Usage:  ./run_jarvis.sh

set -e
cd "$(dirname "$0")"

# 1. venv ──────────────────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
  echo "❌ No venv found. Create one with: python3 -m venv venv && pip install -r requirements.txt"
  exit 1
fi
# shellcheck disable=SC1091
source venv/bin/activate

# 2. Ollama ────────────────────────────────────────────────────────────────
if ! curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null; then
  echo "🟡 Ollama not running — starting in background..."
  # `ollama serve` is the daemon. The Ollama.app does the same thing.
  if command -v ollama >/dev/null; then
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    # Wait up to 10s for it to come up
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      sleep 1
      curl -s --max-time 1 http://localhost:11434/api/tags >/dev/null && break
    done
  else
    echo "❌ 'ollama' binary not found. Install from https://ollama.com/download"
    exit 1
  fi
fi

# 3. Make sure the chat model is pulled ────────────────────────────────────
MODEL="${OLLAMA_CHAT_MODEL:-$(grep -E '^OLLAMA_CHAT_MODEL=' .env 2>/dev/null | cut -d= -f2-)}"
MODEL="${MODEL:-llama3.2:1b}"
if ! ollama list | awk '{print $1}' | grep -qx "$MODEL"; then
  echo "🟡 Pulling Ollama model: $MODEL ..."
  ollama pull "$MODEL"
fi

# 4. Free port 8000 if a previous Jarvis is still bound ───────────────────
if lsof -tiTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "🟡 Port 8000 in use — killing old process..."
  kill "$(lsof -tiTCP:8000 -sTCP:LISTEN)" 2>/dev/null || true
  sleep 1
fi

# 5. Run with VISIBLE logs (uvicorn directly, ignoring server.py's __main__) ──
echo "✅ Starting Jarvis on http://localhost:8000  (Ctrl-C to stop)"
exec python -m uvicorn server:app --host 0.0.0.0 --port 8000 --log-level info
