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
# M5/24GB server tuning (only takes effect when WE start the daemon below;
# if Ollama.app is already running these are harmless no-ops):
#   MAX_LOADED_MODELS=3 → router + chat + embed models resident together
#   NUM_PARALLEL=2      → router and chat calls can overlap instead of queuing
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-3}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-2}"
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

# 3. Make sure all three models are pulled (chat + router + embeddings) ────
_env_model() { grep -E "^$1=" .env 2>/dev/null | cut -d= -f2-; }
CHAT_MODEL="${OLLAMA_CHAT_MODEL:-$(_env_model OLLAMA_CHAT_MODEL)}";  CHAT_MODEL="${CHAT_MODEL:-qwen3:8b}"
ROUTER_MODEL="${OLLAMA_ROUTER_MODEL:-$(_env_model OLLAMA_ROUTER_MODEL)}"; ROUTER_MODEL="${ROUTER_MODEL:-llama3.2:3b}"
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-$(_env_model OLLAMA_EMBED_MODEL)}"; EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text:latest}"
for MODEL in "$CHAT_MODEL" "$ROUTER_MODEL" "$EMBED_MODEL"; do
  if ! ollama list | awk '{print $1}' | grep -qx "$MODEL"; then
    echo "🟡 Pulling Ollama model: $MODEL ..."
    ollama pull "$MODEL"
  fi
done

# 4. Free port 8000 if a previous Jarvis is still bound ───────────────────
if lsof -tiTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "🟡 Port 8000 in use — killing old process..."
  kill "$(lsof -tiTCP:8000 -sTCP:LISTEN)" 2>/dev/null || true
  sleep 1
fi

# 5. Run with VISIBLE logs (uvicorn directly, ignoring server.py's __main__) ──
echo "✅ Starting Jarvis on http://localhost:8000  (Ctrl-C to stop)"
exec python -m uvicorn server:app --host 0.0.0.0 --port 8000 --log-level info
