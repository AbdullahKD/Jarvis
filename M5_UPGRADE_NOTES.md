# M5 / 24 GB upgrade — applied 18 July 2026

## ⚠ One manual step: update `.env`

The tooling couldn't write `.env` directly (secrets file). Open `.env` and make
these changes by hand (the full updated copy was also attached in the chat):

```env
# Was llama3.2:latest — 8B backbone for the M5
OLLAMA_CHAT_MODEL=qwen3:8b

# Was llama3.2:1b — 3b routes far more accurately, still instant on M5
OLLAMA_ROUTER_MODEL=llama3.2:3b

# Was llama3.2:latest,mistral:7b — the dissertation's planned sweep + new backbone
BENCHMARK_MODELS=llama3.2:1b,llama3.2:3b,phi3:mini,gemma2:2b,qwen3:8b

# Was base.en — the accuracy tier the report documents, fast enough on M5
WHISPER_MODEL=small.en

# Was llama3.2:latest
FINEX_MODEL=qwen3:8b

# NEW — add these lines:
OLLAMA_KEEP_ALIVE=-1        # pin models in RAM forever (was 30m, an 8GB-era setting)
LLM_NUM_CTX=8192            # chat context window (was Ollama's 4096 default)
MAX_REPLAN_ATTEMPTS=0       # set to 1 to re-enable the full critic→replan loop
```

Then pull the new models once (run_jarvis.sh now also does this automatically):

```bash
ollama pull qwen3:8b
ollama pull llama3.2:3b
```

## What was changed in code (already applied)

- **gmail_agent.py / calendar_agent.py** — every blocking Google API
  `.execute()` now runs via `asyncio.to_thread`; inbox fetch and
  mark-all-as-read are parallelised (were serial N+1 loops).
- **memory_agent.py** — ChromaDB add/query/count/delete moved off the event
  loop; hash-fallback embeddings are now tagged `embedding_degraded` in
  metadata instead of silently polluting semantic search.
- **orchestrator.py** — removed the duplicate `_try_shortcut` call on the
  Tier-1 miss path (pure wasted latency); `MAX_REPLAN_ATTEMPTS` is now
  env-configurable; elaborate cap raised 450 → 900 tokens.
- **server.py** — CORS pinned to localhost (extend via
  `JARVIS_ALLOWED_ORIGINS`); Basic-auth uses constant-time comparison;
  `/hardware` sanitises the Wi-Fi interface name before shell interpolation.
- **mac_control.py** — new `_as_quote()` AppleScript escaper (backslashes
  then quotes) used in `open_app`, `quit_app`, `hide_app`, `notify` —
  closes an AppleScript-injection path.
- **config/llm_client.py + settings.py** — `keep_alive` and `num_ctx` are
  env-driven (`OLLAMA_KEEP_ALIVE`, `LLM_NUM_CTX`); stream default max_tokens
  512 → 1024.
- **run_jarvis.sh** — exports `OLLAMA_MAX_LOADED_MODELS=3` and
  `OLLAMA_NUM_PARALLEL=2` when it starts the Ollama daemon; auto-pulls chat,
  router AND embed models.
- **.env.example** — updated to match the new recommended config.

## Still outstanding (not code-fixable from here)

1. **`finex/` source files are missing** — the folder holds only an empty
   `__pycache__`. Restore `LLM_SQL.py`, `db_query.py`, `db_insert.py`,
   `extract_pdf.py` from git history or a backup, or FinEx stays in
   "unavailable" mode.
2. **Move the project out of iCloud-synced Desktop** (or exclude it) — the
   `venv.icloud.bak` folder, `.fuse_hidden*` files and duplicate
   `jarvis 2.db`/`jarvis 3.db` are iCloud eviction damage; it likely ate
   `finex/` too.
3. Optional next round (deliberately not done in this pass to keep the demo
   stable): parallel DAG subtask execution, shared aiohttp session reuse,
   print→logging migration, `_try_shortcut` registry refactor, trialling
   `qwen3:30b-a3b` (30B MoE) as the Tier-3 backbone.
