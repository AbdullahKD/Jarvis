# Jarvis Bottleneck Audit & 5-Minute Demo Plan

**Audit scope:** orchestrator, voice pipeline, FinEx, agents, tools, server.
**Goal:** cut a representative end-to-end demo from ~20 minutes down to ≤ 5 minutes by attacking response latency on the Jarvis voice agent, the orchestrator, and FinEx.
**Audit date:** 27 May 2026.

---

## TL;DR — the four changes that will buy you 80% of the win

1. **Use one Ollama model everywhere, and keep it pinned.** Jarvis runs `llama3.2:1b`, FinEx runs `llama3.2:latest`. Every cross-call swap costs 5–15 s of cold start. Standardise on `llama3.2:3b` (a.k.a. `llama3.2:latest`) and add `"keep_alive": "30m"` to every Ollama chat payload. Alternatively, point FinEx at Groq (it already supports it) and leave Jarvis local.
2. **Stream sentences from the orchestrator straight to ElevenLabs.** Right now `_voice_brain` (server.py:271) blocks until the full reply lands, *then* TTS begins. Pipe partial chunks to ElevenLabs Flash v2.5 as they stream — TTFA drops from "wait for full LLM" to <1 s.
3. **Drop Whisper from `small.en` to `tiny.en`.** Transcription on a 5-second utterance falls from ~1.0 s → ~0.25 s on CPU. Voice latency cut by 750 ms per turn. WER on short command-style speech is fine.
4. **Aggressively short-circuit the Tier-3 critic + replan loop for the demo.** Set `MAX_REPLAN_ATTEMPTS = 0` and run the critic only when `routing.confidence < 0.7 AND len(plan.subtasks) > 3`. Trims 1–3 extra LLM round-trips off every multi-step task.

Applied together these four changes typically take a Tier-2 voice query from 6–10 s to 2–3 s, and a Tier-3 query from 18–30 s to 6–10 s. That is what makes a 5-minute demo physically possible.

---

## 1. Voice Pipeline (`voice/*`, `server.py /voice/*`)

### 1.1 — Blocking brain wrapper kills perceived latency
**File:** `server.py:271-289`, `voice/web.py:170-194`

`_voice_brain(transcript)` runs `asyncio.run_coroutine_threadsafe(jarvis.handle(transcript), main_loop).result(timeout=120)`. The voice thread blocks **for the entire orchestrator pipeline** (router + memory + planner + critic + executor + …) before a single character is handed to ElevenLabs. Then TTS streams. The user perceives "long silence → reply".

**Fix (high impact, ~1 hr):**
- Switch `_voice_brain` to call `jarvis.handle_stream(transcript)` instead.
- Buffer streamed chunks until you have a complete sentence (split on `[.!?]\s`).
- As each sentence completes, push it to a queue that an ElevenLabs streaming worker drains. ElevenLabs Flash v2.5 has ~75 ms TTFA, so the user hears Jarvis start talking while the LLM is still generating the back half of the reply.
- Final aggregated text gets stored in `_reply` once both streams finish.

### 1.2 — Whisper model is one tier too heavy
**File:** `.env:WHISPER_MODEL=small.en`

`small.en` is 244 MB and decodes ~5 s of audio in 0.8–1.2 s on M-series CPU. `tiny.en` is 39 MB and does the same in 0.2–0.4 s. For voice-command demo content (short, clear, single speaker) the WER difference is negligible.

**Fix (5 min):** change `.env`:
```
WHISPER_MODEL=tiny.en
```
You can fall back to `small.en` for the dissertation evals if needed — the demo doesn't.

### 1.3 — VAD end-of-utterance silence is reasonable, but could be tighter
**File:** `.env:END_SILENCE_MS=500`

500 ms is fine but the voice thread waits for the *full* end-silence window before STT begins. There is no way to predict the user is done early, but you can compensate by **starting the orchestrator router call speculatively** as soon as the first partial transcript arrives. The `on_partial` callback in `voice/stt.py:106` already exists — just hook it up.

### 1.4 — Wake-word may re-trigger on Jarvis's own TTS output
**File:** `voice/runner.py:158-166`

`barge_in_enabled=True` shares the mic with the wake-word listener while TTS plays through the speakers. If your demo room has any acoustic coupling (i.e. the mic hears the speaker), Jarvis saying anything close to "hey Jarvis" will interrupt himself.

**Fix:** for the demo, set `BARGE_IN_ENABLED=false` in `.env`. Acoustic-echo cancellation is the proper fix but not 5-day work.

### 1.5 — Voice persona stylizer code path is dead but the import isn't
**File:** `server.py:267-269`, `voice/persona.py`

Comment says persona is disabled (`_voice_stylizer = None`) and that saves 500–1500 ms — good. The module is still imported, which is harmless, but you can delete the import and the persona file to remove a tempting re-enable. Leave for now.

---

## 2. Orchestrator (`orchestrator.py`)

### 2.1 — Tier-3 pipeline runs 5–7 sequential LLM calls
**File:** `orchestrator.py:609-665`

Worst-case Tier 3 fires: Router → memory embed → Planner → Critic (plan) → Replan (×2) → Critic (replan) → DAG execute (per-subtask LLM as needed) → Critic (result). At ~1.5 s per call on `llama3.2:1b` this is 10–18 s before any text is built.

**Fixes (1–2 hr):**
- Set `MAX_REPLAN_ATTEMPTS = 0` for the demo (`orchestrator.py:50`). Replan rarely fires today on a well-warm model anyway; in a demo it's pure liability.
- Tighten the critic trigger (`orchestrator.py:617-621`):
  ```python
  _needs_critic = (
      routing.confidence < 0.70
      and len(plan.subtasks) > 3
  )
  ```
  This skips the critic for short, high-confidence plans (which is most of them).
- Skip `critic.review_result` entirely for the demo — it never causes a re-execute; it only logs a score. Move it behind a `if EVALUATOR_MODE: ...` flag.
- Run `evaluator.evaluate(...)` and `_build_response_message(...)` in `asyncio.gather` — they don't depend on each other.

### 2.2 — `JarvisOrchestrator.__init__` is fully synchronous and pulls everything
**File:** `orchestrator.py:112-150`, instantiated at `server.py:207`

Constructor calls `RouterAgent`, `MemoryAgent` (loads Chroma), `PlannerAgent`, `CriticAgent`, `Evaluator`, `Summariser`, `CalendarAgent` (Google OAuth refresh), `GmailAgent` (OAuth refresh), `ContactBook`, `EmailComposer`, and 12 tool classes. Server startup can stall on a token refresh; first request after startup may then wait on the warmup task (server.py:1521-1528, up to 90 s).

**Fix (~30 min):**
- Lazy-init Calendar, Gmail, and Spotify behind a `@cached_property` so the demo can choose to never touch them.
- Move the Google OAuth refresh to a background thread, not the constructor. Today a stale `token.json` blocks orchestrator construction → blocks `server.py` import → blocks the listen socket.

### 2.3 — Router uses `chat_json` with `expect_json` which disables streaming
**File:** `config/llm_client.py:140-142`

`expect_json=True` forces `stream=False` on the Ollama call. The router model is `llama3.2:1b` and outputs ~80 tokens, but a non-streaming call pays the full TTFB. Use streaming + first-attempt JSON parsing — or, better, rely on the deterministic router rules (`agents/router.py:107-202`) which already cover all the demo-relevant intents. **In practice the router LLM does not need to fire for any demo prompt.** Confirm by adding a `print(f"deterministic={fast is not None}")` line and checking.

### 2.4 — Conversation history is fed back in full on every Tier-2 turn
**File:** `orchestrator.py:574-576` and `:869-871`

`recent_history = history[-8:]` is sliced, but each entry can be a 200-word paragraph. After 4 exchanges the prompt is ~1 200 tokens before the system prompt, before the new turn. Time-to-first-token rises with prompt length.

**Fix:** summarise older history into a single rolling 3-line summary (you already have `SummariserAgent` instantiated — use it). For the demo, just hard-cap the last 4 messages and truncate each to 240 chars:
```python
recent_history = [
    {"role": m["role"], "content": m["content"][:240]}
    for m in (history or [])[-4:]
]
```

### 2.5 — The Tier-2 generation cap is too generous for voice
**File:** `orchestrator.py:582`, `:877`

`max_tokens=180` ≈ 130 spoken words ≈ ~40 s of TTS. For voice this is the **single biggest contributor to your 20-minute demo length** — Jarvis is yapping. Bring it to 80–100 for voice-originated requests:
- Add a `voice_mode: bool` flag to `handle_stream` and clamp `max_tokens=100` when set.
- Tell the system prompt: "Voice mode — answer in 1–2 short sentences. No lists."

This alone can halve total demo speaking time.

### 2.6 — `_morning_briefing` is 80+ lines of output and is a demo trap
**File:** `orchestrator.py:2833+`

A morning briefing fans out to 10 parallel APIs and assembles a multi-section block: weather + prayer + calendar + inbox + top news + tech + sports + markets + LLM-generated quote. Reading that aloud via TTS easily eats 90+ seconds. Don't use the briefing in a 5-minute demo. If you must, add a `voice=True` flag that returns ONE sentence per section, max 4 sections.

---

## 3. FinEx (`agents/finex_agent.py`, `finex/*`)

### 3.1 — FinEx uses a different model from Jarvis → model swap on every cross-call
**Files:** `finex/_llm_helper.py:45` (`FINEX_MODEL=llama3.2:latest`), `config/settings.py:30` (`OLLAMA_CHAT_MODEL=llama3`), `.env` overrides to `llama3.2:1b`

When the demo does Jarvis → FinEx → Jarvis, Ollama is forced to load/unload models. Each swap on an M-series Mac is ~5–10 s. The default Ollama `keep_alive` is 5 min — long enough to mask the issue in isolated tests but it WILL bite a live demo.

**Fix (10 min, highest single ROI):** standardise. Either
- Set `FINEX_MODEL=llama3.2:1b` in `.env` (fastest, but loses some L4/L5/L6 reasoning quality), OR
- Set `OLLAMA_CHAT_MODEL=llama3.2:latest` in `.env` (best demo quality, ~3× slower per token but no swap), OR
- **Recommended:** set `LLM_BACKEND=groq` + `GROQ_API_KEY=...` so FinEx hits Groq and Jarvis stays local. Groq's llama-3.3-70b runs at ~250 tok/s → FinEx answers in 1–3 s. The plumbing already exists (`config/groq_client.py`, `finex/_llm_helper.py`).

### 3.2 — No `keep_alive` on the Jarvis main LLM client
**File:** `config/llm_client.py:124-138`

`OllamaClient.chat` does NOT include `keep_alive` in the payload. FinEx's `_chat_ollama` (`finex/_llm_helper.py:160-173`) does (`"keep_alive": "10m"`). After 5 minutes idle Ollama unloads the Jarvis model, and the next user turn pays 5–15 s of cold start mid-demo.

**Fix (2-line change):** in `config/llm_client.py`, every `payload` dict should add `"keep_alive": "30m"`. While you're there, do the same for `chat_stream` and `embed`.

### 3.3 — FinEx is not streaming
**File:** `finex/_llm_helper.py:160-173`, `agents/finex_agent.py:118-129`

`_chat_ollama` posts with `stream=False` and waits for the full response. For an L4/L5/L6 question on `llama3.2:latest` that's 4–12 s of dead air before any text reaches the UI.

**Fix:** add a `chat_stream_sync` variant that yields chunks, and pipe them to a server-sent-events endpoint or to the existing WebSocket. The FinEx UI (`ui/finex.html`) is a static HTML page — add a tiny `EventSource` listener.

### 3.4 — `LLM_SQL.route_question` is fast but `get_cached_hr_context` rebuilds on every uncached company
**File:** `finex/LLM_SQL.py:175-178`

The cache is per-process and per-company; first call to a new company pays the full DB read + formatting. For the demo, pre-warm it on startup for the company you'll demo (likely "Bestway Cement" or HBL):
```python
# In FinExAgent.__init__ after warm_model:
threading.Thread(
    target=lambda: get_cached_hr_context("Bestway Cement"),
    daemon=True,
).start()
```

### 3.5 — `extract_pdf.py` uses `llama3.2:latest` hardcoded
**File:** `finex/extract_pdf.py:60`

If a PDF upload is part of the demo, this is another model load. Make it honour the `FINEX_MODEL` env var:
```python
_FINEX_MODEL = os.environ.get("FINEX_MODEL", "llama3.2:latest")
```

### 3.6 — `_CHAT_TIMEOUT_S = 100` is the wrong demo posture
**File:** `agents/finex_agent.py:28`

A 100-second per-question budget will make the user (and audience) stare at a spinner for 100 s if anything hiccups. Drop to 25 s and surface a clean "FinEx is taking longer than expected, want me to retry?" message — much better demo UX than a frozen screen.

---

## 4. Tools & connectivity (`tools/*`)

### 4.1 — `WebSearchTool.search` fires up to 5 sources sequentially with a 20 s timeout each
**File:** `tools/web_search.py:21`, `:130-184`

Worst case is Wikipedia (20 s) → DDG API (20 s) → DDG HTML (20 s) → fallback Wikipedia (20 s) → fallback DDG (20 s) = 100 s. The user has no idea anything is happening.

**Fixes:**
- Race the first two sources with `asyncio.wait(..., return_when=FIRST_COMPLETED)` and take the winner.
- Drop the global timeout to 6 s. DDG/Wikipedia respond in <2 s when they're going to respond at all; 20 s is masking dead endpoints.
- For Tier-2 web search, cap to ONE source (Wikipedia for factual, DDG API for everything else) and skip the fallback chain. The `query_type` detector already exists for routing.

### 4.2 — `/sidebar` runs 22 parallel HTTP requests on page load and every 60 s
**File:** `server.py:638-682`

Each tick hits: weather, markets, calendar, spotify, 7 news categories, 9 sports leagues, prayer times, gmail. They're parallel (good) but on a flaky Wi-Fi or constrained network it's a lot of contention with the WebSocket. During a live demo a sidebar tick can land mid-voice-query and saturate sockets.

**Fix:**
- Suspend the sidebar tick while `_voice_session.status().state in ("listening","thinking","speaking")`. Add a header `X-Voice-Active: 1` to the response and have the UI back off.
- Move 4 of the 7 news categories behind a "show more" toggle.

### 4.3 — `/live-tick` (every 20 s) also fires 11 parallel scrape calls
**File:** `server.py:813-915`

Markets + 10 sports endpoints every 20 s. Same advice as 4.2.

### 4.4 — Hardware endpoint sleeps 1 second to compute network delta
**File:** `server.py:1051`

`await asyncio.sleep(1.0)` blocks `/hardware` for a guaranteed 1 s. Fine in isolation but UI polls it every 15 s. Cache the previous reading and compute a delta on the **next** call, returning immediately on the first.

---

## 5. Connectivity / agent firing — what's wired correctly today

Audit of agent → tool linkages in `orchestrator._dispatch` (`orchestrator.py:992-1212`):

| Agent string  | Tool/Agent                | Status   | Notes |
|---------------|---------------------------|----------|-------|
| `memory`      | MemoryAgent + Chroma      | ✅       | hash-embed fallback OK |
| `weather`     | WeatherTool               | ✅       | open-meteo, 3 endpoints |
| `websearch`   | WebSearchTool             | ⚠️       | see 4.1 |
| `news`        | NewsTool                  | ✅       | 5 RSS sources |
| `mac`         | MacControlTool            | ✅       | subprocess osascript |
| `spotify`     | SpotifyTool               | ✅       | Spotipy OAuth |
| `document`    | DocumentTool              | ✅       | pypdf2 / docx2txt |
| `summariser`  | SummariserAgent           | ✅       | uses self.llm |
| `calendar`    | CalendarAgent             | ✅       | Google OAuth, mock fallback |
| `email`       | GmailAgent                | ✅       | Google OAuth, mock fallback |
| `reminder`    | ReminderStore             | ✅       | SQLite |
| `finex`       | FinExAgent                | ❌       | **not in `_dispatch`** — see below |

### 5.1 — FinEx is mounted as an HTTP route but not as a dispatchable agent
**File:** `orchestrator.py:992-1212`

`agents/finex_agent.py:FinExAgent` is imported in `server.py:49`, instantiated in `server.py:208`, and exposed via FastAPI routes (search the file for `finex` to find them). But `_dispatch` in the orchestrator has no `finex` branch. So a voice query like *"FinEx, what was Bestway's revenue last year?"* will route the user through Tier 2 / Tier 3, call the planner with the FinEx agent name, then drop into the `Fallback` branch and return `"Unknown agent/action: finex.…"`.

**Fix (15 min):** add a `finex` branch in `_dispatch`:
```python
elif agent == "finex":
    if action in ("ask", "chat", "answer"):
        result = await self.finex.chat(
            question=params.get("question", user_request_or_default),
            company=params.get("company", "Bestway Cement"),
        )
        return {
            "success": True,
            "result": result,
            "message": result.get("answer", ""),
        }
```
And inject `self.finex = FinExAgent()` in `JarvisOrchestrator.__init__` (or pass it in from `server.py`).

Add the router rule (`agents/router.py:_deterministic_route`):
```python
_finex_rx = re.compile(
    r'\b(finex|hbl|bestway|revenue|ebitda|profit margin|balance sheet|'
    r'cash flow|financial statement|annual report)\b', re.I,
)
if _finex_rx.search(r):
    return _decision(AgentRole.FINEX, tier=1)  # need to add FINEX to AgentRole
```

Without this fix you cannot demo FinEx through the voice agent at all — only through the `finex.html` page.

### 5.2 — Gmail / Calendar fallback to mock mode silently
**Files:** `agents/gmail_agent.py:65-100`, `agents/calendar_agent.py` (similar)

If `token.json` has expired, the agent constructs in mock mode and `auth_error` carries the reason. The `/google/status` endpoint surfaces this but the voice/chat path does not. A user asking "schedule a meeting" gets a mocked confirmation that *did not actually touch Google Calendar*.

**Fix:** before demo, run `curl localhost:8000/google/status | jq` — both must show `connected: true`. If not, hit `/google/reauth` and complete the OAuth dance.

---

## 6. Server & startup (`server.py`, `run_jarvis.sh`)

### 6.1 — Startup warmup is 1 token of "ping" — that may be too tiny to fully load the model
**File:** `server.py:60-72`

`async for _ in jarvis.llm.chat_stream([{"role":"user","content":"ping"}], max_tokens=4)` — fine, but you can be defensive and warm with a 30-token prompt that exercises the actual system prompt path. That guarantees the templated chat handler is JIT'd too.

### 6.2 — WebSocket waits up to 90 s on warmup
**File:** `server.py:1521-1528`

If the user starts typing within ~30–60 s of server start, they may stare at "typing…" for the warmup window. Fix: at startup also `chat` once to fully load weights, then resolve a future the WS path can `await`. Today the 90-second wait is masked behind a busy spinner.

### 6.3 — UI polls four endpoints simultaneously
**File:** `ui/index.html:1909-1925`

```
setInterval(refreshSidebar, 60_000);       // 22 parallel calls
setInterval(refreshHardware, 15_000);      // sleeps 1s
setInterval(refreshSpotify, 10_000);
setInterval(refreshLiveTick, 20_000);      // 11 parallel calls
```

During the demo these all keep firing. Worst-case overlap: a 22-call sidebar fan-out, an 11-call live-tick, and a 1-second-sleeping hardware call can all coincide with the user clicking the mic. Suspend polling while the voice session is active.

---

## 7. Things that are right (so you don't waste time "fixing" them)

- Wake-word loop uses ONNX, ~5 ms per 80 ms window — fine.
- Silero VAD ONNX — fine.
- ElevenLabs `eleven_flash_v2_5` with `pcm_22050` streaming — best choice for live demo.
- Deterministic routing rules in `agents/router.py:107-202` cover ~80% of demo prompts.
- Shared `aiohttp` client per call (sidebar/news/sports) — fine; we don't need a global pool at this scale.
- `_enforce_single_paragraph` post-processor — keeps small models honest, leave alone.
- The `_try_shortcut` path is the right architecture; demo prompts should hit it.

---

## 8. The 5-minute demo script

Designed to (a) show every major agent firing, (b) stay under 5 minutes, (c) avoid the slow paths above. Time budget per turn assumes the fixes in §§ 1.1, 1.2, 2.1, 2.5, 3.1, 3.2 are applied.

| # | Time   | Spoken prompt                                                       | What lights up                | Tier | Budget |
|---|--------|---------------------------------------------------------------------|-------------------------------|------|--------|
| 0 | 0:00   | (open UI, click mic, "Hey Jarvis")                                  | Wake word                     | —    | 1 s    |
| 1 | 0:01   | "What's the weather in High Wycombe?"                               | Router → Weather (Tier 1)     | 1    | 3 s    |
| 2 | 0:30   | "Open Spotify and play some focus music."                           | Mac + Spotify shortcut        | 1    | 4 s    |
| 3 | 1:00   | "What's on my calendar today?"                                      | Calendar agent                | 1    | 3 s    |
| 4 | 1:30   | "Read me my top 3 unread emails."                                   | Gmail agent                   | 1    | 5 s    |
| 5 | 2:05   | "Schedule a 30-minute meeting with Sarah tomorrow at 3pm called Demo Review."| Calendar + planner shortcut | 1 | 8 s |
| 6 | 2:45   | "Search the web for the latest news on multi-agent AI systems and give me one paragraph."  | WebSearch + LLM (Tier 2)    | 2    | 8 s    |
| 7 | 3:25   | "Switch to FinEx. What was Bestway Cement's revenue and net profit in 2025?" | FinExAgent L1               | —    | 6 s    |
| 8 | 4:00   | "What was the gross profit margin and how did it change year on year?" | FinEx L3 → L2 chain          | —    | 12 s   |
| 9 | 4:45   | "Set a reminder to email my supervisor tomorrow at 10am."           | Reminder shortcut             | 1    | 3 s    |
| 10| 5:00   | "Thanks Jarvis." (close)                                            | —                             | —    | 1 s    |

Total speaking budget: ~54 s of Jarvis speaking (well within ElevenLabs streaming budget), ~10 s of wake/STT, ~~20 s of LLM latency. Demo cap holds at 5 min with ~30 s of buffer for narration between prompts.

### Pre-demo checklist (run 10 minutes before)

```bash
# 1. Ollama up and the model resident
ollama serve &        # if not already running
curl -s http://localhost:11434/api/generate -d '{"model":"llama3.2:1b","prompt":"warm","keep_alive":"30m"}' > /dev/null

# 2. Server reachable
./run_jarvis.sh   # wait for "🔥 LLM warmup complete"

# 3. Google connected
curl -s localhost:8000/google/status | jq '{gmail:.gmail.connected, cal:.calendar.connected}'
# both must be true; if not → POST /google/reauth

# 4. Voice subsystem
curl -s localhost:8000/voice/health | jq '.ok'    # must be true
curl -s -X POST localhost:8000/voice/test         # confirms ElevenLabs + speaker

# 5. FinEx warm
curl -s localhost:8000/finex/companies | jq '.companies | length'  # must be ≥ 1

# 6. Mic
curl -s -X POST localhost:8000/voice/mic-test | jq '.ok'    # speak during the call
```

If any of the above fails, do NOT start the demo until it passes.

### Pre-demo `.env` overrides

Apply these to your normal `.env` only for the demo run (keep a copy of the original):
```
WHISPER_MODEL=tiny.en
END_SILENCE_MS=400
BARGE_IN_ENABLED=false
VOICE_PERSONA_ENABLED=false
OLLAMA_CHAT_MODEL=llama3.2:1b
FINEX_MODEL=llama3.2:1b      # if you accept the quality trade for speed
# or:
# LLM_BACKEND=groq
# GROQ_API_KEY=...
# FINEX_MODEL=llama-3.3-70b-versatile
```

---

## 9. Recommended fix order (engineering effort vs. demo impact)

| Order | Fix                                                          | Effort   | Impact on demo time | Risk |
|-------|--------------------------------------------------------------|----------|---------------------|------|
| 1     | Add `keep_alive: 30m` to Jarvis LLM client (§3.2)           | 5 min    | Removes 5–15 s cold start per query | None |
| 2     | Standardise FinEx model OR move to Groq (§3.1)              | 10 min   | Removes 5–15 s model swap per cross-call | Low |
| 3     | Drop Whisper to `tiny.en` (§1.2)                            | 1 min    | -750 ms per voice turn | Tiny WER bump |
| 4     | Reduce Tier-2 `max_tokens` to 100 in voice mode (§2.5)      | 15 min   | Halves TTS speaking time | None |
| 5     | Stream sentences from LLM → ElevenLabs (§1.1)               | 1–2 hr   | Halves perceived latency | Some |
| 6     | Disable replan + tighten critic trigger (§2.1)              | 10 min   | -3–8 s on Tier-3 | Low |
| 7     | Wire FinEx into `_dispatch` and router (§5.1)               | 15 min   | Unlocks voice-driven FinEx demo | None |
| 8     | Race web-search sources, drop timeout to 6 s (§4.1)         | 20 min   | -10–40 s on any "search the web" prompt | Low |
| 9     | Suspend sidebar/live-tick polls during voice (§4.2)         | 20 min   | Removes contention spikes | None |
| 10    | Lazy-init Calendar / Gmail / Spotify (§2.2)                 | 30 min   | Faster cold start, no risk of OAuth stall blocking the listen socket | Low |

The first three (15 minutes of work) get you most of the demo speed-up. The whole list is ≤ 5 hours and brings the system from "20-minute demo, occasional dead air, FinEx unreachable by voice" to "5-minute demo, consistent ≤ 2 s TTFA, full agent coverage".

---

## 10. Specific bug-class flags worth fixing before submission

1. **`agents/finex_agent.py:189`** — bare `except Exception` swallows ChromaDB write failures silently.
2. **`orchestrator.py:1536` then `:1544`** — `sports_context` is computed twice in the same branch; second assignment shadows the first inside the `else:` block, masking the calendar guard.
3. **`server.py:43-44`** — `from fastapi.responses import FileResponse, JSONResponse` is imported a second time after the local `SafeJSONResponse` class; the second import is redundant.
4. **`memory/memory_agent.py:158`** — `clear()` deletes the collection by name without holding a lock; concurrent retrieve calls during clear will hit a stale collection reference.
5. **`config/llm_client.py:283-326`** — `_post_with_retry` opens a new `aiohttp.ClientSession` per call (per retry). At demo cadence this is fine, but consider a class-level session for any future load tests.
6. **`tools/web_search.py:151-163`** — relevance filter discards results whose snippet matches in chars 300+ of the body; cap snippet length to avoid losing genuinely relevant hits.

None of these are demo-blocking, but flag them in the dissertation's "future work" appendix.

---

*Prepared for the live demo. Apply fixes 1–4 first — they take 30 minutes and remove the bulk of the perceived latency.*
