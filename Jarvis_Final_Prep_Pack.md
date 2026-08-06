# Jarvis — Final Prep Pack
**Abdullah Khan Durrani · COM6001 · presentation 12 June 2026, 5pm**
One document for tomorrow: (A) the presentation plan, (B) condensed notes on everything we covered, (C) the **code-verified** mismatch list. The demo script is in its own file: `Jarvis_5min_Demo_Plan.md`.

> Verification status: I read the real `~/Desktop/Jarvis` source (orchestrator.py, agents/, config/, tools/, voice/, server.py, .env). Every mismatch in Section C below is confirmed against actual lines. This matches the existing `Jarvis_Code_Audit.md` and adds the routing subtlety in C7.

---

# A. PRESENTATION PLAN

## The single thesis — say it on Slide 2, repeat on Slide 11
> **When an LLM assistant fails at a real multi-step task, the limiting factor is usually the architecture around the model, not the model itself — proven by getting a ~3B local model to 0.83 mean / 100% pass purely through structure.**

Defend the *precise* version: structure took a small model from unreliable to reliable on representative tasks. Do **not** claim "architecture beats model size" outright — the clean multi-model benchmark that would prove that is built but not fully run.

## Timing (≈18–20 min talk + 5 min demo)
| Slide | Content | Target | Compress to |
|---|---|---|---|
| 1 | Title | 20s | 15s |
| 2 | Problem & research question | 2m | 90s |
| 3 | Literature / the gap | 2m | 90s |
| 4 | Aim & objectives | 1.5m | 1m |
| 5 | Methodology, ethics, risk | 2m | 90s |
| 6 | Requirements | 1m | 30s (skim) |
| 7 | **Architecture + tiers (core)** | 3.5m | 3m — never below |
| 8 | Development / hard parts | 2m | 90s |
| 9 | Results | 2m | 90s |
| 10 | **Live demo** | 5m | 4m |
| 11 | Conclusion & contributions | 1.5m | 1m |
| 12 | Further work | 1m | 45s |
| 13 | Thank you | 20s | 15s |

**If running long, compress Slides 6 and 8. Never compress Slide 7 or the demo.**

## Per-slide beats
- **S1 Title (20s):** "I'm Abdullah; this is Jarvis, a local-first multi-agent assistant. In 20 minutes I'll show how it works and what it proves." Move on.
- **S2 Problem (2m):** LLMs reason fluently but assistants around them act like one-shot chatbots. End slow on the research question: *"Is the model the limit, or the architecture around it?"*
- **S3 Gap (2m):** three fields — reasoning (CoT, ReAct, Reflexion), multi-agent+tools (CrewAI, LangGraph, AutoGen), memory (RAG, MemGPT, HNSW). Honest line: "No single part is new; combining all three into something you run locally and *evaluate end to end* is the point."
- **S4 Aim (1.5m):** modular, local, decomposes, remembers, acts, self-checks. Call out three objectives: separate agents, DAG planning, critic-driven replanning. "I've used it as my own assistant for ~6 months — that's where the fixes came from."
- **S5 Method/ethics (2m):** DSR — built the artefact as the study, which is why the Evaluator logged from early. Two honest scope changes: single-agent baseline → multi-model benchmark; FinEx replaced Notion. Three ethics principles each mapped to a mechanism: data sovereignty (local stack), autonomy (confirmation gate), epistemic honesty (Critic).
- **S6 Requirements (1m, skim):** don't read FR numbers. "The three that shaped the build: confirm before destructive actions (FR7), swap the model with no code change (FR10), per-tier latency targets — which drove the tiering."
- **S7 Architecture + tiers (3.5m — protect):** three moves —
  1. *The split:* nine agents so no single model plans, remembers, acts and checks at once. Walk it once: Router → Memory → Planner → Critic → Executor → Evaluator.
  2. *Tier routing (headline):* "Not every request deserves the full pipeline. Early on everything went through it and trivial requests took 30–45s." Tier 1 ~800ms (regex, no LLM), Tier 2 ~3.5s (one LLM + tool), Tier 3 ~95s (Planner→Critic→DAG→Evaluator).
  3. *Why it's research, not optimisation:* "There's a real trade-off between coordination and speed; tiering turns it into a dial you can adjust." Then drop templated injection and hybrid routing.
- **S8 Hard parts (2m):** stack in one breath, then **two** war stories: (a) clean JSON from small models (markdown fences → repair layers), (b) replanning that actually changes the plan (feed Critic's specific complaints back in).
- **S9 Results (2m):** 0.83 mean, 16/16 ≥0.7, ~1 in 3 replanned, 9 tasks/7 intents/3 tiers. Three readings: plans (0.84) > execution (0.78) — better at deciding than landing every API call (mostly Calendar flakiness); latency dominated by inference not tools; replans converge, don't thrash. **State your own limits first:** small task set, mostly one model, one laptop, Critic is itself a model.
- **S10 Demo (5m):** see `Jarvis_5min_Demo_Plan.md`. Narrate; never go silent.
- **S11 Conclusion (1.5m):** close the loop from S2. Three contributions: tier-based routing, templated injection, critic-replanning loop. End on the dial line.
- **S12 Further work (1m):** top three — Pydantic schema validation + explicit topological check (~1 week), the full multi-model benchmark (harness already exists — highest value), a real user study (consent forms drafted).
- **S13 (20s):** "That's Jarvis — happy to take questions." Stop cleanly, look up.

## Three lines to know in your sleep
1. **Thesis:** "The limiting factor is the architecture around the model, not the model itself."
2. **Tier insight:** "Tiering turns the coordination-vs-speed trade-off into a dial you can adjust — that's the transferable finding."
3. **Demo-config honesty:** "The Critic is tuned down for the live demo because it adds latency on short prompts; the benchmark used the full config. That's the latency-vs-coordination trade-off in action — a tunable parameter, not a missing feature."

## If you blank
Pause, look at the slide, ask "what does this prove about my thesis?" Every slide has an answer. Two seconds of silence beats filler.

---

# B. CONDENSED NOTES — everything we covered

**Local-first:** thinking happens on your machine via **Ollama** (LLMs over HTTP). Privacy + it proves the thesis. The switch: `LLM_BACKEND=groq` → cloud Llama 70B, **zero code changes** (FR10), because every agent talks to the LLM through one client interface.

**Nine agents** (count them): Router (llama3.2:1b, classify intent+tier), Planner (LLM, builds subtask DAG), Memory (embeddings only, ChromaDB), Critic (LLM, reviews plans), Evaluator (pure Python, scores→SQLite), Summariser (LLM), Calendar (OAuth2, no LLM), Gmail (hybrid — drafting uses LLM), FinEx (LLM, financial Q&A). They communicate through **typed dataclasses in `config/models.py`** (RouterDecision, TaskPlan, Subtask, CriticVerdict, EvaluationResult, MemoryItem, JarvisResponse) — never shared state. Coordination is **explicit method calls in the orchestrator**, not a message bus — chosen for debuggability over CrewAI/AutoGen's looser passing.

**"Just functions with a grand name?"** Each agent has its own LLM-facing role and typed I/O contract, is independently testable/swappable; Router/Planner/Critic reason independently, the rest are deterministic *by design* — determinism placed exactly where reliability matters. "Coordinated specialisation, not a society of minds."

**The query route (`orchestrator.py:handle()`, ~line 480):**
- *Stage 1 — intercepts (before routing), each returns early:* pending-state (`_try_pending_state_intercept`, line 246) → elaborate/follow-up → memory command → morning briefing → multi-action split. These exist because the tiny Router predictably misclassifies short continuations.
- *Stage 2 — Router:* Layer A `_deterministic_route` (30+ regex, <1ms, no LLM) → Layer B `route` (llama3.2:1b, ~80-token JSON) → Layer C `_fallback_decision` (rules; routing can't crash). Produces RouterDecision with tier.
- *Stage 3 — branch by tier:* Tier 1 (~800ms, `_try_shortcut`, no memory/LLM; escalates if no match) / Tier 2 (~3.5s, memory + one LLM; FinEx and web fast-paths; single-paragraph enforced) / Tier 3 (~95s, Planner → Critic conditionally → `_execute_dag` → Evaluator → store episodic memory → JarvisResponse).
- *Subtle attack — "schedule a meeting" routed to Tier 1?* On purpose: real booking lives in `_try_shortcut`, which only runs at Tier 1. At Tier 2 the lone LLM would hallucinate "booked!" without touching Google. (`router.py:171-176` comment confirms.)

**Reliable JSON from a small model:** (1) Ollama `format:json` + temp 0.1 + schema example; (2) repair step extracts the `{...}` block; (3) Critic as a *semantic* backstop. Syntactic fix ≠ semantic fix.

**Memory:** ChromaDB vector DB; embeddings via Ollama `nomic-embed-text`; cosine similarity, drop <0.3, top-5; HNSW index (cosine space). Types: episodic (auto), semantic (facts you tell it), procedural (reserved stub). ChromaDB chosen for in-process API + persistence + zero-config HNSW (single-user prototype). Embedding down → 8s timeout → deterministic SHA-256 hash fallback (responsive, lower quality). `forget()` uses a stricter 0.45 threshold so it doesn't delete loosely-related memories.

**Critic:** reviews plan (and optionally results); 0–1 score, issues, suggestions, replan flag. Approved ≥0.6, replan <0.5. Five criteria: addresses request, deps ordered, nothing missing, params sensible, not over-engineered. On LLM error → approves at 0.7 (never stalls). Smart replanning inlines the Critic's *specific* complaints into the next Planner prompt. **"LLM judging an LLM — circular?"** Stated limitation; mitigation = narrower fixed-criteria job, approves-on-error; inter-rater study is further work.

**Evaluator formula (cold):** `overall = 0.4·planning + 0.4·execution + 0.2·(1 if any subtask succeeded else 0)`. Execution = fraction of subtasks succeeded; planning = Critic's score; pass ≥0.6. Persists to SQLite (`data/jarvis.db`): latency_ms, subtask_count, replan_count. Export helpers power the multi-model comparison. Weights = a reasoned choice (plan and execution equally important; 0.2 bonus stops partial success scoring zero), not derived from theory.

**DAG execution (`_execute_dag`):** iterative topological sort; loop up to **2× subtask count**; run subtasks whose deps are complete; failed dep → dependent **BLOCKED** (no crash); a round that runs nothing while tasks remain → circular dependency → stop. Two guarantees: failure isolation + cycle guard. **`_inject_deps`:** before each subtask, swap `{subtask_X.result.field}` for the real upstream value — decouples Planner (symbolic refs at plan time) from Executor (resolves at run time). Contribution #2.

**Confirmation / autonomy:** `_pending_email`, `_pending_meeting`, `_pending_file_op`. Email: draft → show → wait → send only on explicit "yes". A code comment documents a real bug fixed where "yes" was misrouted and the LLM faked a send — now confirmation actually calls `gmail.send_email`. Concrete beats abstract in a viva.

**LLM client (`config/llm_client.py`):** one shared `OllamaClient` (async chat + embeddings); `chat_stream` (live UI), `chat_json` (repair fallback); `_post_with_retry` (backoff + jitter); defaults temp 0.1, 1 retry, 60s timeout, keep_alive 30m. `LLM_BACKEND=groq` rebinds the client to GroqClient transparently.

**Tools (~17):** pure-Python no-auth (weather/Open-Meteo, news/RSS, markets, sports, prayer times, web search); OAuth (Calendar, Gmail, Spotify); macOS-bound (file_manager, mac_control); local store (reminders→SQLite). `platform_guard.py` detects Mac vs cloud; Mac-only tools return a clean "macOS-only" message in cloud (NFR7). New tool = ~3 files, no architecture change (NFR3) — how FinEx and voice were added late.

**Voice:** mic → STT (faster-whisper, int8, local) → orchestrator → TTS. VAD = Silero. TTS = **ElevenLabs** (cloud) — the one deliberate local-first exception, for sub-100ms time-to-first-audio. Voice mode clamps to ≤2 sentences. Currently **push-to-talk**, not wake-word.

**FinEx:** replaced planned Notion to prove the design generalises. Extract PDF → structured figures in Postgres (Neon) + raw text in ChromaDB → answer via text-to-SQL + retrieval. Six auto-routed levels (L1–L6). Runs in a thread pool (`run_in_executor` + ~100s timeout) so the sync engine doesn't block the event loop.

**Front end ↔ backend:** HTTP loads the page once (FastAPI serves the single-page HUD: HTML/CSS/JS); then the JS opens a **WebSocket** (`server.py:1666 @app.websocket("/ws")`) and the response **streams back token by token**. WebSocket not plain HTTP because the response streams — HTTP is one-shot. **The WebSocket is the front-end/back-end boundary.** JS does only three things: the WebSocket client, UI interactions, browser mic capture — **no reasoning/routing/planning in JS**, all Python. (Web server is `server.py`, not `main.py`; `main.py` is the CLI/benchmark entry point.)

**Results (local model, n=16 across 9 tasks):** overall 0.83 mean (0.76–0.96), planning 0.84, execution 0.78, latency mean ~128,500ms (48k–183k), replan 0.56 mean, 16/16 ≥0.7. Median by tier: T1 ~800ms, T2 ~3.5s, T3 ~95s. Threats: small task set (9), developer-authored, mostly one model, one M2 MacBook Air/16GB, Critic is itself an LLM. "n=16 is tiny — I don't claim statistical generality; it shows the architecture works as designed on representative tasks; larger set + multi-model run are top further work; the harness exists."

**Positioning:** vs AutoGPT/CrewAI/LangGraph (cloud-bound, developer-facing, synthetic benchmarks) → Jarvis is local-first, user-facing, daily-use, evaluated end to end. Novelty = the integrated locally-hosted *evaluated* whole + tier-based routing as a tunable answer to latency/coordination.

**Carry yourself under pressure:** find a limitation → agree fast, explain the reasoning, name the mitigation/further work. You've mapped every weakness yourself — that's the armour.

---

# C. CODE-VERIFIED MISMATCHES (defend these before they're found)

> Ranked by how much an examiner could hurt you. All confirmed against actual source.

**C1 — Tier-3 latency contradicts the report (highest risk).** NFR1 says Tier 3 <30s; §11.3 says "10–30s"; §13 results say 48–183s (mean ~128.5s). The report disagrees with itself. **Answer:** "The <30s target was set for the Groq cloud backbone. On the local ~3B model Tier 3 runs ~95s, which I report honestly; `LLM_BACKEND=groq` meets the target with no architecture change. I chose to benchmark the harder fully-local case and be upfront." Note Tier 1 (<1s) and Tier 2 (<5s) targets **are met** — say so.

**C2 — Critic & replanning tuned down in the running build.** Confirmed: `orchestrator.py:55 MAX_REPLAN_ATTEMPTS = 0`; `orchestrator.py:782` Critic only runs when `routing.confidence < 0.70 AND len(plan.subtasks) > 3` (or research agent); post-execution result review **commented out** (line ~822, "Re-enable for the dissertation evaluation runs only"). **Extra detail to know:** when the Critic is skipped, `planning_score` is **hard-coded to 0.8** — so in demo mode most tasks' planning component is a fixed 0.8, not a real Critic score. If asked where the 0.84 planning mean comes from: "the reported figures are from the eval config with the Critic on; demo mode defaults planning to 0.8." **Frame:** the latency/coordination dial — your thesis as a tunable parameter.

**C3 — Voice is push-to-talk, not wake-word.** Confirmed: `voice/runner.py` header "Wake-word has been removed…"; `.env` says the same. **But Slide 10 / Slide 8 still show "Wake word →" and openWakeWord**, and the report (FR1/FR12, §11.4) describes openWakeWord. **Action: change the Slide 10 bullet to "press mic → spoken request" tonight,** or pre-empt it. **Answer:** "I implemented openWakeWord but switched to push-to-talk for demo reliability — false triggers mid-presentation are a real risk; rest of the pipeline (Whisper, Silero, ElevenLabs) is identical."

**C4 — Whisper is `base.en`, not `small.en` (NEW — the old study guide got this wrong).** Confirmed: `.env WHISPER_MODEL=base.en`; `settings.py:74` default `base`. The report repeatedly says `small.en` and claims a medium→small downgrade. **Answer:** "The report documents the small.en config; the demo machine runs base.en for speed — smaller, faster, same pipeline, one env var." Do **not** claim small.en is running.

**C5 — Multi-model benchmark not run with the planned models.** Confirmed: `settings.py:35` default `llama3,mistral`; `.env` sets `llama3.2:latest,mistral:7b`; report's planned set is `llama3.2:1b/3b, phi3, gemma`. Neither matches. **Answer:** "Infrastructure exists; the single-model n=16 results are solid; the full cross-model study is my top further-work item." Do **not** claim a completed cross-model comparison.

**C6 — Live DB ≠ the reported benchmark.** Confirmed: `data/jarvis.db` now has **28 rows, mean ~0.845, 27/28 ≥0.7** (one run at 0.34). The headline "16/16, 0.83" is the **frozen** benchmark suite (`benchmark_results.csv`), not the live log. **Answer if they open the DB:** "That's the live usage log including off-hand tests; the reported n=16 is the controlled benchmark suite."

**C7 — Routing subtlety that affects your demo (NEW).** Your old worked-example query "get the news, check the weather, give me a morning briefing" does **not** reach the Planner — `is_morning_briefing` (`tools/briefing.py`) catches any phrase containing "briefing"/"morning brief" *before* routing, so it runs the **deterministic briefing composer**, not the Tier-3 DAG. To genuinely demo the Planner DAG, use **"research … and summarise the key points"** (routes to RESEARCH → Tier 3 → Planner). The demo plan already uses the correct queries — just don't call the briefing "the planner."

**C8 — Default model strings are placeholders (low risk).** `settings.py:30` default `llama3`; `.env` overrides to `llama3.2:latest` (the ~3B model, matches the report's stated default). Report line 421 calls `llama3.2:3b` "the default chat backbone" while the abstract calls it `llama3.2:latest` — minor internal slip; "the default is llama3.2:latest, the ~3B model."

**C9 — Tool count nuance.** 17 files in `tools/` *including* `platform_guard.py` (a helper); a strict count gives 16. "Depending on whether you count the platform guard" — don't die on the number.

## The one-paragraph honesty pre-empt to open Q&A
"Two things I'll flag before you find them: for the live demo the Critic and replanning are tuned down — a config switch, and the benchmark numbers used the full config; and the voice path is push-to-talk rather than wake-word, again for demo reliability. Both are the latency-versus-coordination trade-off the project is about, exposed as tunable parameters rather than hard-coded behaviour." Said first, it reads as mastery.

## File-location quick map (so "show me X" never trips you)
routing → `agents/router.py` · whole flow → `orchestrator.py:handle()` (~480) · intercept/confirmation → `_try_pending_state_intercept` (246) · DAG → `_execute_dag` (~1235) · dispatch → `_dispatch` (~1303) · injection → `_inject_deps` (~1548) · scoring → `agents/evaluator.py` · memory → `memory/memory_agent.py` · model swap → `config/llm_client.py:380-400` · WebSocket boundary → `server.py:1666`.
