# Jarvis — Viva / Presentation Study Guide

**For:** Abdullah Khan Durrani · COM6001 Final Year Project · BNU
**Purpose:** Everything you need to explain how Jarvis works and defend every design decision in person. Grounded in the actual source code, with file references so you can say "that's in `orchestrator.py`" with confidence.

> How to use this: read Parts 1–9 to understand the system end to end, read **Part 10 (Honesty Notes)** carefully because that's where lecturers catch people out, then drill **Part 11 (Q&A bank)**. The one-line definitions in Part 13 are your safety net if you blank on a term.

---

## Part 1 — The 30-second pitch (say this in your sleep)

Jarvis is a **local-first, multi-agent executive assistant** built on large language models. Instead of one model trying to do everything, the work is split across **nine specialised agents** coordinated by a central **orchestrator**, with **seventeen tools** for real-world actions (calendar, email, web, news, weather, music, Mac control, finance, etc.).

The core argument of the whole project: **when an LLM assistant fails at a real multi-step task, the limiting factor is usually the architecture around the model, not the model itself.** I show this by getting strong, reliable behaviour out of a small (~3B parameter) local model purely through how the system is structured.

Three things I'd call genuine contributions:

1. **Tier-based routing** — match how much machinery you use to how hard the request is, so you only pay the coordination cost when it's worth it.
2. **Templated dependency injection** in the task plans — `{subtask_1.result.field}` — which keeps the planner separate from the actual data and lets plans be reused.
3. **A critic-driven replanning loop** — a self-check that reviews plans (and results) before they reach the user.

---

## Part 2 — The big picture

### What it is
A daily-use personal assistant you run on your own machine. It takes natural language (typed or spoken), works out what you want, does it, and replies. It can hold multi-turn conversations, remember facts about you, and carry out authenticated actions in Gmail and Google Calendar.

### Why it exists (the gap)
The individual ingredients — reasoning frameworks (Chain-of-Thought, ReAct), multi-agent systems (CrewAI, LangGraph, AutoGen), and memory (RAG) — are each well studied. What's rare is **all three combined into a user-facing, locally-hosted assistant that's actually evaluated end to end**, rather than a cloud-bound research demo tested on synthetic benchmarks. That intersection is where Jarvis sits.

### Two ways to run it
- **Local (full):** on a Mac, with the voice stack and Mac-control tools. Uses **Ollama** for local inference. Nothing goes to a third-party AI by default.
- **Cloud (demo):** containerised with **Docker**, deployed on **Fly.io** (London). Mac-only tools degrade gracefully; you can optionally switch the LLM backbone to **Groq** (hosted Llama 3.3 70B) via one environment variable.

---

## Part 3 — Architecture at a glance

```
                         ┌──────────────────────────────────┐
   user text / voice ──▶ │        JarvisOrchestrator         │
                         │  (orchestrator.py — the brain)    │
                         └──────────────────────────────────┘
                                       │
   Pre-router intercepts (run in order, before any routing):
     • pending-state (waiting for email/duration/file confirm)
     • elaborate / follow-up ("tell me more")
     • memory command (remember / forget / recall)
     • morning briefing
     • multi-action split ("dim screen, play song, remind me")
                                       │
                                  ┌────▼─────┐
                                  │  Router  │  intent + tier (1/2/3)
                                  └────┬─────┘
                ┌──────────────────────┼───────────────────────┐
            Tier 1                  Tier 2                    Tier 3
        tool shortcut          memory → 1 LLM call        memory → Planner →
        (~800ms)               + tool (~3.5s)             Critic → Executor(DAG)
                                                          → Evaluator (~95s)
```

**The nine agents** (all in `agents/`, except Memory in `memory/`):

| Agent | File | Job | Uses an LLM? |
|---|---|---|---|
| Router | `agents/router.py` | Classify intent + difficulty tier | Yes (small model, `llama3.2:1b`) |
| Planner | `agents/planner.py` | Break a request into a subtask DAG | Yes |
| Memory | `memory/memory_agent.py` | Store/retrieve context (ChromaDB) | Embeddings only |
| Critic | `agents/critic.py` | Review plans/results, trigger replan | Yes |
| Evaluator | `agents/evaluator.py` | Score every task, persist to SQLite | No (pure Python) |
| Summariser | `agents/summariser.py` | Condense long outputs | Yes |
| Calendar | `agents/calendar_agent.py` | Google Calendar (OAuth2) | No |
| Gmail | `agents/gmail_agent.py` | Gmail (OAuth2); drafting uses LLM | Hybrid |
| FinEx | `agents/finex_agent.py` | Financial-statement Q&A | Yes |

**The data contracts** live in `config/models.py` — every agent imports these dataclasses, never redefines them. The important ones:
- `RouterDecision` (primary_agent, supporting_agents, confidence, **tier**)
- `TaskPlan` (task_id, intent, **subtasks**, reasoning, model_used, replan_count)
- `Subtask` (id, action, agent, params, **depends_on**, status, result)
- `CriticVerdict` (approved, score, issues, suggestions, **replan_needed**)
- `EvaluationResult` (score, planning_score, execution_score, latency_ms, …)
- `MemoryItem`, `JarvisResponse`

This shared-types design is itself a defensible point: it's what makes the agents loosely coupled and independently testable.

---

## Part 4 — Every agent, explained

### 4.1 Router (`agents/router.py`)
**Job:** the entry point. Turns a raw request into a `RouterDecision` with an intent, the primary agent, supporting agents (memory is always included), a confidence, and a **tier (1/2/3)**.

**How it works — two layers:**
1. **Deterministic pre-routing** (`_deterministic_route`): ~30+ regular expressions for the common shapes — email send/inbox/reply, calendar create/check, reminders, FinEx finance keywords, and factual `who/what/when…` questions. Returns in well under a millisecond, **no LLM call**. Each regex exists because of a real misrouting I saw during use.
2. **LLM fallback** (`route`): anything the regexes don't catch goes to a small model (`llama3.2:1b`) with an ~80-token JSON classification prompt. There's also a pure rule-based `_fallback_decision` if the LLM errors, so routing never crashes.

**Why two layers?** Speed. LLM-only routing added 1–3s to *every* request; the regex layer removes that for the majority of real requests and keeps the LLM for the genuinely ambiguous ones.

**Watch-out you should know:** some patterns deliberately route to **tier 1** even though they "feel" complex (e.g. "schedule a meeting", "send an email"). That's because the real booking/sending flow lives in the orchestrator's `_try_shortcut` handler, which only runs at tier 1. If those were tier 2, the LLM would *hallucinate* a "done!" reply without ever touching Google. The comments in the file say exactly this.

### 4.2 Planner (`agents/planner.py`)
**Job:** decompose a complex request into a **DAG of subtasks** as JSON, using ReAct-style reasoning (Thought → Observation → Thought → Output).

**Output schema** (this is the heart of the system — memorise the shape):
```json
{
  "intent": "schedule_meeting",
  "reasoning": "step-by-step ReAct trace",
  "subtasks": [
    {"id": "subtask_1", "action": "retrieve_context", "agent": "memory",
     "params": {"query": "..."}, "depends_on": []},
    {"id": "subtask_2", "action": "create_event", "agent": "calendar",
     "params": {"start_time": "{subtask_1.result.preferred_time}"},
     "depends_on": ["subtask_1"]}
  ]
}
```
**Rules baked into the prompt:** atomic subtasks (one action each), ISO-8601 datetimes, max 8 subtasks, start with memory retrieval when context helps, and use the `{subtask_id.result.field}` template syntax for passing data between steps.

**Why decouple planning from execution?** The literature (and my own experience) shows a pure-planner LLM whose JSON is consumed by a deterministic executor is more reliable than one monolithic prompt that tries to reason and act at once.

### 4.3 Memory (`memory/memory_agent.py`)
**Job:** persistent memory using **ChromaDB** (a vector database) with cosine similarity.

- Three memory types (`config/models.py`): **EPISODIC** (things that happened — past tasks), **SEMANTIC** (facts/preferences you told it), **PROCEDURAL** (declared but mainly reserved for future "skill library" work).
- **Embeddings:** Ollama's `nomic-embed-text` locally. Every stored item and every query is embedded; retrieval ranks by cosine similarity, drops anything below the threshold (default **0.3**), returns top-k (default **5**).
- **Robustness:** embedding calls have an **8-second timeout**; if Ollama stalls it falls back to a deterministic SHA-256 **hash embedding** so a request never hangs on memory.
- **User-facing memory commands:** `remember` (store a semantic fact), `recall_facts`, and `forget` (delete matching semantic memories) — wired to the "remember/forget" intercept in the orchestrator.
- Under the hood ChromaDB uses **HNSW** (Hierarchical Navigable Small World) indexing, configured for cosine space.

### 4.4 Critic (`agents/critic.py`)
**Job:** the self-reflection layer. Reviews a plan **before** execution and (optionally) the results **after**, returning a `CriticVerdict` with a 0–1 score, issues, suggestions, and a `replan_needed` flag.

- Thresholds (`config/settings.py`): `approved` if score ≥ 0.6; `replan_needed` if score < **0.5** (`CRITIC_REPLAN_THRESHOLD`).
- Reviews against five criteria: does it address the request, are dependencies ordered correctly, are steps missing, are params sensible, is it over-engineered.
- If the LLM errors, it **approves by default** (score 0.7) so the pipeline never stalls on the critic.
- **Smart replanning:** when a replan is triggered, the orchestrator inlines the critic's *specific* issues/suggestions into the next planner prompt — not just "try again" — which produces a genuinely different plan rather than a near-copy.

> ⚠️ **Read Part 10.1** — in the current build the critic and replanning are tuned down for demo latency. You must be ready to explain this.

### 4.5 Evaluator (`agents/evaluator.py`)
**Job:** score every Tier-3 task and persist to **SQLite** (`data/jarvis.db`). This is where the dissertation's benchmark numbers come from.

- **The exact overall-score formula** (know this cold):
  `overall = 0.4·planning_score + 0.4·execution_score + 0.2·(1 if any subtask succeeded else 0)`
- `execution_score` = proportion of subtasks that succeeded. `planning_score` comes from the Critic. `success` if overall ≥ **0.6** (`EVALUATOR_MIN_SCORE`).
- Also records latency_ms, subtask_count, replan_count, plus human-readable feedback.
- Has export helpers: `get_model_summary()` (per-model aggregates for a comparison table), `export_json()`, `export_csv()`. This is what makes the multi-model benchmark possible.

### 4.6 Summariser (`agents/summariser.py`)
Condenses long outputs (research, documents) to a target word count using the LLM. Used as a subtask agent and for long responses.

### 4.7 Calendar (`agents/calendar_agent.py`)
Google Calendar via **OAuth2**. Create/search/delete events and **conflict detection**. Credentials at `~/.jarvis/credentials.json`, token cached at `token.json`; expired tokens auto-refresh and re-save. **Falls back to "mock mode"** (in-memory fake events) if credentials are missing, so the system still runs for a demo without Google set up. Built **lazily** (on first use) so an expired token never blocks server startup.

### 4.8 Gmail (`agents/gmail_agent.py`)
Gmail via OAuth2 (scope `gmail.modify`): read inbox, search, draft, send. Drafting is the **hybrid** bit — the `EmailComposer` (`tools/email_composer.py`) uses the LLM to write the body, but **sending is gated behind explicit confirmation** (see 5.4).

### 4.9 FinEx (`agents/finex_agent.py` + `finex/`)
A domain sub-agent for **financial-statement Q&A** (e.g. Bestway Cement, HBL). This replaced the originally-planned Notion integration to prove the same orchestration generalises to a very different domain.

- Pipeline: extract a financial-statement PDF (`finex/extract_pdf.py`) → store structured figures in **Postgres** (Neon-hosted) and the raw text in **ChromaDB** → answer questions via **text-to-SQL** generation plus text retrieval (`finex/LLM_SQL.py`).
- **Six sophistication levels** (auto-routed by deterministic phrase matching): L1 Basic Retrieval, L2 Comparative, L3 Ratio Analysis, L4 Analytical Reasoning, L5 Investor Insight, L6 Strategic Reasoning (plus TEXT / DETAIL / OFF_TOPIC).
- Runs in a **thread pool** because the underlying engine is synchronous; the async agent wraps it with `run_in_executor` and a ~100s timeout so it doesn't block the event loop.

---

## Part 5 — The orchestrator deep dive (`orchestrator.py`, ~4,300 lines)

This is the file you'll get the most questions about. The `JarvisOrchestrator` class owns one shared `OllamaClient` and instantiates every agent and tool in `__init__`. Google + FinEx agents are **lazy** (built on first use).

### 5.1 The request lifecycle (`handle()`, line ~480)
Before routing, a chain of **intercepts** runs (each returns early if it matches). This is a deliberate design point — the small router model confidently *misclassifies* short continuations, so these are caught first:

1. **Pending-state intercept** — if we're mid-flow waiting for an email address, a meeting duration, or a file-op confirmation, the next message is a *continuation*, not a new request.
2. **Elaborate / follow-up** — "tell me more", "go into detail" only make sense relative to the previous turn, so they re-ask the prior question with an "expand" prompt.
3. **Memory command** — "remember that…", "forget…", "what do you know about…".
4. **Morning briefing** — caught here because the router would mislabel it as general chat and the LLM would invent a fake briefing.
5. **Multi-action split** — "dim the screen, play Despacito and remind me at 5" is three commands; the router only picks one agent, so this splits and runs each.

Then **routing** produces a tier, and the flow branches:

### 5.2 Tier routing (the key contribution)
- **Tier 1 (tool-only, ~800ms):** skip memory and the LLM entirely; run `_try_shortcut` and return. If the shortcut doesn't match, retrieve memory, try once more, then **escalate to Tier 2**.
- **Tier 2 (single LLM hop, ~3.5s):** retrieve memory, then one LLM call. Special fast-paths: FinEx goes straight to its own engine; web-search/news fetch context first, then the LLM answers in one paragraph. Voice mode clamps the answer hard (≤2 short sentences). Output is forced to a single paragraph by `_enforce_single_paragraph`.
- **Tier 3 (full pipeline, ~95s):** shortcut check → **Planner** → **Critic** (conditionally) → **`_execute_dag`** → **Evaluator** → store episodic memory → build response.

**Why tiers?** Early on, running trivial requests through the full pipeline took 30–45s for results a hardcoded rule returns in under a second. Tiering is the principled middle path between "everything through the slow pipeline" and "dumb hardcoded rules everywhere": pay the orchestration cost only when the task warrants it. This is the project's central, transferable finding — the **latency-versus-coordination trade-off**.

### 5.3 DAG execution (`_execute_dag`, line ~1209)
Plain-English algorithm (this is essentially an **iterative topological sort**):

```
completed = {}; pending = {all subtasks}
repeat up to (2 × number_of_subtasks) times:
    for each pending subtask:
        if all its depends_on are in completed:
            if any dependency FAILED → mark BLOCKED, record failure
            else → dispatch it, store result, mark done
    remove the ones that ran from pending
    if nothing ran this round but tasks remain → CIRCULAR DEPENDENCY → stop
return completed
```
- Each subtask is sent to `_dispatch` (line ~1277), a big `if/elif` that maps `agent.action` to the real tool/agent method and returns `{"success": ..., "result": ..., "message": ...}`.
- **Failure isolation:** a failed dependency *blocks* its dependents rather than crashing the whole plan.
- **Circular-dependency guard:** the `max_iterations = len(pending) * 2` cap and the "nothing ran this round" check catch cycles that small models occasionally produce.

### 5.4 Dependency injection (`_inject_deps`, line ~1522)
Before a subtask runs, any param string containing `{subtask_X.result.field}` is replaced with the actual value from the upstream subtask's result dict. This is **templated parameter injection** — it decouples the planner (which writes symbolic references) from the executor (which resolves them at run time), and is one of the three named contributions.

### 5.5 Confirmation / user autonomy (the ethics mechanism)
Nothing destructive or externally visible happens without explicit confirmation. The orchestrator holds **pending-state** objects: `_pending_email`, `_pending_meeting`, `_pending_file_op`. For email: it drafts → shows you the draft → waits → only sends when you reply "yes/send it" (handled in `_try_pending_state_intercept`, line ~242). There's even a comment explaining a bug they fixed where "yes" was being misrouted and the LLM faked a send — now the confirmation actually triggers `gmail.send_email`. This is a concrete, demonstrable implementation of the **user-autonomy** ethics principle.

---

## Part 6 — The LLM layer (`config/llm_client.py`)

- **`OllamaClient`** wraps Ollama's REST API for async chat + embeddings. One shared instance across all agents (each agent can be given a different model for benchmarking).
- **Streaming** (`chat_stream`) for the live web UI; **`chat_json`** for structured calls with a JSON-repair fallback (if parsing fails it extracts the `{...}` block).
- **Reliability:** `_post_with_retry` with exponential backoff + jitter. Defaults: temperature **0.1** (low, for consistent JSON), 1 retry, 60s per-call timeout, `keep_alive: 30m` so the model stays warm between turns (avoids cold-start mid-demo).
- **Backbone swap:** set env `LLM_BACKEND=groq` and the `OllamaClient` name is transparently rebound to `GroqClient` — every agent uses the cloud model with **zero code changes**. This realises requirement FR10.
- **System prompt:** a detailed Jarvis persona is injected into conversational calls, including strict format rules (one paragraph, no markdown, never claim to be the user, etc.).

Key defaults (`config/settings.py`): `OLLAMA_CHAT_MODEL` (default `llama3`, overridden by env to your `llama3.2`), `OLLAMA_ROUTER_MODEL = llama3.2:1b`, `OLLAMA_EMBED_MODEL = nomic-embed-text`, memory top-k 5 / threshold 0.3, `EVALUATOR_MIN_SCORE = 0.6`, `CRITIC_REPLAN_THRESHOLD = 0.5`.

---

## Part 7 — The tool layer (`tools/`, 17 modules)

Tools are grouped by dependency profile:
- **Pure Python (no auth):** weather (Open-Meteo), news (RSS from BBC/Reuters/Guardian/HN/TechCrunch), markets, sports, prayer times, web search.
- **Authenticated (OAuth):** Calendar, Gmail, Spotify.
- **Platform-bound (macOS):** `file_manager`, `mac_control` (AppleScript/screencapture/volume/brightness).
- **Local store:** reminders persisted to SQLite.

**`platform_guard.py`** detects Mac vs cloud (checks `FLY_APP_NAME`, Kubernetes, Docker env vars, else non-Mac). Mac-only tools return a clean "this is macOS-only" message in the cloud instead of crashing — this is how requirement NFR7 (portability with graceful degradation) is met.

**Adding a new tool** touches ~3 files (the tool itself, its registration in the orchestrator's `_dispatch`, and the planner's tool catalogue) with no architectural change — that's requirement NFR3 (extensibility), and it's how the voice subsystem and FinEx were added late without touching the core.

---

## Part 8 — Voice pipeline (`voice/`)

Stages: **microphone → speech-to-text → orchestrator → text-to-speech.**
- **STT:** `faster-whisper` (local), int8 quantised. (`config/settings.py` default `WHISPER_MODEL=base`; report discusses `small.en`.)
- **VAD:** Silero voice-activity detection (`voice/vad.py`) for utterance boundaries.
- **TTS:** **ElevenLabs** Flash model (cloud) — chosen for sub-100ms time-to-first-audio because no open-source TTS matched its naturalness. This is the one deliberate exception to "local-first", and you should own that as a justified trade-off.
- The runner adapts the async `orchestrator.handle(..., voice_mode=True)` into the voice loop; voice mode tightens responses to ≤2 sentences (`VOICE_MAX_TOKENS=220`).

> ⚠️ **Read Part 10.2** — the current `voice/runner.py` is **push-to-talk** (press Enter / click mic), not wake-word. The report describes openWakeWord wake-word detection. Be ready for this.

---

## Part 9 — Methodology, ethics, risks, results (the report layer)

**Methodology — Design Science Research (DSR).** The artefact (Jarvis) *is* the object of study; each development cycle had to produce evidence, not just a working build — which is why the Evaluator logged scores from early on. Honest scope changes: the single-agent baseline became a multi-model benchmark (a single-agent build would have compared *tooling*, not architecture); FinEx replaced Notion.

**Ethics — three first-order principles.** Data sovereignty (on-device by default), user autonomy (confirm before destructive actions), epistemic honesty (the Critic). Plus minimised OAuth scopes and env-only secrets.

**Risks & mitigations.** Hallucination → JSON-only prompts + low temperature + Critic + bounded replanning. API limits → exponential backoff + per-tool graceful degradation. Token leakage → env isolation. Small-model JSON variance → post-processing/repair.

**Results (default local model, n=16 runs across 9 tasks):**
| Metric | Value |
|---|---|
| Overall score | 0.83 mean (0.76–0.96) |
| Planning score | 0.84 |
| Execution score | 0.78 |
| Latency mean | 128,500 ms (48k–183k) |
| Replan count | 0.56 mean |
| Passed ≥0.7 | 16/16 (100%) |

Median latency by tier: Tier 1 ≈ 800ms, Tier 2 ≈ 3.5s, Tier 3 ≈ 95s. **Key reading:** plans score higher than execution (better at deciding than at landing every external API call — mostly Google Calendar flakiness); latency is dominated by LLM inference, not tools; replanned tasks take longer but converge to similar scores.

**Threats to validity (say these before they do):** small task set (9), developer-authored, mostly one model, one hardware config (M2 MacBook Air, 16GB), and the Critic is itself an LLM with its own biases.

---

## Part 10 — ⚠️ Honesty notes (where the code differs from the report — DON'T get caught out)

These are the questions that separate "I read my report" from "I built this". Be the second person.

### 10.1 The Critic and replanning are tuned DOWN in the current build
- In `orchestrator.py`: `MAX_REPLAN_ATTEMPTS = 0`, the Critic only runs when `confidence < 0.70 AND subtasks > 3`, **or** for the research agent; and the post-execution result-review is commented out ("Re-enable for the dissertation evaluation runs only").
- **What this means:** there are effectively **two configurations** — an **evaluation config** (Critic on, replanning up to 2) that produced the dissertation benchmark numbers, and a **demo config** tuned for responsiveness. The architecture fully supports both; it's a tuning decision, not a missing feature.
- **How to say it:** "For the live demo I tuned the critic down because, as my own results show, it adds 1–2 seconds per call and most demo prompts are high-confidence and short. The benchmark figures in the report were generated with it enabled. This is literally the latency-versus-coordination trade-off my project is about — and it being a tunable parameter is the point." That turns a potential 'gotcha' into a demonstration of your thesis.

### 10.2 Voice is push-to-talk now, not wake-word
- `voice/runner.py` header: "Wake-word has been removed — each turn is initiated by pressing Enter (CLI) or clicking the mic button (web UI)."
- **How to say it:** "I implemented wake-word detection with openWakeWord, but for demo reliability I switched to push-to-talk — wake-word false-triggers during a presentation are a real risk, and the rest of the pipeline (Whisper STT, Silero VAD, ElevenLabs TTS) is identical." Honest and sensible.

### 10.3 Default model strings
- `config/settings.py` defaults to `OLLAMA_CHAT_MODEL=llama3` and `WHISPER_MODEL=base`; your report references `llama3.2` and `small.en`. These are **environment-overridable** (your `.env` sets the real values). If asked, say: "Defaults in code are placeholders; the actual run config is set in `.env` — `llama3.2` for chat, `small.en` for STT."

### 10.4 Benchmark model list
- `BENCHMARK_MODELS` default in code is `llama3,mistral`; the report's planned comparison set is `llama3.2:1b/3b, phi3, gemma`. The multi-model comparison is **set up but not fully run** — you flag this in the report as the highest-value further work. Don't claim the full cross-model study is complete; claim the infrastructure exists and the single-model results are solid.

### 10.5 "Nine agents / seventeen tools"
Be ready to *count* them. Agents: Router, Planner, Memory, Critic, Evaluator, Summariser, Calendar, Gmail, FinEx = 9. Tools: weather, websearch, news, mac_control, spotify, document, sports, markets, prayer_times, briefing, file_manager, reminders, contacts, email_composer, query_parser, web_search, finex_clear (+ platform_guard as a helper). If a lecturer counts differently, say "depending on whether you count helpers like the platform guard and contact book" — don't die on the exact number.

---

## Part 11 — The Q&A bank (drill these)

### A. Architecture & "why multi-agent"
**Q: Why not just one big prompt / one model?**
A monolithic LLM asked to plan, remember, act and critique in a single pass shows well-documented failure modes — runaway loops, hallucinated tool calls, no recovery. Splitting concerns means each component has one narrow job, and I can put *deterministic* execution where determinism is achievable. Reliability comes from the structure, which is my whole thesis.

**Q: Isn't "multi-agent" just functions with a fancy name?**
Fair challenge. The distinction is that each agent has its own LLM-facing role, structured I/O contract (`config/models.py`), and can be tested/replaced in isolation. The Router, Planner and Critic genuinely reason independently; they're not just utility functions. But I don't over-claim emergent behaviour — it's *coordinated specialisation*, not a society of minds.

**Q: What actually coordinates them?**
The orchestrator (`orchestrator.py`). It owns the shared LLM client, runs the intercept chain, routes by tier, executes the DAG, and gates confirmations. There's no message bus — coordination is explicit method calls with typed data objects, which I chose for debuggability over the looser message-passing in CrewAI/AutoGen.

### B. The tier system
**Q: How does the router decide the tier?**
Two layers: regex rules assign a tier directly for common shapes; otherwise the small router model returns a tier in its JSON (clamped to 1–3). Tier 1 = single deterministic tool. Tier 2 = one LLM hop + maybe one tool. Tier 3 = multi-step, needs planning.

**Q: What if it picks the wrong tier?**
There's graceful escalation: a Tier-1 request whose shortcut doesn't match falls through to Tier 2 automatically. Over-classifying down is cheap to recover from; the bigger risk is under-classifying a complex task as Tier 1, which the shortcut simply won't match, so it escalates.

**Q: Tier 3 is 95 seconds — isn't that unusable?**
On a local ~3B model, yes it's slow, and I'm upfront about that. The point of tiering is that *only genuinely multi-step tasks* pay it — the vast majority of daily requests are Tier 1/2 and feel instant. Switching the backbone to Groq (one env var) cuts Tier 3 dramatically; the architecture doesn't change.

### C. Planning & execution
**Q: Walk me through how a plan executes.**
The planner emits JSON subtasks with `depends_on`. `_execute_dag` does an iterative topological sort: each round it runs every subtask whose dependencies are complete, stores results, and repeats. `_inject_deps` substitutes `{subtask_x.result.field}` placeholders with real upstream values just before each subtask runs. Failed dependencies block dependents; a "nothing ran this round" check catches cycles.

**Q: What stops an infinite loop or a cycle?**
A hard iteration cap of `2 × subtask_count`, plus the circular-dependency detector that stops if a round executes nothing while tasks remain.

**Q: How do you get reliable JSON from a small model?**
Three layers: Ollama's `format: json` mode + temperature 0.1 + a concrete schema example in the prompt; then a repair step that extracts the `{...}` block if strict parsing fails; and the Critic/replan as a backstop for semantically (not just syntactically) bad plans.

### D. Memory
**Q: Semantic vs episodic — what's the difference here?**
Semantic = facts/preferences you explicitly told it ("remember I prefer mornings"). Episodic = auto-stored records of past tasks and outcomes. Both are embedded and retrieved by similarity; semantic ones are also directly manageable via remember/forget.

**Q: Why ChromaDB and not FAISS / pgvector / Pinecone?**
In-process Python API, persistence out of the box, and HNSW indexing with no configuration — ideal for a single-user prototype. I note in the report that multi-user or larger scale would justify a dedicated vector DB; that's listed as further work.

**Q: Does the similarity threshold ever hurt you?**
Yes — too high and you miss relevant memories, too low and noise pollutes the prompt. I settled on 0.3 empirically. Below-threshold matches are silently dropped so weak hits don't degrade reasoning.

**Q: What happens if the embedding model is down?**
An 8-second timeout then a deterministic hash-embedding fallback, so a request never blocks on memory. It's non-semantic, so retrieval quality drops, but the system stays responsive.

### E. The Critic & self-correction
**Q: How does replanning actually improve things vs just retrying?**
A naïve retry gives near-identical output. I inline the critic's *specific* issues and suggestions into the next planner prompt, prefixed with the flawed plan, which produces a qualitatively different attempt. Bounded to 2 attempts so it can't loop forever.
*(If they push: see Part 10.1 — be honest that it's tuned down in the demo build.)*

**Q: The critic is an LLM judging an LLM — isn't that circular?**
A real limitation, and I say so in my threats-to-validity. The mitigation is that the critic has a *different, narrow* job (review against five fixed criteria) than the planner, and it defaults to approving on error so it can't silently block everything. An inter-rater study comparing critic scores to human judgement is named as further work.

### F. Evaluation
**Q: How is the overall score computed?**
`0.4·planning + 0.4·execution + 0.2·(any-success bonus)`. Execution score is the fraction of subtasks that succeeded; planning score comes from the critic; pass threshold is 0.6.

**Q: Why is execution lower than planning (0.78 vs 0.84)?**
The system is better at *producing* a good plan than at *landing every external call*. The gap is mostly transient Google Calendar API failures and the occasional subtask the critic lets through. It's an honest, explainable gap, not noise.

**Q: n=16 is tiny. Why should I believe it?**
I don't over-claim. The numbers demonstrate the architecture functions as designed on representative tasks; they don't claim statistical generality. A larger task set and full multi-model run are my top further-work items, and the benchmark harness already exists to do them.

### G. Ethics, privacy, security
**Q: How is privacy actually guaranteed, not just asserted?**
By default all reasoning, embeddings and storage are local (Ollama + ChromaDB + SQLite on your machine); nothing goes to a third-party AI unless you explicitly set `LLM_BACKEND=groq`. Google access uses OAuth with scopes minimised to Calendar + Gmail-modify, and secrets live in env vars / `.env` (never in source).

**Q: Show me where you prevent an accidental send/delete.**
The pending-state mechanism in `orchestrator.py` (`_pending_email`, `_pending_meeting`, `_pending_file_op`). Email drafts are shown and only sent on an explicit "yes". There's a comment documenting a bug I fixed where confirmation was being misrouted.

### H. Comparisons / positioning
**Q: How is this different from AutoGPT / CrewAI / LangGraph?**
Those are mostly cloud-bound, developer-oriented, and evaluated on synthetic benchmarks. Jarvis is local-first, user-facing, daily-use, and evaluated end to end. I borrow ideas (ReAct planning, role separation) but the contribution is the *integration* plus tier-based routing.

**Q: What's genuinely novel?**
Not any single subsystem — planning is ReAct-derived, memory is standard RAG. The novelty is (1) the integrated, locally-hosted, evaluated whole, and (2) tier-based routing as a tunable answer to the latency/coordination trade-off.

### I. Curveballs
**Q: If you started again, what would you change?**
Schema-validate plans with Pydantic and an explicit topological-sort step (catch bad plans earlier); run the full multi-model benchmark; do a proper user study (consent forms are drafted). I'd also formalise the demo-vs-evaluation config as an explicit flag rather than edited constants.

**Q: What was the hardest part?**
JSON reliability from small models, and making replanning actually change the plan. Both are documented in the report's challenges section.

**Q: What did you learn?**
That architecture beats model size for reliability on real tasks, and that latency is the silent killer of agent UX — which is what pushed me to invent tiering.

**Q: Show me where X happens** (be able to name the file):
routing → `agents/router.py`; tiers & the whole flow → `orchestrator.py:handle()`; DAG → `orchestrator.py:_execute_dag()`; dependency injection → `_inject_deps()`; confirmation → `_try_pending_state_intercept()`; scoring → `agents/evaluator.py`; memory → `memory/memory_agent.py`; model swap → bottom of `config/llm_client.py`.

---

## Part 12 — Demo survival kit

**Pre-flight:** Ollama running + model pulled and warm (do one throwaway query first), server up, browser open, mic permission granted, Google token valid (or accept mock mode), **backup screen-recording queued**.

**Run order:** Tier 1 (weather / "play music, volume 60") → Tier 2 (a factual question) → Tier 3 ("get the news, check the weather, give me a morning briefing" — narrate the plan/subtasks) → Voice + confirmation (stop on the confirm step to show autonomy) → FinEx (a finance question against a stored report).

**If it stalls:** switch to the recording and keep talking — never wait silently on a cold model. The most likely failure is a cold-start delay on the first Tier-3 call, which is exactly why you warm the model first.

---

## Part 13 — One-line definitions (your safety net)

- **LLM** — large language model; a neural net trained on huge text corpora.
- **Agent** — a component with one defined role and a structured interface.
- **Orchestrator** — the controller that coordinates all agents per request.
- **DAG** — directed acyclic graph; the subtask plan, with dependencies and no cycles.
- **Tier** — Jarvis's difficulty class (1 tool-only, 2 single-LLM, 3 full pipeline) that sets how much machinery runs.
- **ReAct** — "Reason + Act"; interleaves reasoning steps with tool actions.
- **RAG** — retrieval-augmented generation; ground the LLM in retrieved documents/memories.
- **Embedding** — a vector representing text meaning; similar meanings → nearby vectors.
- **Cosine similarity** — how aligned two vectors are; the memory relevance metric.
- **HNSW** — the fast approximate-nearest-neighbour index ChromaDB uses.
- **ChromaDB** — the local vector database for memory.
- **Ollama** — the local LLM runtime serving models over HTTP.
- **OAuth2** — the auth flow letting Jarvis use Gmail/Calendar without storing your password.
- **Templated dependency injection** — replacing `{subtask.result.field}` with real upstream values at run time.
- **Critic / replanning loop** — self-review of a plan that can trigger a bounded redo.
- **DSR** — Design Science Research; building-and-evaluating an artefact as the research method.
- **Tier-based routing** — matching pipeline depth to task complexity; the project's headline contribution.

---

*Built from a direct read of the Jarvis source (orchestrator.py, agents/, memory/, config/, tools/, voice/, finex/) cross-checked against the dissertation report. Where the two differ, Part 10 is the source of truth — study it.*
