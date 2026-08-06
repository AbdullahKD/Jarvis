# Jarvis — Revision & Cross‑Examination Notes
**Everything we've covered, organised to defend under fire — now grounded in a real read of the source code (9 June 2026).**
Companion files: `Jarvis_Code_Audit.md` (slides/report vs code) and `Jarvis_Presentation_Plan.md` (slide‑by‑slide). Where code and report differ, the audit doc is the source of truth.

---

## 1. The thesis (say it in your sleep)
When an LLM assistant fails at a real multi‑step task, the limiting factor is usually the **architecture around the model, not the model itself.** Evidence: a ~3B local model (`llama3.2:latest`) hit **0.83 mean** with **100% pass** on the benchmark, purely through structural design.

**Defend the precise version, not the overclaim.** Examiner: "You've shown a small model works *in your system*; you haven't held architecture constant and varied the model." Correct answer: the clean multi‑model benchmark that would prove the strong claim is **built but not fully run** (top further‑work item; confirmed in code — `BENCHMARK_MODELS` exists but the planned `phi3`/`gemma` set wasn't run). What you *can* defend: structure took a small model from unreliable to reliable on representative tasks.

---

## 2. "Local‑first" — what and why
Most assistants send your words to a company's servers. Local‑first = the reasoning happens **on your machine** via **Ollama** (runs LLMs locally over HTTP). Nothing leaves by default. Two payoffs: **privacy** and **proving the thesis** (good local results with a small model = the architecture did the work).

**The switch:** set `LLM_BACKEND=groq` and Jarvis uses Llama‑3.3‑70B via Groq with **zero code changes** (FR10). *Confirmed:* `config/llm_client.py:380‑400` rebinds the `OllamaClient` name to `GroqClient`. *Why zero‑change?* Every agent talks to the LLM through one shared client interface, so swapping the implementation behind it is invisible.

---

## 3. The nine agents
Be able to **count them** and say **which use an LLM**.

| # | Agent | File | Job | LLM? |
|---|---|---|---|---|
| 1 | Router | `agents/router.py` | Classify intent + tier (1/2/3) | Yes (`llama3.2:1b`) |
| 2 | Planner | `agents/planner.py` | Break request into a subtask DAG | Yes |
| 3 | Memory | `memory/memory_agent.py` | Store/retrieve context (ChromaDB) | Embeddings only |
| 4 | Critic | `agents/critic.py` | Review plans/results, trigger replan | Yes |
| 5 | Evaluator | `agents/evaluator.py` | Score every task → SQLite | No (pure Python) |
| 6 | Summariser | `agents/summariser.py` | Condense long outputs | Yes |
| 7 | Calendar | `agents/calendar_agent.py` | Google Calendar (OAuth2) | No |
| 8 | Gmail | `agents/gmail_agent.py` | Gmail (OAuth2); drafting uses LLM | Hybrid |
| 9 | FinEx | `agents/finex_agent.py` | Financial‑statement Q&A | Yes |

*Confirmed:* 8 files in `agents/` + Memory in `memory/` = 9.

**"Isn't multi‑agent just functions with a grand name?"** Three‑leg defence: (1) each agent has its own LLM‑facing role and typed I/O contract (the dataclasses in `config/models.py`) — they communicate through typed objects, not shared state; (2) each is independently testable/swappable; (3) honest qualifier — "It's coordinated specialisation, not emergent 'society of minds'. Router, Planner and Critic reason independently; the rest are deterministic *by design* — I put determinism exactly where I want reliability."

**"What coordinates them — a message bus?"** No. Explicit method calls in the orchestrator passing typed objects. Chosen over CrewAI/AutoGen message‑passing **deliberately, for debuggability** — you can trace exactly what happened. Tradeoff: less flexibility/scale for traceability, which suits a single‑user prototype.

**Shared contracts (`config/models.py`):** RouterDecision, TaskPlan, Subtask, CriticVerdict, EvaluationResult, MemoryItem, JarvisResponse. Every agent imports these, never redefines them — that's what makes them loosely coupled and independently testable.

---

## 4. THE FULL QUERY ROUTE (know this cold)
**Entry point: `orchestrator.py` → `handle()` (~line 480).** One front door for typed or spoken input.

### Stage 1 — Intercept chain (BEFORE routing)
Runs in order; each returns early if it matches. Exists because the tiny Router model misclassifies these predictably.
1. **Pending‑state** (`_try_pending_state_intercept`, **line 246**) — mid‑flow? ("what email address?" → your reply is an answer). Also meeting duration, file‑op confirmation.
2. **Elaborate / follow‑up** — "tell me more" re‑runs the prior question with an expand prompt.
3. **Memory command** — "remember…", "forget…", "what do you know about…".
4. **Morning briefing** — caught here or the LLM invents a fake briefing.
5. **Multi‑action split** — "dim screen, play song, remind me at 5" = three commands; Router only picks one agent, so this splits them.

*Why not let the Router handle these?* These are exactly the cases a tiny classifier fails on. Handling them deterministically before routing is a reliability decision — the thesis in miniature.

### Stage 2 — Routing (`agents/router.py`)
Produces a **RouterDecision**: intent, primary agent, supporting agents (memory always included), confidence, **tier**.
- **Layer A — `_deterministic_route`:** 30+ regex patterns (email, calendar, reminders, FinEx keywords, factual who/what/when). Returns in <1ms, no LLM. Each pattern exists because of a real misroute observed in use.
- **Layer B — `route`:** anything regex misses → `llama3.2:1b` with an ~80‑token JSON classification prompt.
- **Layer C — `_fallback_decision`:** pure rule‑based default if the LLM errors. Routing **cannot crash.**

*Why 1b for routing?* Classification is easier than generation, the 1b is faster, and routing runs on **every** request so its latency compounds.

**The subtle attack:** "You route 'schedule a meeting' to Tier 1 — misclassification?" **No.** The real booking/sending logic lives in the orchestrator's `_try_shortcut`, which **only runs at Tier 1**. At Tier 2 the lone LLM would hallucinate "done, booked!" without touching Google. Routing it to Tier 1 *guarantees* the real action fires. *Confirmed:* `router.py:147‑150` comments say exactly this. This detail proves you built it.

### Stage 3 — Branch by tier
- **Tier 1 (~800ms):** skip memory + LLM; `_try_shortcut` maps straight to a tool. No match → retrieve memory, retry once, then escalate to Tier 2.
- **Tier 2 (~3.5s):** retrieve memory, then ONE LLM call. Fast‑paths: FinEx → own engine; web/news fetch context first then answer in one paragraph. Voice mode clamps to ≤2 sentences. Output forced to one paragraph by `_enforce_single_paragraph`.
- **Tier 3 (~95s):** shortcut check → Planner → Critic (conditional) → `_execute_dag` → Evaluator → store episodic memory → build response.

### Stage 3c — Inside Tier 3
- **Planner** emits a JSON DAG (ReAct: Thought→Observation→Thought→Output). Each subtask: id, action, agent, params, depends_on. Rules: atomic subtasks, ISO‑8601 dates, **max 8 subtasks**, start with memory when context helps, `{subtask_id.result.field}` for data passing.
- **Critic** (conditional) scores 0–1 vs five criteria; approved ≥0.6, replan <0.5; inlines specific complaints into the next Planner prompt so the retry is genuinely different. ⚠️ **In the demo build the critic is gated off for most prompts — see §7 and the audit.**
- **`_execute_dag`** (**line 1235**): iterative topological sort; loop up to **2× subtask count**; run subtasks whose deps are complete; failed dep → dependent **BLOCKED** (not a crash); a round that runs nothing while tasks remain = circular dependency → stop. Dispatch via `_dispatch` (**line 1303**). *All confirmed in code.*
- **`_inject_deps`** (**line 1548**): just before each subtask, swap `{subtask_X.result.field}` for the real upstream value.
- **Evaluator:** score, persist to SQLite, record latency/replan count.
- Store result as **episodic memory**; assemble and return **JarvisResponse**.

### Worked example
**"Get the news, check the weather, give me a morning briefing":**
1. `handle()` receives it.
2. Intercepts run — as a *composed* multi‑step request it proceeds to routing as Tier 3 (a bare "morning briefing" would be caught by the intercept directly — good contrast to mention).
3. Router: 1b model → intent ~briefing, **Tier 3**.
4. Planner DAG: s1 retrieve_context (memory); s2 get_news; s3 get_weather; s4 compose_briefing (summariser, `depends_on:[s2,s3]`, params `{subtask_2.result.headlines}`, `{subtask_3.result.summary}`).
5. Critic checks ordering → approved (when enabled).
6. `_execute_dag`: round 1 runs s1, s2, s3; round 2 runs s4. `_inject_deps` fills s4's placeholders first.
7. Evaluator scores; stored as episodic memory.
8. JarvisResponse returns.

---

## 5. Reliable JSON from a small model (they WILL probe this)
Three layers: (1) Ollama `format: json` + **temperature 0.1** + a concrete schema example in the prompt; (2) a **repair step** — if strict parsing fails, extract the `{...}` block (small models add markdown fences and waffle); (3) the **Critic** as a *semantic* backstop for plans that parse fine but are logically bad. Syntactic fixes ≠ semantic fixes — say that distinction. *Confirmed:* `chat_json` + `_post_with_retry` (exponential backoff + jitter) in `config/llm_client.py`.

---

## 6. Memory — the questioned parts
- Store: **ChromaDB** (vector DB). Embeddings via Ollama's **nomic‑embed‑text**, locally.
- Retrieval: **cosine similarity**, drop below **0.3 threshold**, return **top 5**. (`forget()` uses a stricter 0.45.)
- Index: **HNSW**, cosine space (`metadata={"hnsw:space":"cosine"}`).
- Types: **episodic** (auto‑stored past tasks), **semantic** (facts you told it), **procedural** (declared but mostly a stub for a future "skill library" — be honest it's reserved).

**"Why ChromaDB over FAISS/pgvector/Pinecone?"** In‑process Python API, persistence out of the box, HNSW with zero config — ideal for a single‑user prototype. Multi‑user/scale would justify a dedicated vector DB (further work).

**"Does the 0.3 threshold hurt you?"** Own the tradeoff: too high misses relevant memories, too low pollutes the prompt. Settled empirically. Below‑threshold matches silently dropped.

**"Embedding model down?"** 8‑second timeout, then a deterministic **SHA‑256 hash‑embedding** fallback (`self.llm._hash_embed`) — request never hangs. Quality drops (non‑semantic) but stays responsive. This *robustness‑over‑purity* choice recurs system‑wide — name it as a coherent philosophy. *All confirmed in code.*

---

## 7. The Critic + the BIGGEST honesty point
- Reviews the plan before execution (and optionally results after). Returns 0–1 score, issues, suggestions, `replan_needed`.
- **Thresholds:** approved ≥0.6; replan <0.5 (`CRITIC_REPLAN_THRESHOLD`). *Confirmed.*
- Five criteria: addresses request, deps ordered, nothing missing, params sensible, not over‑engineered.
- On LLM error: **approves by default at 0.7** — pipeline never stalls. *Confirmed:* `critic.py:135‑139`.
- Smart replanning: inlines the Critic's *specific* complaints + the flawed plan into the next Planner prompt → qualitatively different attempt.

**HONESTY POINT — the most likely trap (CONFIRMED in code).** In the demo build: `MAX_REPLAN_ATTEMPTS = 0` (`orchestrator.py:55`); the Critic only runs when **confidence < 0.70 AND subtasks > 3**, or for the research agent (`orchestrator.py:782`); post‑execution review is **commented out** (`:817‑825`). Two configs: an **evaluation config** (Critic on, replanning up to 2) that produced the benchmark numbers, and a **demo config** tuned for speed. The architecture supports both.

> Also know: when the critic is skipped, `planning_score` defaults to a hard‑coded **0.8** (`orchestrator.py:785`). So in demo mode the planning component isn't a live critic score.

> If they open the file: *"For the live demo I tuned the critic down — as my own results show it adds 1–2s per call and most demo prompts are short and high‑confidence. The report figures came from the full config with it enabled. This is literally the latency‑versus‑coordination trade‑off the whole project is about — and it being a tunable parameter, not hardcoded behaviour, is the point."* Turns a gotcha into a win.

**"The Critic is an LLM judging an LLM — circular?"** Real limitation, stated in threats‑to‑validity. Mitigation: narrower job (review vs five fixed criteria) than the Planner; defaults to approving on error so it can't silently block everything. An inter‑rater study vs human judgement is named as further work.

---

## 8. The Evaluator + the exact formula (know it cold)
**overall = 0.4 × planning_score + 0.4 × execution_score + 0.2 × (1 if any subtask succeeded else 0)** — *Confirmed exactly:* `evaluator.py:71‑74`.
- execution_score = proportion of subtasks that succeeded.
- planning_score = from the Critic (or the 0.8 default when the critic is skipped — see §7).
- Pass threshold = **overall ≥ 0.6** (`EVALUATOR_MIN_SCORE`).
- Persists to SQLite (`data/jarvis.db`): latency_ms, subtask_count, replan_count, feedback.
- Export helpers: `get_model_summary()`, `export_json()`, `export_csv()` — the infrastructure that *would* power the multi‑model comparison.

**"Why 0.4/0.4/0.2 — arbitrary?"** A reasoned choice, not derived from theory. Plan and execution weighted equally because a great plan that fails to execute is as useless as a bad plan, and vice versa; the 0.2 bonus stops a task that achieved *something* from scoring zero. Weight sensitivity is something a fuller evaluation would test.

> ⚠️ The **live** `data/jarvis.db` now has **28 runs** (mean 0.845, 27/28 ≥0.7, one at 0.34). The reported **"16/16, 0.83"** is the frozen benchmark in `data/benchmark_results.csv`. If asked, distinguish the live usage log from the controlled benchmark.

---

## 9. DAG execution + templated injection (contribution #2)
**`_execute_dag`:** keep `completed` and `pending` sets; loop up to **2× subtask count** (`max_iterations = len(pending)*2`); each round run every pending subtask whose deps are all complete; failed dep → dependent BLOCKED + failure recorded; nothing ran but tasks remain → circular dependency → stop. Two guarantees: **failure isolation** and the **cycle guard**. *Why 2×?* Generous finite bound for the staggered rounds dependencies create. *Confirmed in code.*

**`_inject_deps` (templated dependency injection):** before each subtask, replace `{subtask_X.result.field}` with the real upstream value. **Why it's a contribution:** decouples the Planner (writes symbolic references at planning time, knowing nothing about the eventual data) from the Executor (resolves them at runtime). Plans become reusable and the two concerns stay clean — a small mechanism with a real architectural payoff.

---

## 10. Confirmation / user autonomy (ethics → code)
Orchestrator holds pending‑state objects: `_pending_email`, `_pending_meeting`, `_pending_file_op`. Email flow: draft → show you → wait → only send on explicit "yes/send it" (`_try_pending_state_intercept`, line 246). A comment documents a **real bug fixed**: "yes" was misrouted and the LLM faked a send — now confirmation actually calls `gmail.send_email`. When asked "how do you *guarantee* the user stays in control," point to this exact mechanism and bug fix. Concrete beats abstract.

---

## 11. LLM client layer (`config/llm_client.py`)
- `OllamaClient` wraps Ollama's REST API (async chat + embeddings). One shared instance; each agent can be given a different model for benchmarking.
- `chat_stream` → streams tokens to the live UI; `chat_json` → structured calls with JSON‑repair fallback.
- Reliability: `_post_with_retry` with exponential backoff + jitter. Defaults: **temp 0.1, 60s timeout, keep_alive 30m** (model stays warm — avoids cold‑start mid‑demo). *Confirmed.*
- Backbone swap: `LLM_BACKEND=groq` rebinds `OllamaClient` → `GroqClient` transparently (FR10).
- A detailed Jarvis persona system prompt is injected into conversational calls (one paragraph, no markdown, never claim to be the user).

---

## 12. Tools (17) + portability/extensibility
- **Pure Python (no auth):** weather (Open‑Meteo), news (RSS: BBC/Reuters/Guardian/HN/TechCrunch), markets, sports, prayer_times, web_search.
- **Authenticated (OAuth):** Calendar, Gmail, Spotify.
- **macOS‑bound:** file_manager, mac_control (AppleScript/screencapture/volume/brightness).
- **Local store:** reminders → SQLite.

`platform_guard.py` detects Mac vs cloud (Fly/Kubernetes/Docker env vars). Mac‑only tools return a clean "macOS‑only" message in cloud instead of crashing → meets **NFR7 (graceful degradation)**.

**Extensibility (NFR3):** a new tool touches ~3 files — the tool, its registration in `_dispatch`, the planner's tool catalogue — no architecture change. It's how FinEx and the voice subsystem were added late. *Confirmed:* 17 files in `tools/` incl. `platform_guard.py`.

**Counting tools:** if a marker counts differently, say "depending on whether you count helpers like the platform guard and contact book." Don't die on the exact number (17 files, ~16 if you exclude the platform guard helper).

---

## 13. Voice pipeline
Mic → STT (faster‑whisper, int8, local) → orchestrator → TTS. Utterance boundaries via **Silero VAD**. TTS = **ElevenLabs Flash (cloud)** — the **one deliberate exception** to local‑first, for sub‑100ms time‑to‑first‑audio (no open‑source TTS matched its naturalness). Own it as a justified trade‑off. Voice mode clamps to ≤2 sentences.

**HONESTY POINTS (confirmed):**
- Current `voice/runner.py` is **push‑to‑talk**, not wake‑word ("Wake‑word has been removed"); the report/Slide 8 describe `openWakeWord`. Reason: wake‑word false‑triggers during a live presentation are a real risk; the rest of the pipeline (Whisper, Silero, ElevenLabs) is identical. Say it before they find it. ⚠️ **Slide 10's demo bullet still says "Wake word →" — fix it or say "press mic."**
- STT model is **`base.en`** in the running `.env`, though the report says `small.en`. If asked: "report documents small.en; the demo box runs base.en for speed — same pipeline, one env var."

---

## 14. FinEx (proof the design generalises)
Replaced a planned **Notion** integration to prove the orchestration carries to a very different domain. Pipeline: extract financial‑statement PDF (`finex/extract_pdf.py`) → structured figures in **Postgres (Neon‑hosted)** + raw text in ChromaDB → answer via **text‑to‑SQL** generation + text retrieval (`finex/LLM_SQL.py`). Six auto‑routed sophistication levels: L1 Basic Retrieval, L2 Comparative, L3 Ratio Analysis, L4 Analytical Reasoning, L5 Investor Insight, L6 Strategic Reasoning (+ TEXT/DETAIL/OFF_TOPIC). Runs in a thread pool (engine is synchronous) via `run_in_executor` + ~100s timeout so it doesn't block the event loop. (Loaded report PDFs live in `FinEx Data/` — JLR, Tesco, Unilever, Vodafone, etc.)

**Viva value:** answers "does this only work for toy tasks like weather?" — no, it does text‑to‑SQL financial reasoning through the same orchestration.

---

## 15. Front end ↔ backend connection
**Two layers:**
1. **HTTP, once, to load the page.** Browser requests the URL; **FastAPI** (in `server.py`) serves the single‑page HUD (HTML + CSS + JavaScript).
2. **WebSocket, for the conversation.** The page's JS opens a WebSocket to **`server.py:1666` `@app.websocket("/ws")`** — a persistent two‑way channel. Your query goes down it; the orchestrator's response **streams back up token by token** via `chat_stream`, and the JS appends each chunk.

**Why WebSocket not plain HTTP?** Because the response **streams**. Plain HTTP is one‑shot — it can't push incremental output. A WebSocket lets the server push as many messages as it likes. On a slow Tier‑3 task, streaming is what lets the user see progress instead of a dead screen — the same latency/UX concern that motivated tiering.

**The boundary:** everything left of the WebSocket (HTML/CSS/JS) is presentation; everything right (FastAPI, orchestrator, agents, tools, memory) is Python and holds all the intelligence. The front end is a thin client with no application logic — swap it for the CLI and nothing in the agent system changes.

**JavaScript specifically:** lives **only** in the single‑page web front end, doing three jobs — (1) the WebSocket client, (2) UI interactions (mic button, Finance/Documents tabs, theme, HUD rendering), (3) browser mic capture for push‑to‑talk. **No reasoning, routing, planning or orchestration is in JS — that's all Python.** (Front‑end files: `ui/index.html`, `ui/finex.html`. The web server is `server.py`, not `main.py` — `main.py` is the CLI/benchmark entry point.)

---

## 16. Methodology, results, threats
**DSR:** the artefact is the object of study; each cycle had to produce evidence — why the Evaluator logged from early on. Honest scope changes: single‑agent baseline → multi‑model benchmark; FinEx replaced Notion.

**Results (default local model, n=16 across 9 tasks):**
- Overall **0.83 mean** (range 0.76–0.96); Planning **0.84**, Execution **0.78**
- Latency mean **~128,500 ms** (range 48k–183k); Replan count **0.56 mean**
- **16/16 passed ≥0.7 (100%)**
- Median latency by tier: T1 ≈ 800ms, T2 ≈ 3.5s, T3 ≈ 95s

**Three readings:** plans > execution (better at deciding than landing every external call — mostly Google Calendar flakiness); latency dominated by LLM inference, not tools; replanned tasks take longer but converge to similar scores (the loop works, doesn't thrash).

**Threats to validity (say first):** small task set (9), developer‑authored tasks, mostly one model, one hardware config (M2 MacBook Air, 16GB), and the Critic is itself an LLM.

**"n=16 is tiny — why believe it?"** "I don't claim statistical generality. The numbers show the architecture functions as designed on representative tasks. A larger task set and the full multi‑model run are my top further‑work items, and the harness to do them already exists."

> ⚠️ **Latency target vs measured — the report contradicts itself.** NFR1 says Tier 3 < 30s; §11.3 says "10–30s"; §13 reports 48–183s (mean ~128.5s). **Prepared answer:** "The <30s target was set for a cloud backbone (Groq); on the local ~3B model Tier 3 runs ~95s, which I'm upfront about — switching the backbone with one env var meets the target without any architecture change. I benchmarked the harder fully‑local case and report the real figure honestly." Tier 1 (<1s) and Tier 2 (<5s) targets **are** met — say so.

---

## 17. Comparisons / positioning
**vs AutoGPT/CrewAI/LangGraph:** mostly cloud‑bound, developer‑oriented, evaluated on synthetic benchmarks. Jarvis is local‑first, user‑facing, daily‑use, evaluated end to end. You borrow ideas (ReAct planning, role separation); the contribution is the **integration + tier‑based routing**.

**"What's genuinely novel?"** Not any single subsystem (planning is ReAct‑derived, memory is standard RAG). Novelty = (1) the integrated, locally‑hosted, *evaluated* whole, and (2) **tier‑based routing** as a tunable answer to the latency/coordination trade‑off.

---

## 18. Curveballs
- **"Start again — what changes?"** Pydantic‑validate plans + explicit topological‑sort step (catch bad plans earlier); run the full multi‑model benchmark; proper user study (consent forms drafted); formalise demo‑vs‑eval as an explicit flag rather than edited constants.
- **"Hardest part?"** JSON reliability from small models, and making replanning actually change the plan.
- **"What did you learn?"** Architecture beats model size for reliability on real tasks; latency is the silent killer of agent UX — which pushed you to invent tiering.
- **"Show me where X happens":** routing → `agents/router.py`; whole flow → `orchestrator.py:handle()` (~480); DAG → `_execute_dag` (1235); injection → `_inject_deps` (1548); confirmation → `_try_pending_state_intercept` (246); scoring → `agents/evaluator.py`; memory → `memory/memory_agent.py`; model swap → bottom of `config/llm_client.py` (380); WebSocket → `server.py:1666`.

---

## 19. One‑line definitions (safety net)
- **LLM** — large language model; neural net trained on huge text corpora.
- **Agent** — a component with one defined role and a structured interface.
- **Orchestrator** — the controller coordinating all agents per request.
- **DAG** — directed acyclic graph; the subtask plan, ordered, no cycles.
- **Tier** — difficulty class (1 tool‑only, 2 single‑LLM, 3 full pipeline) setting how much machinery runs.
- **ReAct** — Reason + Act; interleaves reasoning with tool actions.
- **RAG** — retrieval‑augmented generation; ground the LLM in retrieved docs/memories.
- **Embedding** — a vector representing text meaning; similar meaning → nearby vectors.
- **Cosine similarity** — how aligned two vectors are; the memory relevance metric.
- **HNSW** — the fast approximate‑nearest‑neighbour index ChromaDB uses.
- **ChromaDB** — the local vector database for memory.
- **Ollama** — the local LLM runtime serving models over HTTP.
- **OAuth2** — auth flow letting Jarvis use Gmail/Calendar without storing your password.
- **Templated dependency injection** — replacing `{subtask.result.field}` with real upstream values at runtime.
- **Critic / replanning loop** — self‑review of a plan that can trigger a bounded redo.
- **DSR** — Design Science Research; building‑and‑evaluating an artefact as the method.
- **WebSocket** — a persistent two‑way browser↔server channel; carries the streamed conversation.
- **FastAPI** — the Python web framework serving the page and hosting the WebSocket.

---

## 20. Carry yourself under pressure
When they find a limitation: **agree fast, explain the reasoning, name the mitigation or further work.** You've documented every weakness — that's your armour. A defensive "no, it's fine" loses; a crisp "yes, known limitation, here's why I made the call and here's what fixes it" wins. The examiner's job is to find the gap between claim and build; your defence is that you've already mapped it yourself — see `Jarvis_Code_Audit.md`.
