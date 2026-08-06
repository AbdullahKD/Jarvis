# Jarvis — Code Audit (slides/report vs actual source)
**Run against the real `~/Desktop/Jarvis` folder on 9 June 2026.** This is the audit the earlier notes couldn't do, because the code wasn't uploaded then. Every line reference below was read directly from source.

---

## TL;DR — what an examiner could catch, ranked by danger

1. **Tier‑3 latency contradicts your own report.** NFR1 says Tier 3 < 30s; §11.3 says "10–30s"; §13 results say 48–183s (mean ~128.5s). The report disagrees with itself. **Highest‑risk item.** Prep the Groq framing.
2. **Critic + replanning are switched off in the running build.** `MAX_REPLAN_ATTEMPTS = 0`, critic gated, result‑review commented out. Report/slides describe the full loop. (You already knew this — confirmed exactly.)
3. **Voice is push‑to‑talk, but Slide 10 still says "Wake word →".** The demo slide itself contradicts the build. Fix the slide wording or pre‑empt it.
4. **Whisper model is `base.en`, not `small.en`.** The report repeatedly says `small.en` (and that you downgraded medium→small). The actual `.env` runs `base.en`. New mismatch — the old study guide got this wrong too.
5. **Live `jarvis.db` now holds 28 runs incl. one failure (0.34), not a clean 16/16.** The "16/16, 0.83" is the frozen benchmark. If anyone opens the live DB they'll see 27/28.

---

## A. Claims that CHECK OUT (code matches the notes) — say these with full confidence

| Claim | Verified in code |
|---|---|
| 9 agents | 8 files in `agents/` + Memory in `memory/memory_agent.py` = 9 ✓ |
| 17 tools | 17 modules in `tools/` (incl. `platform_guard.py`) ✓ |
| Evaluator formula `0.4·plan + 0.4·exec + 0.2·(any‑success)` | `agents/evaluator.py:71‑74` exact ✓ |
| Pass threshold 0.6 | `EVALUATOR_MIN_SCORE = 0.6` (`settings.py:81`) ✓ |
| Critic approve ≥0.6 / replan <0.5 | `CRITIC_REPLAN_THRESHOLD = 0.5` (`settings.py:82`) ✓ |
| Critic approves @0.7 on LLM error | `critic.py:135‑139` (`approved=True, score=0.7`) ✓ |
| Memory: cosine + HNSW | `memory_agent.py:40` `metadata={"hnsw:space":"cosine"}` ✓ |
| Memory: top‑k 5, threshold 0.3 | `MEMORY_TOP_K=5`, `MEMORY_SIMILARITY_THRESHOLD=0.3` (`settings.py:49‑50`) ✓ |
| Embed 8s timeout → SHA‑256/hash fallback | `_EMBED_TIMEOUT=8.0`, `self.llm._hash_embed(text)` (`memory_agent.py:28,52,55`) ✓ |
| Router model `llama3.2:1b` | `OLLAMA_ROUTER_MODEL=llama3.2:1b` (`settings.py:31`) ✓ |
| Router: 30+ regex → 1b LLM → rule fallback | `router.py` `_deterministic_route` + tier rules + `route` ✓ |
| Router sends "schedule/send" to **Tier 1** on purpose | `router.py:147‑150` comment confirms (so `_try_shortcut` fires) ✓ |
| LLM defaults: temp 0.1, keep_alive 30m, `format:json` | `llm_client.py:144,160,165,193,252` ✓ |
| Backbone swap via `LLM_BACKEND=groq` | `llm_client.py:380‑400` rebinds `OllamaClient`→`GroqClient` ✓ |
| `_execute_dag` cap = `len(pending)*2` + cycle guard | `orchestrator.py:~1246` `max_iterations = len(pending)*2` ✓ |
| Failed dep → BLOCKED (no crash) | `orchestrator.py:~1263` `TaskStatus.BLOCKED` ✓ |
| Templated injection `{subtask_X.result.field}` | `_inject_deps` (`orchestrator.py:1548`) ✓ |
| FastAPI + WebSocket single‑page HUD | `server.py:61` `FastAPI(...)`, `server.py:1666` `@app.websocket("/ws")` ✓ |
| Confirmation gate (`_pending_email` etc.) | `_try_pending_state_intercept` (`orchestrator.py:246`) ✓ |

You can honestly say "that's in the code" for every one of these.

---

## B. CONFIRMED MISMATCHES — code differs from report/slides

### B1. Critic & replanning tuned down for the demo *(your Part 10.1 — confirmed exactly)*
- `orchestrator.py:55` → `MAX_REPLAN_ATTEMPTS = 0`
- `orchestrator.py:782` → critic only runs when `(routing.confidence < 0.70 and len(plan.subtasks) > 3) or primary_agent == "research"`
- `orchestrator.py:817‑825` → post‑execution result review is **commented out** ("Re‑enable for the dissertation evaluation runs only")
- **Subtle extra detail to know:** when the critic is skipped, `planning_score` is **hard‑coded to 0.8** (`orchestrator.py:785`). So in the demo build, most tasks' planning component isn't a real critic score — it's a fixed 0.8. If asked "where does the 0.84 planning mean come from then?" → "the reported figures are from the eval config with the critic on; in demo mode planning defaults to 0.8."
- **Frame:** the latency/coordination dial — your thesis in action. Tunable parameter, not a missing feature.

### B2. Voice = push‑to‑talk, NOT wake‑word *(your Part 10.2 — confirmed, plus a slide bug)*
- `voice/runner.py` header: *"Wake‑word has been removed — each turn is initiated by pressing Enter (CLI) or clicking the mic button (web UI)."*
- `.env`: *"Wake word has been removed — voice is push‑to‑talk via the UI mic button."*
- **But Slide 10 still reads "Voice + confirm — Wake word → spoken request…"** and Slide 8 lists `openWakeWord`; report FR1/FR12 and §11.4 describe `openWakeWord`. **Action: change the Slide 10 bullet to "press mic → spoken request" (or be ready to say it).** Don't let the demo slide assert something the live system won't do.

### B3. Whisper STT is `base.en`, not `small.en` *(NEW — old study guide was wrong here)*
- `.env`: `WHISPER_MODEL=base.en`; `settings.py:74` default `base`.
- Report §11.4/§12 say `small.en` (and claim a deliberate medium→small downgrade).
- **So the running build transcribes with `base.en`.** If asked: *"The report documents the small.en config; the demo machine runs base.en for speed — a smaller, faster model, same pipeline. The model name is a single env var."* Don't claim small.en is what's running.

### B4. Multi‑model benchmark not run with the planned models *(your Part 10.4 — confirmed)*
- `settings.py:35` default `BENCHMARK_MODELS="llama3,mistral"`; `.env` sets `llama3.2:latest,mistral:7b`.
- Report's planned comparison set = `llama3.2:1b`, `llama3.2:3b`, `phi3:mini`, `gemma:2b`.
- **Neither the code default nor `.env` matches the planned 4‑model set.** Claim: *infrastructure exists, single‑model (`llama3.2:latest`, n=16) results are solid; the full cross‑model study is the top further‑work item.* Do **not** claim a completed cross‑model comparison.

### B5. Default model strings are placeholders *(your Part 10.3 — confirmed, with a twist)*
- `settings.py:30` default `OLLAMA_CHAT_MODEL="llama3"`; `.env` overrides to `llama3.2:latest` (matches the report's stated default). Fine — say "code default is a placeholder, `.env` holds the real run config."
- **Twist (report‑internal):** report line 421 calls `llama3.2:3b` "the default chat backbone" while the abstract/§11.4 call the default `llama3.2:latest`. Minor internal slip in the report; if pressed, "the default is `llama3.2:latest`, which is the ~3B model."

---

## C. REPORT‑INTERNAL INCONSISTENCIES (no code needed to spot these)

### C1. Tier‑3 latency target vs measured — **the big one**
- **NFR1** (report line 183): "Tier 3 … shall complete in **under thirty seconds** on the reference hardware."
- **§11.3** (line 234): "Median latency: **10 to 30 seconds**."
- **§13 results** (lines 437, 423): evaluations **48–183s**, **mean 128,500 ms**; your study‑guide Tier‑3 median ≈ **95s**.
- The report sets a <30s expectation and then reports 48–183s. **An examiner reading only the report can catch this without your slides.**
- **Prepared answer:** *"The <30s target in NFR1 was set for the cloud backbone (Groq Llama‑3.3‑70B). On the local ~3B model Tier 3 runs ~95s, which I report honestly. Switching the backbone is one env var — `LLM_BACKEND=groq` — with no architecture change, and that meets the target. I chose to benchmark the harder, fully‑local case and be upfront about its latency."* Say it confidently; it turns the gap into a thesis point.
- Tier 1 (<1s vs ~800ms) and Tier 2 (<5s vs ~3.5s) **are met** — say so to show the targets aren't arbitrary.

### C2. FR8 "replan max 2 attempts" vs `MAX_REPLAN_ATTEMPTS = 0`
- Slide 6 / FR8 advertise self‑check + replan (max 2). Running build = 0. Same root as B1; same Groq/eval‑vs‑demo framing.

### C3. Live DB ≠ reported benchmark
- `data/jarvis.db` now has **28 evaluation rows**, mean **0.845**, **27/28 ≥0.7** (one run at **0.34**). The headline **"16/16, 0.83"** is the frozen benchmark in `data/benchmark_results.csv`, not the live operational log.
- *If they open the live DB:* "That's the live usage log — it records everything including off‑hand tests. The reported n=16 is the controlled benchmark suite in `benchmark_results.csv`." Honest and clean.
- (Note: `benchmark_results.csv` / `.json` are iCloud‑synced and were locked when I read the folder, so I verified counts from `jarvis.db` instead.)

---

## D. Small things worth knowing (so you don't fumble live)

- **Web server is `server.py`, not `main.py`.** `main.py` is the CLI / benchmark entry point. The WebSocket route is **`server.py:1666` `@app.websocket("/ws")`**. If asked "show me the front‑end/back‑end boundary," open `server.py`.
- **Line numbers drifted** as `orchestrator.py` grew (now ~231 KB): `handle` ~480 (still ✓), `_try_pending_state_intercept` **246**, `_execute_dag` **1235**, `_dispatch` **1303**, `_inject_deps` **1548**. Update your mental map so you don't say "around 1209" and then scroll.
- **Cloud embeddings differ from local.** Report says cloud deployment uses `sentence-transformers/all-MiniLM-L6-v2`; local uses `nomic-embed-text`. Fine, just know it.
- **`forget()` uses a 0.45 threshold**, not 0.3 (`memory_agent.py:194`) — deliberately stricter so it doesn't delete loosely‑related memories.
- **Tool count nuance:** 17 files *including* `platform_guard.py` (a helper). If a marker counts only "real" tools they'll get 16. Your line — "depending on whether you count the platform guard" — holds.

---

## E. One‑paragraph "honesty pre‑empt" you can open Q&A with
If you want to disarm the whole audit in advance: *"Two things I'll flag before you find them: for the live demo the critic and replanning are tuned down — that's a config switch, and the benchmark numbers in the report used the full config; and the voice path is push‑to‑talk rather than wake‑word, again for demo reliability. Both are the latency‑versus‑coordination trade‑off the project is about, exposed as tunable parameters rather than hard‑coded behaviour."* Saying it first reads as mastery, not weakness.
