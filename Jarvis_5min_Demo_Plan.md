# Jarvis — 5-Minute Live Demo Plan
**Abdullah Khan Durrani · COM6001 · 12 June 2026, 5pm**
Every query below is checked against the **actual code** (`agents/router.py`, `tools/briefing.py`, `orchestrator.py`) so it fires the path I claim. Read the ⚠️ notes — one of them stops you misnaming a path live.

---

## The one-line frame before you touch the keyboard
"I'll run five requests of increasing difficulty so you can see the tier system pick the right amount of machinery for each — instant tool calls, single-model answers, then the full plan-and-execute pipeline."

That sentence makes the demo *about the thesis*, not about features.

---

## Pre-flight (do in this order — the cold-model stall is the #1 failure)
1. Ollama running; `llama3.2:latest` **pulled and warm** — fire one throwaway query so it's in memory (`keep_alive:30m` holds it).
2. `python server.py` up; browser open on the HUD; **refresh once** to confirm the WebSocket (`/ws`) connects.
3. Mic permission granted in the browser (voice is **push-to-talk** — click the mic, don't say "Hey Jarvis").
4. Google token valid, or accept Calendar **mock mode** (fine for the demo).
5. FinEx: the Bestway/HBL annual-report PDF already loaded so the finance answer returns instantly.
6. **Backup screen-recording of the full demo queued.** If anything stalls >5s, cut to it and keep narrating.

---

## The run order (≈5 min)

### 1 — Tier 1, instant (≈45s) — "no model even runs"
Type, one after another:
- **"What's the weather?"**
- **"Play some focus music, volume 60"**

What to say: "These came straight back — under a second. The router matched a regex, hit the tool directly, and **never called an LLM.** That's Tier 1." *(Verified: `weather`/`spotify` are tier-1 agents in `router.py:339`; `_try_shortcut` runs the tool with no model.)*

> Spotify needs the OAuth token live. If it's not, swap the second query for **"Take a screenshot"** or **"Set brightness to 50"** (both `mac_control`, tier 1, no auth).

### 2 — Tier 2, one model hop (≈10s) — "single LLM + a tool"
- **"Who won the Champions League final this year?"** (or any who/what/when factual)

What to say: "A who/what/when question routes to web search at Tier 2 — it fetches context, then **one** LLM call writes a one-paragraph answer. No planner, no DAG." *(Verified: factual interrogatives → websearch, tier 2; output forced single-paragraph by `_enforce_single_paragraph`.)*

### 3 — FinEx, Tier 2 into its own engine (≈10–15s) — "the design generalises"
- **"What was Bestway Cement's revenue last year?"** (or **"What was HBL's net profit?"**)

What to say: "Same orchestration, completely different domain. This one matched a finance keyword, routed straight into FinEx, which does **text-to-SQL** over a financial statement in Postgres plus text retrieval. So this isn't a toy weather bot — it does financial reasoning through the same pipeline." *(Verified: `_finex_rx` in `router.py:187` matches `bestway|hbl|revenue|net profit|…` → FINEX, tier 2.)*

### 4 — The composition showpiece (≈20–30s) — multiple tools, one answer
- **"Give me my morning briefing"**

What to say: "This pulls news, weather and my calendar in parallel and composes them into one short brief."

> ⚠️ **Be precise here — this is the one place the wording matters.** This is the **deterministic briefing path** (`is_morning_briefing` catches any phrase containing "briefing"/"morning brief" *before* routing), **not** the Planner DAG. Do **not** call it "the planner." If an examiner asks "is that the multi-agent planner?" answer honestly: "No — briefings are common enough that I handle them with a dedicated deterministic composer; the next query shows the actual planner." That honesty is a point in your favour.

### 5 — The real Tier-3 money shot (≈30–60s) — the Planner DAG
- **"Research the latest developments in small language models and summarise the key points for me"**

What to say, **narrating as subtasks appear**: "Now the router tagged this Tier 3, so the **Planner** emitted a JSON DAG — retrieve context, run the search, then summarise, with the summary depending on the search. `_execute_dag` runs them in dependency order and `_inject_deps` passes the search result into the summariser. This is the full pipeline, and it's the slow tier — which is exactly why the cheap requests above never come here."

> ⚠️ **Why this query, not "get the news, check the weather, give me a morning briefing"** (the one in your old worked example): that phrasing contains "morning briefing", so it's swallowed by the briefing intercept in #4 and **never reaches the planner.** A `research … and summarise` request routes to the RESEARCH agent → Tier 3 → Planner (`router.py` tier-3 rules). Use this one for the DAG.
> Research hits the web, so it's the most likely to be slow — this is the query you most want the **backup recording** ready for.

### Optional 6 — Voice + confirmation (≈30s, only if time) — autonomy in code
- Click the mic, speak: **"Draft an email to my supervisor saying I'll send the final report tonight."**
- **Stop on the confirmation step.**

What to say: "Notice it drafted and is now **waiting** — it won't send anything external until I explicitly say 'yes'. That's the user-autonomy principle enforced in `_try_pending_state_intercept`, not just asserted in the report." Then either say "yes" to send, or "cancel" to keep it clean.

---

## If anything stalls
Say: *"While that loads, here's the recording"* → cut to backup → keep talking. **Never wait silently on a cold model.** The single most likely failure is a cold-start on the first Tier-3 call — which is why you warmed the model in pre-flight.

## The 30-second recovery line if the whole live system dies
"The architecture is what I'm defending, not the uptime — let me walk the recording and show you the plan executing step by step." Then narrate the DAG from the recording. You lose nothing.

---

## Cheat-sheet (stick this on a sticky note)
| # | Query | Path it fires | ~time |
|---|---|---|---|
| 1 | "What's the weather?" / "Play focus music, volume 60" | Tier 1, no LLM | <1s |
| 2 | "Who won the Champions League final this year?" | Tier 2, websearch + 1 LLM | ~10s |
| 3 | "What was Bestway Cement's revenue last year?" | Tier 2 → FinEx text-to-SQL | ~10–15s |
| 4 | "Give me my morning briefing" | Deterministic briefing (NOT planner) | ~20s |
| 5 | "Research small language models and summarise the key points" | **Tier 3 Planner DAG** | ~30–60s |
| 6 | (voice) "Draft an email to my supervisor…" | Confirmation gate | ~30s |
