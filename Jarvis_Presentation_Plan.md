# Jarvis — Presentation Plan
**Abdullah Khan Durrani · COM6001 Pathway A · 12 June 2026, 5pm · 13 slides**
Target: ~18–20 min talk + ~5 min live demo, leaving room for Q&A.

---

## The one sentence everything serves
> **The bottleneck for a reliable LLM assistant is the architecture around the model, not the model itself — proven by getting a small ~3B local model to perform reliably (0.83 mean, 100% pass) purely through how the system is structured.**

Say it on Slide 2, again on Slide 11. Every slide in between is evidence for it.

---

## Pre‑flight checklist (do in this order — the #1 failure is a cold model stalling the first Tier‑3 call)
- [ ] Ollama running, `llama3.2:latest` pulled **and warmed** — fire one throwaway query (`keep_alive: 30m` keeps it warm after).
- [ ] `server.py` running; browser open on the HUD; refresh once to confirm the WebSocket (`/ws`) connects.
- [ ] Mic permission granted in the browser (voice is **push‑to‑talk via the mic button** — not wake‑word).
- [ ] Google token valid, or accept Calendar mock mode (fine for demo).
- [ ] FinEx: an annual‑report PDF already loaded/stored so the finance question returns instantly.
- [ ] **Backup screen‑recording of the full demo queued** — if anything stalls, cut to it and keep talking.
- [ ] Water, clicker tested, these notes + the audit doc to hand.

---

## Timing table

| Slide | Content | Target | Compress to |
|---|---|---|---|
| 1 | Title | 20s | 15s |
| 2 | Problem & research question | 2m | 90s |
| 3 | Literature / the gap | 2m | 90s |
| 4 | Aim & objectives | 1.5m | 1m |
| 5 | Methodology, ethics, risk | 2m | 90s |
| 6 | Requirements | 1m | 30s (skim) |
| 7 | **Architecture + tiers (CORE)** | 3.5m | 3m — never less |
| 8 | Development / hard parts | 2m | 90s |
| 9 | Results | 2m | 90s |
| 10 | **Live demo** | ~5m | 4m |
| 11 | Conclusion & contributions | 1.5m | 1m |
| 12 | Further work | 1m | 45s |
| 13 | Thank you / references | 20s | 15s |

**If running long, compress Slide 6 (skim requirements) and Slide 8 (pick two hard parts, not five). Never compress Slide 7 or the demo.**

---

## Per‑slide talking points

**Slide 1 — Title (20s).** Don't read it. "Good afternoon — I'm Abdullah, this is Jarvis, a local‑first multi‑agent personal assistant. Over the next twenty minutes I'll show how it works and what I think it proves." Move on.

**Slide 2 — Problem (2m).** The gap between *reasoning* and *doing*: "Modern LLMs reason fluently, but the assistants around them still behave like one‑shot chatbots — they think in isolation, struggle past one step, can't reliably take the actions real tasks need." Walk the timeline (pattern‑matching → general LLMs → agentic cloud demos → Jarvis: local, daily‑use, evaluated end to end). **End on the research question, slowly:** "Is the model the limit, or the architecture around it?"

**Slide 3 — Literature / the gap (2m).** Three fields: reasoning (CoT, ReAct, Reflexion), multi‑agent + tools (CrewAI, LangGraph, AutoGen), memory (RAG, MemGPT, HNSW vector indexing). **The trust‑earning line:** "No single part of Jarvis is new. Bringing all three together into something you actually run locally and use daily — then evaluating the whole thing — is the point." Pre‑empts "what's novel?".

**Slide 4 — Aim & objectives (1.5m).** Aim in one sentence (modular assistant, runs locally, decomposes into steps, remembers, acts, checks itself, no closed cloud model). Don't read all seven objectives — "seven objectives map onto those capabilities," then call out three: separate agents, DAG planning, critic‑driven replanning. Personal rationale: "I've used Jarvis as my own assistant for ~six months — that's where most of the fixes came from."

**Slide 5 — Methodology, ethics, risk (2m).** DSR in plain words: "I studied the system by building it — each cycle had to produce evidence, which is why the Evaluator logged scores from early on." Two honest scope changes: single‑agent baseline → multi‑model benchmark; FinEx replaced Notion. Three ethics principles → each maps to a real mechanism: data sovereignty (local stack), user autonomy (confirmation gate), epistemic honesty (the Critic). Risks: hallucination → JSON‑only + low temp + Critic; API limits → backoff + graceful degradation.

**Slide 6 — Requirements (1m, skim).** Don't read FR numbers. "Fifteen functional, eight non‑functional — the ones that shaped the architecture are: confirm before destructive actions (FR7), swap the model with no code change (FR10), and the per‑tier latency targets, which pushed me to tiering." Move on fast. ⚠️ FR8 advertises "replan max 2" — that's the eval config; have the B1/C2 answer ready if probed (see audit).

**Slide 7 — Architecture + tiers (3.5m — THE CORE, protect this).** Three moves:
1. **The split:** "Nine agents share the work so no single model has to plan, remember, act and check at once." Walk the flow once: Router → Memory → Planner → Critic → Executor → Critic → Evaluator. Say what the Critic does (checks plan before, result after).
2. **Tier‑based routing (headline):** "Not every request deserves the full pipeline. Early on everything went through it and trivial requests took 30–45s. Tiering matches machinery to difficulty." Tier 1 ~800ms (regex shortcut, no LLM); Tier 2 ~3.5s (one LLM hop + tool); Tier 3 ~95s (full Planner→Critic→Executor DAG).
3. **The research insight:** "There's a real trade‑off between how much the agents coordinate and how fast they respond. Tiering turns that into a dial you can adjust." Then the two supporting mechanisms: templated injection (`{subtask_1.result.field}`) and hybrid routing (30+ regex <1ms, then a 1B fallback).

**Slide 8 — Development / hard parts (2m).** Stack in one breath: Python/asyncio, Ollama local + Groq cloud, ChromaDB + SQLite, FastAPI + WebSocket, Whisper/Silero/ElevenLabs, Docker/Fly.io. Then **two** war stories: (a) clean JSON from small models (they wrap it in markdown → repair layers), (b) replanning that actually changes (feed the Critic's *specific* complaints back in, not "try again"). ⚠️ Slide lists `openWakeWord` — if asked, use the push‑to‑talk framing (audit B2).

**Slide 9 — Results (2m).** Headline: 0.83 mean overall, 16/16 passed ≥0.7, ~1 in 3 tasks replanned, across 9 tasks / 7 intents / 3 tiers. Three readings: plans > execution (0.84 vs 0.78 — better at deciding than landing every API call, mostly Google Calendar flakiness); latency dominated by model inference not tools; replanned tasks take longer but converge to similar scores. **State your own limits first:** small task set (9), mostly one model, one laptop, Critic is itself an LLM. ⚠️ If the latency target comes up, use the Groq framing (audit C1).

**Slide 10 — Live demo (~5m).** Narrate throughout — never go silent.
1. **Tier 1:** "What's the weather?" then "Play focus music, volume 60" — point out it returns instantly, no model.
2. **Tier 2:** a factual question — one LLM hop + web search.
3. **Tier 3:** "Get the news, check the weather, give me a morning briefing" — **narrate the subtasks/DAG as they run.** The money shot.
4. **Voice + confirm:** press the mic, speak a calendar/email request, **stop on the confirmation step** — "notice it asks before doing anything external — user autonomy enforced in code." ⚠️ Say "press the mic," not "wake word."
5. **FinEx:** ask a question against the loaded annual‑report PDF.
If anything stalls: "while that loads, here's the recording" — cut to backup, keep talking.

**Slide 11 — Conclusion (1.5m).** Close the Slide‑2 loop: "A small ~3B local model reached 0.83 with every task passing — from how the system is put together, not a bigger model. The bottleneck is the architecture." Three contributions: tier‑based routing, templated injection, critic‑replanning loop. End on the trade‑off line: "There's a real tension between coordination and speed — tiering makes it a dial."

**Slide 12 — Further work (1m).** "Working, but not finished." Top items: schema validation + topological checks (~1 week, catch bad plans earlier); the **full multi‑model benchmark** (infrastructure already exists — highest value); a real user study (consent forms drafted). Mentioning the study is drafted shows planning.

**Slide 13 — Thank you (20s).** "That's Jarvis — happy to take questions." Stop cleanly, look up. Don't trail off.

---

## Three lines to say in your sleep
1. **Thesis:** "When an LLM assistant fails at a real task, the limiting factor is usually the architecture around the model, not the model itself."
2. **Tier insight:** "Tiering turns the trade‑off between coordination and speed into a dial you can adjust — that's the transferable finding."
3. **Demo‑config honesty:** "The critic is tuned down for the live demo because it adds latency on short prompts; the benchmark numbers used the full config. That's the latency‑vs‑coordination trade‑off in action — a tunable parameter, not a missing feature."

## Two slide fixes to make tonight (5 minutes)
- **Slide 10:** change "Wake word → spoken request" to "**Press mic → spoken request**" (the build is push‑to‑talk).
- **Slide 8 (optional):** if you don't want a wake‑word question, footnote `openWakeWord` as "(wake‑word implemented; demo runs push‑to‑talk)".

## If you blank mid‑sentence
Pause, look at the slide, ask yourself "what does this slide prove about my thesis?" Every slide has an answer; saying it gets you moving. Two seconds of silence beats filler waffle.
