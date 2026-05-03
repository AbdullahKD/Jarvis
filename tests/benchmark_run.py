"""
Jarvis Benchmark Suite
Runs 10 representative queries through the orchestrator across two models,
collects EvaluationResult data, and exports JSON + CSV for dissertation evidence.

Usage:
    cd /Users/akd/Desktop/Jarvis
    python3 -m tests.benchmark_run
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import JarvisOrchestrator
from agents.evaluator import EvaluatorAgent

# ── Test cases ─────────────────────────────────────────────────────────────────
# 10 queries spanning all major intents
TEST_QUERIES = [
    # (intent_label, query)
    ("weather",       "What's the weather like in High Wycombe right now?"),
    ("web_search",    "Who is the current Prime Minister of the UK?"),
    ("news",          "Give me the top 3 BBC headlines today"),
    ("sports",        "What are the latest Premier League results?"),
    ("markets",       "What is Bitcoin trading at right now?"),
    ("spotify",       "What's currently playing on Spotify?"),
    ("calendar",      "What do I have on my calendar this week?"),
    ("mac_control",   "What is my Mac's battery level?"),
    ("reminders",     "List all my pending reminders"),
    ("general_chat",  "Summarise what a multi-agent AI system is in 2 sentences"),
]

# Models to benchmark — add/remove as needed
MODELS = ["llama3.2:latest", "mistral:7b"]


async def run_benchmark():
    print("\n" + "=" * 70)
    print("  J.A.R.V.I.S BENCHMARK SUITE")
    print("  Dissertation evaluation data collection")
    print("=" * 70 + "\n")

    evaluator = EvaluatorAgent()
    results_by_model: dict = {}

    for model in MODELS:
        print(f"\n{'─'*60}")
        print(f"  Model: {model}")
        print(f"{'─'*60}")

        # Fresh orchestrator for each model
        try:
            jarvis = JarvisOrchestrator(model=model)
        except Exception as e:
            print(f"  ⚠  Could not initialise with {model}: {e}")
            continue

        model_results = []

        for intent_label, query in TEST_QUERIES:
            print(f"\n  [{intent_label}] {query[:60]}...")
            try:
                start = time.time()
                response = await asyncio.wait_for(
                    jarvis.handle(query, model_override=model),
                    timeout=90.0,
                )
                elapsed = (time.time() - start) * 1000
                icon = "✅" if response.success else "❌"
                print(f"  {icon}  {elapsed:.0f}ms | score={response.evaluation.get('score', 0):.3f}")
                model_results.append({
                    "intent": intent_label,
                    "query": query,
                    "success": response.success,
                    "latency_ms": elapsed,
                    "score": response.evaluation.get("score", 0),
                })
            except asyncio.TimeoutError:
                print(f"  ⏱  TIMEOUT after 90s")
                model_results.append({
                    "intent": intent_label,
                    "query": query,
                    "success": False,
                    "latency_ms": 90000,
                    "score": 0,
                })
            except Exception as e:
                print(f"  💥  ERROR: {e}")
                model_results.append({
                    "intent": intent_label,
                    "query": query,
                    "success": False,
                    "latency_ms": 0,
                    "score": 0,
                })

        results_by_model[model] = model_results

    # ── Print summary table ───────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<22} {'Tasks':>5} {'Success':>8} {'Avg Score':>10} {'Avg Latency':>13}")
    print("─" * 65)

    for model, results in results_by_model.items():
        total = len(results)
        successes = sum(1 for r in results if r["success"])
        avg_score = sum(r["score"] for r in results) / total if total else 0
        avg_lat = sum(r["latency_ms"] for r in results) / total if total else 0
        print(
            f"{model:<22} {total:>5} {successes:>8} {avg_score:>10.3f} {avg_lat:>11.0f}ms"
        )

    print()

    # ── Export ───────────────────────────────────────────────────────────
    json_path = evaluator.export_json()
    csv_path  = evaluator.export_csv()

    print(f"\n📊 JSON exported → {json_path}")
    print(f"📊 CSV  exported → {csv_path}")
    print("\nDone. Drop data/benchmark_results.csv into Excel for dissertation charts.\n")

    return results_by_model


if __name__ == "__main__":
    asyncio.run(run_benchmark())
