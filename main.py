"""
Jarvis — Multi-Agent AI Executive Assistant
Main entry point. Run directly for a demo, or import JarvisOrchestrator
in your web UI / voice layer.

Usage:
    python main.py # interactive CLI demo
    python main.py --benchmark # run full benchmark suite
    python main.py --export # export benchmark results to JSON/CSV
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import List

from orchestrator import JarvisOrchestrator
from config.settings import BENCHMARK_MODELS, OLLAMA_CHAT_MODEL


# ── Demo tasks ─────────────────────────────────────────────────────────────
DEMO_TASKS = [
    "What's the weather like today?",
    "Search the web for recent advances in multi-agent AI systems",
    "Get me the latest tech headlines",
    "Play some focus music on Spotify",
    "Open Safari on my Mac",
    "Set my Mac volume to 40",
    "Send me a notification: time to take a break",
    "What time is it and remind me to review my notes in 10 minutes",
]

# ── Benchmark tasks (used for benchmark metrics) ────────────────────────
BENCHMARK_TASKS = [
    # Simple single-agent tasks
    "What's the weather today?",
    "Get me the top BBC headlines",
    "Search for information about ReAct prompting framework",
    # Multi-step tasks
    "Search the web for the latest AI news and summarise the top 3 stories",
    "What's the weather this week and should I bring an umbrella?",
    "Find me a focus music playlist on Spotify and set volume to 60",
    # Complex tasks
    "Get the news, check the weather, and give me a morning briefing",
    "Search for information about ChromaDB and save a note about it",
    "Open my notes app and set a reminder to check emails in 30 minutes",
]


async def run_demo(model: str = OLLAMA_CHAT_MODEL) -> None:
    """Run a quick interactive demo."""
    print("\n" + "="*60)
    print("🤖 JARVIS — Multi-Agent AI Executive Assistant")
    print("="*60)

    jarvis = JarvisOrchestrator(model=model)

    # Verify Ollama is running
    if not await jarvis.llm.health_check():
        print("\n❌ Ollama is not running. Start it with: ollama serve")
        print(" Then pull a model: ollama pull llama3")
        sys.exit(1)

    print("\nType a request (or 'quit' to exit, 'benchmark' to run tests):\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("👋 Goodbye!")
            break
        if user_input.lower() == "benchmark":
            await run_benchmark()
            continue

        response = await jarvis.handle(user_input)
        print(f"\nJarvis: {response.message}")
        if not response.success:
            print(f" [Error: {response.error}]")
        print()


async def run_benchmark(
    models: List[str] = None,
    tasks: List[str] = None,
) -> None:
    """
    Run the full benchmark suite across multiple models.
    Results are automatically stored in SQLite and can be exported.

    This generates the benchmark evaluation data:
    - Per-model accuracy, latency, and planning quality scores
    - Task success rates across intent types
    - Replan frequency by model
    """
    models = models or BENCHMARK_MODELS
    tasks = tasks or BENCHMARK_TASKS

    print("\n" + "="*60)
    print("📊 JARVIS BENCHMARK SUITE")
    print(f" Models: {', '.join(models)}")
    print(f" Tasks: {len(tasks)}")
    print(f" Total: {len(models) * len(tasks)} evaluations")
    print("="*60 + "\n")

    # We need one orchestrator per model for clean benchmarking
    for model in models:
        print(f"\n{'─'*40}")
        print(f"Model: {model}")
        print(f"{'─'*40}")

        jarvis = JarvisOrchestrator(model=model)

        if not await jarvis.llm.is_model_available(model):
            print(f"⚠️ Model '{model}' not pulled. Run: ollama pull {model}")
            continue

        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] {task}")
            response = await jarvis.handle(task, model_override=model)
            icon = "✅" if response.success else "❌"
            print(f" {icon} {response.message[:80]}")
            await asyncio.sleep(0.5) # Avoid hammering Ollama

    # Export results
    from agents.evaluator import EvaluatorAgent
    evaluator = EvaluatorAgent()

    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("="*60)

    summary = evaluator.get_model_summary()
    for model, stats in summary.items():
        print(f"\n Model: {model}")
        print(f" Tasks: {stats['total_tasks']}")
        print(f" Success rate: {stats['successes']}/{stats['total_tasks']}")
        print(f" Avg score: {stats['avg_score']:.3f}")
        print(f" Avg planning: {stats['avg_planning']:.3f}")
        print(f" Avg execution: {stats['avg_execution']:.3f}")
        print(f" Avg latency: {stats['avg_latency_ms']:.0f}ms")
        print(f" Avg replans: {stats['avg_replans']:.2f}")

    json_path = evaluator.export_json()
    csv_path = evaluator.export_csv()
    print(f"\n📁 Results exported:")
    print(f" JSON: {json_path}")
    print(f" CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Jarvis AI Assistant")
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run full benchmark suite across all models"
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export existing benchmark results to JSON and CSV"
    )
    parser.add_argument(
        "--model", default=OLLAMA_CHAT_MODEL,
        help=f"Model to use (default: {OLLAMA_CHAT_MODEL})"
    )
    parser.add_argument(
        "--models", nargs="+",
        help="Models for benchmarking (e.g. --models llama3 mistral)"
    )
    args = parser.parse_args()

    if args.export:
        from agents.evaluator import EvaluatorAgent
        ev = EvaluatorAgent()
        ev.export_json()
        ev.export_csv()
        print("✅ Export complete")
        return

    if args.benchmark:
        asyncio.run(run_benchmark(models=args.models))
    else:
        asyncio.run(run_demo(model=args.model))


if __name__ == "__main__":
    main()
