"""
benchmark.py — Model benchmarking runner.

Responsibilities:
- run_benchmark(): send a fixed prompt to a model N times and record metrics
- Run same benchmark across multiple models sequentially
- Export results to benchmarks/results/ as JSON
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chat import OllamaClient
from app.config import FALLBACK_MODELS
from app.metrics import MetricsCollector

RESULTS_DIR = Path(__file__).parent / "results"

BENCHMARK_PROMPT = (
    "Explain the difference between supervised and unsupervised machine learning "
    "in three concise sentences."
)


def run_benchmark(
    prompt: str, model: str, n: int = 5, verbose: bool = True
) -> list[dict]:
    """Send prompt to model N times via OllamaClient; return list of metric dicts."""
    collector = MetricsCollector()
    client = OllamaClient()

    for i in range(n):
        if verbose:
            print(f"  [{model}] run {i + 1}/{n} ...", end=" ", flush=True)
        try:
            for _ in client.stream(prompt, model, collector=collector):
                pass
            if verbose:
                last = collector.get_results()[-1]
                print(f"{last['token_count']} tokens, {last['ttft']:.3f}s TTFT")
        except Exception as exc:
            if verbose:
                print(f"FAILED: {exc}")

    return collector.get_results()


def run_multi_model_benchmark(
    prompt: str, models: list[str], n: int = 5, verbose: bool = True
) -> dict[str, list[dict]]:
    """Run run_benchmark for each model sequentially. Returns {model: [metric dicts]}."""
    results: dict[str, list[dict]] = {}
    for model in models:
        if verbose:
            print(f"\nBenchmarking {model} ({n} runs)…")
        results[model] = run_benchmark(prompt, model, n, verbose=verbose)
    return results


def export_results(results: dict[str, list[dict]], output_path: str | None = None) -> Path:
    """Write benchmark results to a JSON file in benchmarks/results/.

    Returns the path of the written file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = RESULTS_DIR / f"benchmark_{timestamp}.json"
    else:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

    dest.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run model benchmarks")
    parser.add_argument(
        "--models",
        nargs="+",
        default=FALLBACK_MODELS,
        help="Ollama model names to benchmark (default: mistral llama3)",
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of runs per model (default: 5)"
    )
    parser.add_argument("--output", default=None, help="Output JSON path (optional)")
    args = parser.parse_args()

    results = run_multi_model_benchmark(BENCHMARK_PROMPT, args.models, n=args.runs)
    dest = export_results(results, args.output)
    print(f"\nResults saved to {dest}")

    # Print summary table
    print(f"\n{'Model':<20} {'TTFT (s)':<12} {'Tokens/s':<12} {'Latency (s)':<12} Runs")
    print("-" * 64)
    for model, runs in results.items():
        if not runs:
            continue
        avg_ttft = sum(r["ttft"] for r in runs) / len(runs)
        avg_tps = sum(r["tokens_per_sec"] for r in runs) / len(runs)
        avg_lat = sum(r["total_latency"] for r in runs) / len(runs)
        print(f"{model:<20} {avg_ttft:<12.3f} {avg_tps:<12.1f} {avg_lat:<12.3f} {len(runs)}")
