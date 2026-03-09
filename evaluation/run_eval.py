"""
run_eval.py — Run all evaluators sequentially and print a consolidated summary.

Usage:
    python evaluation/run_eval.py
    python evaluation/run_eval.py --model llama3 --k 3
    python evaluation/run_eval.py --dataset evaluation/dataset.json
    python evaluation/run_eval.py --mode retrieval
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `app.*` and `evaluation.*` resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import (
    run_faithfulness_evaluation,
    run_quality_evaluation,
    run_retrieval_evaluation,
)

_SEP = "═" * 68
_DIV = "─" * 68


def _row(label: str, value: str, note: str = "") -> None:
    note_str = f"  ({note})" if note else ""
    print(f"  {label:<32} {value:>10}{note_str}")


def run_all(
    dataset_path: str = "evaluation/dataset.json",
    model: str = "mistral",
    k: int = 5,
    modes: list[str] | None = None,
) -> dict:
    """Run selected evaluators sequentially and print a consolidated summary.

    Args:
        modes: subset of ["retrieval", "faithfulness", "quality"].
               Defaults to all three when None.

    Returns:
        Consolidated summary dict with all available metric keys.
    """
    if modes is None:
        modes = ["retrieval", "faithfulness", "quality"]

    retrieval = faithfulness = quality = None

    if "retrieval" in modes:
        retrieval = run_retrieval_evaluation(dataset_path=dataset_path, k=k)
    if "faithfulness" in modes:
        faithfulness = run_faithfulness_evaluation(
            dataset_path=dataset_path, model=model, k=k
        )
    if "quality" in modes:
        quality = run_quality_evaluation(dataset_path=dataset_path, model=model, k=k)

    n = (retrieval or faithfulness or quality or {}).get("num_questions", 0)
    recall_key = f"mean_recall_at_{k}"

    mean_quality: float | None = None
    if quality:
        mean_quality = round(
            (
                quality["mean_helpfulness"]
                + quality["mean_accuracy"]
                + quality["mean_completeness"]
            )
            / 3,
            4,
        )

    print(f"\n{_SEP}")
    print("  CONSOLIDATED EVALUATION SUMMARY")
    print(_SEP)
    _row("Metric", "Score", "detail")
    print(f"  {_DIV}")

    if retrieval:
        _row(f"Recall@{k}", f"{retrieval[recall_key]:.4f}", f"{n} questions")

    if faithfulness:
        faith_valid = faithfulness.get("num_valid", n)
        _row(
            "Mean Faithfulness",
            f"{faithfulness['mean_faithfulness_score']:.4f}",
            f"{faith_valid}/{n} scored",
        )

    if quality:
        qual_valid = quality.get("num_valid", n)
        _row(
            "Mean Helpfulness",
            f"{quality['mean_helpfulness']:.4f}",
            f"{qual_valid}/{n} scored",
        )
        _row("Mean Accuracy", f"{quality['mean_accuracy']:.4f}")
        _row("Mean Completeness", f"{quality['mean_completeness']:.4f}")

    if mean_quality is not None:
        print(f"  {_DIV}")
        _row("Overall Quality Score", f"{mean_quality:.4f}", "mean of 3 quality dims")

    print(_SEP + "\n")

    summary: dict = {"model": model, "k": k, "num_questions": n}
    if retrieval:
        summary[recall_key] = retrieval[recall_key]
    if faithfulness:
        summary["mean_faithfulness_score"] = faithfulness["mean_faithfulness_score"]
    if quality:
        summary["mean_helpfulness"] = quality["mean_helpfulness"]
        summary["mean_accuracy"] = quality["mean_accuracy"]
        summary["mean_completeness"] = quality["mean_completeness"]
    if mean_quality is not None:
        summary["mean_quality_score"] = mean_quality

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full RAG evaluation suite")
    parser.add_argument(
        "--dataset",
        default="evaluation/dataset.json",
        help="Path to dataset.json (default: evaluation/dataset.json)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=int(os.getenv("TOP_K", "5")),
        help="Retrieval cutoff k (default: TOP_K env var or 5)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DEFAULT_MODEL", "mistral"),
        help="Ollama model for LLM-as-judge (default: DEFAULT_MODEL env var or mistral)",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=["retrieval", "faithfulness", "quality"],
        default=None,
        help="Evaluations to run (default: all three)",
    )
    args = parser.parse_args()

    try:
        run_all(
            dataset_path=args.dataset,
            model=args.model,
            k=args.k,
            modes=args.mode,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
