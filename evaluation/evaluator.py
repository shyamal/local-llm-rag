"""
evaluator.py — RAG evaluation: retrieval quality and LLM-as-judge.

Responsibilities:
- Load and validate evaluation/dataset.json
- recall_at_k(): compare retrieved chunk IDs against expected relevant chunk IDs
- run_retrieval_evaluation(): run Recall@k over the full dataset, print + log results
- evaluate_faithfulness(): LLM-as-judge scoring response against context (1–5)
- evaluate_response_quality(): helpfulness, accuracy, completeness scores (1–5 each)
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_REQUIRED_FIELDS = {"id", "question", "expected_answer", "source_document", "relevant_chunk_ids"}


def load_dataset(path: str) -> list[dict]:
    """Load and validate dataset.json with schema [{"id", "question", "expected_answer", ...}].

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the JSON is not a list or any entry is missing required fields.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in dataset file: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"dataset.json must be a JSON array, got {type(data).__name__}")

    if not data:
        raise ValueError("dataset.json is empty — must contain at least one entry")

    for i, entry in enumerate(data):
        missing = _REQUIRED_FIELDS - entry.keys()
        if missing:
            raise ValueError(
                f"Entry {i} (id={entry.get('id', '?')!r}) is missing fields: {sorted(missing)}"
            )
        chunk_ids = entry["relevant_chunk_ids"]
        if not isinstance(chunk_ids, list):
            raise ValueError(
                f"Entry {i} (id={entry['id']!r}): 'relevant_chunk_ids' must be a list"
            )
        if not all(isinstance(cid, int) for cid in chunk_ids):
            raise ValueError(
                f"Entry {i} (id={entry['id']!r}): 'relevant_chunk_ids' elements must be integers"
            )

    return data


def recall_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    """Return Recall@k: fraction of relevant chunks found in top-k retrieved results.

    Formula: |relevant_ids ∩ retrieved_ids[:k]| / |relevant_ids|

    Returns 1.0 when relevant_ids is empty (nothing to recall).
    Returns 0.0 when k <= 0.
    """
    if not relevant_ids:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for rid in relevant_ids if rid in top_k)
    return hits / len(relevant_ids)


_PROJECT_ROOT = Path(__file__).parent.parent
_RESULTS_DIR = _PROJECT_ROOT / "evaluation" / "results"
_RETRIEVAL_RESULTS_FILE = _RESULTS_DIR / "retrieval_eval.json"
_FAITHFULNESS_RESULTS_FILE = _RESULTS_DIR / "faithfulness_eval.json"
_QUALITY_RESULTS_FILE = _RESULTS_DIR / "quality_eval.json"

# ---------------------------------------------------------------------------
# Faithfulness prompt template (LLM-as-judge)
# ---------------------------------------------------------------------------
_FAITHFULNESS_PROMPT = """\
You are an expert evaluator. Your task is to assess whether an AI-generated \
response is supported by the provided context.

Question: {question}

Context:
{context}

Response:
{response}

Rate the faithfulness of the response to the context using the scale below.
A faithful response only uses information present in the context.

1 = Response contradicts or completely ignores the context
2 = Response is mostly unsupported; contains significant hallucination
3 = Response is partially supported; some claims lack grounding in the context
4 = Response is mostly supported; only minor gaps or additions
5 = Response is fully supported by the context; no hallucination

Output a single integer between 1 and 5. Do not add any explanation.

Score:\
"""


def run_retrieval_evaluation(
    dataset_path: str = "evaluation/dataset.json",
    k: int = 5,
) -> dict:
    """Run Recall@k over the full dataset using vector search.

    For each dataset entry:
    - Runs vector_search against the persisted FAISS index
    - Extracts retrieved chunk IDs
    - Computes Recall@k against expected relevant_chunk_ids

    Prints per-question results and aggregate Recall@k, then logs to
    evaluation/results/retrieval_eval.json.

    Raises:
        RuntimeError: if no FAISS index exists (ingest a document first).
    """
    # Import here to keep the module importable without app dependencies installed
    from app.rag import load_index
    from app.retriever import vector_search

    dataset = load_dataset(dataset_path)

    loaded = load_index()
    if loaded is None:
        raise RuntimeError(
            "No FAISS index found. Run document ingestion before retrieval evaluation."
        )
    index, metadata, _corpus = loaded

    per_question: list[dict] = []
    print(f"\nRetrieval Evaluation — Recall@{k}\n{'─' * 60}")

    for entry in dataset:
        results = vector_search(entry["question"], index, metadata, top_k=k)
        retrieved_ids = [r["chunk_id"] for r in results]
        score = recall_at_k(retrieved_ids, entry["relevant_chunk_ids"], k)
        per_question.append(
            {
                "id": entry["id"],
                "question": entry["question"],
                "relevant_chunk_ids": entry["relevant_chunk_ids"],
                "retrieved_chunk_ids": retrieved_ids,
                f"recall_at_{k}": round(score, 4),
            }
        )
        hit = "✓" if score == 1.0 else "✗"
        print(f"  {hit} [{entry['id']}] recall@{k}={score:.2f}  {entry['question'][:60]}")

    scores = [r[f"recall_at_{k}"] for r in per_question]
    mean_recall = sum(scores) / len(scores) if scores else 0.0
    print(f"{'─' * 60}")
    print(f"  Mean Recall@{k}: {mean_recall:.4f}  ({len(dataset)} questions)\n")

    output = {
        "k": k,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(dataset),
        f"mean_recall_at_{k}": round(mean_recall, 4),
        "results": per_question,
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _RETRIEVAL_RESULTS_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Results logged to {_RETRIEVAL_RESULTS_FILE}\n")

    return output


def evaluate_faithfulness(question: str, context: str, response: str, model: str) -> int:
    """Score 1–5 whether the response is supported by the provided context.

    Sends the (question, context, response) triple to the local LLM using
    _FAITHFULNESS_PROMPT and parses the returned integer score.

    Returns:
        Integer score in [1, 5].

    Raises:
        ValueError: if the LLM output contains no parseable 1–5 score.
    """
    from app.chat import OllamaClient

    prompt = _FAITHFULNESS_PROMPT.format(
        question=question.strip(),
        context=context.strip(),
        response=response.strip(),
    )
    raw = OllamaClient().generate(prompt, model).strip()
    match = re.search(r"[1-5]", raw)
    if not match:
        raise ValueError(
            f"LLM returned no parseable 1–5 score for faithfulness. Raw output: {raw!r}"
        )
    return int(match.group())


def run_faithfulness_evaluation(
    dataset_path: str = "evaluation/dataset.json",
    model: str = "mistral",
    k: int = 5,
) -> dict:
    """Generate a RAG response for each dataset entry and score its faithfulness.

    For each entry:
    1. Retrieves top-k context chunks via vector_search
    2. Builds a RAG prompt and generates a response via OllamaClient.generate()
    3. Calls evaluate_faithfulness() to score the response against the context

    Prints per-question scores and aggregate mean, then logs to
    evaluation/results/faithfulness_eval.json.

    Raises:
        RuntimeError: if no FAISS index exists.
    """
    from app.chat import OllamaClient
    from app.rag import build_context_prompt, load_index
    from app.retriever import vector_search

    dataset = load_dataset(dataset_path)

    loaded = load_index()
    if loaded is None:
        raise RuntimeError(
            "No FAISS index found. Run document ingestion before faithfulness evaluation."
        )
    index, metadata, _corpus = loaded

    client = OllamaClient()
    per_question: list[dict] = []
    print(f"\nFaithfulness Evaluation (LLM-as-Judge) — model={model}\n{'─' * 60}")

    for entry in dataset:
        results = vector_search(entry["question"], index, metadata, top_k=k)
        chunks = [r["text"] for r in results]
        context = "\n\n".join(f"[{i + 1}] {c.strip()}" for i, c in enumerate(chunks))

        rag_prompt = build_context_prompt(entry["question"], chunks)
        response = client.generate(rag_prompt, model)

        try:
            score = evaluate_faithfulness(entry["question"], context, response, model)
        except ValueError as e:
            print(f"  ! [{entry['id']}] score parse failed: {e}")
            score = -1

        per_question.append(
            {
                "id": entry["id"],
                "question": entry["question"],
                "context": context,
                "response": response,
                "faithfulness_score": score,
            }
        )
        print(f"  [{entry['id']}] score={score}  {entry['question'][:60]}")

    valid_scores = [r["faithfulness_score"] for r in per_question if r["faithfulness_score"] > 0]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    print(f"{'─' * 60}")
    print(f"  Mean Faithfulness: {mean_score:.4f}  ({len(valid_scores)}/{len(dataset)} scored)\n")

    output = {
        "model": model,
        "k": k,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(dataset),
        "mean_faithfulness_score": round(mean_score, 4),
        "results": per_question,
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _FAITHFULNESS_RESULTS_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Results logged to {_FAITHFULNESS_RESULTS_FILE}\n")

    return output


# ---------------------------------------------------------------------------
# Response quality prompt template (LLM-as-judge)
# ---------------------------------------------------------------------------
_QUALITY_PROMPT = """\
You are an expert evaluator assessing the quality of an AI-generated response.

Question: {question}

Response:
{response}

Rate the response on each dimension below using a score from 1 to 5.

Helpfulness: Does the response actually address and answer the question?
1 = Completely unhelpful or off-topic
2 = Slightly relevant but fails to answer the question
3 = Partially helpful; answers part of the question
4 = Mostly helpful; minor gaps
5 = Fully helpful; directly and completely addresses the question

Accuracy: Is the information in the response factually correct?
1 = Mostly or entirely incorrect
2 = More incorrect than correct
3 = Mix of correct and incorrect information
4 = Mostly correct; minor inaccuracies
5 = Fully accurate; no factual errors

Completeness: Does the response cover all aspects of the question?
1 = Extremely incomplete; almost nothing covered
2 = Covers only a small fraction of what was asked
3 = Covers the main point but misses important aspects
4 = Mostly complete; minor omissions
5 = Fully complete; all aspects addressed

Output exactly three lines in this format with no extra text:
Helpfulness: <score>
Accuracy: <score>
Completeness: <score>\
"""

_QUALITY_DIMENSIONS = ("helpfulness", "accuracy", "completeness")
_QUALITY_PATTERN = re.compile(
    r"Helpfulness\s*:\s*([1-5]).*?Accuracy\s*:\s*([1-5]).*?Completeness\s*:\s*([1-5])",
    re.DOTALL | re.IGNORECASE,
)


def evaluate_response_quality(question: str, response: str, model: str) -> dict:
    """Score helpfulness, accuracy, and completeness (1–5 each) via LLM-as-judge.

    Returns:
        dict with keys "helpfulness", "accuracy", "completeness" (int values 1–5).

    Raises:
        ValueError: if the LLM output cannot be parsed into three 1–5 scores.
    """
    from app.chat import OllamaClient

    prompt = _QUALITY_PROMPT.format(
        question=question.strip(),
        response=response.strip(),
    )
    raw = OllamaClient().generate(prompt, model).strip()
    match = _QUALITY_PATTERN.search(raw)
    if not match:
        raise ValueError(
            f"LLM returned no parseable quality scores. Raw output: {raw!r}"
        )
    return {
        "helpfulness": int(match.group(1)),
        "accuracy": int(match.group(2)),
        "completeness": int(match.group(3)),
    }


def run_quality_evaluation(
    dataset_path: str = "evaluation/dataset.json",
    model: str = "mistral",
    k: int = 5,
) -> dict:
    """Generate a RAG response for each dataset entry and score its quality.

    For each entry:
    1. Retrieves top-k context chunks via vector_search
    2. Builds a RAG prompt and generates a response via OllamaClient.generate()
    3. Calls evaluate_response_quality() to score helpfulness, accuracy, completeness

    Prints per-question scores and per-dimension means, then logs to
    evaluation/results/quality_eval.json.

    Raises:
        RuntimeError: if no FAISS index exists.
    """
    from app.chat import OllamaClient
    from app.rag import build_context_prompt, load_index
    from app.retriever import vector_search

    dataset = load_dataset(dataset_path)

    loaded = load_index()
    if loaded is None:
        raise RuntimeError(
            "No FAISS index found. Run document ingestion before quality evaluation."
        )
    index, metadata, _corpus = loaded

    client = OllamaClient()
    per_question: list[dict] = []
    print(f"\nResponse Quality Evaluation (LLM-as-Judge) — model={model}\n{'─' * 60}")

    for entry in dataset:
        results = vector_search(entry["question"], index, metadata, top_k=k)
        chunks = [r["text"] for r in results]

        rag_prompt = build_context_prompt(entry["question"], chunks)
        response = client.generate(rag_prompt, model)

        try:
            scores = evaluate_response_quality(entry["question"], response, model)
        except ValueError as e:
            print(f"  ! [{entry['id']}] score parse failed: {e}")
            scores = {"helpfulness": -1, "accuracy": -1, "completeness": -1}

        per_question.append(
            {
                "id": entry["id"],
                "question": entry["question"],
                "response": response,
                **scores,
            }
        )
        h, a, c = scores["helpfulness"], scores["accuracy"], scores["completeness"]
        print(
            f"  [{entry['id']}] helpfulness={h} accuracy={a} completeness={c}"
            f"  {entry['question'][:50]}"
        )

    summary: dict[str, float] = {}
    num_valid = 0
    for dim in _QUALITY_DIMENSIONS:
        valid = [r[dim] for r in per_question if r[dim] > 0]
        if dim == "helpfulness":
            num_valid = len(valid)
        summary[f"mean_{dim}"] = round(sum(valid) / len(valid), 4) if valid else 0.0

    print(f"{'─' * 60}")
    for dim in _QUALITY_DIMENSIONS:
        print(f"  Mean {dim.capitalize()}: {summary[f'mean_{dim}']:.4f}")
    print(f"  ({num_valid}/{len(dataset)} scored)\n")

    output = {
        "model": model,
        "k": k,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(dataset),
        "num_valid": num_valid,
        **summary,
        "results": per_question,
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _QUALITY_RESULTS_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Results logged to {_QUALITY_RESULTS_FILE}\n")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG evaluation suite")
    parser.add_argument(
        "--mode",
        choices=["retrieval", "faithfulness", "quality"],
        default="retrieval",
        help="Evaluation to run (default: retrieval)",
    )
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
    args = parser.parse_args()

    try:
        if args.mode == "retrieval":
            run_retrieval_evaluation(dataset_path=args.dataset, k=args.k)
        elif args.mode == "faithfulness":
            run_faithfulness_evaluation(dataset_path=args.dataset, model=args.model, k=args.k)
        elif args.mode == "quality":
            run_quality_evaluation(dataset_path=args.dataset, model=args.model, k=args.k)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
