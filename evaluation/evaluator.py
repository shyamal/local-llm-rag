"""
evaluator.py — RAG evaluation: retrieval quality and LLM-as-judge.

Responsibilities:
- Load and validate evaluation/dataset.json
- recall_at_k(): compare retrieved chunk IDs against expected relevant chunk IDs
- evaluate_faithfulness(): LLM-as-judge scoring response against context (1–5)
- evaluate_response_quality(): helpfulness, accuracy, completeness scores (1–5 each)
"""

import json
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
    """Return Recall@k: fraction of relevant chunks found in top-k retrieved results."""
    raise NotImplementedError


def evaluate_faithfulness(question: str, context: str, response: str, model: str) -> int:
    """Score 1–5 whether the response is supported by the provided context."""
    raise NotImplementedError


def evaluate_response_quality(question: str, response: str, model: str) -> dict:
    """Score helpfulness, accuracy, completeness (1–5 each). Returns dict of scores."""
    raise NotImplementedError
