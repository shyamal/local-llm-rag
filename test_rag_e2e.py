"""
End-to-end RAG test: ingest a document, ask questions, verify grounded responses.

Run from the project root (Ollama must be running):
    python3 test_rag_e2e.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.rag import (
    load_document,
    chunk_document,
    embed_chunks,
    build_index,
    save_index,
    load_index,
    build_context_prompt,
    rag_query,
)

DOC_PATH = "data/documents/ai_systems.txt"

# (question, keyword that must appear in the response)
TEST_CASES = [
    ("What is retrieval-augmented generation?", "retrieval"),
    ("How does FAISS perform similarity search?", "FAISS"),
    ("What are sentence transformers used for?", "embedding"),
]


def ingest(doc_path: str) -> tuple[int, float]:
    """Run full ingestion pipeline. Returns (chunk_count, elapsed_seconds)."""
    t0 = time.monotonic()
    text = load_document(doc_path)
    chunks = chunk_document(text)
    embeddings = embed_chunks(chunks)
    index = build_index(embeddings)
    metadata = [
        {"chunk_id": i, "text": chunk, "source": Path(doc_path).name}
        for i, chunk in enumerate(chunks)
    ]
    save_index(index, metadata)
    return len(chunks), time.monotonic() - t0


def run_rag(question: str, model: str = "mistral") -> tuple[str, float, float]:
    """Stream rag_query and return (full_response, ttft, total_elapsed)."""
    tokens = []
    ttft = None
    t0 = time.monotonic()
    for token in rag_query(question, model=model):
        if ttft is None:
            ttft = time.monotonic() - t0
        tokens.append(token)
    total = time.monotonic() - t0
    return "".join(tokens), ttft or 0.0, total


def main():
    print(f"=== RAG End-to-End Test ===\n")

    # --- Step 1: Ingest ---
    print(f"[1/3] Ingesting: {DOC_PATH}")
    n_chunks, ingest_time = ingest(DOC_PATH)
    print(f"      {n_chunks} chunks indexed in {ingest_time:.2f}s")

    # --- Step 2: Verify index loaded ---
    print("\n[2/3] Verifying index reload from disk…")
    loaded = load_index()
    assert loaded is not None, "FAIL: load_index() returned None after ingest"
    index, metadata, corpus = loaded
    assert index.ntotal == n_chunks, (
        f"FAIL: index has {index.ntotal} vectors, expected {n_chunks}"
    )
    assert len(metadata) == n_chunks, (
        f"FAIL: metadata has {len(metadata)} entries, expected {n_chunks}"
    )
    print(f"      Index OK — {index.ntotal} vectors, {len(metadata)} metadata entries")

    # --- Step 3: RAG queries ---
    print("\n[3/3] Running RAG queries against mistral…\n")
    passed = 0
    failed = 0

    for question, keyword in TEST_CASES:
        print(f"  Q: {question!r}")
        try:
            response, ttft, total = run_rag(question)
        except Exception as exc:
            print(f"  [FAIL] Exception: {exc}\n")
            failed += 1
            continue

        hit = keyword.lower() in response.lower()
        status = "PASS" if hit else "FAIL"
        print(f"  [{status}] keyword={keyword!r}  TTFT={ttft:.2f}s  total={total:.2f}s")
        print(f"         Response: {response[:200].strip()!r}")
        print()

        if hit:
            passed += 1
        else:
            failed += 1

    print(f"Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
