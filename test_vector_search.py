"""
Smoke test: vector_search() against an indexed document.

Run from the project root:
    python test_vector_search.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `app.*` imports resolve.
sys.path.insert(0, str(Path(__file__).parent))

from app.rag import load_document, chunk_document, embed_chunks, build_index, save_index, load_index
from app.retriever import vector_search

DOC_PATH = "data/documents/ai_systems.txt"

# Each tuple: (query, keyword that must appear in the top-1 result)
TEST_CASES = [
    ("What is FAISS and how does vector similarity search work?", "FAISS"),
    ("How do large language models generate text?", "transformer"),
    ("What is retrieval-augmented generation?", "RAG"),
    ("How do sentence transformers create embeddings?", "embeddings"),
    ("How does BM25 rank documents?", "BM25"),
]


def build_metadata(chunks: list[str], source: str) -> list[dict]:
    """Associate each chunk with its text and source filename."""
    return [{"chunk_id": i, "text": chunk, "source": source} for i, chunk in enumerate(chunks)]


def run():
    print(f"Loading document: {DOC_PATH}")
    text = load_document(DOC_PATH)

    print("Chunking...")
    chunks = chunk_document(text)
    print(f"  {len(chunks)} chunks produced")

    print("Embedding...")
    embeddings = embed_chunks(chunks)
    print(f"  Embedding shape: {embeddings.shape}")

    print("Building and saving FAISS index...")
    index = build_index(embeddings)
    metadata = build_metadata(chunks, DOC_PATH)
    save_index(index, metadata)
    print("  Index saved to vector_store/")

    print("Reloading index from disk...")
    loaded = load_index()
    assert loaded is not None, "load_index() returned None — index files missing"
    index, metadata, corpus = loaded
    print(f"  Loaded index with {index.ntotal} vectors and {len(metadata)} metadata entries\n")

    passed = 0
    failed = 0

    for query, expected_keyword in TEST_CASES:
        results = vector_search(query, index, metadata, top_k=3)
        assert results, f"No results returned for query: {query!r}"

        top_chunk = results[0]["text"]
        score = results[0]["score"]
        hit = expected_keyword.lower() in top_chunk.lower()

        status = "PASS" if hit else "FAIL"
        print(f"[{status}] Query: {query!r}")
        print(f"       Expected keyword : {expected_keyword!r}")
        print(f"       Top-1 score (L2) : {score:.4f}")
        print(f"       Top-1 chunk      : {top_chunk[:120].strip()!r}")
        print()

        if hit:
            passed += 1
        else:
            failed += 1

    print(f"Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    run()
