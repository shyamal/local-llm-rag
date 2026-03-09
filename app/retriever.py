"""
retriever.py — Hybrid retrieval with FAISS, BM25, and RRF fusion.

Responsibilities:
- vector_search(): embed query and retrieve top-k chunks from FAISS
- bm25_search(): keyword search over the chunk corpus using BM25Okapi
- reciprocal_rank_fusion(): merge ranked lists using RRF (k=60)
- hybrid_search(): run both retrievers and return fused top-k results
"""

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from app.config import RRF_K
from app.embedder import get_embedder


def vector_search(query: str, index: faiss.Index, metadata: list[dict], top_k: int = 5) -> list[dict]:
    """Embed query and return top-k chunks from FAISS ranked by L2 distance.

    Returns dicts from metadata with an added 'score' key (L2 distance; lower = more similar).
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    # Clamp to avoid FAISS returning -1 sentinel indices
    top_k = min(top_k, index.ntotal)
    if top_k == 0:
        return []

    embedder = get_embedder()
    query_vec = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
    query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)

    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        entry = dict(metadata[idx])
        entry["score"] = float(dist)
        results.append(entry)
    return results


def bm25_search(query: str, corpus: list[str], metadata: list[dict], top_k: int = 5) -> list[dict]:
    """Return top-k chunks from BM25 keyword search over the corpus.

    Tokenises by lowercased whitespace split. Results are sorted by BM25 score
    (higher = more relevant). Returns dicts from metadata with an added 'score' key.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if not corpus:
        return []

    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_k = min(top_k, len(corpus))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score == 0.0:
            break  # remaining scores are also 0 — no keyword match
        entry = dict(metadata[idx])
        entry["score"] = score
        results.append(entry)
    return results


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Fuse multiple ranked lists using RRF: score = Σ 1 / (k + rank).

    Deduplicates by 'text' field. Returns all fused results sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = (doc["chunk_id"], doc["source"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in docs:
                docs[key] = doc

    fused = []
    for key, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = dict(docs[key])
        entry["score"] = rrf_score
        fused.append(entry)
    return fused


def hybrid_search(query: str, index: faiss.Index, metadata: list[dict], corpus: list[str], top_k: int = 5) -> list[dict]:
    """Run vector_search and bm25_search, fuse with RRF, return top-k chunks."""
    vector_results = vector_search(query, index, metadata, top_k=top_k)
    bm25_results = bm25_search(query, corpus, metadata, top_k=top_k)
    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    return fused[:top_k]
