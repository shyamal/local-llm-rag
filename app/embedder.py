"""
embedder.py — Shared SentenceTransformer singleton.

Centralised here so both rag.py and retriever.py use the same model instance
without creating a circular import between those modules.
"""

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance (lazy-loaded)."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder
