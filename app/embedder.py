"""
embedder.py — Shared SentenceTransformer singleton.

Centralised here so both rag.py and retriever.py use the same model instance
without creating a circular import between those modules.
"""

import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance (lazy-loaded)."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder
