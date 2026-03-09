"""
config.py — Single source of truth for all application configuration.

Loads .env once and exposes typed constants. All application modules should
import configuration from here instead of calling os.getenv() directly.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "mistral")
FALLBACK_MODELS: list[str] = ["mistral", "llama3"]

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Vector store ──────────────────────────────────────────────────────────────
VECTOR_STORE_PATH: Path = Path(os.getenv("VECTOR_STORE_PATH", "vector_store/"))

# ── Document storage ──────────────────────────────────────────────────────────
DOCUMENTS_DIR: Path = Path(os.getenv("DOCUMENTS_DIR", "data/documents"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))
RRF_K: int = int(os.getenv("RRF_K", "60"))

# ── Semantic cache ────────────────────────────────────────────────────────────
CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92"))
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))

# ── Performance targets ───────────────────────────────────────────────────────
TARGET_TTFT: float = 2.0    # seconds — Time To First Token
TARGET_TPS: float = 20.0    # tokens/sec — throughput
TARGET_LATENCY: float = 5.0  # seconds — end-to-end
