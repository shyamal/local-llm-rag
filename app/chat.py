"""
chat.py — Ollama inference client and semantic cache.

Responsibilities:
- OllamaClient: non-streaming generate() and streaming stream() methods
- SemanticCache: Redis-backed cache with cosine similarity lookup
- Conversation history management (in-memory, per-session)
"""

import json
import time
import uuid

import numpy as np
import redis as redis_lib
import requests

from app.config import CACHE_SIMILARITY_THRESHOLD, CACHE_TTL, OLLAMA_BASE_URL, REDIS_URL
from app.embedder import get_embedder
from app.metrics import MetricsCollector, get_collector

_CACHE_INDEX_KEY = "semantic_cache:index"


class OllamaClient:
    """Connects to the local Ollama REST API and wraps inference calls."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._generate_url = f"{self.base_url}/api/generate"

    def generate(self, prompt: str, model: str, collector: MetricsCollector | None = None) -> str:
        """Send a prompt and return the full response (non-streaming)."""
        collector = collector if collector is not None else get_collector()
        start_time = time.monotonic()
        payload = {"model": model, "prompt": prompt, "stream": False}
        response = requests.post(self._generate_url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        if "response" not in data:
            raise ValueError(f"Unexpected Ollama response shape: {data}")
        text = data["response"]
        total_latency = time.monotonic() - start_time
        token_count = len(text.split())
        # TTFT approximated as total_latency for non-streaming calls
        collector.record(model, total_latency, token_count, total_latency)
        return text

    def stream(self, prompt: str, model: str, collector: MetricsCollector | None = None):
        """Yield tokens as they arrive from Ollama (streaming)."""
        collector = collector if collector is not None else get_collector()
        start_time = time.monotonic()
        first_token_time: float | None = None
        end_time: float | None = None
        token_count = 0

        payload = {"model": model, "prompt": prompt, "stream": True}
        try:
            with requests.post(
                self._generate_url, json=payload, stream=True, timeout=120
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("response", "")
                    if token:
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                        token_count += 1
                        yield token
                    if chunk.get("done", False):
                        end_time = time.monotonic()
                        break
        finally:
            if first_token_time is not None:
                total_latency = (end_time or time.monotonic()) - start_time
                collector.record(model, first_token_time - start_time, token_count, total_latency)

    def chat_stream(self, messages: list[dict], model: str, collector: MetricsCollector | None = None):
        """Yield tokens from a multi-turn chat conversation via /api/chat."""
        collector = collector if collector is not None else get_collector()
        start_time = time.monotonic()
        first_token_time: float | None = None
        end_time: float | None = None
        token_count = 0

        payload = {"model": model, "messages": messages, "stream": True}
        try:
            with requests.post(
                f"{self.base_url}/api/chat", json=payload, stream=True, timeout=120
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                        token_count += 1
                        yield token
                    if chunk.get("done", False):
                        end_time = time.monotonic()
                        break
        finally:
            if first_token_time is not None:
                total_latency = (end_time or time.monotonic()) - start_time
                collector.record(model, first_token_time - start_time, token_count, total_latency)

    def list_models(self) -> list[str]:
        """Return names of models available in the local Ollama instance."""
        response = requests.get(f"{self.base_url}/api/tags", timeout=10)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]


class SemanticCache:
    """Redis-backed semantic cache using embedding cosine similarity.

    Falls back to a no-op cache if Redis is unavailable, so the app works
    fully offline without Redis running.
    """

    def __init__(
        self,
        redis_url: str = REDIS_URL,
        threshold: float = CACHE_SIMILARITY_THRESHOLD,
        ttl: int = CACHE_TTL,
    ):
        self._threshold = threshold
        self._ttl = ttl
        try:
            self._redis = redis_lib.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._available = True
        except Exception:
            self._redis = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def get(self, query: str) -> str | None:
        """Return cached response if a similar query exists above threshold, else None."""
        if not self._available:
            return None
        embedder = get_embedder()
        query_vec = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        try:
            keys = list(self._redis.smembers(_CACHE_INDEX_KEY))
        except Exception:
            return None

        if not keys:
            return None

        # Batch all hgetall calls into a single pipeline round-trip
        try:
            pipe = self._redis.pipeline()
            for key in keys:
                pipe.hgetall(key)
            entries = pipe.execute()
        except Exception:
            return None

        expired_keys = []
        best_score = -1.0
        best_response: str | None = None
        for key, entry in zip(keys, entries):
            if not entry:
                expired_keys.append(key)
                continue
            stored_vec = np.array(json.loads(entry["embedding"]), dtype=np.float32)
            norm = np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)
            score = float(np.dot(query_vec, stored_vec) / (norm + 1e-8))
            if score > best_score:
                best_score = score
                best_response = entry["response"]

        if expired_keys:
            try:
                self._redis.srem(_CACHE_INDEX_KEY, *expired_keys)
            except Exception:
                pass

        return best_response if best_score >= self._threshold else None

    def set(self, query: str, response: str) -> None:
        """Store query embedding and response in Redis with TTL."""
        if not self._available:
            return
        embedder = get_embedder()
        query_vec = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        key = f"semantic_cache:{uuid.uuid4().hex}"
        try:
            self._redis.hset(key, mapping={
                "embedding": json.dumps(query_vec.tolist()),
                "response": response,
            })
            self._redis.expire(key, self._ttl)
            self._redis.sadd(_CACHE_INDEX_KEY, key)
        except Exception:
            pass
