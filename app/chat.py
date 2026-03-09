"""
chat.py — Ollama inference client and semantic cache.

Responsibilities:
- OllamaClient: non-streaming generate() and streaming stream() methods
- SemanticCache: Redis-backed cache with cosine similarity lookup
- Conversation history management (in-memory, per-session)
"""

import json
import os
import time

import requests
from dotenv import load_dotenv

from app.metrics import MetricsCollector, get_collector

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


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
    """Redis-backed semantic cache using embedding cosine similarity."""

    def get(self, query: str) -> str | None:
        """Return cached response if a similar query exists, else None."""
        raise NotImplementedError

    def set(self, query: str, response: str) -> None:
        """Store query embedding and response in Redis with TTL."""
        raise NotImplementedError
