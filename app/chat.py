"""
chat.py — Ollama inference client and semantic cache.

Responsibilities:
- OllamaClient: non-streaming generate() and streaming stream() methods
- SemanticCache: Redis-backed cache with cosine similarity lookup
- Conversation history management (in-memory, per-session)
"""