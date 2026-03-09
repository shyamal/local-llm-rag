#!/usr/bin/env python3
"""CLI smoke test for OllamaClient.generate() and OllamaClient.stream()."""

import sys
from app.chat import OllamaClient

MODELS = ["mistral", "llama3"]
TEST_PROMPT = "Reply with exactly one sentence: what is the capital of France?"


def test_generate(client: OllamaClient, model: str) -> None:
    print(f"\n[generate] model={model}")
    response = client.generate(TEST_PROMPT, model)
    print(f"  → {response.strip()}")


def test_stream(client: OllamaClient, model: str) -> None:
    print(f"\n[stream]   model={model}")
    print("  → ", end="", flush=True)
    for token in client.stream(TEST_PROMPT, model):
        print(token, end="", flush=True)
    print()


def main() -> None:
    client = OllamaClient()
    models = sys.argv[1:] if len(sys.argv) > 1 else MODELS
    for model in models:
        try:
            test_generate(client, model)
            test_stream(client, model)
        except Exception as exc:
            print(f"\n  ERROR ({model}): {exc}", file=sys.stderr)
    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
