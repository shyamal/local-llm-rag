"""
metrics.py — Per-inference metrics collection and storage.

Responsibilities:
- MetricsCollector: record TTFT, token count, tokens/sec, total latency
- Store results in-memory as list of dicts for dashboard display
- Targets: TTFT < 2s, tokens/sec > 20, total latency < 5s
"""

import datetime


class MetricsCollector:
    """Collects and stores inference metrics for benchmarking and dashboard display."""

    def __init__(self):
        self.results: list[dict] = []

    def record(self, model: str, ttft: float, token_count: int, total_latency: float) -> dict:
        """Compute tokens/sec, build result dict, append to results, and return it."""
        tokens_per_sec = token_count / total_latency if total_latency > 0 else 0.0
        entry = {
            "model": model,
            "ttft": round(ttft, 4),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "total_latency": round(total_latency, 4),
            "token_count": token_count,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.results.append(entry)
        return entry

    def get_results(self) -> list[dict]:
        """Return all stored inference metrics."""
        return self.results

    def clear(self) -> None:
        """Reset stored metrics."""
        self.results = []


_collector = MetricsCollector()


def get_collector() -> MetricsCollector:
    """Return the global MetricsCollector singleton."""
    return _collector
