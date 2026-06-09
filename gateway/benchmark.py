"""Throughput benchmarking across all three compute paths.

Tracks per-backend statistics:
- Request count (total, successful, failed)
- Average latency
- P50 / P95 / P99 latency percentiles
- Requests per second (over a configurable rolling window)
- Tokens per second (if the caller supplies token counts)
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional


@dataclass
class _RequestRecord:
    timestamp: float   # monotonic clock (seconds)
    latency_s: float
    success: bool
    tokens: int = 0


@dataclass
class BackendStats:
    """Aggregated statistics for a single backend."""

    backend_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0

    # Latency (seconds)
    avg_latency_s: float = 0.0
    p50_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    p99_latency_s: float = 0.0
    min_latency_s: float = 0.0
    max_latency_s: float = 0.0

    # Throughput
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
            "latency": {
                "avg_s": round(self.avg_latency_s, 4),
                "p50_s": round(self.p50_latency_s, 4),
                "p95_s": round(self.p95_latency_s, 4),
                "p99_s": round(self.p99_latency_s, 4),
                "min_s": round(self.min_latency_s, 4),
                "max_s": round(self.max_latency_s, 4),
            },
            "throughput": {
                "requests_per_second": round(self.requests_per_second, 2),
                "tokens_per_second": round(self.tokens_per_second, 2),
            },
        }


def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile (0–100) of *data* using linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    rank = (p / 100) * (n - 1)
    lower = int(rank)
    upper = lower + 1
    if upper >= n:
        return sorted_data[-1]
    frac = rank - lower
    return sorted_data[lower] + frac * (sorted_data[upper] - sorted_data[lower])


class ThroughputBenchmark:
    """Accumulates per-backend request telemetry and computes aggregate stats.

    Args:
        window_seconds: Rolling window (seconds) used for requests-per-second
            calculation.  Default: 60 s.
    """

    def __init__(self, window_seconds: float = 60.0) -> None:
        self._window = window_seconds
        self._records: dict[str, Deque[_RequestRecord]] = {}

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def record(
        self,
        backend_id: str,
        latency_s: float,
        success: bool = True,
        tokens: int = 0,
    ) -> None:
        """Record a completed (or failed) request for *backend_id*."""
        if backend_id not in self._records:
            self._records[backend_id] = deque()
        self._records[backend_id].append(
            _RequestRecord(
                timestamp=time.monotonic(),
                latency_s=latency_s,
                success=success,
                tokens=tokens,
            )
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def stats(self, backend_id: str) -> BackendStats:
        """Compute aggregate statistics for *backend_id*."""
        s = BackendStats(backend_id=backend_id)
        records = list(self._records.get(backend_id, []))
        if not records:
            return s

        now = time.monotonic()
        window_start = now - self._window
        recent = [r for r in records if r.timestamp >= window_start]

        s.total_requests = len(records)
        s.successful_requests = sum(1 for r in records if r.success)
        s.failed_requests = s.total_requests - s.successful_requests
        s.total_tokens = sum(r.tokens for r in records)

        latencies = [r.latency_s for r in records if r.success]
        if latencies:
            s.avg_latency_s = statistics.mean(latencies)
            s.p50_latency_s = _percentile(latencies, 50)
            s.p95_latency_s = _percentile(latencies, 95)
            s.p99_latency_s = _percentile(latencies, 99)
            s.min_latency_s = min(latencies)
            s.max_latency_s = max(latencies)

        if recent:
            elapsed = now - recent[0].timestamp if len(recent) > 1 else self._window
            elapsed = max(elapsed, 1e-9)
            s.requests_per_second = len(recent) / elapsed
            s.tokens_per_second = sum(r.tokens for r in recent) / elapsed

        return s

    def all_stats(self) -> list[BackendStats]:
        """Return stats for every backend that has recorded at least one request."""
        return [self.stats(bid) for bid in self._records]

    def report(self) -> list[dict[str, Any]]:
        """Return a JSON-serialisable benchmark report."""
        return [s.to_dict() for s in self.all_stats()]

    def reset(self, backend_id: Optional[str] = None) -> None:
        """Clear recorded data.  If *backend_id* is None, clears all data."""
        if backend_id is None:
            self._records.clear()
        else:
            self._records.pop(backend_id, None)
