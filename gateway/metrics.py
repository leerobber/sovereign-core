"""
gateway/metrics.py — Prometheus Metrics for Sovereign Core Gateway

Counters, histograms, and gauges for:
  - Inference requests (by backend, model, status)
  - Request latency (by backend)
  - Active backend count
  - Auction outcomes
  - KAIROS evolution cycles
  - HTTP request rate (by method, path, status)

Usage:
    from gateway.metrics import INFERENCE_COUNTER, INFERENCE_LATENCY, metrics_output
    INFERENCE_COUNTER.labels(backend="rtx5050", model="qwen", status="success").inc()
    INFERENCE_LATENCY.labels(backend="rtx5050").observe(0.42)
"""
from __future__ import annotations

import time
from typing import Optional

# ── Prometheus client (optional dep — graceful no-op if not installed) ─────

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False


# ── No-op shims (used when prometheus_client not installed) ────────────────

class _NoopLabels:
    def inc(self, amount=1): pass
    def set(self, value): pass
    def observe(self, value): pass
    def time(self): return _NoopCtx()

class _NoopCtx:
    def __enter__(self): return self
    def __exit__(self, *a): pass

class _NoopMetric:
    def labels(self, **kw): return _NoopLabels()
    def inc(self, amount=1): pass
    def set(self, value): pass
    def observe(self, value): pass


def _counter(name, doc, labelnames=()):
    if _PROM_AVAILABLE:
        return Counter(name, doc, labelnames)
    return _NoopMetric()

def _gauge(name, doc, labelnames=()):
    if _PROM_AVAILABLE:
        return Gauge(name, doc, labelnames)
    return _NoopMetric()

def _histogram(name, doc, labelnames=(), buckets=None):
    if _PROM_AVAILABLE:
        kw = {"buckets": buckets} if buckets else {}
        return Histogram(name, doc, labelnames, **kw)
    return _NoopMetric()


# ── Metrics ────────────────────────────────────────────────────────────────

INFERENCE_COUNTER = _counter(
    "sovereign_inference_total",
    "Total inference requests by backend, model, and status",
    ["backend", "model", "status"],
)

INFERENCE_LATENCY = _histogram(
    "sovereign_inference_latency_seconds",
    "Inference request latency in seconds",
    ["backend"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

ACTIVE_BACKENDS = _gauge(
    "sovereign_active_backends",
    "Number of currently healthy GPU backends",
)

AUCTION_BIDS_TOTAL = _counter(
    "sovereign_auction_bids_total",
    "Total auction bids submitted",
    ["resource", "outcome"],
)

AUCTION_CREDITS_SPENT = _counter(
    "sovereign_auction_credits_spent_total",
    "Total credits spent in auctions",
    ["resource"],
)

KAIROS_CYCLES_TOTAL = _counter(
    "sovereign_kairos_cycles_total",
    "Total KAIROS ARSO evolution cycles completed",
    ["verdict"],
)

KAIROS_ELITE_PROMOTIONS = _counter(
    "sovereign_kairos_elite_promotions_total",
    "Total agents promoted to Elite tier",
)

KAIROS_SCORE = _gauge(
    "sovereign_kairos_best_score",
    "Best KAIROS agent score in current session",
)

HTTP_REQUESTS_TOTAL = _counter(
    "sovereign_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

HTTP_LATENCY = _histogram(
    "sovereign_http_latency_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

LEDGER_ENTRIES_TOTAL = _counter(
    "sovereign_ledger_entries_total",
    "Total Aegis-Vault ledger entries written",
    ["operation_type"],
)

WS_CONNECTIONS = _gauge(
    "sovereign_ws_connections",
    "Current WebSocket client connections",
)

# ── Helpers ────────────────────────────────────────────────────────────────

def record_request(method: str, path: str, status: int, latency: float):
    """Record an HTTP request in both counter and histogram."""
    # Normalize path to avoid high-cardinality label explosion
    normalized = _normalize_path(path)
    HTTP_REQUESTS_TOTAL.labels(method=method, path=normalized, status=str(status)).inc()
    HTTP_LATENCY.labels(method=method, path=normalized).observe(latency)


def _normalize_path(path: str) -> str:
    """Collapse dynamic path segments to reduce label cardinality."""
    import re
    # Replace UUID segments
    path = re.sub(
        r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "/{id}", path
    )
    # Replace numeric segments
    path = re.sub(r"/\d+", "/{id}", path)
    return path


def metrics_output() -> str:
    """Return Prometheus text-format metrics for the /metrics endpoint."""
    if _PROM_AVAILABLE:
        return generate_latest(REGISTRY).decode("utf-8")
    return (
        "# Prometheus client not installed.\n"
        "# pip install prometheus-client\n"
        "# sovereign_prom_available 0\n"
    )
