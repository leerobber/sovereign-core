"""Sovereign Core — Heterogeneous Compute Gateway.

FastAPI application that:
- Accepts inference requests on ``localhost:8000``
- Routes them to the best available backend (RTX 5050 | Radeon 780M | Ryzen 7)
- Exposes health, metrics, and benchmark endpoints
- Manages the background HealthMonitor lifecycle

Environment variables (prefix ``GATEWAY_``)
───────────────────────────────────────────
GATEWAY_HOST                 Bind host (default: 0.0.0.0)
GATEWAY_PORT                 Bind port (default: 8000)
GATEWAY_HEALTH_CHECK_INTERVAL  Seconds between health probes (default: 5)
GATEWAY_BACKEND_TIMEOUT        Per-request forwarding timeout (default: 30)
GATEWAY_FAILURE_THRESHOLD      Failures before marking backend unhealthy (default: 3)
GATEWAY_RECOVERY_THRESHOLD     Successes needed to restore a backend (default: 2)
GATEWAY_LATENCY_EMA_ALPHA      EMA smoothing factor (default: 0.2)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from gateway.auction import (
    Auctioneer,
    InsufficientCreditsError,
    ResourceType,
    auctioneer as _auctioneer,
)
from gateway.benchmark import ThroughputBenchmark
from gateway.config import BACKENDS, BACKEND_MAP, settings
from gateway.context import AgentRole, SharedContextLayer, init_context_layer
from gateway.health import HealthMonitor
from gateway.mem_evolve import ABTestManager, MemEvolveEngine
from gateway.models import ModelAssigner
from gateway.pattern_memory import PatternRecord, PatternStore
from gateway.router import GatewayRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
_health_monitor: HealthMonitor
_benchmark: ThroughputBenchmark
_router: GatewayRouter
_pattern_store: PatternStore
_mem_evolve: MemEvolveEngine
_ab_test: ABTestManager
_context: SharedContextLayer


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _health_monitor, _benchmark, _router, _pattern_store, _mem_evolve, _ab_test, _context

    _health_monitor = HealthMonitor(cfg=settings)
    _benchmark = ThroughputBenchmark()
    _router = GatewayRouter(
        health_monitor=_health_monitor,
        assigner=ModelAssigner(),
        benchmark=_benchmark,
        cfg=settings,
    )
    _context = init_context_layer()

    _pattern_store = PatternStore()
    _mem_evolve = MemEvolveEngine(_pattern_store)
    _ab_test = ABTestManager(_mem_evolve)

    await _health_monitor.start()
    await _router.start()
    logger.info("Gateway ready on %s:%d", settings.host, settings.port)

    yield  # application serves requests here

    await _router.stop()
    await _health_monitor.stop()
    _pattern_store.close()
    logger.info("Gateway shut down cleanly")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sovereign Core — Compute Gateway",
    description=(
        "Heterogeneous tri-GPU inference mesh gateway.\n\n"
        "Backends:\n"
        "- RTX 5050 → localhost:8001 (primary GPU)\n"
        "- Radeon 780M → localhost:8002 (secondary GPU)\n"
        "- Ryzen 7 CPU → localhost:8003 (CPU fallback)\n"
    ),
    version="0.1.0",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Health & diagnostics endpoints
# ---------------------------------------------------------------------------
@app.get("/health", summary="Gateway health summary")
async def gateway_health() -> JSONResponse:
    """Return health status of the gateway and all backends."""
    backend_statuses = _health_monitor.status_report()
    healthy_count = sum(1 for s in backend_statuses if s["status"] == "healthy")
    return JSONResponse(
        {
            "status": "ok" if healthy_count > 0 else "degraded",
            "healthy_backends": healthy_count,
            "total_backends": len(backend_statuses),
            "backends": backend_statuses,
        }
    )


@app.get("/metrics", summary="Latency EMA per backend")
async def metrics() -> JSONResponse:
    """Return current EMA latency readings and throughput metrics."""
    latencies = _router.latency_snapshot()
    return JSONResponse(
        {
            "latency_ema_s": latencies,
            "benchmark": _benchmark.report(),
        }
    )


@app.get("/benchmark", summary="Full throughput benchmark report")
async def benchmark_report() -> JSONResponse:
    """Return detailed per-backend throughput statistics."""
    return JSONResponse({"benchmark": _benchmark.report()})


@app.post("/benchmark/reset", summary="Reset benchmark counters")
async def benchmark_reset(
    backend_id: Optional[str] = Query(default=None, description="Reset a specific backend only"),
) -> JSONResponse:
    """Clear accumulated benchmark data (optionally for a single backend)."""
    if backend_id is not None and backend_id not in BACKEND_MAP:
        raise HTTPException(status_code=404, detail=f"Unknown backend: {backend_id}")
    _benchmark.reset(backend_id)
    return JSONResponse({"reset": True, "backend_id": backend_id})


# ---------------------------------------------------------------------------
# Inference proxy endpoint
# ---------------------------------------------------------------------------
@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    summary="Proxy inference request to best available backend",
)
async def proxy_inference(
    request: Request,
    path: str,
    model_id: Optional[str] = Query(default=None, description="Target model identifier"),
    vram_gib: float = Query(default=0.0, ge=0.0, description="Minimum VRAM required (GiB)"),
) -> Response:
    """Forward the request to the best available backend.

    Query parameters ``model_id`` and ``vram_gib`` inform the routing decision.
    All other query parameters and the request body are forwarded verbatim.
    """
    body = await request.body()
    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length"}
    }

    # Preserve original query string (minus our routing hints) on the forwarded URL
    fwd_path = f"/{path}"

    # Obtain auction-derived routing priority so settled auctions influence
    # which backend is selected for this request.
    auction_priority = _auctioneer.allocation_priority()

    status, resp_headers, resp_body = await _router.route(
        path=fwd_path,
        method=request.method,
        headers=forward_headers,
        body=body,
        model_id=model_id,
        vram_required_gib=vram_gib,
        auction_priority=auction_priority,
    )

    # Strip hop-by-hop headers before returning
    skip_headers = {"transfer-encoding", "connection", "keep-alive", "content-encoding"}
    clean_headers = {k: v for k, v in resp_headers.items() if k.lower() not in skip_headers}

    return Response(
        content=resp_body,
        status_code=status,
        headers=clean_headers,
    )


# ---------------------------------------------------------------------------
# Auction endpoints
# ---------------------------------------------------------------------------


@app.post("/auction/credits", summary="Top up agent credits")
async def auction_top_up(
    agent_id: str = Query(..., description="Agent identifier"),
    amount: int = Query(..., ge=1, description="Credits to add"),
) -> JSONResponse:
    """Register (or top up) an agent with the given credit amount."""
    try:
        result = _auctioneer.top_up(agent_id, amount)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(result)


@app.post("/auction/bid", summary="Place a quadratic bid")
async def auction_bid(
    agent_id: str = Query(..., description="Bidding agent identifier"),
    resource_type: ResourceType = Query(..., description="Resource category"),
    backend_id: str = Query(..., description="Target compute backend ID"),
    votes: int = Query(..., ge=1, description="Quadratic votes to cast (cost = votes²)"),
) -> JSONResponse:
    """Submit a bid for a resource slot on the specified backend.

    The credit cost of *votes* votes is ``votes²``.  Credits are reserved
    at bid time and charged to the winner at settlement.
    """
    if backend_id not in BACKEND_MAP:
        raise HTTPException(status_code=404, detail=f"Unknown backend: {backend_id!r}")
    try:
        auction_id, bid = _auctioneer.place_bid(agent_id, resource_type, backend_id, votes)
    except InsufficientCreditsError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        {
            "auction_id": auction_id,
            "agent_id": bid.agent_id,
            "resource_type": bid.resource_type.value,
            "backend_id": bid.backend_id,
            "votes": bid.votes,
            "credit_cost_if_winner": bid.credit_cost,
        }
    )


@app.get("/auction/status", summary="Current auction state")
async def auction_status() -> JSONResponse:
    """Return all open auctions and per-agent credit balances."""
    return JSONResponse(_auctioneer.status())


@app.post("/auction/settle", summary="Settle one or all open auctions")
async def auction_settle(
    auction_id: Optional[str] = Query(
        default=None, description="Settle a specific auction; omit to settle all"
    ),
) -> JSONResponse:
    """Settle the specified auction (or all open auctions).

    The Vickrey second-price rule is applied: the highest-vote bidder wins
    but pays only ``second_highest_votes²`` credits.
    """
    try:
        if auction_id is not None:
            result = _auctioneer.settle_auction(auction_id)
            results = [result]
        else:
            results = _auctioneer.settle_all()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return JSONResponse(
        {
            "settled": [
                {
                    "auction_id": r.auction_id,
                    "resource_type": r.resource_type.value,
                    "backend_id": r.backend_id,
                    "winner_agent_id": r.winner_agent_id,
                    "winning_votes": r.winning_votes,
                    "payment_credits": r.payment_credits,
                    "bid_count": len(r.all_bids),
                }
                for r in results
            ]
        }
    )


@app.get("/auction/metrics", summary="Allocation fairness metrics")
async def auction_metrics() -> JSONResponse:
    """Return Gini coefficient and utilization statistics for all settled auctions."""
    m = _auctioneer.metrics()
    return JSONResponse(
        {
            "total_auctions": m.total_auctions,
            "total_credits_spent": m.total_credits_spent,
            "gini_coefficient": m.gini_coefficient,
            "unique_winners": m.unique_winners,
            "unique_winners_ratio": m.unique_winners_ratio,
            "utilization_by_backend": m.utilization_by_backend,
            "utilization_by_resource": m.utilization_by_resource,
        }
    )


# ---------------------------------------------------------------------------
# Context endpoints
# ---------------------------------------------------------------------------


@app.post("/context/write", summary="Write an agent context entry")
async def context_write(
    role: AgentRole = Query(..., description="Cognitive role of the writing agent"),
    backend_id: str = Query(..., description="Compute backend that produced this result"),
    document: str = Query(..., description="Free-text conclusion or summary from the agent"),
    trace_id: Optional[str] = Query(default=None, description="Optional request trace ID"),
) -> JSONResponse:
    """Persist an agent conclusion to the shared ChromaDB context layer.

    Any agent (generator, verifier, safety, reasoner, planner) writes its
    output here so that all other agents can read it on subsequent calls.
    """
    entry_id = _context.write(
        role=role,
        backend_id=backend_id,
        document=document,
        trace_id=trace_id,
    )
    return JSONResponse(
        {
            "entry_id": entry_id,
            "role": role.value,
            "backend_id": backend_id,
            "trace_id": trace_id,
        }
    )


@app.get("/context/read", summary="Read context entries")
async def context_read(
    role: Optional[AgentRole] = Query(default=None, description="Filter by agent role"),
    backend_id: Optional[str] = Query(default=None, description="Filter by backend ID"),
    trace_id: Optional[str] = Query(default=None, description="Filter by trace ID"),
    limit: int = Query(default=20, ge=1, le=500, description="Maximum entries to return"),
) -> JSONResponse:
    """Read context entries with optional filtering.

    Filters are applied in priority order: ``trace_id`` → ``role`` →
    ``backend_id`` → all entries.  At most one filter is applied per call.
    """
    if trace_id is not None:
        entries = _context.read_by_trace(trace_id)
    elif role is not None:
        entries = _context.read_by_role(role, limit=limit)
    elif backend_id is not None:
        entries = _context.read_by_backend(backend_id, limit=limit)
    else:
        entries = _context.read_all(limit=limit)

    return JSONResponse(
        {
            "entries": [e.as_dict() for e in entries],
            "count": len(entries),
        }
    )


@app.get(
    "/context/cross-gpu/{backend_id}",
    summary="Cross-GPU context visibility for a backend",
)
async def context_cross_gpu(
    backend_id: str,
    limit: int = Query(default=20, ge=1, le=500, description="Maximum peer entries to return"),
) -> JSONResponse:
    """Return context produced by *peer* backends for cross-GPU visibility.

    The Radeon 780M calls this to see what the RTX 5050 concluded, and vice
    versa.  Entries produced by the requesting backend itself are excluded.
    """
    entries = _context.read_cross_gpu(backend_id, limit=limit)
    return JSONResponse(
        {
            "backend_id": backend_id,
            "peer_entries": [e.as_dict() for e in entries],
            "count": len(entries),
        }
    )


@app.get("/context/count", summary="Total context entry count")
async def context_count() -> JSONResponse:
    """Return the total number of entries in the shared context store."""
    return JSONResponse({"count": _context.count()})


@app.delete("/context/clear", summary="Clear all context entries")
async def context_clear() -> JSONResponse:
    """Remove all entries from the shared context store.

    Use with care — this deletes all cross-agent memory accumulated since
    the last clear or gateway restart.
    """
    removed = _context.clear()
    return JSONResponse({"cleared": removed})


# ---------------------------------------------------------------------------
# Pattern Memory endpoints (RES-12 MemEvolve)
# ---------------------------------------------------------------------------


@app.post("/memory/patterns", summary="Store an optimization pattern")
async def memory_store_pattern(
    model_id: str = Query(..., description="Model identifier"),
    backend_id: str = Query(..., description="Backend identifier"),
    pattern_type: str = Query(..., description="Pattern category (e.g. latency, routing)"),
    context: Optional[str] = Query(default=None, description="JSON context object"),
    recommendation: Optional[str] = Query(default=None, description="JSON recommendation object"),
) -> JSONResponse:
    """Persist an optimization pattern to Pattern Memory."""
    import json as _json

    ctx: dict[str, Any] = {}
    rec: dict[str, Any] = {}
    if context:
        try:
            ctx = _json.loads(context)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid context JSON: {exc}") from exc
    if recommendation:
        try:
            rec = _json.loads(recommendation)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid recommendation JSON: {exc}") from exc

    try:
        record = PatternRecord(
            model_id=model_id,
            backend_id=backend_id,
            pattern_type=pattern_type,
            context=ctx,
            recommendation=rec,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    stored = _pattern_store.store(record)
    return JSONResponse(stored.to_dict(), status_code=201)


@app.get("/memory/patterns", summary="Lookup optimization patterns")
async def memory_lookup_patterns(
    model_id: Optional[str] = Query(default=None, description="Filter by model"),
    backend_id: Optional[str] = Query(default=None, description="Filter by backend"),
    pattern_type: Optional[str] = Query(default=None, description="Filter by category"),
    limit: int = Query(default=10, ge=1, le=100, description="Max results"),
    strategy: str = Query(default="evolved", description="Retrieval strategy: evolved or static"),
    context: Optional[str] = Query(default=None, description="JSON query context for context-match scoring"),
) -> JSONResponse:
    """Retrieve and rank optimization patterns using evolved or static retrieval."""
    import json as _json

    if strategy not in ("evolved", "static"):
        raise HTTPException(status_code=422, detail="strategy must be 'evolved' or 'static'")

    query_ctx: Optional[dict[str, Any]] = None
    if context:
        try:
            query_ctx = _json.loads(context)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid context JSON: {exc}") from exc

    patterns = _pattern_store.lookup(
        model_id=model_id,
        backend_id=backend_id,
        pattern_type=pattern_type,
        limit=limit,
    )
    ranked = _mem_evolve.rank_patterns(patterns, strategy=strategy, query_context=query_ctx)
    return JSONResponse(
        {
            "strategy": strategy,
            "count": len(ranked),
            "patterns": [p.to_dict() for p in ranked],
        }
    )


@app.post("/memory/outcome", summary="Record a pattern lookup outcome")
async def memory_record_outcome(
    pattern_id: str = Query(..., description="Pattern that was applied"),
    success: bool = Query(..., description="Whether the optimization succeeded"),
    latency_s: float = Query(default=0.0, ge=0.0, description="Observed latency in seconds"),
    request_id: Optional[str] = Query(default=None, description="A/B test request ID"),
) -> JSONResponse:
    """Record the success or failure of an applied optimization pattern.

    If a *request_id* is provided the result is also forwarded to the A/B test
    manager so evolved vs. static hit rates are tracked.
    """
    if _pattern_store.get_pattern(pattern_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown pattern_id: {pattern_id!r}")

    outcome = _pattern_store.record_outcome(
        pattern_id, success=success, latency_s=latency_s
    )

    variant: Optional[str] = None
    if request_id is not None:
        variant = _ab_test.record_result(request_id, success=success)

    return JSONResponse(
        {
            "outcome_id": outcome.outcome_id,
            "pattern_id": outcome.pattern_id,
            "success": outcome.success,
            "latency_s": outcome.latency_s,
            "ab_variant": variant,
        }
    )


@app.get("/memory/stats", summary="Pattern Memory statistics")
async def memory_stats() -> JSONResponse:
    """Return aggregated statistics about the Pattern Memory store."""
    return JSONResponse(_pattern_store.get_stats().to_dict())


@app.post("/memory/evolve", summary="Run one MemEvolve meta-evolution step")
async def memory_evolve() -> JSONResponse:
    """Trigger one meta-evolution step to update retrieval weights.

    The engine reads outcome data accumulated since the last evolution and
    adjusts weights to amplify dimensions correlated with successful lookups.
    """
    result = _mem_evolve.evolve()
    return JSONResponse(result)


@app.get("/memory/evolve/status", summary="Current MemEvolve strategy status")
async def memory_evolve_status() -> JSONResponse:
    """Return the current state of both retrieval strategies."""
    return JSONResponse(
        {
            "static": _mem_evolve.static_strategy.to_dict(),
            "evolved": _mem_evolve.evolved_strategy.to_dict(),
        }
    )


@app.get("/memory/ab-test", summary="A/B test comparison: evolved vs. static retrieval")
async def memory_ab_test() -> JSONResponse:
    """Return comparative hit-rate statistics for evolved and static retrieval."""
    return JSONResponse(_ab_test.comparison())


@app.post("/memory/ab-test/assign", summary="Assign a request to an A/B variant")
async def memory_ab_assign(
    request_id: str = Query(..., description="Unique request identifier"),
) -> JSONResponse:
    """Return the A/B variant assigned to *request_id* (deterministic)."""
    variant = _ab_test.assign(request_id)
    return JSONResponse({"request_id": request_id, "variant": variant})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def run() -> None:  # pragma: no cover
    uvicorn.run(
        "gateway.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    run()
