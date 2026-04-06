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
from gateway.health import HealthMonitor
from gateway.kairos import (
    AgentTier,
    KAIROSAgent,
    KAIROSEvolutionEngine,
    SkillDomain,
    EliteRegistry,
    elite_registry as _elite_registry,
    _kairos_engine,
)
from gateway.models import ModelAssigner
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



@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _health_monitor, _benchmark, _router

    _health_monitor = HealthMonitor(cfg=settings)
    _benchmark = ThroughputBenchmark()
    _router = GatewayRouter(
        health_monitor=_health_monitor,
        assigner=ModelAssigner(),
        benchmark=_benchmark,
        cfg=settings,
    )

    await _health_monitor.start()
    await _router.start()
    logger.info("Gateway ready on %s:%d", settings.host, settings.port)

    yield  # application serves requests here

    await _router.stop()
    await _health_monitor.stop()
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
# KAIROS endpoints
# ---------------------------------------------------------------------------


def _agent_to_dict(agent: KAIROSAgent) -> dict[str, Any]:
    """Convert a KAIROSAgent to a JSON-serialisable dict."""
    return {
        "agent_id": agent.agent_id,
        "generation": agent.generation,
        "tier": agent.tier.value,
        "optimizations_successful": agent.optimizations_successful,
        "auction_wins": agent.auction_wins,
        "auction_participations": agent.auction_participations,
        "ancestor_id": agent.ancestor_id,
        "skill_domains": [d.value for d in agent.skill_domains],
        "retrieval_weights": {
            "recency": agent.retrieval_weights.recency,
            "relevance": agent.retrieval_weights.relevance,
            "frequency": agent.retrieval_weights.frequency,
        },
        "skill_transfer_successes": agent.skill_transfer_successes,
        "created_at": agent.created_at,
        "last_evolved_at": agent.last_evolved_at,
        "fitness_score": agent.fitness_score,
        "auction_win_rate": agent.auction_win_rate,
        "optimization_rate": agent.optimization_rate,
    }


@app.get("/kairos/elites", summary="List Elite and nextElite agents")
async def kairos_list_elites() -> JSONResponse:
    """Return all Elite and nextElite agents sorted by fitness."""
    agents = _elite_registry.list_elites()
    return JSONResponse({"agents": [_agent_to_dict(a) for a in agents]})


@app.get("/kairos/agent/{agent_id}", summary="Get KAIROS agent details")
async def kairos_get_agent(agent_id: str) -> JSONResponse:
    """Return details of a specific KAIROS agent. 404 if not found."""
    try:
        agent = _elite_registry.get(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(_agent_to_dict(agent))


@app.post("/kairos/evolve/{agent_id}", summary="Trigger evolution cycle")
async def kairos_evolve(agent_id: str) -> JSONResponse:
    """Run one evolution cycle on the specified agent. 404 if not found."""
    try:
        evolved = _elite_registry.promote(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(_agent_to_dict(evolved))


@app.post("/kairos/reconstruct/{ancestor_id}", summary="Reconstruct agent from archive")
async def kairos_reconstruct(ancestor_id: str) -> JSONResponse:
    """Rebuild a new agent from an archived ancestor. 404 if ancestor not in archive."""
    try:
        new_agent = _kairos_engine.reconstruct_from_archive(ancestor_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _elite_registry.register(new_agent)
    return JSONResponse(_agent_to_dict(new_agent))


@app.get("/kairos/metrics", summary="KAIROS population statistics")
async def kairos_metrics() -> JSONResponse:
    """Return population stats: total, elite_count, next_elite_count, avg_fitness."""
    return JSONResponse(_elite_registry.metrics())


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