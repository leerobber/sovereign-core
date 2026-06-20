"""
Sovereign Core — Heterogeneous Compute Gateway
===============================================
FastAPI application:
  - Routes inference across RTX 5050 | Radeon 780M | Ryzen 7
  - Exposes OpenAI-compatible /v1/chat/completions (for llm_local.py + SAGE)
  - KAIROS agent economy: /kairos/sage, /evolve, /leaderboard
  - Prometheus metrics: /metrics
  - SSE health stream: /status/stream
  - WebSocket event bus: /ws/events
  - Request tracing via X-Request-ID

Environment variables (prefix GATEWAY_):
  GATEWAY_HOST, GATEWAY_PORT, GATEWAY_API_KEY, GATEWAY_CORS_ORIGINS
  GATEWAY_HEALTH_CHECK_INTERVAL, GATEWAY_BACKEND_TIMEOUT
  GATEWAY_FAILURE_THRESHOLD, GATEWAY_RECOVERY_THRESHOLD
  GATEWAY_LATENCY_EMA_ALPHA, GATEWAY_LEDGER_HMAC_SECRET
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from pathlib import Path as _Path

from gateway.auction import InsufficientCreditsError, ResourceType, auctioneer as _auctioneer
from gateway.benchmark import ThroughputBenchmark
from gateway.config import BACKENDS, BACKEND_MAP, settings
from gateway.health import HealthMonitor
from gateway.inference import InferenceRequest, InferenceResponse, route_inference
from gateway.kairos_routes import router as kairos_router
from gateway.mem_evolve_routes import router as mem_evolve_router
from gateway.metrics import INFERENCE_COUNTER, INFERENCE_LATENCY, ACTIVE_BACKENDS, record_request, metrics_output
from gateway.models import ModelAssigner
from gateway.router import GatewayRouter
from gateway.status import router as status_router
from gateway.v1_compat import router as v1_router
from gateway.ws import event_bus, router as ws_router
from gateway.db import get_db, log_event  # persistent SQLite layer
from gateway.auth import AuthMiddleware    # API key auth + rate limiting
from gateway.iron_dome_middleware import iron_dome_guard  # injection screening

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_health_monitor: HealthMonitor
_benchmark: ThroughputBenchmark
_router: GatewayRouter
_boot_time: float = time.time()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _health_monitor, _benchmark, _router

    # ── Initialize persistent database ───────────────────────────────────
    try:
        _db = get_db()
        log_event("gateway_boot", "main", "Sovereign Core Gateway starting up",
                  metadata={"backends": len(BACKENDS), "port": settings.port})
        logger.info("Persistent database initialized")
    except Exception as _db_exc:
        logger.warning("Database init failed (non-fatal): %s", _db_exc)

    _health_monitor = HealthMonitor(cfg=settings)
    _benchmark = ThroughputBenchmark()
    _router = GatewayRouter(
        health_monitor=_health_monitor,
        assigner=ModelAssigner(),
        benchmark=_benchmark,
        cfg=settings,
    )

    # Attach to app state so all routers can access
    app.state.health_monitor = _health_monitor
    app.state.router = _router
    app.state.benchmark = _benchmark
    app.state.boot_time = _boot_time

    await _health_monitor.start()
    await _router.start()
    asyncio.create_task(event_bus.broadcast_loop(), name="ws-broadcast")

    healthy = sum(1 for b in BACKENDS if _health_monitor.is_healthy(b.id))
    ACTIVE_BACKENDS.set(healthy)

    logger.info(
        "Sovereign Core Gateway ready — %s:%d  |  %d backends  |  "
        "OpenAI compat: /v1/chat/completions  |  SAGE: /kairos/sage",
        settings.host, settings.port, len(BACKENDS),
    )

    yield

    await _health_monitor.stop()
    await _router.stop()
    logger.info("Gateway shutdown complete.")


def create_app() -> FastAPI:
    cors_origins = (
        [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
        if hasattr(settings, "cors_origins") and settings.cors_origins != "*"
        else ["*"]
    )

    app = FastAPI(
        title="Sovereign Core Gateway",
        description=(
            "Heterogeneous Compute Gateway — routes inference across RTX 5050, "
            "Radeon 780M, and Ryzen 7. OpenAI-compatible API. KAIROS agent economy."
        ),
        version="2.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request tracing + auth
    @app.middleware("http")
    async def request_tracing(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = time.time()

        # API key auth (optional)
        api_key = getattr(settings, "api_key", None)
        if api_key:
            public = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
            if request.url.path not in public:
                auth = request.headers.get("Authorization", "")
                if not auth.startswith("Bearer ") or auth[7:] != api_key:
                    return JSONResponse({"detail": "Unauthorized"}, status_code=401)

        response = await call_next(request)
        latency = time.time() - request.state.start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{latency * 1000:.2f}ms"
        record_request(request.method, request.url.path, response.status_code, latency)
        return response

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(status_router)
    app.include_router(kairos_router)
    app.include_router(mem_evolve_router)
    app.include_router(ws_router)
    app.include_router(v1_router)   # ← OpenAI-compat: /v1/chat/completions + /v1/models

    # ── Core endpoints ────────────────────────────────────────────────────────

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard() -> HTMLResponse:
        """Real-time command interface — live backend health, KAIROS, latency graphs."""
        html_path = _Path(__file__).parent / "dashboard.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(), status_code=200)
        return HTMLResponse(content="<h1>Dashboard file not found</h1>", status_code=404)

    @app.get("/health", tags=["core"])
    async def health() -> dict:
        healthy = sum(1 for b in BACKENDS if _health_monitor.is_healthy(b.id))
        ACTIVE_BACKENDS.set(healthy)
        return {
            "status": "healthy" if healthy > 0 else "degraded",
            "backends_healthy": healthy,
            "backends_total": len(BACKENDS),
            "uptime_s": round(time.time() - _boot_time, 2),
            "version": "2.1.0",
        }

    @app.get("/v1/repos", tags=["mesh"])
    async def mesh_repos() -> dict:
        """Probe sovereign ecosystem repos (GH05T3 dual-runtime aware)."""
        try:
            from src.orchestration.registry import RepoRegistry
        except ImportError:
            from orchestration.registry import RepoRegistry
        reg = RepoRegistry()
        statuses = await reg.probe_all()
        return {
            "runtime": __import__("os").environ.get("GH05T3_RUNTIME", "wsl"),
            "repos": [
                {
                    "name": s.name,
                    "role": s.role,
                    "healthy": s.healthy,
                    "detail": s.detail,
                    "ports": s.ports,
                }
                for s in statuses
            ],
        }

    @app.get("/metrics", tags=["core"])
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(
            content=metrics_output(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/inference", response_model=InferenceResponse, tags=["inference"])
    async def inference(req: InferenceRequest, request: Request) -> InferenceResponse:
        """Route inference via Ollama format. Use /v1/chat/completions for OpenAI format."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        t0 = time.time()
        try:
            result = await route_inference(req=req, router=_router, request_id=request_id)
            latency = time.time() - t0
            INFERENCE_COUNTER.labels(backend=result.backend_id, model=result.model, status="success").inc()
            INFERENCE_LATENCY.labels(backend=result.backend_id).observe(latency)
            await event_bus.emit("inference.completed", {
                "backend_id": result.backend_id,
                "model": result.model,
                "latency_ms": round(latency * 1000, 2),
                "tokens": result.eval_count,
                "request_id": request_id,
            })
            return result
        except Exception as exc:
            INFERENCE_COUNTER.labels(backend="unknown", model=req.model, status="error").inc()
            logger.error("Inference error [%s]: %s", request_id, exc)
            raise HTTPException(status_code=503, detail=str(exc))

    @app.get("/ledger/tail", tags=["ledger"])
    async def ledger_tail(n: int = 20) -> dict:
        try:
            from gateway.ledger import SemanticLedger
            ledger = SemanticLedger.instance()
            return {"entries": ledger.tail(n), "total": ledger.count()}
        except Exception as exc:
            return {"entries": [], "total": 0, "error": str(exc)}

    @app.post("/benchmark/run", tags=["benchmark"])
    async def benchmark_run(model_id: str = "default") -> dict:
        try:
            return await _benchmark.run(model_id=model_id, router=_router)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return app


app = create_app()


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

    # Derive an auction-based routing hint: prefer the backend that most recently
    # won a settled auction so that allocation decisions influence request dispatch.
    priority_map = _auctioneer.allocation_priority()
    priority_backend_id: Optional[str] = next(iter(priority_map), None)

    status, resp_headers, resp_body = await _router.route(
        path=fwd_path,
        method=request.method,
        headers=forward_headers,
        body=body,
        model_id=model_id,
        vram_required_gib=vram_gib,
        priority_backend_id=priority_backend_id,
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
# Entrypoint
# ---------------------------------------------------------------------------
def run() -> None:  # pragma: no cover
    uvicorn.run(
        "gateway.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        loop="asyncio",
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    run()