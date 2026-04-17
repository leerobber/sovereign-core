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
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from gateway.auction import InsufficientCreditsError, ResourceType, auctioneer as _auctioneer
from gateway.benchmark import ThroughputBenchmark
from gateway.config import BACKENDS, BACKEND_MAP, settings
from gateway.health import HealthMonitor
from gateway.inference import InferenceRequest, InferenceResponse, route_inference
from gateway.kairos_routes import router as kairos_router
from gateway.metrics import INFERENCE_COUNTER, INFERENCE_LATENCY, ACTIVE_BACKENDS, record_request, metrics_output
from gateway.models import ModelAssigner
from gateway.router import GatewayRouter
from gateway.status import router as status_router
from gateway.v1_compat import router as v1_router
from gateway.ws import event_bus, router as ws_router
from gateway.db import get_db, log_event  # persistent SQLite layer

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
    app.include_router(ws_router)
    app.include_router(v1_router)   # ← OpenAI-compat: /v1/chat/completions + /v1/models

    # ── Core endpoints ────────────────────────────────────────────────────────

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

    @app.post("/auction/bid", tags=["auction"])
    async def auction_bid(resource_type: str, votes: int, agent_id: str = "anonymous") -> dict:
        try:
            rt = ResourceType(resource_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown resource: {resource_type}")
        try:
            result = _auctioneer.bid(agent_id=agent_id, resource=rt, votes=votes)
            await event_bus.emit("auction.completed", result)
            return result
        except InsufficientCreditsError as e:
            raise HTTPException(status_code=402, detail=str(e))

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

if __name__ == "__main__":
    uvicorn.run(
        "gateway.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        loop="asyncio",
        log_level="info",
    )
