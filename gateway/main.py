"""
Sovereign Core — Heterogeneous Compute Gateway
===============================================
FastAPI application that:
  - Routes inference requests across RTX 5050 | Radeon 780M | Ryzen 7
  - Exposes KAIROS agent economy HTTP + WebSocket APIs
  - Emits real-time events via the WebSocket event bus
  - Provides Prometheus-compatible /metrics endpoint
  - Traces every request with a unique X-Request-ID header

Environment variables (prefix GATEWAY_)
────────────────────────────────────────
GATEWAY_HOST                    Bind host (default: 0.0.0.0)
GATEWAY_PORT                    Bind port (default: 8000)
GATEWAY_HEALTH_CHECK_INTERVAL   Seconds between health probes (default: 5)
GATEWAY_BACKEND_TIMEOUT         Per-request forwarding timeout (default: 30)
GATEWAY_FAILURE_THRESHOLD       Failures before unhealthy (default: 3)
GATEWAY_RECOVERY_THRESHOLD      Successes needed to recover (default: 2)
GATEWAY_LATENCY_EMA_ALPHA       EMA smoothing factor (default: 0.2)
GATEWAY_CORS_ORIGINS            Comma-separated allowed origins (default: *)
GATEWAY_API_KEY                 Optional bearer token to protect the API
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from gateway.auction import (
    InsufficientCreditsError,
    ResourceType,
    auctioneer as _auctioneer,
)
from gateway.benchmark import ThroughputBenchmark
from gateway.config import BACKENDS, BACKEND_MAP, settings
from gateway.health import HealthMonitor
from gateway.inference import InferenceRequest, InferenceResponse, route_inference
from gateway.kairos_routes import router as kairos_router
from gateway.metrics import (
    INFERENCE_COUNTER,
    INFERENCE_LATENCY,
    ACTIVE_BACKENDS,
    record_request,
    metrics_output,
)
from gateway.models import ModelAssigner
from gateway.router import GatewayRouter
from gateway.status import router as status_router
from gateway.ws import event_bus, router as ws_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Application state ────────────────────────────────────────────────────────
_health_monitor: HealthMonitor
_benchmark: ThroughputBenchmark
_router: GatewayRouter
_boot_time: float = time.time()


# ── Lifespan ─────────────────────────────────────────────────────────────────

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

    # Attach to app state so routers can access them via request.app.state
    app.state.health_monitor = _health_monitor
    app.state.router = _router
    app.state.benchmark = _benchmark
    app.state.boot_time = _boot_time

    # Start background tasks
    await _health_monitor.start()
    await _router.start()
    asyncio.create_task(event_bus.broadcast_loop(), name="ws-broadcast")

    # Signal initial backend count to metrics
    healthy = sum(1 for b in BACKENDS if _health_monitor.is_healthy(b.id))
    ACTIVE_BACKENDS.set(healthy)

    logger.info(
        "Sovereign Core Gateway ready — %s:%d  |  %d backends registered",
        settings.host, settings.port, len(BACKENDS),
    )

    yield  # ── serve requests ──

    # Graceful shutdown
    await _health_monitor.stop()
    await _router.stop()
    logger.info("Gateway shutdown complete.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    cors_origins = [
        o.strip() for o in settings.cors_origins.split(",") if o.strip()
    ] if hasattr(settings, "cors_origins") and settings.cors_origins != "*" else ["*"]

    app = FastAPI(
        title="Sovereign Core Gateway",
        description=(
            "Heterogeneous Compute Gateway — routes inference across RTX 5050, "
            "Radeon 780M, and Ryzen 7 with KAIROS agent economy and Aegis-Vault ledger."
        ),
        version="2.0.0",
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

    # Request tracing middleware
    @app.middleware("http")
    async def request_tracing(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = time.time()

        # Optional API key check
        if hasattr(settings, "api_key") and settings.api_key:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != settings.api_key:
                # Allow health + metrics without auth
                if request.url.path not in ("/health", "/metrics", "/docs", "/redoc", "/openapi.json"):
                    return JSONResponse({"detail": "Unauthorized"}, status_code=401)

        response = await call_next(request)
        latency = time.time() - request.state.start_time

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{latency * 1000:.2f}ms"

        record_request(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency=latency,
        )
        return response

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(status_router)
    app.include_router(kairos_router)
    app.include_router(ws_router)

    # ── Core endpoints ────────────────────────────────────────────────────────

    @app.get("/health", tags=["core"], summary="Liveness probe")
    async def health() -> dict:
        healthy_count = sum(
            1 for b in BACKENDS
            if _health_monitor.is_healthy(b.id)
        )
        ACTIVE_BACKENDS.set(healthy_count)
        return {
            "status": "healthy" if healthy_count > 0 else "degraded",
            "backends_healthy": healthy_count,
            "backends_total": len(BACKENDS),
            "uptime_s": round(time.time() - _boot_time, 2),
        }

    @app.get("/metrics", tags=["core"], summary="Prometheus metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(
            content=metrics_output(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post(
        "/inference",
        response_model=InferenceResponse,
        tags=["inference"],
        summary="Route an inference request across the GPU mesh",
    )
    async def inference(req: InferenceRequest, request: Request) -> InferenceResponse:
        """
        Submit a text generation request. The gateway selects the optimal
        backend based on model affinity, health status, and latency EMA.

        Priority: RTX 5050 → Radeon 780M → Ryzen 7 CPU
        """
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        t0 = time.time()

        try:
            result = await route_inference(
                req=req,
                router=_router,
                request_id=request_id,
            )
            latency = time.time() - t0
            INFERENCE_COUNTER.labels(
                backend=result.backend_id,
                model=result.model,
                status="success",
            ).inc()
            INFERENCE_LATENCY.labels(backend=result.backend_id).observe(latency)

            # Emit real-time event
            await event_bus.emit("inference.completed", {
                "backend_id": result.backend_id,
                "model": result.model,
                "latency_ms": round(latency * 1000, 2),
                "tokens": result.eval_count,
                "request_id": request_id,
            })
            return result

        except Exception as exc:
            INFERENCE_COUNTER.labels(
                backend="unknown", model=req.model, status="error"
            ).inc()
            logger.error("Inference error [%s]: %s", request_id, exc)
            raise HTTPException(status_code=503, detail=str(exc))

    @app.post("/auction/bid", tags=["auction"], summary="Bid on a compute resource")
    async def auction_bid(
        resource_type: str,
        votes: int,
        agent_id: str = "anonymous",
        request: Request = None,
    ) -> dict:
        """
        Submit a Vickrey-Quadratic auction bid for a compute resource.
        Cost = votes². Winner pays second_highest_votes² credits.
        """
        try:
            rt = ResourceType(resource_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown resource type: {resource_type}. "
                       f"Valid: {[e.value for e in ResourceType]}",
            )
        try:
            result = _auctioneer.bid(agent_id=agent_id, resource=rt, votes=votes)
            await event_bus.emit("auction.completed", result)
            return result
        except InsufficientCreditsError as e:
            raise HTTPException(status_code=402, detail=str(e))

    @app.get("/ledger/tail", tags=["ledger"], summary="Tail Aegis-Vault ledger entries")
    async def ledger_tail(n: int = 20) -> dict:
        """Return the N most recent Aegis-Vault ledger entries."""
        try:
            from gateway.ledger import SemanticLedger
            ledger = SemanticLedger.instance()
            return {"entries": ledger.tail(n), "total": ledger.count()}
        except Exception as exc:
            return {"entries": [], "total": 0, "error": str(exc)}

    @app.post("/benchmark/run", tags=["benchmark"], summary="Run throughput benchmark")
    async def benchmark_run(model_id: str = "default") -> dict:
        """Run a throughput benchmark against all healthy backends."""
        try:
            results = await _benchmark.run(model_id=model_id, router=_router)
            return results
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return app


app = create_app()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "gateway.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        loop="asyncio",
        access_log=True,
        log_level="info",
    )
