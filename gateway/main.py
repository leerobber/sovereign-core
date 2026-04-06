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

from gateway.benchmark import ThroughputBenchmark
from gateway.config import BACKENDS, BACKEND_MAP, settings
from gateway.health import HealthMonitor
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

    status, resp_headers, resp_body = await _router.route(
        path=fwd_path,
        method=request.method,
        headers=forward_headers,
        body=body,
        model_id=model_id,
        vram_required_gib=vram_gib,
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
