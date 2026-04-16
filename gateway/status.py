"""
gateway/status.py — System Status Router

Routes:
  GET /status/           — full system snapshot (JSON)
  GET /status/backends   — per-backend health + latency
  GET /status/stream     — SSE real-time health stream
  GET /status/kairos/{id} — agent detail (delegates to kairos_routes)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/status", tags=["status"])

_START_TIME = time.time()
_VERSION = "2.0.0"


@router.get("/", summary="Full system snapshot")
async def system_status(request: Request) -> Dict[str, Any]:
    """Return a complete system snapshot: backends, KAIROS, auction, metrics."""
    monitor = getattr(request.app.state, "health_monitor", None)
    benchmark = getattr(request.app.state, "benchmark", None)

    backends = []
    if monitor:
        from gateway.config import BACKENDS
        for b in BACKENDS:
            healthy = monitor.is_healthy(b.id)
            lat = getattr(monitor, "get_latency", lambda _: None)(b.id)
            backends.append({
                "name": b.id,
                "label": b.label,
                "healthy": healthy,
                "latency_ms": round(lat * 1000, 2) if lat else None,
                "meta": {
                    "url": b.url,
                    "device_type": b.device_type.value,
                    "vram_gib": b.vram_gib,
                    "weight": b.weight,
                },
            })

    # KAIROS summary
    kairos_summary: Dict[str, Any] = {}
    try:
        from pathlib import Path
        import os
        agents_dir = Path(os.getenv("KAIROS_AGENTS_DIR", "data/kairos/agents"))
        if agents_dir.exists():
            agent_files = [p for p in agents_dir.glob("*.json") if "_archive" not in p.name]
            kairos_summary = {
                "agent_count": len(agent_files),
                "elite_count": 0,
            }
            for p in agent_files:
                try:
                    import json as _j
                    with open(p) as f:
                        d = _j.load(f)
                    if d.get("tier") in ("elite", "next_elite"):
                        kairos_summary["elite_count"] += 1
                except Exception:
                    pass
    except Exception:
        pass

    # Auction summary
    auction_summary: Dict[str, Any] = {}
    try:
        from gateway.auction import auctioneer
        auction_summary = {"credit_pool": getattr(auctioneer, "_credits", "unknown")}
    except Exception:
        pass

    # WS stats
    ws_stats: Dict[str, Any] = {}
    try:
        from gateway.ws import event_bus
        ws_stats = event_bus.stats
    except Exception:
        pass

    return {
        "version": _VERSION,
        "timestamp": time.time(),
        "uptime_s": round(time.time() - _START_TIME, 2),
        "backends": backends,
        "healthy_count": sum(1 for b in backends if b["healthy"]),
        "total_backends": len(backends),
        "kairos": kairos_summary,
        "auction": auction_summary,
        "websocket": ws_stats,
    }


@router.get("/backends", summary="Per-backend health and latency")
async def backend_detail(request: Request) -> Dict[str, Any]:
    """Return detailed per-backend state including latency EMA and last error."""
    monitor = getattr(request.app.state, "health_monitor", None)
    if not monitor:
        return {"error": "Health monitor not initialized"}

    from gateway.config import BACKENDS
    result = {}
    for b in BACKENDS:
        healthy = monitor.is_healthy(b.id)
        state = getattr(monitor, "_states", {}).get(b.id, None)
        result[b.id] = {
            "label": b.label,
            "url": b.url,
            "device_type": b.device_type.value,
            "vram_gib": b.vram_gib,
            "healthy": healthy,
            "latency_ms": None,
            "consecutive_failures": 0,
            "consecutive_successes": 0,
            "last_error": None,
            "last_checked": None,
        }
        if state:
            lat = getattr(state, "last_latency_ms", None)
            result[b.id].update({
                "latency_ms": round(lat, 2) if lat else None,
                "consecutive_failures": getattr(state, "consecutive_failures", 0),
                "consecutive_successes": getattr(state, "consecutive_successes", 0),
                "last_error": getattr(state, "last_error", None),
                "last_checked": getattr(state, "last_checked", None),
                "status": getattr(state, "status", {}).value
                    if hasattr(getattr(state, "status", None), "value") else str(getattr(state, "status", "unknown")),
            })

    return result


@router.get("/stream", summary="SSE real-time health stream")
async def health_stream(request: Request) -> StreamingResponse:
    """
    Server-Sent Events stream of backend health.
    Emits a JSON snapshot every 2 seconds.
    Connect with: curl -N http://localhost:8000/status/stream
    """
    async def generator() -> AsyncGenerator[str, None]:
        while True:
            if await request.is_disconnected():
                break
            try:
                snapshot = await system_status(request)
                data = json.dumps(snapshot)
                yield f"data: {data}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/kairos/{agent_id}", summary="KAIROS agent detail")
async def kairos_agent_status(agent_id: str) -> Dict[str, Any]:
    """Proxy to kairos_routes agent detail."""
    from gateway.kairos_routes import get_agent
    return await get_agent(agent_id)
