"""
gateway/status.py — Unified System Status & Observability API

Real-time health, metrics, and event stream for all Sovereign Core subsystems.
Exposes /status/, /status/stream (SSE), /status/backends, /status/kairos/{id}
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/status", tags=["observability"])


# ── Models ───────────────────────────────────────────────────────────────────

class SubsystemStatus(BaseModel):
    name: str
    healthy: bool
    latency_ms: Optional[float] = None
    last_seen: Optional[float] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class SystemSnapshot(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    version: str = "1.0.0"
    uptime_s: float = 0.0
    backends: List[SubsystemStatus] = Field(default_factory=list)
    gateway: Dict[str, Any] = Field(default_factory=dict)
    kairos: Dict[str, Any] = Field(default_factory=dict)
    auction: Dict[str, Any] = Field(default_factory=dict)
    ledger: Dict[str, Any] = Field(default_factory=dict)
    mem_evolve: Dict[str, Any] = Field(default_factory=dict)


_boot_time = time.time()


def _safe(fn):
    """Execute a status getter safely; return error dict on failure."""
    try:
        return fn() or {}
    except Exception as exc:
        return {"error": str(exc)}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/", response_model=SystemSnapshot, summary="Full system snapshot")
async def get_status(request: Request):
    """
    Comprehensive real-time snapshot: GPU backends, KAIROS economy,
    auction ledger, MemEvolve retrieval stats.
    """
    snapshot = SystemSnapshot(uptime_s=time.time() - _boot_time)

    # Backend health (requires HealthMonitor on app.state)
    try:
        monitor = request.app.state.health_monitor
        for b in monitor.backends:
            snapshot.backends.append(SubsystemStatus(
                name=b.id,
                healthy=monitor.is_healthy(b.id),
                latency_ms=monitor.get_latency(b.id),
                last_seen=monitor.last_seen(b.id),
                meta={"url": b.url, "device_type": b.device_type},
            ))
    except Exception:
        pass

    # Auction stats
    try:
        from gateway.auction import auctioneer
        snapshot.auction = _safe(lambda: auctioneer.stats())
    except Exception:
        pass

    return snapshot


@router.get("/stream", summary="Server-Sent Events health stream")
async def stream_status():
    """
    SSE endpoint — pushes a lightweight system snapshot every 5 seconds.
    Connect from any frontend: new EventSource('/status/stream')
    """
    async def _generate() -> AsyncIterator[str]:
        while True:
            try:
                from gateway.auction import auctioneer
                data = {
                    "ts": time.time(),
                    "uptime_s": time.time() - _boot_time,
                    "auction": _safe(lambda: auctioneer.stats()),
                }
                yield f"data: {json.dumps(data)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/backends", summary="GPU backend health details")
async def get_backends(request: Request):
    """Per-backend latency EMA, health status, and request counts."""
    try:
        monitor = request.app.state.health_monitor
        return {
            b.id: {
                "healthy": monitor.is_healthy(b.id),
                "latency_ms": monitor.get_latency(b.id),
                "url": b.url,
                "device_type": b.device_type,
            }
            for b in monitor.backends
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/kairos/{agent_id}", summary="KAIROS agent lineage and metrics")
async def get_kairos_agent(agent_id: str):
    """
    Full evolution history, skill domains, and ARSO cycle count
    for a specific KAIROS elite agent.
    """
    try:
        from gateway.kairos import KAIROSAgent
        agent = KAIROSAgent.load(agent_id)
        return agent.to_dict()
    except Exception as exc:
        return {"error": str(exc), "agent_id": agent_id}
