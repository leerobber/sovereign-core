"""
gateway/ws.py — WebSocket Event Bus

Real-time bidirectional events: auction updates, backend health changes,
KAIROS evolution milestones, ledger entries.

Usage (from anywhere in the codebase):
    from gateway.ws import event_bus
    await event_bus.emit("auction.completed", {"winner": agent_id, "cost": 42})
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])


# ── Event Bus ─────────────────────────────────────────────────────────────────

class EventBus:
    """
    Lightweight in-process pub/sub.
    Producers call await emit(); all connected WebSocket clients receive it.
    """

    def __init__(self):
        self._clients: Set[WebSocket] = set()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=2000)

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.add(ws)
        logger.info("WS client connected — total: %d", len(self._clients))

    def disconnect(self, ws: WebSocket):
        self._clients.discard(ws)
        logger.info("WS client disconnected — total: %d", len(self._clients))

    async def emit(self, event_type: str, payload: dict):
        """Enqueue an event to broadcast to all connected clients."""
        envelope = {"type": event_type, "ts": time.time(), "data": payload}
        try:
            self._queue.put_nowait(envelope)
        except asyncio.QueueFull:
            logger.warning("EventBus queue full — dropping %s", event_type)

    async def broadcast_loop(self):
        """
        Background task to run in the FastAPI lifespan.
        Drains the queue and fans out to all connected WebSocket clients.
        """
        while True:
            envelope = await self._queue.get()
            dead: Set[WebSocket] = set()
            for ws in list(self._clients):
                try:
                    await ws.send_json(envelope)
                except Exception:
                    dead.add(ws)
            for ws in dead:
                self._clients.discard(ws)
            self._queue.task_done()


# Module-level singleton — import and call emit() from anywhere
event_bus = EventBus()


# ── WebSocket Route ────────────────────────────────────────────────────────────

@router.websocket("/events")
async def ws_events(websocket: WebSocket):
    """
    WebSocket endpoint for real-time Sovereign Core events.

    Event types emitted by the platform:
      backend.health_changed    — a GPU backend changed health state
      auction.completed         — Vickrey-Quadratic auction round concluded
      kairos.cycle_complete     — KAIROS ARSO evolution cycle finished
      kairos.elite_promoted     — agent reached Elite tier
      ledger.entry_written      — new Aegis-Vault entry
      mem_evolve.weights_updated — MemEvolve adjusted retrieval weights
      verifier.verdict          — self-verification pass/fail

    Client → server messages:
      "ping"  → server responds with {"type":"pong","ts":<float>}
    """
    await event_bus.connect(websocket)
    try:
        await websocket.send_json({"type": "connected", "ts": time.time(),
                                   "message": "Sovereign Core event stream active"})
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong", "ts": time.time()})
    except WebSocketDisconnect:
        event_bus.disconnect(websocket)
