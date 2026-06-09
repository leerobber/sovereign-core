"""
gateway/ws.py — WebSocket Event Bus

Real-time bidirectional event streaming for:
  - Backend health changes
  - Inference completions
  - KAIROS cycle results + elite promotions
  - Auction outcomes
  - Ledger writes
  - MemEvolve weight updates

Clients connect to ws://host:8000/ws/events
Send "ping" → receive {"type":"pong",...}
All other messages are broadcast events in JSON format.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


# ── Event Bus ─────────────────────────────────────────────────────────────────

class EventBus:
    """
    Central async pub/sub bus.
    Any gateway component can call await event_bus.emit(type, data)
    All connected WebSocket clients receive the event.
    """

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._event_counts: Dict[str, int] = {}

    def add_client(self, ws: WebSocket) -> None:
        self._clients.add(ws)
        logger.info("WS client connected. Total: %d", len(self._clients))
        # Update prometheus gauge if available
        try:
            from gateway.metrics import WS_CONNECTIONS
            WS_CONNECTIONS.set(len(self._clients))
        except Exception:
            pass

    def remove_client(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        logger.info("WS client disconnected. Total: %d", len(self._clients))
        try:
            from gateway.metrics import WS_CONNECTIONS
            WS_CONNECTIONS.set(len(self._clients))
        except Exception:
            pass

    async def emit(self, event_type: str, data: Any = None) -> None:
        """Put an event on the broadcast queue."""
        event = {
            "type": event_type,
            "ts": time.time(),
            "data": data or {},
        }
        self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop oldest event if queue is full
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except Exception:
                pass

    async def broadcast_loop(self) -> None:
        """Background task — drains queue and broadcasts to all clients."""
        logger.info("WebSocket broadcast loop started")
        while True:
            try:
                event = await self._queue.get()
                payload = json.dumps(event)

                dead: Set[WebSocket] = set()
                for ws in list(self._clients):
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        dead.add(ws)

                for ws in dead:
                    self.remove_client(ws)

            except asyncio.CancelledError:
                logger.info("Broadcast loop cancelled — shutting down")
                break
            except Exception as exc:
                logger.error("Broadcast loop error: %s", exc)
                await asyncio.sleep(0.1)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "connected_clients": len(self._clients),
            "queue_size": self._queue.qsize(),
            "event_counts": dict(self._event_counts),
        }


# Module-level singleton — imported by main.py and all routers
event_bus = EventBus()


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@router.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    """
    WebSocket endpoint for real-time gateway events.

    Protocol:
      Client → "ping"                        → server replies {"type":"pong"}
      Client → "stats"                       → server replies current stats
      Server → {"type": EVENT_TYPE, "ts": float, "data": {...}}

    Event types:
      backend.health_changed      — a backend flipped healthy/unhealthy
      inference.completed         — an inference request finished
      kairos.cycle_complete       — a KAIROS ARSO cycle finished
      kairos.elite_promoted       — an agent promoted to Elite tier
      auction.completed           — an auction was resolved
      ledger.entry_written        — Aegis-Vault ledger entry written
      mem_evolve.weights_updated  — MemEvolve updated retrieval weights
      verifier.verdict            — self-verification result
    """
    await ws.accept()
    event_bus.add_client(ws)

    # Send connected event
    await ws.send_text(json.dumps({
        "type": "connected",
        "ts": time.time(),
        "data": {
            "message": "Sovereign Core event stream connected",
            "client_count": event_bus.client_count,
        }
    }))

    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=30.0)

                if msg.strip().lower() == "ping":
                    await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))

                elif msg.strip().lower() == "stats":
                    await ws.send_text(json.dumps({
                        "type": "stats",
                        "ts": time.time(),
                        "data": event_bus.stats,
                    }))

            except asyncio.TimeoutError:
                # Send keepalive
                await ws.send_text(json.dumps({"type": "keepalive", "ts": time.time()}))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WS error: %s", exc)
    finally:
        event_bus.remove_client(ws)
