"""Latency-aware, capability-aware request router.

The router combines three signals to select the best available backend for
each inference request:

1. **Health** — unhealthy backends are skipped entirely.
2. **Model affinity** — the :class:`~gateway.models.ModelAssigner` provides
   a capability-ordered preference list.
3. **Latency** — an Exponential Moving Average (EMA) tracks observed round-
   trip latency per backend.  Within a device-type tier, the backend with the
   lowest EMA latency is selected.

Failover
────────
If the preferred backend is unreachable at request time, the router
transparently retries the next candidate in the preference list until either a
response is obtained or all backends are exhausted (→ 503).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import aiohttp

from gateway.benchmark import ThroughputBenchmark
from gateway.config import BACKENDS, BackendConfig, GatewaySettings, settings
from gateway.health import HealthMonitor
from gateway.models import ModelAssigner

logger = logging.getLogger(__name__)


class LatencyTracker:
    """Per-backend Exponential Moving Average latency tracker.

    Args:
        alpha: EMA smoothing factor (0 < alpha ≤ 1).  Higher values weight
            recent observations more heavily.
        initial_latency: Seed value (seconds) before any observations are
            recorded.  Set high to allow all backends to be tried initially.
    """

    def __init__(self, alpha: float = 0.2, initial_latency: float = 0.5) -> None:
        self._alpha = alpha
        self._ema: dict[str, float] = {}
        self._initial = initial_latency

    def record(self, backend_id: str, latency_s: float) -> None:
        if backend_id not in self._ema:
            self._ema[backend_id] = latency_s
        else:
            self._ema[backend_id] = (
                self._alpha * latency_s + (1 - self._alpha) * self._ema[backend_id]
            )

    def get(self, backend_id: str) -> float:
        return self._ema.get(backend_id, self._initial)

    def all_latencies(self) -> dict[str, float]:
        return dict(self._ema)


class GatewayRouter:
    """Routes inference requests across the heterogeneous compute mesh.

    Parameters
    ----------
    health_monitor:
        Provides real-time backend health status.
    assigner:
        Provides capability-ordered backend preference lists.
    benchmark:
        Collects throughput statistics (updated on every request).
    cfg:
        Gateway runtime settings.
    backends:
        Backend list (defaults to the global ``BACKENDS``).
    """

    def __init__(
        self,
        health_monitor: HealthMonitor,
        assigner: Optional[ModelAssigner] = None,
        benchmark: Optional[ThroughputBenchmark] = None,
        cfg: GatewaySettings = settings,
        backends: Optional[list[BackendConfig]] = None,
    ) -> None:
        self._health = health_monitor
        self._assigner = assigner or ModelAssigner(backends)
        self._benchmark = benchmark or ThroughputBenchmark()
        self._cfg = cfg
        self._latency = LatencyTracker(alpha=cfg.latency_ema_alpha)
        self._session: Optional[aiohttp.ClientSession] = None
        self._backends = backends if backends is not None else BACKENDS

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        connector = aiohttp.TCPConnector(limit=100)
        timeout = aiohttp.ClientTimeout(total=self._cfg.backend_timeout)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Public routing API
    # ------------------------------------------------------------------
    async def route(
        self,
        path: str,
        method: str,
        headers: dict[str, str],
        body: Optional[bytes],
        model_id: Optional[str] = None,
        vram_required_gib: float = 0.0,
        priority_backend_id: Optional[str] = None,
    ) -> tuple[int, dict[str, str], bytes]:
        """Forward *path* to the best available backend and return the response.

        Args:
            priority_backend_id: If provided and the backend is among the
                healthy candidates, it is promoted to first position so that
                settled auction outcomes influence backend selection.

        Returns:
            A tuple of ``(status_code, response_headers, response_body)``.

        Raises:
            RuntimeError: If the router has not been started via
                :meth:`start`.
        """
        if self._session is None:
            raise RuntimeError("GatewayRouter.start() must be called before routing")

        candidates = self._select_candidates(model_id, vram_required_gib, priority_backend_id)
        if not candidates:
            logger.warning("No healthy backends available for request path=%s", path)
            return 503, {}, b'{"error": "no healthy backends available"}'

        last_error: Optional[str] = None
        for backend in candidates:
            status, resp_headers, resp_body = await self._try_backend(
                backend, path, method, headers, body
            )
            if status != 0:
                return status, resp_headers, resp_body
            last_error = f"backend {backend.id} unreachable"

        logger.error("All backends failed for path=%s last_error=%s", path, last_error)
        return 503, {}, b'{"error": "all backends failed"}'

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_candidates(
        self,
        model_id: Optional[str],
        vram_required_gib: float,
        priority_backend_id: Optional[str] = None,
    ) -> list[BackendConfig]:
        """Return healthy candidates ordered by capability + latency.

        If *priority_backend_id* is provided and the backend is present in the
        healthy candidate list, it is promoted to first position so that
        auction settlement outcomes influence which backend receives the
        request.
        """
        preferred = self._assigner.assign(
            model_id=model_id, vram_required_gib=vram_required_gib
        )
        # Filter to healthy only, then sort within each tier by EMA latency
        healthy = {b.id for b in self._health.get_healthy_backends()}
        healthy_candidates = [b for b in preferred if b.id in healthy]

        if not healthy_candidates:
            # Degrade gracefully: include UNKNOWN backends before giving up
            unknown_candidates = [
                b
                for b in preferred
                if b.id not in healthy
                and self._health.states[b.id].status.value == "unknown"
            ]
            if unknown_candidates:
                logger.warning(
                    "No HEALTHY backends; falling back to %d UNKNOWN backend(s)",
                    len(unknown_candidates),
                )
                return self._sort_by_latency(unknown_candidates)
            return []

        sorted_candidates = self._sort_by_latency(healthy_candidates)

        # Promote the auction-priority backend to the front of the list so
        # that settled auctions determine which backend is tried first.
        if priority_backend_id is not None:
            priority_idx = next(
                (i for i, b in enumerate(sorted_candidates) if b.id == priority_backend_id),
                None,
            )
            if priority_idx is not None and priority_idx != 0:
                promoted = sorted_candidates.pop(priority_idx)
                sorted_candidates.insert(0, promoted)
                logger.debug(
                    "Auction priority: promoted backend %s to first position",
                    priority_backend_id,
                )

        return sorted_candidates

    def _sort_by_latency(self, backends: list[BackendConfig]) -> list[BackendConfig]:
        """Sort backends by EMA latency (ascending) preserving relative tier order."""
        # Group by device weight so high-capability tiers are still preferred
        # but within a tier we pick the faster backend.
        return sorted(backends, key=lambda b: (round(b.weight * -1, 0), self._latency.get(b.id)))

    async def _try_backend(
        self,
        backend: BackendConfig,
        path: str,
        method: str,
        headers: dict[str, str],
        body: Optional[bytes],
    ) -> tuple[int, dict[str, str], bytes]:
        """Attempt a single backend call.  Returns ``(0, {}, b'')`` on failure."""
        assert self._session is not None
        url = f"{backend.url}{path}"
        start = time.monotonic()
        try:
            async with self._session.request(
                method,
                url,
                headers=headers,
                data=body,
            ) as resp:
                latency = time.monotonic() - start
                resp_body = await resp.read()
                resp_headers = dict(resp.headers)

                self._latency.record(backend.id, latency)
                self._benchmark.record(backend.id, latency, success=True)

                logger.debug(
                    "Routed to %s %s → HTTP %d (%.3fs)",
                    backend.id,
                    path,
                    resp.status,
                    latency,
                )
                return resp.status, resp_headers, resp_body

        except asyncio.TimeoutError:
            latency = time.monotonic() - start
            self._latency.record(backend.id, self._cfg.unhealthy_latency_penalty)
            self._benchmark.record(backend.id, latency, success=False)
            logger.warning("Timeout reaching backend %s after %.1fs", backend.id, latency)
            return 0, {}, b""

        except aiohttp.ClientError as exc:
            latency = time.monotonic() - start
            self._latency.record(backend.id, self._cfg.unhealthy_latency_penalty)
            self._benchmark.record(backend.id, latency, success=False)
            logger.warning("ClientError reaching backend %s: %s", backend.id, exc)
            return 0, {}, b""

    def latency_snapshot(self) -> dict[str, float]:
        """Return current EMA latency readings keyed by backend ID."""
        return self._latency.all_latencies()