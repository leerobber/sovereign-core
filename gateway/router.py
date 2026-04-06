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
        auction_priority: Optional[dict[str, str]] = None,
    ) -> tuple[int, dict[str, str], bytes]:
        """Forward *path* to the best available backend and return the response.

        Args:
            path: Request path to forward.
            method: HTTP method.
            headers: Request headers to forward.
            body: Request body bytes.
            model_id: Optional model identifier for capability matching.
            vram_required_gib: Minimum VRAM requirement in GiB.
            auction_priority: Optional mapping of backend_id -> agent_id from
                recent auction results. Backends with matching priority agents
                are ranked higher in candidate selection.

        Returns:
            A tuple of ``(status_code, response_headers, response_body)``.

        Raises:
            RuntimeError: If the router has not been started via
                :meth:`start`.
        """
        if self._session is None:
            raise RuntimeError("GatewayRouter.start() must be called before routing")

        candidates = self._select_candidates(model_id, vram_required_gib, auction_priority)
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
        auction_priority: Optional[dict[str, str]] = None,
    ) -> list[BackendConfig]:
        """Return healthy candidates ordered by auction priority, capability + latency.

        Args:
            model_id: Optional model identifier.
            vram_required_gib: Minimum VRAM requirement.
            auction_priority: Optional backend_id -> agent_id mapping from auctions.

        Returns:
            List of backends ordered by priority (auction winners first), then
            capability and latency.
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
                return self._sort_by_latency(unknown_candidates, auction_priority)
            return []

        return self._sort_by_latency(healthy_candidates, auction_priority)

    def _sort_by_latency(
        self,
        backends: list[BackendConfig],
        auction_priority: Optional[dict[str, str]] = None,
    ) -> list[BackendConfig]:
        """Sort backends by auction priority, then EMA latency (ascending) preserving tier order.

        Args:
            backends: List of backends to sort.
            auction_priority: Optional backend_id -> agent_id mapping from auctions.
                Backends in this mapping are prioritized first.

        Returns:
            Sorted list with auction winners first, then by capability tier and latency.
        """
        # Group by device weight so high-capability tiers are still preferred
        # but within a tier we pick the faster backend.
        # If auction_priority is provided, backends in that map get priority=0, others get 1
        def sort_key(b: BackendConfig) -> tuple[int, float, float]:
            has_auction_priority = 0 if (auction_priority and b.id in auction_priority) else 1
            capability_tier = round(b.weight * -1, 0)
            latency = self._latency.get(b.id)
            return (has_auction_priority, capability_tier, latency)

        return sorted(backends, key=sort_key)

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