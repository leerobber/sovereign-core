"""Health check & failover protocol for the compute gateway."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, Optional

import aiohttp

from gateway.config import BACKENDS, BackendConfig, GatewaySettings, settings

logger = logging.getLogger(__name__)


class BackendStatus(str, Enum):
    """Operational status of a backend."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class BackendState:
    """Mutable runtime state for one backend instance."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.status: BackendStatus = BackendStatus.UNKNOWN
        self.consecutive_failures: int = 0
        self.consecutive_successes: int = 0
        self.last_checked: Optional[float] = None
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def record_success(self, recovery_threshold: int) -> None:
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        self.last_error = None
        self.last_checked = time.monotonic()
        if (
            self.status != BackendStatus.HEALTHY
            and self.consecutive_successes >= recovery_threshold
        ):
            logger.info(
                "Backend %s recovered → HEALTHY (successes=%d)",
                self.config.id,
                self.consecutive_successes,
            )
            self.status = BackendStatus.HEALTHY

    def record_failure(self, failure_threshold: int, error: str) -> None:
        self.consecutive_successes = 0
        self.consecutive_failures += 1
        self.last_error = error
        self.last_checked = time.monotonic()
        if (
            self.status != BackendStatus.UNHEALTHY
            and self.consecutive_failures >= failure_threshold
        ):
            logger.warning(
                "Backend %s marked UNHEALTHY after %d failures: %s",
                self.config.id,
                self.consecutive_failures,
                error,
            )
            self.status = BackendStatus.UNHEALTHY

    def to_dict(self) -> dict:
        return {
            "id": self.config.id,
            "label": self.config.label,
            "url": self.config.url,
            "status": self.status.value,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_checked": self.last_checked,
            "last_error": self.last_error,
        }


class HealthMonitor:
    """Periodically probes each backend and maintains its health state.

    Implements a simple circuit-breaker:
    - A backend is marked UNHEALTHY after ``failure_threshold`` consecutive
      failures.
    - It is restored to HEALTHY after ``recovery_threshold`` consecutive
      successes.
    """

    def __init__(
        self,
        cfg: GatewaySettings = settings,
        backends: Optional[list[BackendConfig]] = None,
    ) -> None:
        self._cfg = cfg
        _backends = backends if backends is not None else BACKENDS
        self._states: Dict[str, BackendState] = {
            b.id: BackendState(b) for b in _backends
        }
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def states(self) -> Dict[str, BackendState]:
        return self._states

    def get_healthy_backends(self) -> list[BackendConfig]:
        """Return configs for all currently healthy backends."""
        return [
            s.config
            for s in self._states.values()
            if s.status == BackendStatus.HEALTHY
        ]

    def is_healthy(self, backend_id: str) -> bool:
        state = self._states.get(backend_id)
        return state is not None and state.status == BackendStatus.HEALTHY

    def status_report(self) -> list[dict]:
        return [s.to_dict() for s in self._states.values()]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Start the background health-check loop."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop(), name="health-monitor")
        logger.info("HealthMonitor started (interval=%.1fs)", self._cfg.health_check_interval)

    async def stop(self) -> None:
        """Stop the background health-check loop."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("HealthMonitor stopped")

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------
    async def _loop(self) -> None:
        connector = aiohttp.TCPConnector(limit=len(self._states) * 2)
        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            while True:
                await self._check_all(session)
                await asyncio.sleep(self._cfg.health_check_interval)

    async def _check_all(self, session: aiohttp.ClientSession) -> None:
        tasks = [self._check_one(session, state) for state in self._states.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_one(
        self, session: aiohttp.ClientSession, state: BackendState
    ) -> None:
        url = f"{state.config.url}/health"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    state.record_success(self._cfg.recovery_threshold)
                else:
                    state.record_failure(
                        self._cfg.failure_threshold,
                        f"HTTP {resp.status}",
                    )
        except Exception as exc:
            state.record_failure(self._cfg.failure_threshold, str(exc))
