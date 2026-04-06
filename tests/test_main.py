"""Integration tests for the FastAPI gateway application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from gateway.config import BACKEND_MAP
from gateway.health import BackendStatus


# ---------------------------------------------------------------------------
# Helpers: patch the lifespan so we control health state in tests
# ---------------------------------------------------------------------------
def _patch_app_state(healthy_ids: list[str]):
    """Return mocked monitor/benchmark/router with the given backends healthy."""
    import gateway.main as gm
    from gateway.benchmark import ThroughputBenchmark
    from gateway.config import GatewaySettings
    from gateway.health import HealthMonitor
    from gateway.models import ModelAssigner
    from gateway.router import GatewayRouter

    cfg = GatewaySettings(failure_threshold=1, recovery_threshold=1)
    monitor = HealthMonitor(cfg=cfg)
    for bid, state in monitor.states.items():
        state.status = (
            BackendStatus.HEALTHY if bid in healthy_ids else BackendStatus.UNHEALTHY
        )
    benchmark = ThroughputBenchmark()
    router = GatewayRouter(
        health_monitor=monitor,
        assigner=ModelAssigner(),
        benchmark=benchmark,
        cfg=cfg,
    )

    return monitor, benchmark, router


# ---------------------------------------------------------------------------
# App fixture using ASGITransport (no real server needed)
# ---------------------------------------------------------------------------
@pytest.fixture
def patched_app():
    """Return the FastAPI app with mocked gateway state (no background tasks).

    The lifespan is replaced with a no-op so that TestClient does not
    overwrite the manually-assigned module-level state.
    """
    import gateway.main as gm

    monitor, benchmark, router = _patch_app_state(["rtx5050", "radeon780m"])
    gm._health_monitor = monitor
    gm._benchmark = benchmark
    gm._router = router

    # Swap the lifespan so TestClient doesn't recreate globals on __enter__
    original_lifespan = gm.app.router.lifespan_context

    @asynccontextmanager
    async def _noop_lifespan(app):  # type: ignore[override]
        yield

    gm.app.router.lifespan_context = _noop_lifespan
    yield gm.app
    gm.app.router.lifespan_context = original_lifespan


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health_returns_200(self, patched_app):
        with TestClient(patched_app, raise_server_exceptions=False) as client:
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_json_structure(self, patched_app):
        with TestClient(patched_app, raise_server_exceptions=False) as client:
            data = client.get("/health").json()
        assert "status" in data
        assert "backends" in data
        assert "healthy_backends" in data
        assert "total_backends" in data

    def test_health_all_unhealthy_is_degraded(self):
        import gateway.main as gm
        from contextlib import asynccontextmanager

        monitor, benchmark, router = _patch_app_state([])
        gm._health_monitor = monitor
        gm._benchmark = benchmark
        gm._router = router

        original_lifespan = gm.app.router.lifespan_context

        @asynccontextmanager
        async def _noop(app):  # type: ignore[override]
            yield

        gm.app.router.lifespan_context = _noop
        try:
            with TestClient(gm.app, raise_server_exceptions=False) as client:
                data = client.get("/health").json()
        finally:
            gm.app.router.lifespan_context = original_lifespan

        assert data["status"] == "degraded"
        assert data["healthy_backends"] == 0

    def test_health_some_healthy_is_ok(self, patched_app):
        with TestClient(patched_app, raise_server_exceptions=False) as client:
            data = client.get("/health").json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# /metrics endpoint
# ---------------------------------------------------------------------------
class TestMetricsEndpoint:
    def test_metrics_returns_200(self, patched_app):
        with TestClient(patched_app, raise_server_exceptions=False) as client:
            resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_json_structure(self, patched_app):
        with TestClient(patched_app, raise_server_exceptions=False) as client:
            data = client.get("/metrics").json()
        assert "latency_ema_s" in data
        assert "benchmark" in data


# ---------------------------------------------------------------------------
# /benchmark endpoint
# ---------------------------------------------------------------------------
class TestBenchmarkEndpoint:
    def test_benchmark_returns_200(self, patched_app):
        with TestClient(patched_app, raise_server_exceptions=False) as client:
            resp = client.get("/benchmark")
        assert resp.status_code == 200

    def test_benchmark_reset_all(self, patched_app):
        with TestClient(patched_app, raise_server_exceptions=False) as client:
            resp = client.post("/benchmark/reset")
        assert resp.status_code == 200
        assert resp.json()["reset"] is True

    def test_benchmark_reset_specific_backend(self, patched_app):
        with TestClient(patched_app, raise_server_exceptions=False) as client:
            resp = client.post("/benchmark/reset", params={"backend_id": "rtx5050"})
        assert resp.status_code == 200

    def test_benchmark_reset_unknown_backend_404(self, patched_app):
        with TestClient(patched_app, raise_server_exceptions=False) as client:
            resp = client.post("/benchmark/reset", params={"backend_id": "nonexistent"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /v1/* proxy endpoint
# ---------------------------------------------------------------------------
class TestProxyEndpoint:
    @pytest.mark.asyncio
    async def test_proxy_503_when_no_healthy_backends(self):
        import gateway.main as gm

        monitor, benchmark, router = _patch_app_state([])
        # Router session must be set so it doesn't raise RuntimeError
        router._session = MagicMock()
        gm._health_monitor = monitor
        gm._benchmark = benchmark
        gm._router = router

        async with AsyncClient(
            transport=ASGITransport(app=gm.app), base_url="http://test"
        ) as client:
            resp = await client.post("/v1/chat/completions", content=b"{}")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_proxy_routes_to_backend(self):
        import gateway.main as gm

        monitor, benchmark, router = _patch_app_state(["rtx5050"])
        router._session = MagicMock()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=b'{"choices": []}')
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        router._session.request = MagicMock(return_value=mock_resp)

        gm._health_monitor = monitor
        gm._benchmark = benchmark
        gm._router = router

        async with AsyncClient(
            transport=ASGITransport(app=gm.app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                content=b"{}",
                params={"model_id": "deepseek-v3"},
            )
        assert resp.status_code == 200
