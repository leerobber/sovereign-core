"""Tests for the latency-aware, capability-aware gateway router."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.benchmark import ThroughputBenchmark
from gateway.config import BACKENDS, BackendConfig, DeviceType, GatewaySettings
from gateway.health import BackendStatus, HealthMonitor
from gateway.models import ModelAssigner
from gateway.router import GatewayRouter, LatencyTracker


# ---------------------------------------------------------------------------
# LatencyTracker unit tests
# ---------------------------------------------------------------------------
class TestLatencyTracker:
    def test_initial_value_is_seed(self):
        tracker = LatencyTracker(initial_latency=0.5)
        assert tracker.get("anything") == pytest.approx(0.5)

    def test_first_record_sets_ema(self):
        tracker = LatencyTracker(alpha=0.5)
        tracker.record("backend_a", 1.0)
        assert tracker.get("backend_a") == pytest.approx(1.0)

    def test_ema_smoothing(self):
        tracker = LatencyTracker(alpha=0.5)
        tracker.record("b", 1.0)
        tracker.record("b", 3.0)
        # EMA = 0.5 * 3.0 + 0.5 * 1.0 = 2.0
        assert tracker.get("b") == pytest.approx(2.0)

    def test_multiple_backends_independent(self):
        tracker = LatencyTracker(alpha=0.2)
        tracker.record("a", 0.1)
        tracker.record("b", 1.0)
        assert tracker.get("a") < tracker.get("b")

    def test_all_latencies_returns_all(self):
        tracker = LatencyTracker()
        tracker.record("x", 0.1)
        tracker.record("y", 0.2)
        lats = tracker.all_latencies()
        assert "x" in lats and "y" in lats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_monitor(healthy_ids: list[str]) -> HealthMonitor:
    cfg = GatewaySettings(failure_threshold=1, recovery_threshold=1)
    monitor = HealthMonitor(cfg=cfg)
    for bid, state in monitor.states.items():
        if bid in healthy_ids:
            state.status = BackendStatus.HEALTHY
        else:
            state.status = BackendStatus.UNHEALTHY
    return monitor


def _make_router(healthy_ids: list[str]) -> GatewayRouter:
    cfg = GatewaySettings(backend_timeout=5.0)
    monitor = _make_monitor(healthy_ids)
    benchmark = ThroughputBenchmark()
    router = GatewayRouter(
        health_monitor=monitor,
        assigner=ModelAssigner(),
        benchmark=benchmark,
        cfg=cfg,
    )
    return router


# ---------------------------------------------------------------------------
# GatewayRouter routing logic
# ---------------------------------------------------------------------------
class TestGatewayRouterCandidateSelection:
    def test_healthy_backends_selected(self):
        router = _make_router(["rtx5050", "radeon780m"])
        candidates = router._select_candidates(model_id=None, vram_required_gib=0.0)
        ids = [c.id for c in candidates]
        assert "rtx5050" in ids
        assert "radeon780m" in ids
        assert "ryzen7cpu" not in ids

    def test_no_healthy_falls_back_to_unknown(self):
        cfg = GatewaySettings(failure_threshold=1, recovery_threshold=1)
        monitor = HealthMonitor(cfg=cfg)
        # All backends remain UNKNOWN
        router = GatewayRouter(
            health_monitor=monitor,
            assigner=ModelAssigner(),
            benchmark=ThroughputBenchmark(),
            cfg=cfg,
        )
        candidates = router._select_candidates(model_id=None, vram_required_gib=0.0)
        assert len(candidates) == 3  # all UNKNOWN backends included

    def test_all_unhealthy_returns_empty(self):
        router = _make_router([])  # no healthy backends, all UNHEALTHY
        candidates = router._select_candidates(model_id=None, vram_required_gib=0.0)
        assert candidates == []

    def test_latency_sort_within_tier(self):
        router = _make_router(["rtx5050", "radeon780m", "ryzen7cpu"])
        # Simulate that rtx5050 has higher latency than radeon780m
        router._latency.record("rtx5050", 2.0)
        router._latency.record("radeon780m", 0.1)
        # Large model: rtx5050 > radeon780m by weight, but latency sorts within tier
        candidates = router._select_candidates(model_id="deepseek", vram_required_gib=0.0)
        # rtx5050 has higher weight (3.0) → still should be first in tier
        assert candidates[0].id == "rtx5050"


# ---------------------------------------------------------------------------
# GatewayRouter.route (mocked HTTP)
# ---------------------------------------------------------------------------
class TestGatewayRouterRoute:
    @pytest.mark.asyncio
    async def test_route_not_started_raises(self):
        router = _make_router(["rtx5050"])
        with pytest.raises(RuntimeError, match="start()"):
            await router.route("/v1/chat", "POST", {}, b"{}")

    @pytest.mark.asyncio
    async def test_route_returns_response(self):
        router = _make_router(["rtx5050"])
        router._session = MagicMock()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=b'{"result": "ok"}')
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        router._session.request = MagicMock(return_value=mock_resp)

        status, headers, body = await router.route(
            "/v1/chat", "POST", {}, b"{}", model_id="deepseek"
        )
        assert status == 200
        assert body == b'{"result": "ok"}'

    @pytest.mark.asyncio
    async def test_route_503_when_no_healthy_backends(self):
        router = _make_router([])
        router._session = MagicMock()
        status, _, body = await router.route("/v1/chat", "POST", {}, b"{}")
        assert status == 503

    @pytest.mark.asyncio
    async def test_route_failover_to_next_backend(self):
        import aiohttp

        router = _make_router(["rtx5050", "radeon780m"])
        router._session = MagicMock()

        # First backend (rtx5050) raises ClientError, second succeeds
        call_count = 0

        def request_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_ctx = AsyncMock()
            if call_count == 1:
                mock_ctx.__aenter__ = AsyncMock(
                    side_effect=aiohttp.ClientConnectionError("refused")
                )
            else:
                mock_resp = AsyncMock()
                mock_resp.status = 200
                mock_resp.read = AsyncMock(return_value=b'{"ok": true}')
                mock_resp.headers = {}
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            return mock_ctx

        router._session.request = MagicMock(side_effect=request_side_effect)

        status, _, body = await router.route(
            "/v1/generate", "POST", {}, b"{}", model_id="deepseek"
        )
        assert status == 200
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_latency_recorded_on_success(self):
        router = _make_router(["rtx5050"])
        router._session = MagicMock()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=b'{}')
        mock_resp.headers = {}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        router._session.request = MagicMock(return_value=mock_resp)

        await router.route("/v1/chat", "POST", {}, b"{}", model_id="deepseek")
        assert "rtx5050" in router._latency.all_latencies()

    def test_latency_snapshot_returns_dict(self):
        router = _make_router(["rtx5050"])
        router._latency.record("rtx5050", 0.3)
        snap = router.latency_snapshot()
        assert isinstance(snap, dict)
        assert snap["rtx5050"] == pytest.approx(0.3)
