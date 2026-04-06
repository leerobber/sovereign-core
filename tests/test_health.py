"""Tests for health check & failover protocol."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import BACKENDS, BackendConfig, DeviceType, GatewaySettings
from gateway.health import BackendState, BackendStatus, HealthMonitor


# ---------------------------------------------------------------------------
# BackendState unit tests
# ---------------------------------------------------------------------------
class TestBackendState:
    def _make_state(self) -> BackendState:
        cfg = BackendConfig(
            id="test", url="http://localhost:9001", device_type=DeviceType.CPU
        )
        return BackendState(cfg)

    def test_initial_status_is_unknown(self):
        state = self._make_state()
        assert state.status == BackendStatus.UNKNOWN

    def test_record_success_transitions_to_healthy(self):
        state = self._make_state()
        state.record_success(recovery_threshold=2)
        assert state.status == BackendStatus.UNKNOWN  # not enough successes yet
        state.record_success(recovery_threshold=2)
        assert state.status == BackendStatus.HEALTHY

    def test_record_failure_transitions_to_unhealthy(self):
        state = self._make_state()
        state.record_failure(failure_threshold=3, error="timeout")
        assert state.status == BackendStatus.UNKNOWN
        state.record_failure(failure_threshold=3, error="timeout")
        assert state.status == BackendStatus.UNKNOWN
        state.record_failure(failure_threshold=3, error="timeout")
        assert state.status == BackendStatus.UNHEALTHY

    def test_recovery_from_unhealthy(self):
        state = self._make_state()
        for _ in range(3):
            state.record_failure(failure_threshold=3, error="err")
        assert state.status == BackendStatus.UNHEALTHY
        for _ in range(2):
            state.record_success(recovery_threshold=2)
        assert state.status == BackendStatus.HEALTHY

    def test_success_resets_failure_counter(self):
        state = self._make_state()
        state.record_failure(failure_threshold=3, error="err")
        state.record_success(recovery_threshold=2)
        assert state.consecutive_failures == 0

    def test_failure_resets_success_counter(self):
        state = self._make_state()
        state.record_success(recovery_threshold=2)
        state.record_failure(failure_threshold=3, error="err")
        assert state.consecutive_successes == 0

    def test_to_dict_keys(self):
        state = self._make_state()
        d = state.to_dict()
        for key in ("id", "status", "consecutive_failures", "consecutive_successes"):
            assert key in d

    def test_last_error_stored(self):
        state = self._make_state()
        state.record_failure(failure_threshold=3, error="connection refused")
        assert state.last_error == "connection refused"

    def test_last_error_cleared_on_success(self):
        state = self._make_state()
        state.record_failure(failure_threshold=3, error="err")
        state.record_success(recovery_threshold=2)
        assert state.last_error is None


# ---------------------------------------------------------------------------
# HealthMonitor unit tests
# ---------------------------------------------------------------------------
class TestHealthMonitor:
    def _make_monitor(self, backends=None) -> HealthMonitor:
        cfg = GatewaySettings(
            health_check_interval=1.0,
            failure_threshold=2,
            recovery_threshold=1,
        )
        return HealthMonitor(cfg=cfg, backends=backends or BACKENDS)

    def test_all_backends_initially_unknown(self):
        monitor = self._make_monitor()
        for state in monitor.states.values():
            assert state.status == BackendStatus.UNKNOWN

    def test_get_healthy_backends_initially_empty(self):
        monitor = self._make_monitor()
        assert monitor.get_healthy_backends() == []

    def test_is_healthy_false_for_unknown(self):
        monitor = self._make_monitor()
        assert not monitor.is_healthy("rtx5050")

    def test_is_healthy_true_after_successes(self):
        monitor = self._make_monitor()
        monitor.states["rtx5050"].record_success(recovery_threshold=1)
        assert monitor.is_healthy("rtx5050")

    def test_get_healthy_backends_returns_healthy(self):
        monitor = self._make_monitor()
        monitor.states["rtx5050"].record_success(recovery_threshold=1)
        healthy = monitor.get_healthy_backends()
        assert len(healthy) == 1
        assert healthy[0].id == "rtx5050"

    def test_status_report_contains_all_backends(self):
        monitor = self._make_monitor()
        report = monitor.status_report()
        assert len(report) == len(BACKENDS)

    def test_unknown_backend_is_healthy_returns_false(self):
        monitor = self._make_monitor()
        assert not monitor.is_healthy("nonexistent")

    @pytest.mark.asyncio
    async def test_check_one_marks_healthy_on_200(self):
        monitor = self._make_monitor()
        state = monitor.states["rtx5050"]

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        await monitor._check_one(mock_session, state)
        assert state.consecutive_successes == 1

    @pytest.mark.asyncio
    async def test_check_one_marks_failure_on_non_200(self):
        monitor = self._make_monitor()
        state = monitor.states["rtx5050"]

        mock_resp = AsyncMock()
        mock_resp.status = 503
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        await monitor._check_one(mock_session, state)
        assert state.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_check_one_marks_failure_on_exception(self):
        import aiohttp

        monitor = self._make_monitor()
        state = monitor.states["rtx5050"]

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientConnectionError("refused"))

        await monitor._check_one(mock_session, state)
        assert state.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        monitor = self._make_monitor()
        with patch.object(monitor, "_loop", new_callable=AsyncMock) as mock_loop:
            await monitor.start()
            assert monitor._task is not None
            await monitor.stop()
            assert monitor._task is None

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        monitor = self._make_monitor()
        with patch.object(monitor, "_loop", new_callable=AsyncMock):
            await monitor.start()
            task1 = monitor._task
            await monitor.start()  # second call should be a no-op
            assert monitor._task is task1
            await monitor.stop()
