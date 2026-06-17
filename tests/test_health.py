"""Tests for Health Monitor"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.health import HealthMonitor, SystemHealth


@pytest.fixture
def monitor():
    return HealthMonitor()


class TestHealthMonitor:
    @pytest.mark.asyncio
    async def test_get_system_health_returns_dataclass(self, monitor):
        health = await monitor.get_system_health()
        assert isinstance(health, SystemHealth)
        assert health.cpu_percent >= 0
        assert health.ram_total_gb > 0

    @pytest.mark.asyncio
    async def test_to_dict_structure(self, monitor):
        d = await monitor.to_dict()
        assert "cpu_percent" in d
        assert "ram" in d
        assert "gpu" in d
        assert "ollama_running" in d
        assert "uptime_seconds" in d

    @pytest.mark.asyncio
    async def test_ram_values_reasonable(self, monitor):
        health = await monitor.get_system_health()
        assert health.ram_total_gb > 0
        assert health.ram_used_gb > 0
        assert 0 <= health.ram_percent <= 100

    @pytest.mark.asyncio
    async def test_uptime_positive(self, monitor):
        health = await monitor.get_system_health()
        assert health.uptime_seconds >= 0

    def test_check_ollama_static(self, monitor):
        # Just verify it doesn't crash
        result = HealthMonitor._check_ollama()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_gpu_check_handles_no_nvidia(self, monitor):
        info = await monitor._check_gpu()
        # Should gracefully handle missing nvidia-smi
        assert isinstance(info, dict)
        assert "available" in info
