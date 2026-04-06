"""Tests for gateway configuration."""

import pytest

from gateway.config import (
    BACKENDS,
    BACKEND_MAP,
    BACKEND_PRIORITY,
    BackendConfig,
    DeviceType,
    GatewaySettings,
    GPU_DEVICE_TYPES,
)


class TestBackendConfig:
    def test_all_backends_defined(self):
        assert len(BACKENDS) == 3

    def test_backend_ids(self):
        ids = {b.id for b in BACKENDS}
        assert ids == {"rtx5050", "radeon780m", "ryzen7cpu"}

    def test_backend_ports(self):
        urls = {b.id: b.url for b in BACKENDS}
        assert urls["rtx5050"] == "http://localhost:8001"
        assert urls["radeon780m"] == "http://localhost:8002"
        assert urls["ryzen7cpu"] == "http://localhost:8003"

    def test_device_types(self):
        types = {b.id: b.device_type for b in BACKENDS}
        assert types["rtx5050"] == DeviceType.NVIDIA_GPU
        assert types["radeon780m"] == DeviceType.AMD_GPU
        assert types["ryzen7cpu"] == DeviceType.CPU

    def test_backend_map_keys(self):
        assert set(BACKEND_MAP.keys()) == {"rtx5050", "radeon780m", "ryzen7cpu"}

    def test_backend_priority_order(self):
        assert BACKEND_PRIORITY == ["rtx5050", "radeon780m", "ryzen7cpu"]

    def test_gpu_device_types(self):
        assert DeviceType.NVIDIA_GPU in GPU_DEVICE_TYPES
        assert DeviceType.AMD_GPU in GPU_DEVICE_TYPES
        assert DeviceType.CPU not in GPU_DEVICE_TYPES

    def test_vram_values(self):
        vram = {b.id: b.vram_gib for b in BACKENDS}
        assert vram["rtx5050"] > vram["radeon780m"]
        assert vram["ryzen7cpu"] == 0.0

    def test_weights_descending(self):
        weights = [b.weight for b in BACKENDS]
        assert weights == sorted(weights, reverse=True)

    def test_url_validation_requires_scheme(self):
        with pytest.raises(ValueError, match="url must start with"):
            BackendConfig(
                id="bad",
                url="localhost:9999",
                device_type=DeviceType.CPU,
            )

    def test_url_trailing_slash_stripped(self):
        b = BackendConfig(
            id="ok",
            url="http://localhost:9001/",
            device_type=DeviceType.CPU,
        )
        assert not b.url.endswith("/")


class TestGatewaySettings:
    def test_defaults(self):
        s = GatewaySettings()
        assert s.port == 8000
        assert s.health_check_interval > 0
        assert s.failure_threshold >= 1
        assert s.recovery_threshold >= 1
        assert 0 < s.latency_ema_alpha <= 1.0
