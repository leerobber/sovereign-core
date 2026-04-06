"""Gateway configuration — backend definitions and device capabilities."""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeviceType(str, Enum):
    """Compute device categories."""

    NVIDIA_GPU = "nvidia_gpu"
    AMD_GPU = "amd_gpu"
    CPU = "cpu"


class BackendConfig(BaseModel):
    """Static configuration for a single inference backend."""

    id: str
    url: str
    device_type: DeviceType
    # Advertised VRAM in GiB (0 for CPU backends)
    vram_gib: float = 0.0
    # Relative throughput weight used as a tie-breaker
    weight: float = 1.0
    # Human-readable device label
    label: str = ""

    @field_validator("url")
    @classmethod
    def url_must_have_scheme(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v.rstrip("/")


# ---------------------------------------------------------------------------
# Tri-GPU mesh: RTX 5050 | Radeon 780M | Ryzen 7 CPU
# ---------------------------------------------------------------------------
BACKENDS: list[BackendConfig] = [
    BackendConfig(
        id="rtx5050",
        url="http://localhost:8001",
        device_type=DeviceType.NVIDIA_GPU,
        vram_gib=8.0,
        weight=3.0,
        label="RTX 5050 (primary GPU)",
    ),
    BackendConfig(
        id="radeon780m",
        url="http://localhost:8002",
        device_type=DeviceType.AMD_GPU,
        vram_gib=4.0,
        weight=2.0,
        label="Radeon 780M (secondary GPU)",
    ),
    BackendConfig(
        id="ryzen7cpu",
        url="http://localhost:8003",
        device_type=DeviceType.CPU,
        vram_gib=0.0,
        weight=1.0,
        label="Ryzen 7 (CPU fallback)",
    ),
]

BACKEND_MAP: dict[str, BackendConfig] = {b.id: b for b in BACKENDS}

# Ordered preference for routing when no specific model constraint applies
BACKEND_PRIORITY: list[str] = ["rtx5050", "radeon780m", "ryzen7cpu"]

# Device types capable of GPU acceleration
GPU_DEVICE_TYPES: FrozenSet[DeviceType] = frozenset(
    {DeviceType.NVIDIA_GPU, DeviceType.AMD_GPU}
)


class GatewaySettings(BaseSettings):
    """Runtime-tuneable gateway settings (overridable via environment variables)."""

    model_config = SettingsConfigDict(env_prefix="GATEWAY_", case_sensitive=False)

    host: str = "0.0.0.0"
    port: int = 8000
    # How often (seconds) to probe backend health
    health_check_interval: float = 5.0
    # Per-request timeout when forwarding to a backend (seconds)
    backend_timeout: float = 30.0
    # Number of consecutive failures before marking a backend unhealthy
    failure_threshold: int = 3
    # Number of consecutive successes needed to restore a backend
    recovery_threshold: int = 2
    # EMA smoothing factor for latency tracking (0 < alpha <= 1)
    latency_ema_alpha: float = 0.2
    # Latency penalty (seconds) applied to unhealthy / unknown backends
    unhealthy_latency_penalty: float = 9999.0


settings = GatewaySettings()
