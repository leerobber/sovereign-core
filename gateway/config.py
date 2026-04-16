"""
gateway/config.py — Gateway configuration (production upgrade)
Adds CORS origins, API key, and sovereign-wide settings.
Full backward compat — all new fields have defaults.
"""
from __future__ import annotations

from enum import Enum
from typing import FrozenSet, List, Optional

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeviceType(str, Enum):
    NVIDIA_GPU = "nvidia_gpu"
    AMD_GPU = "amd_gpu"
    CPU = "cpu"


class BackendConfig(BaseModel):
    id: str
    url: str
    device_type: DeviceType
    vram_gib: float = 0.0
    weight: float = 1.0
    label: str = ""

    @field_validator("url")
    @classmethod
    def url_must_have_scheme(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v.rstrip("/")


# ── Backend definitions ───────────────────────────────────────────────────────

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
BACKEND_PRIORITY: list[str] = ["rtx5050", "radeon780m", "ryzen7cpu"]

GPU_DEVICE_TYPES: FrozenSet[DeviceType] = frozenset(
    {DeviceType.NVIDIA_GPU, DeviceType.AMD_GPU}
)


# ── Settings ──────────────────────────────────────────────────────────────────

class GatewaySettings(BaseSettings):
    """
    Runtime-configurable gateway settings.
    All fields overridable via environment variables prefixed GATEWAY_.
    """
    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Bind
    host: str = "0.0.0.0"
    port: int = 8000

    # Health probing
    health_check_interval: float = 5.0
    backend_timeout: float = 30.0
    failure_threshold: int = 3
    recovery_threshold: int = 2
    latency_ema_alpha: float = 0.2

    # Security
    api_key: Optional[str] = None   # None = no auth required
    cors_origins: str = "*"          # comma-separated or "*"

    # Ledger
    ledger_hmac_secret: str = "change-me-in-production"
    ledger_max_memory_entries: int = 10_000

    # KAIROS
    kairos_agents_dir: str = "data/kairos/agents"
    kairos_max_cycles: int = 50

    # MemEvolve
    mem_evolve_ab_ratio: float = 0.5

    # Logging
    log_level: str = "INFO"


settings = GatewaySettings()
