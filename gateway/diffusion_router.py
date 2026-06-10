"""Diffusion-based language model router prototype (RES-10).

WeDLM — Weighted-Diffusion Language Model routing.

Compares parallel (diffusion) decoding against autoregressive decoding on
the heterogeneous compute mesh, tracking per-device tokens-per-watt efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class DecodeMode(str, Enum):
    PARALLEL = "parallel"
    AUTOREGRESSIVE = "autoregressive"


@dataclass
class DiffusionConfig:
    model_params: int = 50_000_000
    denoising_steps: int = 10
    device: str = "nvidia_gpu"


@dataclass
class GenerationResult:
    mode: DecodeMode
    tokens_generated: int
    latency_s: float
    estimated_watts: float
    tokens_per_second: float
    tokens_per_watt: float
    forward_passes: int
    flops_total: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "tokens_generated": self.tokens_generated,
            "latency_s": round(self.latency_s, 6),
            "estimated_watts": self.estimated_watts,
            "tokens_per_second": self.tokens_per_second,
            "tokens_per_watt": self.tokens_per_watt,
            "forward_passes": self.forward_passes,
            "flops_total": self.flops_total,
        }


@dataclass
class ComparisonResult:
    parallel: GenerationResult
    autoregressive: GenerationResult
    tokens_per_watt_speedup: float
    latency_speedup: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "parallel": self.parallel.to_dict(),
            "autoregressive": self.autoregressive.to_dict(),
            "tokens_per_watt_speedup": self.tokens_per_watt_speedup,
            "latency_speedup": self.latency_speedup,
            "advantage": "parallel" if self.tokens_per_watt_speedup > 1.0 else "autoregressive",
        }


class TokensPerWattTracker:
    def __init__(self) -> None:
        self._total_tokens: dict[str, int] = {
            DecodeMode.PARALLEL: 0,
            DecodeMode.AUTOREGRESSIVE: 0,
        }
        self._total_calls: dict[str, int] = {
            DecodeMode.PARALLEL: 0,
            DecodeMode.AUTOREGRESSIVE: 0,
        }
        self._total_watt_seconds: dict[str, float] = {
            DecodeMode.PARALLEL: 0.0,
            DecodeMode.AUTOREGRESSIVE: 0.0,
        }

    def record(self, result: GenerationResult) -> None:
        mode = result.mode
        self._total_tokens[mode] += result.tokens_generated
        self._total_calls[mode] += 1
        self._total_watt_seconds[mode] += result.estimated_watts * result.latency_s

    def average_tokens_per_watt(self, mode: DecodeMode) -> float:
        watt_secs = self._total_watt_seconds[mode]
        if watt_secs == 0.0:
            return 0.0
        return self._total_tokens[mode] / watt_secs

    def summary(self) -> dict[str, Any]:
        par_tpw = self.average_tokens_per_watt(DecodeMode.PARALLEL)
        ar_tpw = self.average_tokens_per_watt(DecodeMode.AUTOREGRESSIVE)
        ratio: Optional[float] = (par_tpw / ar_tpw) if ar_tpw > 0 else None
        return {
            "parallel": {
                "total_tokens": self._total_tokens[DecodeMode.PARALLEL],
                "total_calls": self._total_calls[DecodeMode.PARALLEL],
                "avg_tokens_per_watt": par_tpw,
            },
            "autoregressive": {
                "total_tokens": self._total_tokens[DecodeMode.AUTOREGRESSIVE],
                "total_calls": self._total_calls[DecodeMode.AUTOREGRESSIVE],
                "avg_tokens_per_watt": ar_tpw,
            },
            "tokens_per_watt_ratio": ratio,
        }


_DEVICE_WATTS: dict[str, float] = {
    "nvidia_gpu": 25.0,
    "amd_gpu": 15.0,
    "cpu": 45.0,
}
_TIME_PER_FORWARD_PASS_S: float = 1e-3


class DiffusionRouter:
    def __init__(self, cfg: Optional[DiffusionConfig] = None) -> None:
        self._cfg = cfg or DiffusionConfig()
        self._tracker = TokensPerWattTracker()

    @property
    def device(self) -> str:
        return self._cfg.device

    def generate(self, num_tokens: int, mode: DecodeMode = DecodeMode.PARALLEL) -> GenerationResult:
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")

        watts = _DEVICE_WATTS.get(self._cfg.device, 25.0)
        flops_per_token = 2 * self._cfg.model_params

        if mode == DecodeMode.PARALLEL:
            forward_passes = self._cfg.denoising_steps
            latency_s = self._cfg.denoising_steps * _TIME_PER_FORWARD_PASS_S
            flops_total = float(self._cfg.denoising_steps * num_tokens * flops_per_token)
        else:
            forward_passes = num_tokens
            latency_s = num_tokens * _TIME_PER_FORWARD_PASS_S
            flops_total = float(num_tokens * flops_per_token)

        tps = num_tokens / latency_s
        tpw = tps / watts

        result = GenerationResult(
            mode=mode,
            tokens_generated=num_tokens,
            latency_s=latency_s,
            estimated_watts=watts,
            tokens_per_second=tps,
            tokens_per_watt=tpw,
            forward_passes=forward_passes,
            flops_total=flops_total,
        )
        self._tracker.record(result)
        return result

    def compare(self, num_tokens: int = 256) -> ComparisonResult:
        par = self.generate(num_tokens, DecodeMode.PARALLEL)
        ar = self.generate(num_tokens, DecodeMode.AUTOREGRESSIVE)
        tpw_speedup = par.tokens_per_watt / ar.tokens_per_watt if ar.tokens_per_watt > 0 else 0.0
        latency_speedup = ar.latency_s / par.latency_s if par.latency_s > 0 else 0.0
        return ComparisonResult(
            parallel=par,
            autoregressive=ar,
            tokens_per_watt_speedup=tpw_speedup,
            latency_speedup=latency_speedup,
        )

    def metrics(self) -> dict[str, Any]:
        return {
            "config": {
                "model_params": self._cfg.model_params,
                "denoising_steps": self._cfg.denoising_steps,
                "device": self._cfg.device,
            },
            "tracker": self._tracker.summary(),
        }
