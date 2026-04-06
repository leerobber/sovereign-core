"""Diffusion-based language model router prototype.

RES-10: WeDLM-8B — Diffusion Language Modeling for Micro-Models

Implements a 50 M-parameter-scale diffusion router prototype that generates all
tokens in parallel (via iterative denoising steps) rather than sequentially (one
token per autoregressive forward pass).

Architecture
────────────
Diffusion LMs denoise a full-length token sequence over T fixed steps, regardless
of the output length N.  Each step is a single batched forward pass over all N
positions simultaneously.  Autoregressive (AR) decoding requires N serial forward
passes — one per token.

The key efficiency advantage materialises through hardware utilisation:

* **AR** always executes at batch = 1 (one new token per pass), keeping the
  device permanently *memory-bandwidth-bound* (weight-loading bottleneck).
* **Diffusion** executes at batch = N; for large N the workload crosses the
  roofline into *compute-bound* territory, achieving near-peak GPU throughput.

For N > T tokens, the parallel mode has:

    latency_ratio ≈ T × step_lat_diff / (N × step_lat_ar)

Since T is a fixed constant (e.g. 10), diffusion latency is O(T) while AR is
O(N).  For long sequences the gap is substantial.

Primary figure of merit: **tokens-per-watt** (tokens generated per watt-second of
energy consumed).

Prototype scope
───────────────
* Model scale    : 50 M parameters
* Denoising steps: configurable (default 10)
* Device targets : ``nvidia_gpu`` | ``amd_gpu`` | ``cpu``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------

# Memory bandwidth (GB/s) — limits single-token AR forward-pass throughput
_DEVICE_BANDWIDTH_GBS: dict[str, float] = {
    "nvidia_gpu": 192.0,   # RTX 5050-class discrete GPU
    "amd_gpu":     51.0,   # Radeon 780M integrated GPU
    "cpu":         50.0,   # Ryzen 7 dual-channel DDR5
}

# Peak compute throughput (TFLOPS, FP16) — limits large-batch diffusion steps
_DEVICE_TFLOPS: dict[str, float] = {
    "nvidia_gpu": 10.0,
    "amd_gpu":     2.7,
    "cpu":         0.5,
}

# Sustained power draw during inference (watts)
_DEVICE_WATTS: dict[str, float] = {
    "nvidia_gpu": 25.0,
    "amd_gpu":    15.0,
    "cpu":        45.0,
}

_DEFAULT_DEVICE = "nvidia_gpu"

# Bytes per weight element (float32 prototype weights)
_BYTES_PER_PARAM: int = 4


# ---------------------------------------------------------------------------
# Public enumerations
# ---------------------------------------------------------------------------

class DecodeMode(str, Enum):
    """Token-generation strategy."""

    PARALLEL = "parallel"
    """Diffusion: all tokens denoised simultaneously across T fixed steps."""

    AUTOREGRESSIVE = "autoregressive"
    """Baseline: one token generated per forward pass (N passes for N tokens)."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DiffusionConfig:
    """Configuration for the 50 M-parameter diffusion router prototype.

    Attributes:
        model_params: Number of parameters in the micro-model.
        denoising_steps: Number of iterative denoising steps (T).
            Lower values → faster but potentially lower quality; typical range
            is 5–20 for micro-model experiments.
        device: Target device class for performance estimation.
            One of ``"nvidia_gpu"``, ``"amd_gpu"``, or ``"cpu"``.
    """

    model_params: int = 50_000_000
    denoising_steps: int = 10
    device: str = _DEFAULT_DEVICE


@dataclass
class GenerationResult:
    """Outcome of a single generation call.

    Attributes:
        mode: Decode mode used for this generation.
        tokens_generated: Number of tokens produced.
        latency_s: Estimated wall-clock latency in seconds.
        estimated_watts: Sustained device power draw during generation.
        tokens_per_second: Estimated throughput in tokens per second.
        tokens_per_watt: Efficiency figure of merit (tokens per watt-second).
        forward_passes: Number of model forward passes required.
        flops_total: Approximate total FLOPs consumed.
    """

    mode: DecodeMode
    tokens_generated: int
    latency_s: float
    estimated_watts: float
    tokens_per_second: float
    tokens_per_watt: float
    forward_passes: int
    flops_total: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "mode": self.mode.value,
            "tokens_generated": self.tokens_generated,
            "latency_s": round(self.latency_s, 6),
            "estimated_watts": round(self.estimated_watts, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "tokens_per_watt": round(self.tokens_per_watt, 4),
            "forward_passes": self.forward_passes,
            "flops_total": round(self.flops_total, 0),
        }


@dataclass
class ComparisonResult:
    """Head-to-head comparison of parallel vs autoregressive decoding.

    Attributes:
        parallel: Results from the diffusion (parallel) mode.
        autoregressive: Results from the autoregressive baseline.
        tokens_per_watt_speedup: Ratio of parallel tpw to AR tpw.
            Values > 1.0 indicate that parallel decoding is more efficient.
        latency_speedup: Ratio of AR latency to parallel latency.
            Values > 1.0 indicate that parallel decoding is faster.
    """

    parallel: GenerationResult
    autoregressive: GenerationResult
    tokens_per_watt_speedup: float
    latency_speedup: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "parallel": self.parallel.to_dict(),
            "autoregressive": self.autoregressive.to_dict(),
            "tokens_per_watt_speedup": round(self.tokens_per_watt_speedup, 4),
            "latency_speedup": round(self.latency_speedup, 4),
            "advantage": (
                "parallel" if self.tokens_per_watt_speedup > 1.0 else "autoregressive"
            ),
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class TokensPerWattTracker:
    """Accumulates tokens-per-watt statistics for each :class:`DecodeMode`.

    Maintains running cumulative totals so that average efficiency can be
    compared between parallel and autoregressive decoding across many calls.
    """

    def __init__(self) -> None:
        self._totals: dict[DecodeMode, dict[str, float]] = {
            m: {"tokens": 0.0, "watt_seconds": 0.0, "calls": 0.0}
            for m in DecodeMode
        }

    def record(self, result: GenerationResult) -> None:
        """Accumulate statistics from a completed generation."""
        b = self._totals[result.mode]
        b["tokens"] += result.tokens_generated
        b["watt_seconds"] += result.estimated_watts * result.latency_s
        b["calls"] += 1.0

    def average_tokens_per_watt(self, mode: DecodeMode) -> float:
        """Return the cumulative tokens-per-watt for *mode* (0.0 if no data)."""
        b = self._totals[mode]
        ws = b["watt_seconds"]
        return b["tokens"] / ws if ws > 0.0 else 0.0

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for all modes."""
        parallel_tpw = self.average_tokens_per_watt(DecodeMode.PARALLEL)
        ar_tpw = self.average_tokens_per_watt(DecodeMode.AUTOREGRESSIVE)
        ratio: Optional[float] = (parallel_tpw / ar_tpw) if ar_tpw > 0.0 else None
        return {
            "parallel": {
                "total_tokens": int(self._totals[DecodeMode.PARALLEL]["tokens"]),
                "total_calls": int(self._totals[DecodeMode.PARALLEL]["calls"]),
                "avg_tokens_per_watt": round(parallel_tpw, 4),
            },
            "autoregressive": {
                "total_tokens": int(self._totals[DecodeMode.AUTOREGRESSIVE]["tokens"]),
                "total_calls": int(self._totals[DecodeMode.AUTOREGRESSIVE]["calls"]),
                "avg_tokens_per_watt": round(ar_tpw, 4),
            },
            "tokens_per_watt_ratio": round(ratio, 4) if ratio is not None else None,
        }


# ---------------------------------------------------------------------------
# Core router
# ---------------------------------------------------------------------------

class DiffusionRouter:
    """50 M-parameter diffusion LM router prototype (RES-10).

    Models the compute cost of diffusion-based (parallel) vs autoregressive
    token generation using a **roofline model**:

    * **Memory-bound** (small batch / single-token AR): latency is dominated by
      loading model weights from DRAM each forward pass.
    * **Compute-bound** (large-batch diffusion): latency is dominated by the
      arithmetic throughput of the accelerator.

    Autoregressive decoding always runs at batch = 1 (one new token per forward
    pass), so it is permanently memory-bandwidth-bound.  Parallel (diffusion)
    decoding runs at batch = N, crossing into compute-bound territory for
    large N — which is precisely where the efficiency gain materialises.

    For N > T (sequence length greater than denoising steps), parallel decoding
    is always faster and achieves higher tokens-per-watt on GPU devices.

    Parameters
    ----------
    config:
        Model and device configuration.  Defaults to a 50 M-param model on
        the NVIDIA GPU device class.
    """

    def __init__(self, config: Optional[DiffusionConfig] = None) -> None:
        self._cfg = config or DiffusionConfig()
        self._tracker = TokensPerWattTracker()
        device = self._cfg.device
        self._bandwidth_gbs = _DEVICE_BANDWIDTH_GBS.get(
            device, _DEVICE_BANDWIDTH_GBS[_DEFAULT_DEVICE]
        )
        self._peak_tflops = _DEVICE_TFLOPS.get(device, _DEVICE_TFLOPS[_DEFAULT_DEVICE])
        self._watts = _DEVICE_WATTS.get(device, _DEVICE_WATTS[_DEFAULT_DEVICE])
        self._model_bytes: float = float(self._cfg.model_params * _BYTES_PER_PARAM)

    @property
    def device(self) -> str:
        """Target device class for this router instance."""
        return self._cfg.device

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_latency_s(self, batch_tokens: int) -> float:
        """Estimate latency for one forward pass over *batch_tokens* tokens.

        Uses a roofline model: a forward pass is either memory-bandwidth-limited
        (weight loading dominates when batch is small) or compute-limited
        (arithmetic throughput dominates when batch is large).

        Args:
            batch_tokens: Number of tokens processed in this forward pass.

        Returns:
            Estimated latency in seconds.
        """
        # Memory-bound floor: load all model weights once per forward pass
        mem_latency_s = self._model_bytes / (self._bandwidth_gbs * 1e9)
        # Compute-bound ceiling: 2 FLOPs per parameter per token (matmul)
        flops = 2.0 * self._cfg.model_params * batch_tokens
        compute_latency_s = flops / (self._peak_tflops * 1e12)
        return max(mem_latency_s, compute_latency_s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, num_tokens: int, mode: DecodeMode) -> GenerationResult:
        """Simulate token generation and return performance metrics.

        Args:
            num_tokens: Number of tokens to generate.
            mode: :attr:`DecodeMode.PARALLEL` for diffusion decoding;
                :attr:`DecodeMode.AUTOREGRESSIVE` for the sequential baseline.

        Returns:
            A :class:`GenerationResult` populated with latency, power, and
            tokens-per-watt estimates.

        Raises:
            ValueError: If *num_tokens* is not a positive integer.
        """
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens!r}")

        flops_per_token = 2.0 * self._cfg.model_params

        if mode == DecodeMode.PARALLEL:
            # T denoising steps; each step processes all N tokens in one batched
            # forward pass — full parallelism across the sequence length.
            forward_passes = self._cfg.denoising_steps
            step_lat = self._step_latency_s(num_tokens)
            latency_s = self._cfg.denoising_steps * step_lat
            total_flops = flops_per_token * num_tokens * self._cfg.denoising_steps
        else:
            # One forward pass per token; batch size is always 1, so the pass
            # is permanently memory-bandwidth-bound.
            forward_passes = num_tokens
            step_lat = self._step_latency_s(1)
            latency_s = num_tokens * step_lat
            total_flops = flops_per_token * num_tokens

        latency_s = max(latency_s, 1e-9)
        tokens_per_second = num_tokens / latency_s
        tokens_per_watt = tokens_per_second / self._watts

        result = GenerationResult(
            mode=mode,
            tokens_generated=num_tokens,
            latency_s=latency_s,
            estimated_watts=self._watts,
            tokens_per_second=tokens_per_second,
            tokens_per_watt=tokens_per_watt,
            forward_passes=forward_passes,
            flops_total=total_flops,
        )
        self._tracker.record(result)
        logger.debug(
            "DiffusionRouter.generate mode=%s tokens=%d latency=%.6fs tpw=%.2f",
            mode.value,
            num_tokens,
            latency_s,
            tokens_per_watt,
        )
        return result

    def compare(self, num_tokens: int) -> ComparisonResult:
        """Run both decode modes and return a head-to-head comparison.

        Args:
            num_tokens: Sequence length used for both generation calls.

        Returns:
            A :class:`ComparisonResult` summarising the relative advantage of
            parallel vs autoregressive decoding at this sequence length.
        """
        parallel = self.generate(num_tokens, DecodeMode.PARALLEL)
        autoregressive = self.generate(num_tokens, DecodeMode.AUTOREGRESSIVE)

        tpw_speedup = (
            parallel.tokens_per_watt / autoregressive.tokens_per_watt
            if autoregressive.tokens_per_watt > 0.0
            else 0.0
        )
        latency_speedup = (
            autoregressive.latency_s / parallel.latency_s
            if parallel.latency_s > 0.0
            else 0.0
        )

        logger.info(
            "DiffusionRouter.compare tokens=%d tpw_speedup=%.2f× latency_speedup=%.2f×",
            num_tokens,
            tpw_speedup,
            latency_speedup,
        )
        return ComparisonResult(
            parallel=parallel,
            autoregressive=autoregressive,
            tokens_per_watt_speedup=tpw_speedup,
            latency_speedup=latency_speedup,
        )

    def metrics(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of accumulated statistics."""
        return {
            "config": {
                "model_params": self._cfg.model_params,
                "denoising_steps": self._cfg.denoising_steps,
                "device": self._cfg.device,
            },
            "tracker": self._tracker.summary(),
        }
