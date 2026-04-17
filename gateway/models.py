"""Model-to-device assignment strategy.

Each inference request may declare a ``model_id`` and optional ``model_size``
hint.  The assigner maps these onto the preferred device type and ordered
backend list according to the following strategy:

Priority matrix
───────────────
VRAM requirement  →  preferred device
  ≥ 6 GiB           NVIDIA_GPU (RTX 5050, 8 GiB)
  2–6 GiB           AMD_GPU    (Radeon 780M, 4 GiB)
  < 2 GiB / CPU     CPU        (Ryzen 7)
  unspecified       NVIDIA_GPU → AMD_GPU → CPU (priority cascade)

Primary Brain override
──────────────────────
``nemotron-3-nano`` is designated the Primary Brain candidate (RES-05).
Despite its SMALL size bucket, it is always routed to ``NVIDIA_GPU`` first
(RTX 5050) to maximise throughput and reliability for the agent economy.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from gateway.config import BACKEND_PRIORITY, BACKENDS, BackendConfig, DeviceType

logger = logging.getLogger(__name__)


class ModelSize(str, Enum):
    """Coarse model-size buckets used to infer VRAM requirements."""

    SMALL = "small"    # < 2 GiB  → CPU-capable
    MEDIUM = "medium"  # 2–6 GiB  → AMD GPU or better
    LARGE = "large"    # ≥ 6 GiB  → NVIDIA GPU preferred


# VRAM thresholds in GiB
_MEDIUM_VRAM_GIB = 2.0
_LARGE_VRAM_GIB = 6.0

# Mapping from size bucket → ordered preference of device types
_SIZE_TO_DEVICE_PREFERENCE: dict[ModelSize, list[DeviceType]] = {
    ModelSize.LARGE: [DeviceType.NVIDIA_GPU, DeviceType.AMD_GPU, DeviceType.CPU],
    ModelSize.MEDIUM: [DeviceType.AMD_GPU, DeviceType.NVIDIA_GPU, DeviceType.CPU],
    ModelSize.SMALL: [DeviceType.CPU, DeviceType.AMD_GPU, DeviceType.NVIDIA_GPU],
}

# Primary Brain: model id routed exclusively to NVIDIA_GPU regardless of size bucket
_PRIMARY_BRAIN_MODEL = "nemotron-3-nano"

# Static model registry: model_id prefix → size bucket
_MODEL_REGISTRY: dict[str, ModelSize] = {
    # ── Terry's installed models (sovereign-core local cluster) ───────────────
    "gemma3": ModelSize.LARGE,           # 8.1GB → RTX 5050 primary
    "qwen2.5": ModelSize.MEDIUM,         # 4.7GB → Radeon 780M primary
    "llama3.2": ModelSize.SMALL,         # 2.0GB → Ryzen 7 CPU primary
    "dolphin-llama3": ModelSize.MEDIUM,  # 4.7GB → Radeon 780M / RTX 5050
    "dolphin-phi": ModelSize.SMALL,      # 1.6GB → CPU fast tasks
    "llama3": ModelSize.MEDIUM,          # 4.7GB → Radeon 780M fallback
    "nomic-embed-text": ModelSize.SMALL, # 274MB → embeddings any backend
    # ── Generic large LLMs ────────────────────────────────────────────────────
    "deepseek": ModelSize.LARGE,
    "llama-70b": ModelSize.LARGE,
    "llama-34b": ModelSize.LARGE,
    "mistral-7b": ModelSize.MEDIUM,
    "llama-7b": ModelSize.MEDIUM,
    "phi-3": ModelSize.SMALL,
    "tinyllama": ModelSize.SMALL,
    # Primary Brain candidate (RES-05) — nano model, NVIDIA-first routing
    "nemotron-3-nano": ModelSize.SMALL,
    # Embedding / reranking models
    "bge-large": ModelSize.MEDIUM,
    "bge-small": ModelSize.SMALL,
    "minilm": ModelSize.SMALL,
}


def infer_model_size(model_id: Optional[str], vram_required_gib: float = 0.0) -> ModelSize:
    """Determine a ``ModelSize`` from a model id or explicit VRAM requirement.

    Args:
        model_id: Optional model identifier string (case-insensitive prefix
            matching against the model registry).
        vram_required_gib: Explicit VRAM requirement hint supplied by the
            caller.  Takes precedence when non-zero.

    Returns:
        The inferred :class:`ModelSize` bucket.
    """
    if vram_required_gib >= _LARGE_VRAM_GIB:
        return ModelSize.LARGE
    if vram_required_gib >= _MEDIUM_VRAM_GIB:
        return ModelSize.MEDIUM
    if vram_required_gib > 0:
        return ModelSize.SMALL

    if model_id:
        key = model_id.lower()
        for prefix, size in _MODEL_REGISTRY.items():
            if key.startswith(prefix):
                return size

    return ModelSize.LARGE  # default: try the most capable device first


class ModelAssigner:
    """Maps inference requests to an ordered list of preferred backends.

    The caller is responsible for filtering this list against the currently
    healthy backends (see :class:`gateway.health.HealthMonitor`).
    """

    def __init__(self, backends: Optional[list[BackendConfig]] = None) -> None:
        self._backends: list[BackendConfig] = backends if backends is not None else BACKENDS
        # Build per-device-type index
        self._by_device: dict[DeviceType, list[BackendConfig]] = {}
        for b in self._backends:
            self._by_device.setdefault(b.device_type, []).append(b)
        # Sort each device bucket by descending weight (most capable first)
        for bucket in self._by_device.values():
            bucket.sort(key=lambda b: b.weight, reverse=True)

    def assign(
        self,
        model_id: Optional[str] = None,
        vram_required_gib: float = 0.0,
        device_hint: Optional[DeviceType] = None,
    ) -> list[BackendConfig]:
        """Return an ordered list of backends suitable for the request.

        Args:
            model_id: Optional model identifier used for size inference.
            vram_required_gib: Explicit VRAM requirement (GiB).  Non-zero
                values override model_id-based inference.
            device_hint: If set, place backends of this device type first.
                Applied after the Primary Brain override (see below).

        Returns:
            Ordered list of :class:`BackendConfig` from most to least preferred.
            Falls back through the full priority cascade so a result is always
            returned (the caller decides which backends are currently healthy).

        Note:
            Requests for the Primary Brain model (``nemotron-3-nano``) always
            prefer ``NVIDIA_GPU`` first, overriding the normal SMALL-model CPU
            preference.  An explicit ``device_hint`` can further adjust the
            order on top of this override.
        """
        size = infer_model_size(model_id, vram_required_gib)
        device_preference = _SIZE_TO_DEVICE_PREFERENCE[size]

        # Primary Brain override: nemotron-3-nano always runs on NVIDIA_GPU first
        # regardless of its SMALL size bucket, to maximise performance (RES-05).
        if model_id and model_id.lower().startswith(_PRIMARY_BRAIN_MODEL):
            device_preference = [DeviceType.NVIDIA_GPU] + [
                d for d in device_preference if d != DeviceType.NVIDIA_GPU
            ]

        if device_hint is not None and device_hint in {d for d in DeviceType}:
            # Promote hinted device type to the front
            device_preference = [device_hint] + [
                d for d in device_preference if d != device_hint
            ]

        ordered: list[BackendConfig] = []
        seen_ids: set[str] = set()
        for device_type in device_preference:
            for backend in self._by_device.get(device_type, []):
                if backend.id not in seen_ids:
                    ordered.append(backend)
                    seen_ids.add(backend.id)

        # Append any remaining backends not yet included (safety net)
        for backend_id in BACKEND_PRIORITY:
            if backend_id not in seen_ids:
                b = next((x for x in self._backends if x.id == backend_id), None)
                if b:
                    ordered.append(b)
                    seen_ids.add(backend_id)

        logger.debug(
            "ModelAssigner: model=%s size=%s preference=%s",
            model_id,
            size.value,
            [b.id for b in ordered],
        )
        return ordered
