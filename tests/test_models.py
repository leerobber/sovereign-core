"""Tests for model-to-device assignment strategy."""

from __future__ import annotations

import pytest

from gateway.config import BackendConfig, DeviceType
from gateway.models import (
    ModelAssigner,
    ModelSize,
    _LARGE_VRAM_GIB,
    _MEDIUM_VRAM_GIB,
    _PRIMARY_BRAIN_MODEL,
    infer_model_size,
)


# ---------------------------------------------------------------------------
# infer_model_size
# ---------------------------------------------------------------------------
class TestInferModelSize:
    def test_explicit_large_vram(self):
        assert infer_model_size(None, vram_required_gib=_LARGE_VRAM_GIB) == ModelSize.LARGE

    def test_explicit_medium_vram(self):
        assert infer_model_size(None, vram_required_gib=_MEDIUM_VRAM_GIB) == ModelSize.MEDIUM

    def test_explicit_small_vram(self):
        assert infer_model_size(None, vram_required_gib=0.5) == ModelSize.SMALL

    def test_vram_takes_precedence_over_model_id(self):
        # deepseek would normally be LARGE, but explicit small VRAM overrides
        assert infer_model_size("deepseek", vram_required_gib=0.5) == ModelSize.SMALL

    def test_model_id_deepseek(self):
        assert infer_model_size("deepseek-coder-33b") == ModelSize.LARGE

    def test_model_id_tinyllama(self):
        assert infer_model_size("tinyllama-1b") == ModelSize.SMALL

    def test_model_id_mistral(self):
        assert infer_model_size("mistral-7b-instruct") == ModelSize.MEDIUM

    def test_model_id_bge_small(self):
        assert infer_model_size("bge-small-en-v1.5") == ModelSize.SMALL

    def test_model_id_bge_large(self):
        assert infer_model_size("bge-large-en-v1.5") == ModelSize.MEDIUM

    def test_unknown_model_defaults_to_large(self):
        assert infer_model_size("totally-unknown-model-xyz") == ModelSize.LARGE

    def test_no_model_no_vram_defaults_to_large(self):
        assert infer_model_size(None) == ModelSize.LARGE

    def test_case_insensitive_lookup(self):
        assert infer_model_size("DeepSeek-v3") == ModelSize.LARGE


# ---------------------------------------------------------------------------
# ModelAssigner
# ---------------------------------------------------------------------------
class TestModelAssigner:
    def _make_assigner(self) -> ModelAssigner:
        return ModelAssigner()  # uses default BACKENDS

    def test_large_model_prefers_nvidia(self):
        assigner = self._make_assigner()
        result = assigner.assign(model_id="deepseek-v3")
        assert result[0].device_type == DeviceType.NVIDIA_GPU

    def test_small_model_prefers_cpu(self):
        assigner = self._make_assigner()
        result = assigner.assign(model_id="tinyllama-1b")
        assert result[0].device_type == DeviceType.CPU

    def test_medium_model_prefers_amd(self):
        assigner = self._make_assigner()
        result = assigner.assign(model_id="mistral-7b")
        assert result[0].device_type == DeviceType.AMD_GPU

    def test_all_backends_included_in_fallback(self):
        assigner = self._make_assigner()
        result = assigner.assign(model_id="deepseek-v3")
        assert len(result) == 3

    def test_no_duplicates_in_result(self):
        assigner = self._make_assigner()
        result = assigner.assign(model_id="deepseek-v3")
        ids = [b.id for b in result]
        assert len(ids) == len(set(ids))

    def test_device_hint_promotes_to_front(self):
        assigner = self._make_assigner()
        # Large model would normally prefer NVIDIA, but hint for CPU
        result = assigner.assign(model_id="deepseek-v3", device_hint=DeviceType.CPU)
        assert result[0].device_type == DeviceType.CPU

    def test_vram_large_prefers_nvidia(self):
        assigner = self._make_assigner()
        result = assigner.assign(vram_required_gib=7.0)
        assert result[0].device_type == DeviceType.NVIDIA_GPU

    def test_vram_medium_prefers_amd(self):
        assigner = self._make_assigner()
        result = assigner.assign(vram_required_gib=3.0)
        assert result[0].device_type == DeviceType.AMD_GPU

    def test_custom_backends(self):
        custom_backends = [
            BackendConfig(
                id="cpu1",
                url="http://localhost:9003",
                device_type=DeviceType.CPU,
                weight=1.0,
            )
        ]
        assigner = ModelAssigner(backends=custom_backends)
        result = assigner.assign(model_id="tinyllama")
        assert len(result) == 1
        assert result[0].id == "cpu1"

    def test_weight_ordering_within_same_device_type(self):
        backends = [
            BackendConfig(
                id="gpu_low",
                url="http://localhost:9001",
                device_type=DeviceType.NVIDIA_GPU,
                weight=1.0,
            ),
            BackendConfig(
                id="gpu_high",
                url="http://localhost:9002",
                device_type=DeviceType.NVIDIA_GPU,
                weight=5.0,
            ),
        ]
        assigner = ModelAssigner(backends=backends)
        result = assigner.assign(model_id="deepseek-v3")
        # Higher weight should be first
        assert result[0].id == "gpu_high"


# ---------------------------------------------------------------------------
# Primary Brain (nemotron-3-nano) — RES-05
# ---------------------------------------------------------------------------
class TestNemotronPrimaryBrain:
    """Validate that nemotron-3-nano is registered as SMALL and always
    routes to NVIDIA_GPU first regardless of its size bucket."""

    def test_nemotron_registered_as_small(self):
        assert infer_model_size(_PRIMARY_BRAIN_MODEL) == ModelSize.SMALL

    def test_nemotron_registered_as_small_case_insensitive(self):
        assert infer_model_size("Nemotron-3-Nano") == ModelSize.SMALL

    def test_nemotron_with_suffix_registered_as_small(self):
        assert infer_model_size("nemotron-3-nano-8b") == ModelSize.SMALL

    def test_primary_brain_constant_value(self):
        assert _PRIMARY_BRAIN_MODEL == "nemotron-3-nano"

    def test_nemotron_assign_prefers_nvidia(self):
        assigner = ModelAssigner()
        result = assigner.assign(model_id="nemotron-3-nano")
        assert result[0].device_type == DeviceType.NVIDIA_GPU

    def test_nemotron_assign_prefers_nvidia_with_suffix(self):
        assigner = ModelAssigner()
        result = assigner.assign(model_id="nemotron-3-nano-instruct")
        assert result[0].device_type == DeviceType.NVIDIA_GPU

    def test_nemotron_assign_prefers_nvidia_case_insensitive(self):
        assigner = ModelAssigner()
        result = assigner.assign(model_id="Nemotron-3-Nano")
        assert result[0].device_type == DeviceType.NVIDIA_GPU

    def test_nemotron_assign_all_backends_included(self):
        assigner = ModelAssigner()
        result = assigner.assign(model_id="nemotron-3-nano")
        assert len(result) == 3

    def test_nemotron_assign_no_duplicates(self):
        assigner = ModelAssigner()
        result = assigner.assign(model_id="nemotron-3-nano")
        ids = [b.id for b in result]
        assert len(ids) == len(set(ids))

    def test_nemotron_device_hint_still_overrides(self):
        """An explicit device_hint should still override the Primary Brain preference."""
        assigner = ModelAssigner()
        result = assigner.assign(model_id="nemotron-3-nano", device_hint=DeviceType.CPU)
        assert result[0].device_type == DeviceType.CPU

    def test_nemotron_differs_from_plain_small_model(self):
        """Unlike other SMALL models (CPU-first), nemotron-3-nano prefers NVIDIA."""
        assigner = ModelAssigner()
        tinyllama_result = assigner.assign(model_id="tinyllama")
        nemotron_result = assigner.assign(model_id="nemotron-3-nano")
        assert tinyllama_result[0].device_type == DeviceType.CPU
        assert nemotron_result[0].device_type == DeviceType.NVIDIA_GPU
