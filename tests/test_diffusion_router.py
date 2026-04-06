"""Tests for the diffusion-based language model router prototype (RES-10)."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from gateway.diffusion_router import (
    ComparisonResult,
    DecodeMode,
    DiffusionConfig,
    DiffusionRouter,
    GenerationResult,
    TokensPerWattTracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    mode: DecodeMode = DecodeMode.PARALLEL,
    tokens: int = 128,
    latency: float = 0.01,
    watts: float = 25.0,
) -> GenerationResult:
    tps = tokens / latency
    return GenerationResult(
        mode=mode,
        tokens_generated=tokens,
        latency_s=latency,
        estimated_watts=watts,
        tokens_per_second=tps,
        tokens_per_watt=tps / watts,
        forward_passes=10 if mode == DecodeMode.PARALLEL else tokens,
        flops_total=1e9,
    )


def _make_comparison(tpw_speedup: float = 2.0) -> ComparisonResult:
    p = _make_result(DecodeMode.PARALLEL, tokens=256, latency=0.01)
    a = _make_result(DecodeMode.AUTOREGRESSIVE, tokens=256, latency=0.02)
    return ComparisonResult(
        parallel=p,
        autoregressive=a,
        tokens_per_watt_speedup=tpw_speedup,
        latency_speedup=2.0,
    )


# ---------------------------------------------------------------------------
# DecodeMode
# ---------------------------------------------------------------------------

class TestDecodeMode:
    def test_parallel_value(self):
        assert DecodeMode.PARALLEL.value == "parallel"

    def test_autoregressive_value(self):
        assert DecodeMode.AUTOREGRESSIVE.value == "autoregressive"

    def test_string_equality(self):
        assert DecodeMode.PARALLEL == "parallel"
        assert DecodeMode.AUTOREGRESSIVE == "autoregressive"

    def test_all_members(self):
        members = {m.value for m in DecodeMode}
        assert members == {"parallel", "autoregressive"}


# ---------------------------------------------------------------------------
# DiffusionConfig
# ---------------------------------------------------------------------------

class TestDiffusionConfig:
    def test_defaults(self):
        cfg = DiffusionConfig()
        assert cfg.model_params == 50_000_000
        assert cfg.denoising_steps == 10
        assert cfg.device == "nvidia_gpu"

    def test_custom_params(self):
        cfg = DiffusionConfig(model_params=10_000_000, denoising_steps=5, device="cpu")
        assert cfg.model_params == 10_000_000
        assert cfg.denoising_steps == 5
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------

class TestGenerationResult:
    def test_to_dict_keys(self):
        d = _make_result().to_dict()
        expected = {
            "mode", "tokens_generated", "latency_s", "estimated_watts",
            "tokens_per_second", "tokens_per_watt", "forward_passes", "flops_total",
        }
        assert set(d.keys()) == expected

    def test_to_dict_mode_is_string(self):
        assert _make_result(DecodeMode.AUTOREGRESSIVE).to_dict()["mode"] == "autoregressive"
        assert _make_result(DecodeMode.PARALLEL).to_dict()["mode"] == "parallel"

    def test_to_dict_token_count(self):
        d = _make_result(tokens=64).to_dict()
        assert d["tokens_generated"] == 64

    def test_to_dict_latency_rounded(self):
        d = _make_result(latency=0.123456789).to_dict()
        # Rounded to 6 decimal places
        assert isinstance(d["latency_s"], float)


# ---------------------------------------------------------------------------
# ComparisonResult
# ---------------------------------------------------------------------------

class TestComparisonResult:
    def test_to_dict_structure(self):
        d = _make_comparison().to_dict()
        assert "parallel" in d
        assert "autoregressive" in d
        assert "tokens_per_watt_speedup" in d
        assert "latency_speedup" in d
        assert "advantage" in d

    def test_advantage_parallel_when_speedup_gt_1(self):
        assert _make_comparison(tpw_speedup=2.0).to_dict()["advantage"] == "parallel"

    def test_advantage_autoregressive_when_speedup_lt_1(self):
        assert _make_comparison(tpw_speedup=0.5).to_dict()["advantage"] == "autoregressive"

    def test_advantage_autoregressive_at_exactly_1(self):
        # Exactly 1.0 is not > 1.0, so autoregressive wins the tie
        assert _make_comparison(tpw_speedup=1.0).to_dict()["advantage"] == "autoregressive"

    def test_nested_dicts_are_dicts(self):
        d = _make_comparison().to_dict()
        assert isinstance(d["parallel"], dict)
        assert isinstance(d["autoregressive"], dict)


# ---------------------------------------------------------------------------
# TokensPerWattTracker
# ---------------------------------------------------------------------------

class TestTokensPerWattTracker:
    def test_initial_zero_for_parallel(self):
        assert TokensPerWattTracker().average_tokens_per_watt(DecodeMode.PARALLEL) == 0.0

    def test_initial_zero_for_autoregressive(self):
        assert TokensPerWattTracker().average_tokens_per_watt(DecodeMode.AUTOREGRESSIVE) == 0.0

    def test_record_single_parallel(self):
        tracker = TokensPerWattTracker()
        # 100 tokens / (25W * 0.01s) = 400 tpw
        tracker.record(_make_result(DecodeMode.PARALLEL, tokens=100, latency=0.01, watts=25.0))
        assert tracker.average_tokens_per_watt(DecodeMode.PARALLEL) == pytest.approx(400.0)

    def test_record_multiple_cumulative(self):
        tracker = TokensPerWattTracker()
        tracker.record(_make_result(DecodeMode.PARALLEL, tokens=100, latency=0.01, watts=25.0))
        tracker.record(_make_result(DecodeMode.PARALLEL, tokens=200, latency=0.02, watts=25.0))
        # Cumulative: 300 tokens / (25W * 0.03s) = 400 tpw
        assert tracker.average_tokens_per_watt(DecodeMode.PARALLEL) == pytest.approx(400.0)

    def test_modes_are_independent(self):
        tracker = TokensPerWattTracker()
        tracker.record(_make_result(DecodeMode.PARALLEL, tokens=100, latency=0.01, watts=25.0))
        assert tracker.average_tokens_per_watt(DecodeMode.AUTOREGRESSIVE) == 0.0

    def test_summary_structure(self):
        s = TokensPerWattTracker().summary()
        assert "parallel" in s
        assert "autoregressive" in s
        assert "tokens_per_watt_ratio" in s

    def test_summary_ratio_none_when_no_ar_data(self):
        tracker = TokensPerWattTracker()
        tracker.record(_make_result(DecodeMode.PARALLEL, tokens=100, latency=0.01))
        assert tracker.summary()["tokens_per_watt_ratio"] is None

    def test_summary_ratio_computed(self):
        tracker = TokensPerWattTracker()
        # Parallel: 200 / (25 * 0.01) = 800 tpw
        tracker.record(_make_result(DecodeMode.PARALLEL, tokens=200, latency=0.01, watts=25.0))
        # AR: 100 / (25 * 0.01) = 400 tpw
        tracker.record(_make_result(DecodeMode.AUTOREGRESSIVE, tokens=100, latency=0.01, watts=25.0))
        assert tracker.summary()["tokens_per_watt_ratio"] == pytest.approx(2.0)

    def test_summary_total_tokens_counted(self):
        tracker = TokensPerWattTracker()
        tracker.record(_make_result(DecodeMode.PARALLEL, tokens=50, latency=0.01))
        tracker.record(_make_result(DecodeMode.PARALLEL, tokens=100, latency=0.01))
        s = tracker.summary()
        assert s["parallel"]["total_tokens"] == 150
        assert s["parallel"]["total_calls"] == 2

    def test_summary_zero_calls_initially(self):
        s = TokensPerWattTracker().summary()
        assert s["parallel"]["total_calls"] == 0
        assert s["autoregressive"]["total_calls"] == 0


# ---------------------------------------------------------------------------
# DiffusionRouter — generate()
# ---------------------------------------------------------------------------

class TestDiffusionRouterGenerate:
    def setup_method(self) -> None:
        self.router = DiffusionRouter(
            DiffusionConfig(model_params=50_000_000, denoising_steps=10, device="nvidia_gpu")
        )

    def test_invalid_num_tokens_zero(self):
        with pytest.raises(ValueError, match="num_tokens"):
            self.router.generate(0, DecodeMode.PARALLEL)

    def test_invalid_num_tokens_negative(self):
        with pytest.raises(ValueError):
            self.router.generate(-5, DecodeMode.AUTOREGRESSIVE)

    def test_parallel_returns_generation_result(self):
        result = self.router.generate(256, DecodeMode.PARALLEL)
        assert isinstance(result, GenerationResult)
        assert result.mode == DecodeMode.PARALLEL
        assert result.tokens_generated == 256

    def test_autoregressive_returns_generation_result(self):
        result = self.router.generate(256, DecodeMode.AUTOREGRESSIVE)
        assert isinstance(result, GenerationResult)
        assert result.mode == DecodeMode.AUTOREGRESSIVE

    def test_parallel_forward_passes_equals_denoising_steps(self):
        result = self.router.generate(256, DecodeMode.PARALLEL)
        assert result.forward_passes == 10

    def test_autoregressive_forward_passes_equals_num_tokens(self):
        result = self.router.generate(256, DecodeMode.AUTOREGRESSIVE)
        assert result.forward_passes == 256

    def test_positive_metrics_parallel(self):
        result = self.router.generate(64, DecodeMode.PARALLEL)
        assert result.latency_s > 0
        assert result.tokens_per_second > 0
        assert result.tokens_per_watt > 0

    def test_positive_metrics_autoregressive(self):
        result = self.router.generate(64, DecodeMode.AUTOREGRESSIVE)
        assert result.latency_s > 0
        assert result.tokens_per_second > 0
        assert result.tokens_per_watt > 0

    def test_parallel_faster_than_ar_for_large_n(self):
        """Parallel decoding must be faster than AR when N >> denoising_steps."""
        parallel = self.router.generate(512, DecodeMode.PARALLEL)
        ar = self.router.generate(512, DecodeMode.AUTOREGRESSIVE)
        assert parallel.latency_s < ar.latency_s

    def test_parallel_higher_tpw_for_large_n(self):
        """Parallel decoding must achieve higher tokens-per-watt for large N."""
        parallel = self.router.generate(512, DecodeMode.PARALLEL)
        ar = self.router.generate(512, DecodeMode.AUTOREGRESSIVE)
        assert parallel.tokens_per_watt > ar.tokens_per_watt

    def test_parallel_flops_equal_steps_times_ar_flops(self):
        """Diffusion uses denoising_steps × the FLOPs of a single AR generation."""
        parallel = self.router.generate(100, DecodeMode.PARALLEL)
        ar = self.router.generate(100, DecodeMode.AUTOREGRESSIVE)
        assert parallel.flops_total == pytest.approx(ar.flops_total * 10)

    def test_watts_matches_nvidia_device(self):
        result = self.router.generate(100, DecodeMode.PARALLEL)
        assert result.estimated_watts == pytest.approx(25.0)

    def test_tracker_updated_after_parallel_generate(self):
        self.router.generate(128, DecodeMode.PARALLEL)
        m = self.router.metrics()
        assert m["tracker"]["parallel"]["total_calls"] == 1
        assert m["tracker"]["parallel"]["total_tokens"] == 128

    def test_tracker_updated_after_ar_generate(self):
        self.router.generate(64, DecodeMode.AUTOREGRESSIVE)
        m = self.router.metrics()
        assert m["tracker"]["autoregressive"]["total_calls"] == 1
        assert m["tracker"]["autoregressive"]["total_tokens"] == 64

    def test_single_token_parallel_and_ar_both_succeed(self):
        """Edge case: generating exactly 1 token must not raise."""
        r_par = self.router.generate(1, DecodeMode.PARALLEL)
        r_ar = self.router.generate(1, DecodeMode.AUTOREGRESSIVE)
        assert r_par.tokens_generated == 1
        assert r_ar.tokens_generated == 1

    def test_tokens_per_second_consistent_with_latency(self):
        result = self.router.generate(256, DecodeMode.PARALLEL)
        expected_tps = result.tokens_generated / result.latency_s
        assert result.tokens_per_second == pytest.approx(expected_tps, rel=1e-5)

    def test_tokens_per_watt_consistent_with_tps_and_watts(self):
        result = self.router.generate(256, DecodeMode.AUTOREGRESSIVE)
        expected_tpw = result.tokens_per_second / result.estimated_watts
        assert result.tokens_per_watt == pytest.approx(expected_tpw, rel=1e-5)


# ---------------------------------------------------------------------------
# DiffusionRouter — compare()
# ---------------------------------------------------------------------------

class TestDiffusionRouterCompare:
    def setup_method(self) -> None:
        self.router = DiffusionRouter()

    def test_compare_returns_comparison_result(self):
        assert isinstance(self.router.compare(256), ComparisonResult)

    def test_compare_modes_present(self):
        result = self.router.compare(256)
        assert result.parallel.mode == DecodeMode.PARALLEL
        assert result.autoregressive.mode == DecodeMode.AUTOREGRESSIVE

    def test_compare_same_token_count(self):
        result = self.router.compare(128)
        assert result.parallel.tokens_generated == 128
        assert result.autoregressive.tokens_generated == 128

    def test_parallel_advantage_for_large_n(self):
        """Parallel must have tpw_speedup > 1 for N >> denoising_steps."""
        result = self.router.compare(512)
        assert result.tokens_per_watt_speedup > 1.0

    def test_latency_speedup_positive(self):
        assert self.router.compare(256).latency_speedup > 0.0

    def test_to_dict_structure(self):
        d = self.router.compare(256).to_dict()
        assert "parallel" in d
        assert "autoregressive" in d
        assert "tokens_per_watt_speedup" in d
        assert "latency_speedup" in d
        assert "advantage" in d

    def test_tracker_updated_for_both_modes(self):
        self.router.compare(128)
        m = self.router.metrics()
        assert m["tracker"]["parallel"]["total_calls"] == 1
        assert m["tracker"]["autoregressive"]["total_calls"] == 1

    def test_tpw_speedup_consistent_with_individual_calls(self):
        par = self.router.generate(200, DecodeMode.PARALLEL)
        ar = self.router.generate(200, DecodeMode.AUTOREGRESSIVE)
        expected = par.tokens_per_watt / ar.tokens_per_watt if ar.tokens_per_watt > 0 else 0.0

        # Fresh router for clean compare call
        router2 = DiffusionRouter()
        cmp = router2.compare(200)
        assert cmp.tokens_per_watt_speedup == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# DiffusionRouter — metrics()
# ---------------------------------------------------------------------------

class TestDiffusionRouterMetrics:
    def test_metrics_structure(self):
        m = DiffusionRouter().metrics()
        assert "config" in m
        assert "tracker" in m

    def test_config_reflects_defaults(self):
        cfg = DiffusionRouter().metrics()["config"]
        assert cfg["model_params"] == 50_000_000
        assert cfg["denoising_steps"] == 10
        assert cfg["device"] == "nvidia_gpu"

    def test_config_reflects_custom(self):
        router = DiffusionRouter(DiffusionConfig(denoising_steps=5, device="cpu"))
        cfg = router.metrics()["config"]
        assert cfg["denoising_steps"] == 5
        assert cfg["device"] == "cpu"

    def test_initial_tracker_zeros(self):
        t = DiffusionRouter().metrics()["tracker"]
        assert t["parallel"]["total_calls"] == 0
        assert t["autoregressive"]["total_calls"] == 0
        assert t["tokens_per_watt_ratio"] is None


# ---------------------------------------------------------------------------
# DiffusionRouter — device property
# ---------------------------------------------------------------------------

class TestDiffusionRouterDevice:
    def test_device_property_default(self):
        assert DiffusionRouter().device == "nvidia_gpu"

    def test_device_property_cpu(self):
        assert DiffusionRouter(DiffusionConfig(device="cpu")).device == "cpu"

    def test_device_property_amd(self):
        assert DiffusionRouter(DiffusionConfig(device="amd_gpu")).device == "amd_gpu"


# ---------------------------------------------------------------------------
# DiffusionRouter — device variations
# ---------------------------------------------------------------------------

class TestDiffusionRouterDeviceVariations:
    def test_cpu_watts(self):
        result = DiffusionRouter(DiffusionConfig(device="cpu")).generate(64, DecodeMode.PARALLEL)
        assert result.estimated_watts == pytest.approx(45.0)

    def test_amd_gpu_watts(self):
        result = DiffusionRouter(DiffusionConfig(device="amd_gpu")).generate(64, DecodeMode.PARALLEL)
        assert result.estimated_watts == pytest.approx(15.0)

    def test_nvidia_gpu_watts(self):
        result = DiffusionRouter(DiffusionConfig(device="nvidia_gpu")).generate(64, DecodeMode.PARALLEL)
        assert result.estimated_watts == pytest.approx(25.0)

    def test_unknown_device_falls_back_to_nvidia_watts(self):
        result = DiffusionRouter(DiffusionConfig(device="unknown_device")).generate(64, DecodeMode.PARALLEL)
        assert result.estimated_watts == pytest.approx(25.0)

    def test_parallel_advantage_holds_on_cpu(self):
        router = DiffusionRouter(DiffusionConfig(device="cpu"))
        cmp = router.compare(512)
        assert cmp.tokens_per_watt_speedup > 1.0

    def test_parallel_advantage_holds_on_amd_gpu(self):
        router = DiffusionRouter(DiffusionConfig(device="amd_gpu"))
        cmp = router.compare(512)
        assert cmp.tokens_per_watt_speedup > 1.0


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

@pytest.fixture
def diffusion_app():
    """Return the FastAPI app with _diffusion_router initialised (no lifespan)."""
    import gateway.main as gm

    gm._diffusion_router = DiffusionRouter()

    original_lifespan = gm.app.router.lifespan_context

    @asynccontextmanager
    async def _noop(_app):  # type: ignore[override]
        yield

    gm.app.router.lifespan_context = _noop
    yield gm.app
    gm.app.router.lifespan_context = original_lifespan


class TestDiffusionMetricsEndpoint:
    def test_returns_200(self, diffusion_app):
        with TestClient(diffusion_app, raise_server_exceptions=False) as client:
            resp = client.get("/diffusion/metrics")
        assert resp.status_code == 200

    def test_json_structure(self, diffusion_app):
        with TestClient(diffusion_app, raise_server_exceptions=False) as client:
            data = client.get("/diffusion/metrics").json()
        assert "config" in data
        assert "tracker" in data

    def test_config_fields_present(self, diffusion_app):
        with TestClient(diffusion_app, raise_server_exceptions=False) as client:
            cfg = client.get("/diffusion/metrics").json()["config"]
        assert "model_params" in cfg
        assert "denoising_steps" in cfg
        assert "device" in cfg


class TestDiffusionGenerateEndpoint:
    def test_returns_200_default_params(self, diffusion_app):
        with TestClient(diffusion_app, raise_server_exceptions=False) as client:
            resp = client.post("/diffusion/generate")
        assert resp.status_code == 200

    def test_json_structure(self, diffusion_app):
        with TestClient(diffusion_app, raise_server_exceptions=False) as client:
            data = client.post("/diffusion/generate").json()
        assert "mode" in data
        assert "tokens_generated" in data
        assert "tokens_per_watt" in data
        assert "latency_s" in data

    def test_parallel_mode(self, diffusion_app):
        with TestClient(diffusion_app, raise_server_exceptions=False) as client:
            data = client.post(
                "/diffusion/generate", params={"num_tokens": 128, "mode": "parallel"}
            ).json()
        assert data["mode"] == "parallel"
        assert data["tokens_generated"] == 128

    def test_autoregressive_mode(self, diffusion_app):
        with TestClient(diffusion_app, raise_server_exceptions=False) as client:
            data = client.post(
                "/diffusion/generate",
                params={"num_tokens": 64, "mode": "autoregressive"},
            ).json()
        assert data["mode"] == "autoregressive"

    def test_cpu_device(self, diffusion_app):
        with TestClient(diffusion_app, raise_server_exceptions=False) as client:
            data = client.post(
                "/diffusion/generate", params={"device": "cpu", "num_tokens": 64}
            ).json()
        assert data["estimated_watts"] == pytest.approx(45.0)

    def test_amd_device(self, diffusion_app):
        with TestClient(diffusion_app, raise_server_exceptions=False) as client:
            data = client.post(
                "/diffusion/generate", params={"device": "amd_gpu", "num_tokens": 64}
            ).json()
        assert data["estimated_watts"] == pytest.approx(15.0)


class TestDiffusionCompareEndpoint:
    def test_returns_200(self, diffusion_app):
        with TestClient(diffusion_app, raise_server_exceptions=False) as client:
            resp = client.post("/diffusion/compare")
        assert resp.status_code == 200

    def test_json_structure(self, diffusion_app):
        with TestClient(diffusion_app, raise_server_exceptions=False) as client:
            data = client.post("/diffusion/compare").json()
        assert "parallel" in data
        assert "autoregressive" in data
        assert "tokens_per_watt_speedup" in data
        assert "latency_speedup" in data
        assert "advantage" in data

    def test_parallel_advantage_large_n(self, diffusion_app):
        with TestClient(diffusion_app, raise_server_exceptions=False) as client:
            data = client.post(
                "/diffusion/compare", params={"num_tokens": 512}
            ).json()
        assert data["advantage"] == "parallel"
        assert data["tokens_per_watt_speedup"] > 1.0

    def test_cpu_device(self, diffusion_app):
        with TestClient(diffusion_app, raise_server_exceptions=False) as client:
            data = client.post(
                "/diffusion/compare", params={"device": "cpu", "num_tokens": 256}
            ).json()
        assert "advantage" in data
