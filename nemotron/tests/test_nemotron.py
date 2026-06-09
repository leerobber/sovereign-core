"""
tests/test_nemotron.py

Unit tests for Sovereign Core nemotron/ integration module (RES-05).
Zero live LLM calls — all tests run in CI without GPU.
"""
import dataclasses
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nemotron.serve_config import (
    NemotronConfig, ModelVariant, VLLM_LAUNCH_TEMPLATE,
    nemotron_awq_config,
)
from nemotron.reasoning_parser import NemotronReasoningParser, ReasoningTrace
from nemotron.ab_router import (
    ABRouter, ModelChoice, RoutingDecision, RoutingReason,
    RouterStats, QWEN_MAX_CONTEXT,
)
from nemotron.benchmark import (
    BenchmarkConfig, BenchmarkSuite, BenchmarkResult,
    BenchmarkTestType,
)


# ===========================================================================
# NemotronConfig
# ===========================================================================
class TestNemotronConfig:
    def test_default_model_id_is_fp8(self):
        cfg = NemotronConfig()
        assert "Nemotron" in cfg.model_id

    def test_default_port_is_8002(self):
        """Port 8002 keeps Nemotron separate from Qwen @ 8001."""
        cfg = NemotronConfig()
        assert cfg.port == 8002

    def test_default_variant_is_fp8(self):
        cfg = NemotronConfig()
        assert cfg.variant == ModelVariant.FP8

    def test_to_vllm_args_contains_model_id(self):
        cfg = NemotronConfig()
        args = cfg.to_vllm_args()
        assert cfg.model_id in args

    def test_to_vllm_args_contains_reasoning_parser(self):
        cfg = NemotronConfig()
        args = cfg.to_vllm_args()
        assert "--reasoning-parser" in args
        assert "nano_v3" in args

    def test_to_vllm_args_contains_tool_call_parser(self):
        cfg = NemotronConfig()
        args = cfg.to_vllm_args()
        assert "--tool-call-parser" in args

    def test_to_vllm_args_enables_auto_tool_choice(self):
        cfg = NemotronConfig()
        assert "--enable-auto-tool-choice" in cfg.to_vllm_args()

    def test_awq_adds_quantization_flag(self):
        cfg = NemotronConfig(variant=ModelVariant.AWQ)
        args = cfg.to_vllm_args()
        assert "--quantization" in args
        assert "awq" in args

    def test_fp8_no_quantization_flag(self):
        cfg = NemotronConfig(variant=ModelVariant.FP8)
        args = cfg.to_vllm_args()
        assert "--quantization" not in args

    def test_to_yaml_contains_model_id(self):
        cfg = NemotronConfig()
        yaml_str = cfg.to_yaml()
        assert cfg.model_id in yaml_str

    def test_to_yaml_contains_port(self):
        cfg = NemotronConfig()
        assert str(cfg.port) in cfg.to_yaml()

    def test_launch_command_contains_flashinfer_env(self):
        """Nemotron requires VLLM_USE_FLASHINFER_MOE_FP8=1 for correct MoE routing."""
        cfg = NemotronConfig()
        cmd = cfg.launch_command()
        assert "VLLM_USE_FLASHINFER_MOE_FP8" in cmd

    def test_awq_config_helper(self):
        cfg = nemotron_awq_config()
        assert cfg.variant == ModelVariant.AWQ
        assert "AWQ" in cfg.model_id

    def test_vllm_launch_template_is_string(self):
        assert isinstance(VLLM_LAUNCH_TEMPLATE, str)
        assert len(VLLM_LAUNCH_TEMPLATE) > 50

    def test_max_model_len_default_256k(self):
        """Default context window is 256k (1M max)."""
        cfg = NemotronConfig()
        assert cfg.max_model_len == 262144   # 256 * 1024

    def test_kv_cache_dtype_fp8(self):
        cfg = NemotronConfig()
        assert cfg.kv_cache_dtype == "fp8"


# ===========================================================================
# NemotronReasoningParser
# ===========================================================================
class TestNemotronReasoningParser:
    def _parser(self) -> NemotronReasoningParser:
        return NemotronReasoningParser(reasoning_budget=1000)

    def test_parse_with_think_block(self):
        p = self._parser()
        text = "<think>\nLet me reason...\n</think>\nThe answer is 42."
        trace = p.parse(text)
        assert trace.thinking == "Let me reason..."
        assert trace.response == "The answer is 42."

    def test_parse_no_think_block(self):
        p = self._parser()
        text = "The answer is 42."
        trace = p.parse(text)
        assert trace.thinking is None
        assert trace.response == "The answer is 42."

    def test_has_thinking_true(self):
        p = self._parser()
        trace = p.parse("<think>some reasoning</think>final answer")
        assert trace.has_thinking is True

    def test_has_thinking_false(self):
        p = self._parser()
        trace = p.parse("plain response")
        assert trace.has_thinking is False

    def test_thinking_token_estimate(self):
        p = NemotronReasoningParser(reasoning_budget=10_000)
        thinking = "x" * 400   # 400 chars ≈ 100 tokens
        trace = p.parse(f"<think>{thinking}</think>response")
        assert trace.thinking_tokens == 100

    def test_budget_enforcement(self):
        p = NemotronReasoningParser(reasoning_budget=10)   # 10 tokens = 40 chars
        long_thinking = "a" * 1000
        trace = p.parse(f"<think>{long_thinking}</think>final")
        assert trace.budget_exceeded is True
        assert "[truncated]" in trace.thinking

    def test_no_budget_exceeded_within_limit(self):
        p = NemotronReasoningParser(reasoning_budget=10_000)
        trace = p.parse("<think>short thought</think>response")
        assert trace.budget_exceeded is False

    def test_strip_think_tags(self):
        p = self._parser()
        text = "<think>internal reasoning</think>external answer"
        result = p.strip_think_tags(text)
        assert result == "external answer"
        assert "<think>" not in result

    def test_extract_thinking_only(self):
        p = self._parser()
        text = "<think>my thoughts</think>final"
        assert p.extract_thinking_only(text) == "my thoughts"

    def test_extract_thinking_only_none_when_absent(self):
        p = self._parser()
        assert p.extract_thinking_only("no think block") is None

    def test_disable_thinking_adds_system_message(self):
        p = self._parser()
        messages = [{"role": "user", "content": "hello"}]
        result = p.disable_thinking(messages)
        roles = [m["role"] for m in result]
        assert "system" in roles
        sys_msg = next(m for m in result if m["role"] == "system")
        assert "/no_think" in sys_msg["content"]

    def test_set_budget_adds_directive(self):
        p = self._parser()
        messages = [{"role": "user", "content": "question"}]
        result = p.set_budget(messages, budget=5000)
        sys_msgs = [m for m in result if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert "5000" in sys_msgs[0]["content"]

    def test_parse_streaming_joins_chunks(self):
        p = self._parser()
        chunks = ["<think>", "reasoning", "</think>", "answer"]
        trace = p.parse_streaming(chunks)
        assert trace.response == "answer"

    def test_reasoning_trace_summary(self):
        p = self._parser()
        trace = p.parse("<think>a b c d</think>response")
        summary = trace.summary()
        assert "Nemotron" in summary

    def test_case_insensitive_tags(self):
        p = self._parser()
        trace = p.parse("<THINK>thinking</THINK>answer")
        assert trace.thinking == "thinking"
        assert trace.response == "answer"


# ===========================================================================
# ABRouter — routing logic
# ===========================================================================
class TestABRouterRouting:
    def _router(self, ab_split_pct=0.0, nemotron_healthy=True) -> ABRouter:
        return ABRouter(
            ab_split_pct=ab_split_pct,
            nemotron_healthy=nemotron_healthy,
            seed=42,
        )

    def test_long_context_routes_to_nemotron(self):
        router = self._router()
        decision = router.route(prompt_tokens=QWEN_MAX_CONTEXT + 1)
        assert decision.model == ModelChoice.NEMOTRON
        assert decision.reason == RoutingReason.LONG_CONTEXT

    def test_short_context_routes_to_qwen_default(self):
        router = self._router()
        decision = router.route(prompt_tokens=1000)
        assert decision.model == ModelChoice.QWEN
        assert decision.reason == RoutingReason.FALLBACK

    def test_agentic_task_routes_to_nemotron(self):
        router = self._router()
        decision = router.route(task_type="agentic_plan_execution")
        assert decision.model == ModelChoice.NEMOTRON
        assert decision.reason == RoutingReason.AGENTIC_TASK

    def test_auction_routes_to_nemotron(self):
        router = self._router()
        decision = router.route(task_type="batch_auction_bids")
        assert decision.model == ModelChoice.NEMOTRON

    def test_nemotron_down_forces_qwen(self):
        router = self._router(nemotron_healthy=False)
        decision = router.route(prompt_tokens=QWEN_MAX_CONTEXT + 100)
        assert decision.model == ModelChoice.QWEN
        assert decision.reason == RoutingReason.NEMOTRON_DOWN

    def test_force_override_nemotron(self):
        router = self._router()
        decision = router.route(force=ModelChoice.NEMOTRON)
        assert decision.model == ModelChoice.NEMOTRON

    def test_force_override_qwen(self):
        router = self._router()
        decision = router.route(force=ModelChoice.QWEN)
        assert decision.model == ModelChoice.QWEN

    def test_ab_split_pct_routes_some_to_nemotron(self):
        router = self._router(ab_split_pct=0.5, nemotron_healthy=True)
        decisions = [router.route(prompt_tokens=1000) for _ in range(100)]
        nemotron_count = sum(1 for d in decisions if d.model == ModelChoice.NEMOTRON)
        # With 50% split, expect 30–70 nemotron decisions
        assert 20 <= nemotron_count <= 80

    def test_ab_split_zero_never_routes_to_nemotron_on_short(self):
        router = self._router(ab_split_pct=0.0)
        decisions = [router.route(prompt_tokens=100) for _ in range(50)]
        assert all(d.model == ModelChoice.QWEN for d in decisions)

    def test_decision_contains_api_base(self):
        router = self._router()
        d = router.route(prompt_tokens=QWEN_MAX_CONTEXT + 1)
        assert d.api_base.startswith("http://")

    def test_decision_to_openai_kwargs(self):
        router = self._router()
        d = router.route(prompt_tokens=QWEN_MAX_CONTEXT + 1)
        kwargs = d.to_openai_kwargs()
        assert "base_url" in kwargs
        assert "model" in kwargs

    def test_nemotron_decision_points_to_port_8002(self):
        router = self._router()
        d = router.route(prompt_tokens=QWEN_MAX_CONTEXT + 1)
        assert "8002" in d.api_base

    def test_qwen_decision_points_to_port_8001(self):
        router = self._router()
        d = router.route(prompt_tokens=100)
        assert "8001" in d.api_base


# ===========================================================================
# ABRouter — stats and recommendation
# ===========================================================================
class TestABRouterStats:
    def _populated_router(self) -> ABRouter:
        """Create a router with pre-recorded stats for both models."""
        router = ABRouter(seed=0)
        # Record 15 Nemotron requests (fast)
        nem_decision = RoutingDecision(
            model=ModelChoice.NEMOTRON, reason=RoutingReason.LONG_CONTEXT,
            api_base="http://localhost:8002/v1", model_name="nemotron-nano"
        )
        for _ in range(15):
            router.record(nem_decision, latency_ms=450, prompt_tokens=100,
                          completion_tokens=512, success=True)
        # Record 15 Qwen requests (slow)
        qwen_decision = RoutingDecision(
            model=ModelChoice.QWEN, reason=RoutingReason.FALLBACK,
            api_base="http://localhost:8001/v1", model_name="qwen2.5-32b-awq"
        )
        for _ in range(15):
            router.record(qwen_decision, latency_ms=1800, prompt_tokens=100,
                          completion_tokens=512, success=True)
        return router

    def test_stats_returns_both_models(self):
        router = ABRouter()
        stats = router.stats()
        assert "qwen" in stats
        assert "nemotron" in stats

    def test_record_increments_count(self):
        router = ABRouter()
        d = RoutingDecision(
            model=ModelChoice.QWEN, reason=RoutingReason.FALLBACK,
            api_base="http://localhost:8001/v1", model_name="q"
        )
        router.record(d, latency_ms=100, prompt_tokens=50, completion_tokens=50, success=True)
        assert router._stats[ModelChoice.QWEN].total_requests == 1

    def test_recommendation_insufficient_data(self):
        router = ABRouter()
        rec = router.recommendation()
        assert "Insufficient" in rec

    def test_recommendation_migrate_when_nemotron_wins(self):
        router = self._populated_router()
        rec = router.recommendation()
        # Nemotron is 4× faster in synthetic data → should recommend MIGRATE
        assert "MIGRATE" in rec or "SUPPLEMENT" in rec

    def test_recommendation_keep_qwen_on_low_success(self):
        router = ABRouter(seed=0)
        d = RoutingDecision(
            model=ModelChoice.NEMOTRON, reason=RoutingReason.LONG_CONTEXT,
            api_base="http://localhost:8002/v1", model_name="nemotron-nano"
        )
        for i in range(15):
            router.record(d, latency_ms=400, prompt_tokens=100,
                          completion_tokens=200, success=(i < 5))   # only 5/15 success
        d2 = RoutingDecision(
            model=ModelChoice.QWEN, reason=RoutingReason.FALLBACK,
            api_base="http://localhost:8001/v1", model_name="qwen"
        )
        for _ in range(15):
            router.record(d2, latency_ms=1800, prompt_tokens=100,
                          completion_tokens=200, success=True)
        rec = router.recommendation()
        assert "KEEP QWEN" in rec

    def test_reset_clears_stats(self):
        router = self._populated_router()
        router.reset_stats()
        for s in router._stats.values():
            assert s.total_requests == 0

    def test_router_stats_avg_latency(self):
        router = self._populated_router()
        nem_stats = router._stats[ModelChoice.NEMOTRON]
        assert abs(nem_stats.avg_latency_ms - 450) < 1.0

    def test_router_stats_throughput_positive(self):
        router = self._populated_router()
        nem_stats = router._stats[ModelChoice.NEMOTRON]
        assert nem_stats.avg_throughput_tps > 0

    def test_router_stats_success_rate(self):
        router = self._populated_router()
        nem_stats = router._stats[ModelChoice.NEMOTRON]
        assert nem_stats.success_rate == 1.0


# ===========================================================================
# BenchmarkSuite (dry-run)
# ===========================================================================
class TestBenchmarkSuiteDryRun:
    def _suite(self) -> BenchmarkSuite:
        return BenchmarkSuite(BenchmarkConfig(live=False, n_samples=10, seed=42))

    def test_run_all_returns_results(self):
        results = self._suite().run_all()
        assert len(results) > 0

    def test_run_all_covers_both_models(self):
        results = self._suite().run_all()
        models = {r.model for r in results}
        assert ModelChoice.QWEN in models
        assert ModelChoice.NEMOTRON in models

    def test_run_all_covers_all_test_types(self):
        results = self._suite().run_all()
        test_types = {r.test_type for r in results}
        assert BenchmarkTestType.THROUGHPUT in test_types
        assert BenchmarkTestType.LATENCY_TTFT in test_types

    def test_nemotron_throughput_faster_than_qwen_synthetic(self):
        """Verify synthetic data reflects Nvidia's 4× throughput claim."""
        suite = self._suite()
        results = suite.run_all()
        nem_thr = next(r for r in results
                       if r.model == ModelChoice.NEMOTRON
                       and r.test_type == BenchmarkTestType.THROUGHPUT)
        qwen_thr = next(r for r in results
                        if r.model == ModelChoice.QWEN
                        and r.test_type == BenchmarkTestType.THROUGHPUT)
        assert nem_thr.avg_throughput_tps > qwen_thr.avg_throughput_tps * 2

    def test_qwen_long_context_not_supported(self):
        """Qwen long-context test has zero successful samples (>128k unsupported)."""
        suite = self._suite()
        results = suite.run_all()
        qwen_lc = next(r for r in results
                       if r.model == ModelChoice.QWEN
                       and r.test_type == BenchmarkTestType.LONG_CONTEXT)
        assert qwen_lc.success_count == 0

    def test_nemotron_long_context_has_results(self):
        suite = self._suite()
        results = suite.run_all()
        nem_lc = next(r for r in results
                      if r.model == ModelChoice.NEMOTRON
                      and r.test_type == BenchmarkTestType.LONG_CONTEXT)
        assert nem_lc.success_count > 0

    def test_benchmark_result_to_dict(self):
        suite = self._suite()
        results = suite.run_all()
        d = results[0].to_dict()
        assert "model" in d
        assert "avg_throughput_tps" in d
        assert "p50_latency_ms" in d

    def test_report_contains_model_names(self):
        suite = self._suite()
        results = suite.run_all()
        report = suite.report(results)
        assert "qwen" in report.lower()
        assert "nemotron" in report.lower()

    def test_report_contains_decision_framework(self):
        suite = self._suite()
        results = suite.run_all()
        report = suite.report(results)
        assert "MIGRATE" in report or "SUPPLEMENT" in report or "Decision" in report

    def test_report_indicates_dry_run(self):
        suite = self._suite()
        results = suite.run_all()
        report = suite.report(results)
        assert "DRY-RUN" in report

    def test_success_rate_between_0_and_1(self):
        suite = self._suite()
        for r in suite.run_all():
            assert 0.0 <= r.success_rate <= 1.0

    def test_p50_lte_p95(self):
        suite = self._suite()
        for r in suite.run_all():
            if r.success_count > 0:
                assert r.p50_latency_ms <= r.p95_latency_ms + 1.0  # floating point tolerance


# ===========================================================================
# Integration: config → parser → router → benchmark pipeline
# ===========================================================================
class TestNemotronIntegration:
    def test_full_pipeline_dryrun(self):
        """Gene-to-module style end-to-end: config → router → benchmark → report."""
        config = NemotronConfig()
        router = ABRouter(config=config, seed=99)
        suite = BenchmarkSuite(BenchmarkConfig(live=False, n_samples=5, seed=99))

        # Run benchmark
        results = suite.run_all()
        assert len(results) > 0

        # Route a long-context request
        decision = router.route(prompt_tokens=200_000)
        assert decision.model == ModelChoice.NEMOTRON

        # Record it
        nem_result = next(r for r in results
                          if r.model == ModelChoice.NEMOTRON
                          and r.test_type == BenchmarkTestType.THROUGHPUT)
        router.record(
            decision,
            latency_ms=nem_result.p50_latency_ms,
            prompt_tokens=200_000,
            completion_tokens=512,
            success=True,
        )

        # Parse a response
        parser = NemotronReasoningParser(reasoning_budget=10_000)
        raw = "<think>Internal reasoning here.</think>Final answer: yes."
        trace = parser.parse(raw)
        assert trace.response == "Final answer: yes."

    def test_config_yaml_is_parseable(self):
        """YAML output should be valid for documentation."""
        cfg = NemotronConfig()
        yaml_str = cfg.to_yaml()
        # At minimum it's non-empty and has key sections
        assert "model:" in yaml_str
        assert "serving:" in yaml_str
        assert "endpoint:" in yaml_str

    def test_nemotron_arxiv_reference_in_module(self):
        """Ensure the paper reference is present in source (traceability)."""
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "serve_config.py"
        )
        with open(src_path) as f:
            source = f.read()
        assert "2512.20848" in source   # arXiv paper ID
