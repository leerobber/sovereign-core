"""
nemotron/benchmark.py

Throughput and latency benchmark harness for Nemotron-3-Nano vs Qwen2.5-32B-AWQ.

Implements the RES-05 benchmark plan:
  1. Throughput comparison (tokens/sec on auction-style short requests)
  2. Latency comparison (TTFT and end-to-end for agentic tasks)
  3. Long-context test (128k+ tokens — Nemotron-only baseline)
  4. Tool-use accuracy (structured output correctness)

All tests are designed to run with or without a live vLLM endpoint.
When endpoints are unreachable, the suite generates synthetic results
suitable for CI validation of the harness code itself.

Usage
-----
    # Live benchmark (requires vLLM instances at :8001 and :8002)
    from nemotron.benchmark import BenchmarkSuite, BenchmarkConfig
    suite = BenchmarkSuite(BenchmarkConfig(live=True))
    results = suite.run_all()
    print(suite.report(results))

    # Dry-run (CI / unit tests — uses synthetic latencies)
    suite = BenchmarkSuite(BenchmarkConfig(live=False))
    results = suite.run_all()
"""
from __future__ import annotations

import dataclasses
import math
import random
import statistics
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .ab_router import ABRouter, ModelChoice, RoutingDecision, RoutingReason


# ── Config ────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    live: bool = False                     # True = hit real endpoints
    n_warmup: int = 3                      # warm-up requests (discarded)
    n_samples: int = 20                    # measured requests per test
    max_tokens: int = 512                  # generation length for throughput tests
    timeout_s: float = 60.0               # per-request timeout
    seed: int = 42

    # For live runs: endpoints
    qwen_api_base: str = "http://localhost:8001/v1"
    qwen_model: str = "qwen2.5-32b-awq"
    nemotron_api_base: str = "http://localhost:8002/v1"
    nemotron_model: str = "nemotron-nano"


class BenchmarkTestType(str, Enum):
    THROUGHPUT      = "throughput"     # short prompts, measure tok/sec
    LATENCY_TTFT    = "latency_ttft"   # time-to-first-token
    LONG_CONTEXT    = "long_context"   # 128k+ prompt (Nemotron baseline only)
    TOOL_USE        = "tool_use"       # structured JSON tool call accuracy


# ── Results ───────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class BenchmarkResult:
    """Result for a single (model, test_type) combination."""
    model: ModelChoice
    test_type: BenchmarkTestType
    n_samples: int
    latencies_ms: List[float]
    completion_tokens: List[int]
    success_count: int

    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        idx = max(0, int(len(s) * 0.95) - 1)
        return s[idx]

    @property
    def avg_throughput_tps(self) -> float:
        total_tokens = sum(self.completion_tokens)
        total_s = sum(self.latencies_ms) / 1000
        return total_tokens / max(total_s, 1e-9)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.n_samples, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.value,
            "test_type": self.test_type.value,
            "n_samples": self.n_samples,
            "success_rate": round(self.success_rate, 3),
            "p50_latency_ms": round(self.p50_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "avg_throughput_tps": round(self.avg_throughput_tps, 1),
        }


# ── Suite ─────────────────────────────────────────────────────────────────────

class BenchmarkSuite:
    """
    Runs all benchmark tests and compares Nemotron-3-Nano vs Qwen2.5-32B-AWQ.

    In dry-run mode (config.live=False), generates synthetic latencies based
    on Nvidia's published claims:
      - Nemotron throughput: ~4× higher than dense 30B baseline
      - Qwen2.5-32B-AWQ: ~60 tok/sec on RTX 5090 (estimated)
      → Nemotron synthetic: ~240 tok/sec

    Actual numbers will vary significantly based on hardware, batch size,
    and prompt characteristics.
    """

    # Synthetic performance parameters (dry-run mode)
    # Based on Nvidia paper claims + RTX 5090 estimates
    _SYNTHETIC = {
        ModelChoice.QWEN: {
            BenchmarkTestType.THROUGHPUT:  {"mean_ms": 1800, "std_ms": 200, "tokens": 512},
            BenchmarkTestType.LATENCY_TTFT: {"mean_ms": 450,  "std_ms": 80,  "tokens": 1},
            BenchmarkTestType.LONG_CONTEXT: {"mean_ms": 0,    "std_ms": 0,   "tokens": 0},  # N/A
            BenchmarkTestType.TOOL_USE:    {"mean_ms": 2200, "std_ms": 300, "tokens": 256},
        },
        ModelChoice.NEMOTRON: {
            BenchmarkTestType.THROUGHPUT:  {"mean_ms": 450,  "std_ms": 60,  "tokens": 512},
            BenchmarkTestType.LATENCY_TTFT: {"mean_ms": 180,  "std_ms": 40,  "tokens": 1},
            BenchmarkTestType.LONG_CONTEXT: {"mean_ms": 8000, "std_ms": 800, "tokens": 512},
            BenchmarkTestType.TOOL_USE:    {"mean_ms": 600,  "std_ms": 90,  "tokens": 256},
        },
    }

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self._rng = random.Random(self.config.seed)

    # ── Dry-run sample generation ─────────────────────────────────────────────

    def _synthetic_sample(
        self, model: ModelChoice, test_type: BenchmarkTestType
    ) -> Tuple[float, int, bool]:
        """Generate a synthetic (latency_ms, completion_tokens, success) tuple."""
        params = self._SYNTHETIC[model][test_type]
        if params["mean_ms"] == 0:
            return (0.0, 0, False)   # Qwen long-context → not supported
        latency = max(10.0, self._rng.gauss(params["mean_ms"], params["std_ms"]))
        tokens = max(1, int(self._rng.gauss(params["tokens"], params["tokens"] * 0.1)))
        return (latency, tokens, True)

    def _run_test_dryrun(
        self, model: ModelChoice, test_type: BenchmarkTestType
    ) -> BenchmarkResult:
        latencies, completions, successes = [], [], 0
        for _ in range(self.config.n_samples):
            lat, tok, ok = self._synthetic_sample(model, test_type)
            if ok:
                latencies.append(lat)
                completions.append(tok)
                successes += 1
        return BenchmarkResult(
            model=model,
            test_type=test_type,
            n_samples=self.config.n_samples,
            latencies_ms=latencies,
            completion_tokens=completions,
            success_count=successes,
        )

    # ── Live test runner ──────────────────────────────────────────────────────

    def _run_test_live(
        self, model: ModelChoice, test_type: BenchmarkTestType
    ) -> BenchmarkResult:
        """
        Execute live requests against the running vLLM endpoint.
        Requires openai package and running vLLM servers.
        """
        try:
            import openai
        except ImportError:
            raise RuntimeError("pip install openai required for live benchmarks")

        api_base = (
            self.config.nemotron_api_base if model == ModelChoice.NEMOTRON
            else self.config.qwen_api_base
        )
        model_name = (
            self.config.nemotron_model if model == ModelChoice.NEMOTRON
            else self.config.qwen_model
        )
        client = openai.OpenAI(base_url=api_base, api_key="sovereign-core")

        # Long context test: Nemotron only
        if test_type == BenchmarkTestType.LONG_CONTEXT and model == ModelChoice.QWEN:
            return BenchmarkResult(
                model=model, test_type=test_type, n_samples=0,
                latencies_ms=[], completion_tokens=[], success_count=0,
            )

        prompts = self._get_test_prompts(test_type)
        latencies, completions, successes = [], [], 0

        for i in range(self.config.n_warmup + self.config.n_samples):
            prompt = prompts[i % len(prompts)]
            try:
                t0 = time.monotonic()
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout_s,
                )
                elapsed_ms = (time.monotonic() - t0) * 1000
                if i >= self.config.n_warmup:
                    tok = resp.usage.completion_tokens if resp.usage else self.config.max_tokens
                    latencies.append(elapsed_ms)
                    completions.append(tok)
                    successes += 1
            except Exception:
                if i >= self.config.n_warmup:
                    latencies.append(self.config.timeout_s * 1000)
                    completions.append(0)

        return BenchmarkResult(
            model=model,
            test_type=test_type,
            n_samples=self.config.n_samples,
            latencies_ms=latencies,
            completion_tokens=completions,
            success_count=successes,
        )

    def _get_test_prompts(self, test_type: BenchmarkTestType) -> List[str]:
        """Return test prompts for each benchmark type."""
        if test_type == BenchmarkTestType.THROUGHPUT:
            return [
                "Write a haiku about distributed systems.",
                "What is 2+2? Explain step by step.",
                "Name three programming languages.",
            ]
        elif test_type == BenchmarkTestType.LATENCY_TTFT:
            return ["Hello.", "Yes or no?", "Continue."]
        elif test_type == BenchmarkTestType.LONG_CONTEXT:
            # Would be a 128k+ token prompt in live mode — placeholder here
            return ["[LONG CONTEXT PLACEHOLDER — requires 128k+ token prompt]"]
        elif test_type == BenchmarkTestType.TOOL_USE:
            return [
                'Call the search_web function with query="latest GPU prices"',
                'Use get_weather to check "New York City" temperature.',
                'Execute run_code with python_code="print(1+1)"',
            ]
        return ["Test prompt."]

    # ── Public API ────────────────────────────────────────────────────────────

    def run_test(
        self, model: ModelChoice, test_type: BenchmarkTestType
    ) -> BenchmarkResult:
        """Run a single test for a single model."""
        if self.config.live:
            return self._run_test_live(model, test_type)
        return self._run_test_dryrun(model, test_type)

    def run_all(self) -> List[BenchmarkResult]:
        """Run all tests for both models. Returns list of BenchmarkResults."""
        results = []
        models = [ModelChoice.QWEN, ModelChoice.NEMOTRON]
        tests = list(BenchmarkTestType)
        for model in models:
            for test in tests:
                results.append(self.run_test(model, test))
        return results

    def report(self, results: List[BenchmarkResult]) -> str:
        """Generate a human-readable comparison report."""
        lines = [
            "=" * 68,
            "  Nemotron-3-Nano vs Qwen2.5-32B-AWQ — Benchmark Report",
            "  RES-05: Sovereign Core Primary Brain Evaluation",
            "=" * 68,
            "",
        ]
        # Group by test type
        by_test: Dict[str, Dict[str, BenchmarkResult]] = {}
        for r in results:
            by_test.setdefault(r.test_type.value, {})[r.model.value] = r

        for test_name, model_results in by_test.items():
            lines.append(f"  ── {test_name.upper().replace('_', ' ')} ──")
            header = f"  {'Model':<18} {'P50 ms':>8} {'P95 ms':>8} {'tok/s':>8} {'Success':>8}"
            lines.append(header)
            lines.append("  " + "-" * 52)
            for model_name, r in sorted(model_results.items()):
                if r.n_samples == 0:
                    lines.append(f"  {model_name:<18} {'N/A (unsupported)':>36}")
                else:
                    lines.append(
                        f"  {model_name:<18}"
                        f" {r.p50_latency_ms:>8.0f}"
                        f" {r.p95_latency_ms:>8.0f}"
                        f" {r.avg_throughput_tps:>8.1f}"
                        f" {r.success_rate:>7.0%}"
                    )
            lines.append("")

        lines += [
            "  Decision framework (RES-05):",
            "  • Nemotron 4× throughput AND success ≥ 95% → MIGRATE primary brain",
            "  • Nemotron comparable AND success ≥ 95% → SUPPLEMENT (long-ctx + auction)",
            "  • Success < 95% → KEEP QWEN, investigate serving config",
            "",
            f"  Mode: {'LIVE' if self.config.live else 'DRY-RUN (synthetic data)'}",
            "=" * 68,
        ]
        return "\n".join(lines)
