"""
RES-05: Nemotron-3-Nano Benchmark Script
Compares Nemotron-3-Nano vs Qwen2.5-32B-AWQ on the Sovereign Core eval suite.

Steps from issue:
  1. Download and quantize Nemotron-3-Nano to AWQ format    ← setup instructions below
  2. Benchmark against Qwen2.5-32B-AWQ on existing eval suite  ← this script
  3. A/B test in auction system for throughput comparison
  4. If wins: migrate primary brain; if not: use as supplementary model
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests

logger = logging.getLogger(__name__)

GATEWAY_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Eval prompts — covers auction reasoning, code gen, and context window use
# ---------------------------------------------------------------------------
EVAL_SUITE = [
    {
        "id": "auction_bid_reasoning",
        "prompt": "An agent has 100 credits. A VRAM slot costs 25 credits (quadratic: 5 votes). "
                  "Should it bid 5 votes or 3 votes to maximise expected utility given 3 competitors? "
                  "Show your calculation.",
        "category": "auction",
    },
    {
        "id": "code_fix_simple",
        "prompt": "Fix this Python function:\ndef add(a, b):\n    return a - b\nExplain the bug.",
        "category": "code",
    },
    {
        "id": "code_fix_async",
        "prompt": "This FastAPI endpoint leaks connections. Fix it:\n"
                  "async def proxy(req):\n    client = aiohttp.ClientSession()\n"
                  "    resp = await client.get('http://backend')\n    return await resp.json()",
        "category": "code",
    },
    {
        "id": "agent_routing",
        "prompt": "A request needs a 7B model with 4 GiB VRAM. Backends: RTX 5050 (8 GiB, healthy), "
                  "Radeon 780M (4 GiB, healthy), Ryzen 7 (CPU, healthy). Which backend and why?",
        "category": "routing",
    },
    {
        "id": "self_optimization",
        "prompt": "The inference gateway shows p99 latency of 4.2s on the Radeon backend vs 1.1s on NVIDIA. "
                  "Propose 3 concrete optimizations ranked by expected impact.",
        "category": "optimization",
    },
    {
        "id": "long_context_summary",
        "prompt": "Summarize the Sovereign Core architecture in 3 bullet points: "
                  "heterogeneous compute mesh, VQ auction system, SAGE co-evolution loop.",
        "category": "summarization",
    },
]


@dataclass
class BenchmarkResult:
    model_id: str
    prompt_id: str
    category: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    response_preview: str
    error: Optional[str] = None

    @property
    def tokens_per_second(self) -> float:
        if self.latency_s > 0:
            return self.tokens_out / self.latency_s
        return 0.0


@dataclass
class ModelBenchmarkSummary:
    model_id: str
    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def successful(self) -> list[BenchmarkResult]:
        return [r for r in self.results if not r.error]

    @property
    def avg_latency_s(self) -> float:
        lats = [r.latency_s for r in self.successful]
        return statistics.mean(lats) if lats else 0.0

    @property
    def p95_latency_s(self) -> float:
        lats = sorted(r.latency_s for r in self.successful)
        if not lats:
            return 0.0
        idx = int(len(lats) * 0.95)
        return lats[min(idx, len(lats) - 1)]

    @property
    def avg_tokens_per_second(self) -> float:
        tps = [r.tokens_per_second for r in self.successful]
        return statistics.mean(tps) if tps else 0.0

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return len(self.successful) / len(self.results)

    def report(self) -> dict:
        return {
            "model_id": self.model_id,
            "total_prompts": len(self.results),
            "success_rate": round(self.success_rate, 3),
            "avg_latency_s": round(self.avg_latency_s, 3),
            "p95_latency_s": round(self.p95_latency_s, 3),
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 2),
            "by_category": self._by_category(),
        }

    def _by_category(self) -> dict:
        cats: dict[str, list[float]] = {}
        for r in self.successful:
            cats.setdefault(r.category, []).append(r.latency_s)
        return {cat: round(statistics.mean(lats), 3) for cat, lats in cats.items()}


def run_inference(model_id: str, prompt: str, timeout: int = 120) -> tuple[str, float, int, int]:
    """Hit the gateway and return (response_text, latency_s, tokens_in, tokens_out)."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    start = time.perf_counter()
    resp = requests.post(
        f"{GATEWAY_URL}/v1/chat/completions",
        json=payload,
        timeout=timeout,
        params={"model_id": model_id},
    )
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    tokens_in = usage.get("prompt_tokens", len(prompt.split()))
    tokens_out = usage.get("completion_tokens", len(text.split()))
    return text, elapsed, tokens_in, tokens_out


def benchmark_model(model_id: str, runs: int = 3) -> ModelBenchmarkSummary:
    """Run the full eval suite against a model, averaging over `runs` passes."""
    summary = ModelBenchmarkSummary(model_id=model_id)

    for eval_item in EVAL_SUITE:
        latencies = []
        last_response = ""
        error = None

        for run in range(runs):
            try:
                text, latency, tok_in, tok_out = run_inference(model_id, eval_item["prompt"])
                latencies.append(latency)
                last_response = text
            except Exception as e:
                error = str(e)
                logger.warning("Error on %s/%s run %d: %s", model_id, eval_item["id"], run, e)
                break

        avg_lat = statistics.mean(latencies) if latencies else 0.0
        summary.results.append(BenchmarkResult(
            model_id=model_id,
            prompt_id=eval_item["id"],
            category=eval_item["category"],
            latency_s=avg_lat,
            tokens_in=tok_in if latencies else 0,
            tokens_out=tok_out if latencies else 0,
            response_preview=last_response[:150],
            error=error,
        ))
        logger.info("  %-35s avg=%.2fs", eval_item["id"], avg_lat)

    return summary


def compare_and_decide(
    nemotron: ModelBenchmarkSummary,
    qwen: ModelBenchmarkSummary,
) -> str:
    """
    A/B decision logic per issue step 4:
    'If wins: migrate primary brain; if not: use as supplementary model'
    """
    nem_tps = nemotron.avg_tokens_per_second
    qwen_tps = qwen.avg_tokens_per_second
    nem_lat = nemotron.avg_latency_s
    qwen_lat = qwen.avg_latency_s

    throughput_win = nem_tps > qwen_tps * 1.1   # >10% faster
    latency_win    = nem_lat < qwen_lat * 0.9    # >10% lower latency

    if throughput_win and latency_win:
        return "MIGRATE_PRIMARY_BRAIN"
    elif throughput_win or latency_win:
        return "SUPPLEMENTARY_CANDIDATE"
    else:
        return "KEEP_QWEN_PRIMARY"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(description="RES-05 Nemotron vs Qwen benchmark")
    parser.add_argument("--nemotron-model", default="nemotron-3-nano")
    parser.add_argument("--qwen-model", default="qwen2.5-32b-awq")
    parser.add_argument("--runs", type=int, default=3, help="Averaging runs per prompt")
    parser.add_argument("--output", default="res05_benchmark_results.json")
    args = parser.parse_args()

    logger.info("=== RES-05: Nemotron-3-Nano vs Qwen2.5-32B-AWQ Benchmark ===")

    logger.info("Benchmarking %s...", args.nemotron_model)
    nemotron_results = benchmark_model(args.nemotron_model, runs=args.runs)

    logger.info("Benchmarking %s...", args.qwen_model)
    qwen_results = benchmark_model(args.qwen_model, runs=args.runs)

    decision = compare_and_decide(nemotron_results, qwen_results)

    output = {
        "nemotron": nemotron_results.report(),
        "qwen": qwen_results.report(),
        "decision": decision,
        "recommendation": {
            "MIGRATE_PRIMARY_BRAIN":     "✓ Nemotron wins on both throughput and latency — update PRIMARY_BRAIN_MODEL in gateway/models.py",
            "SUPPLEMENTARY_CANDIDATE":   "~ Nemotron wins on one dimension — deploy as supplementary for high-throughput auction paths",
            "KEEP_QWEN_PRIMARY":         "✗ Qwen2.5-32B-AWQ remains primary brain — re-evaluate after AWQ quantization tuning",
        }[decision],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("\n%s", "=" * 60)
    logger.info("NEMOTRON  avg_lat=%.2fs  tps=%.1f", nemotron_results.avg_latency_s, nemotron_results.avg_tokens_per_second)
    logger.info("QWEN      avg_lat=%.2fs  tps=%.1f", qwen_results.avg_latency_s, qwen_results.avg_tokens_per_second)
    logger.info("DECISION: %s", decision)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
