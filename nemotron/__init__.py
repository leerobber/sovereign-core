"""
nemotron/__init__.py

Nemotron-3-Nano integration for Sovereign Core (RES-05).

Nvidia Nemotron-3-Nano-30B-A3B (arXiv:2512.20848, December 2025).
Hybrid Mamba-2 + MoE architecture — 30B total / 3.5B active params.
4× throughput over dense 30B, 1M token context.

Candidate to replace or supplement Qwen2.5-32B-AWQ as Sovereign Core's
primary brain on TatorTot (RTX 5090 / RTX 5050).

Usage
-----
    from nemotron import NemotronConfig, NemotronReasoningParser, ABRouter

    config = NemotronConfig()
    parser = NemotronReasoningParser()
    router = ABRouter(config)
"""
from .serve_config import NemotronConfig, ModelVariant, VLLM_LAUNCH_TEMPLATE
from .reasoning_parser import NemotronReasoningParser, ReasoningTrace
from .ab_router import ABRouter, RoutingDecision, RouterStats
from .benchmark import BenchmarkResult, BenchmarkSuite

__all__ = [
    "NemotronConfig",
    "ModelVariant",
    "VLLM_LAUNCH_TEMPLATE",
    "NemotronReasoningParser",
    "ReasoningTrace",
    "ABRouter",
    "RoutingDecision",
    "RouterStats",
    "BenchmarkResult",
    "BenchmarkSuite",
]
