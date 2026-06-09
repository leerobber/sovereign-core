"""
nemotron/ab_router.py

A/B routing between Nemotron-3-Nano and Qwen2.5-32B-AWQ.

Intelligent routing logic for Sovereign Core's dual-model stack:
  - Port 8001: Qwen2.5-32B-AWQ   (current primary brain, dense, 128k context)
  - Port 8002: Nemotron-3-Nano   (candidate brain, MoE 3.5B active, 1M context)

Routing strategy
----------------
1. Long context  (> 128k tokens) → Nemotron (only viable option)
2. Agentic tasks (tool calls, multi-step reasoning) → Nemotron (4× throughput)
3. Batch auction bids (many short requests) → Nemotron (throughput advantage)
4. Default / fallback → Qwen (known-good baseline, zero risk)

The router tracks per-model stats and can switch to A/B percentage routing
for systematic comparison (set ab_split_pct > 0).

Benchmark integration
---------------------
The router is also the primary data source for BenchmarkSuite comparisons.
After each request it records latency, token count, and task type.
"""
from __future__ import annotations

import dataclasses
import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ── Enums ─────────────────────────────────────────────────────────────────────

class ModelChoice(str, Enum):
    QWEN      = "qwen"
    NEMOTRON  = "nemotron"


class RoutingReason(str, Enum):
    LONG_CONTEXT   = "long_context"      # > 128k tokens
    AGENTIC_TASK   = "agentic_task"      # tool-use / multi-step
    BATCH_AUCTION  = "batch_auction"     # high-volume bid processing
    AB_SPLIT       = "ab_split"          # random A/B percentage
    FALLBACK       = "fallback"          # default Qwen path
    NEMOTRON_DOWN  = "nemotron_down"     # health check failed → force Qwen


# ── Data types ────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class RoutingDecision:
    """Result of a single routing decision."""
    model: ModelChoice
    reason: RoutingReason
    api_base: str
    model_name: str
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def is_nemotron(self) -> bool:
        return self.model == ModelChoice.NEMOTRON

    def to_openai_kwargs(self) -> Dict[str, Any]:
        """
        Returns kwargs suitable for `openai.OpenAI(base_url=..., model=...)`.
        """
        return {
            "base_url": self.api_base,
            "model": self.model_name,
        }


@dataclasses.dataclass
class RequestRecord:
    """Single request outcome recorded for stats."""
    model: ModelChoice
    reason: RoutingReason
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    success: bool
    timestamp: float = dataclasses.field(default_factory=time.time)

    @property
    def throughput_tps(self) -> float:
        """Completion tokens per second."""
        if self.latency_ms <= 0:
            return 0.0
        return self.completion_tokens / (self.latency_ms / 1000)


@dataclasses.dataclass
class RouterStats:
    """Aggregated per-model statistics."""
    model: ModelChoice
    total_requests: int = 0
    successful_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.total_requests)

    @property
    def avg_throughput_tps(self) -> float:
        if self.total_latency_ms <= 0:
            return 0.0
        return self.total_completion_tokens / (self.total_latency_ms / 1000)

    @property
    def success_rate(self) -> float:
        return self.successful_requests / max(1, self.total_requests)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.value,
            "total_requests": self.total_requests,
            "success_rate": round(self.success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_throughput_tps": round(self.avg_throughput_tps, 1),
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }


# ── Router ────────────────────────────────────────────────────────────────────

# Threshold beyond which only Nemotron can handle the context
QWEN_MAX_CONTEXT = 131_072   # 128k tokens

# Task keywords that signal agentic / multi-step workloads
AGENTIC_KEYWORDS = {
    "tool", "function_call", "plan", "step-by-step", "search",
    "execute", "multi-turn", "agent", "orchestrate",
}


class ABRouter:
    """
    Intelligent A/B router for Nemotron-3-Nano vs Qwen2.5-32B-AWQ.

    Parameters
    ----------
    config           : NemotronConfig (provides api_base and model name).
    qwen_api_base    : Endpoint for Qwen2.5-32B-AWQ (default: localhost:8001).
    qwen_model_name  : Model name for Qwen endpoint.
    ab_split_pct     : Float 0.0–1.0. Fraction of eligible requests to send
                       to Nemotron for A/B comparison. 0.0 = routing rules only.
    nemotron_healthy : Manually override health status (for testing).
    seed             : Random seed for reproducible A/B splits.
    """

    def __init__(
        self,
        config=None,
        qwen_api_base: str = "http://localhost:8001/v1",
        qwen_model_name: str = "qwen2.5-32b-awq",
        ab_split_pct: float = 0.0,
        nemotron_healthy: bool = True,
        seed: Optional[int] = None,
    ):
        if config is None:
            from .serve_config import NemotronConfig
            config = NemotronConfig()
        self.config = config
        self.qwen_api_base = qwen_api_base
        self.qwen_model_name = qwen_model_name
        self.ab_split_pct = ab_split_pct
        self.nemotron_healthy = nemotron_healthy
        self._rng = random.Random(seed)

        self._stats: Dict[ModelChoice, RouterStats] = {
            ModelChoice.QWEN:     RouterStats(model=ModelChoice.QWEN),
            ModelChoice.NEMOTRON: RouterStats(model=ModelChoice.NEMOTRON),
        }
        self._history: List[RequestRecord] = []

    # ── Routing logic ─────────────────────────────────────────────────────────

    def route(
        self,
        prompt_tokens: int = 0,
        task_type: Optional[str] = None,
        force: Optional[ModelChoice] = None,
    ) -> RoutingDecision:
        """
        Choose which model to send a request to.

        Args:
            prompt_tokens : Estimated input token count.
            task_type     : Optional hint ("agentic", "auction", etc.).
            force         : Override routing decision.
        Returns:
            RoutingDecision with endpoint config.
        """
        if force is not None:
            model = force
            reason = RoutingReason.AB_SPLIT
        elif not self.nemotron_healthy:
            model = ModelChoice.QWEN
            reason = RoutingReason.NEMOTRON_DOWN
        elif prompt_tokens > QWEN_MAX_CONTEXT:
            model = ModelChoice.NEMOTRON
            reason = RoutingReason.LONG_CONTEXT
        elif task_type and any(kw in (task_type or "").lower() for kw in AGENTIC_KEYWORDS):
            model = ModelChoice.NEMOTRON
            reason = RoutingReason.AGENTIC_TASK
        elif task_type and "auction" in (task_type or "").lower():
            model = ModelChoice.NEMOTRON
            reason = RoutingReason.BATCH_AUCTION
        elif self.ab_split_pct > 0 and self._rng.random() < self.ab_split_pct:
            model = ModelChoice.NEMOTRON
            reason = RoutingReason.AB_SPLIT
        else:
            model = ModelChoice.QWEN
            reason = RoutingReason.FALLBACK

        if model == ModelChoice.NEMOTRON:
            return RoutingDecision(
                model=model,
                reason=reason,
                api_base=self.config.api_base,
                model_name=self.config.served_model_name,
                metadata={"prompt_tokens": prompt_tokens, "task_type": task_type},
            )
        else:
            return RoutingDecision(
                model=model,
                reason=reason,
                api_base=self.qwen_api_base,
                model_name=self.qwen_model_name,
                metadata={"prompt_tokens": prompt_tokens, "task_type": task_type},
            )

    # ── Stats recording ───────────────────────────────────────────────────────

    def record(
        self,
        decision: RoutingDecision,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        success: bool = True,
    ) -> None:
        """Record a completed request outcome for benchmark comparison."""
        rec = RequestRecord(
            model=decision.model,
            reason=decision.reason,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            success=success,
        )
        self._history.append(rec)
        s = self._stats[decision.model]
        s.total_requests += 1
        s.total_latency_ms += latency_ms
        s.total_prompt_tokens += prompt_tokens
        s.total_completion_tokens += completion_tokens
        if success:
            s.successful_requests += 1

    # ── Reporting ─────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return current per-model stats dict."""
        return {m.value: s.to_dict() for m, s in self._stats.items()}

    def recommendation(self) -> str:
        """
        Return a human-readable recommendation based on accumulated stats.
        Implements the RES-05 decision rule:
          'If Nemotron wins: migrate primary brain; if not: use as supplementary.'
        """
        q = self._stats[ModelChoice.QWEN]
        n = self._stats[ModelChoice.NEMOTRON]

        if n.total_requests < 10 or q.total_requests < 10:
            return "Insufficient data — collect at least 10 requests per model before deciding."

        nem_wins_throughput = n.avg_throughput_tps > q.avg_throughput_tps * 1.5
        nem_wins_latency    = n.avg_latency_ms < q.avg_latency_ms * 0.8
        nem_success         = n.success_rate >= 0.95

        if nem_wins_throughput and nem_success:
            return (
                f"MIGRATE: Nemotron-3-Nano wins throughput ({n.avg_throughput_tps:.1f} "
                f"vs {q.avg_throughput_tps:.1f} tok/s). Promote to primary brain."
            )
        elif nem_wins_latency and nem_success:
            return (
                f"MIGRATE: Nemotron-3-Nano wins latency ({n.avg_latency_ms:.0f} "
                f"vs {q.avg_latency_ms:.0f} ms avg). Promote to primary brain."
            )
        elif nem_success:
            return (
                "SUPPLEMENT: Nemotron-3-Nano performs comparably but does not dominate. "
                "Use for long-context (>128k) and high-concurrency auction workloads only."
            )
        else:
            return (
                f"KEEP QWEN: Nemotron-3-Nano success rate too low ({n.success_rate:.1%}). "
                "Investigate serving config before considering migration."
            )

    def reset_stats(self) -> None:
        """Clear all accumulated stats (useful between benchmark runs)."""
        for s in self._stats.values():
            s.total_requests = 0
            s.successful_requests = 0
            s.total_prompt_tokens = 0
            s.total_completion_tokens = 0
            s.total_latency_ms = 0.0
        self._history.clear()
