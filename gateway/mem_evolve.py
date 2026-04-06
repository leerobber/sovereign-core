"""MemEvolve — Meta-Evolution of Agent Memory Systems (RES-12).

Evolves the Pattern Memory retrieval architecture itself — adjusting how
patterns are indexed, retrieved, and prioritized based on which lookups
led to successful optimizations.

Research path (RES-12)
──────────────────────
1. Instrument Pattern Memory lookups with success/failure tracking  → gateway/pattern_memory.py
2. Build meta-evolution loop that adjusts indexing/retrieval weights → MemEvolveEngine
3. A/B test evolved retrieval vs. static retrieval on hit rate      → ABTestManager

Components
──────────
RetrievalWeights — Per-dimension weights for scoring patterns during retrieval.
RetrievalStrategy — A named strategy (static or evolved) with its weights and
                    accumulated performance metrics.
MemEvolveEngine  — Meta-evolution loop: reads Pattern Memory outcome data and
                   updates retrieval weights so high-signal dimensions gain more
                   influence.
ABTestManager    — Deterministically splits requests across evolved/static
                   strategies and records comparative hit-rate statistics.
"""

from __future__ import annotations

import logging
import math
import random
import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from gateway.pattern_memory import PatternRecord, PatternStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval weight model
# ---------------------------------------------------------------------------


class RetrievalWeights(BaseModel):
    """Weights applied when scoring candidate patterns during retrieval.

    Higher weight = stronger influence on a pattern's retrieval rank.
    All weights must be non-negative; they are L1-normalised before use.

    Dimensions
    ──────────
    recency       — Prefer patterns created more recently.
    frequency     — Prefer patterns that have been looked up often.
    success_rate  — Prefer patterns with a high success / lookup ratio.
    context_match — Prefer patterns whose stored context keys overlap with
                    the query context (Jaccard similarity).
    """

    recency: float = Field(default=0.25, ge=0.0)
    frequency: float = Field(default=0.25, ge=0.0)
    success_rate: float = Field(default=0.40, ge=0.0)
    context_match: float = Field(default=0.10, ge=0.0)

    @field_validator("recency", "frequency", "success_rate", "context_match", mode="before")
    @classmethod
    def _clamp_non_negative(cls, v: float) -> float:
        return max(0.0, float(v))

    def normalised(self) -> "RetrievalWeights":
        """Return a copy with all weights normalised to sum to 1.0.

        Returns the default distribution if the total is effectively zero.
        """
        total = self.recency + self.frequency + self.success_rate + self.context_match
        if total < 1e-9:
            return RetrievalWeights()
        return RetrievalWeights(
            recency=self.recency / total,
            frequency=self.frequency / total,
            success_rate=self.success_rate / total,
            context_match=self.context_match / total,
        )


# ---------------------------------------------------------------------------
# Retrieval strategy
# ---------------------------------------------------------------------------


class RetrievalStrategy(BaseModel):
    """A named retrieval strategy with its weights and performance history.

    Instances are immutable; updated copies are produced via
    :meth:`pydantic.BaseModel.model_copy`.
    """

    strategy_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str  # "static" | "evolved"
    generation: int = 0
    weights: RetrievalWeights = Field(default_factory=RetrievalWeights)
    fitness_score: float = 0.0
    total_trials: int = 0
    successful_trials: int = 0
    created_at: float = Field(default_factory=time.time)

    @property
    def hit_rate(self) -> float:
        """Fraction of trials that resulted in a successful optimization."""
        if self.total_trials == 0:
            return 0.0
        return self.successful_trials / self.total_trials

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        d["hit_rate"] = self.hit_rate
        return d


# ---------------------------------------------------------------------------
# Pattern scorer (internal utility)
# ---------------------------------------------------------------------------


class _PatternScorer:
    """Score and rank patterns using a given set of retrieval weights."""

    def score(
        self,
        pattern: PatternRecord,
        weights: RetrievalWeights,
        query_context: Optional[dict[str, Any]] = None,
        _now: Optional[float] = None,
    ) -> float:
        """Return a non-negative relevance score for *pattern*.

        Args:
            pattern:       The candidate pattern to score.
            weights:       Retrieval weights to apply.
            query_context: Optional query context for context-match scoring.
            _now:          Override for the current timestamp (used in tests).
        """
        w = weights.normalised()
        now = _now if _now is not None else time.time()

        # Recency: exponential decay — patterns newer than 1 day score ≈ 1.0
        age_days = max(0.0, (now - pattern.created_at) / 86_400)
        recency_score = math.exp(-age_days)

        # Frequency: log-scaled, capped at 100 lookups → 1.0
        frequency_score = min(1.0, math.log1p(pattern.lookup_count) / math.log1p(100))

        # Success rate: direct ratio
        sr_score = pattern.success_count / max(1, pattern.lookup_count)

        # Context match: Jaccard similarity on context keys
        ctx_score = 0.0
        if query_context:
            q_keys = set(query_context.keys())
            p_keys = set(pattern.context.keys())
            union = q_keys | p_keys
            if union:
                ctx_score = len(q_keys & p_keys) / len(union)

        return (
            w.recency * recency_score
            + w.frequency * frequency_score
            + w.success_rate * sr_score
            + w.context_match * ctx_score
        )

    def rank(
        self,
        patterns: list[PatternRecord],
        weights: RetrievalWeights,
        query_context: Optional[dict[str, Any]] = None,
        _now: Optional[float] = None,
    ) -> list[PatternRecord]:
        """Return *patterns* sorted by descending relevance score."""
        scored = [
            (self.score(p, weights, query_context, _now), p) for p in patterns
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored]


# ---------------------------------------------------------------------------
# Meta-evolution engine
# ---------------------------------------------------------------------------

#: Default weights used by the immutable static strategy.
_STATIC_WEIGHTS = RetrievalWeights(
    recency=0.25,
    frequency=0.25,
    success_rate=0.40,
    context_match=0.10,
)


class MemEvolveEngine:
    """Meta-evolution loop that adapts retrieval weights from outcome data.

    Each call to :meth:`evolve` inspects the Pattern Memory for patterns with
    accumulated outcome data and computes a new set of retrieval weights that
    amplifies dimensions correlated with high-success patterns.

    Args:
        store:         The :class:`~gateway.pattern_memory.PatternStore` to
                       read pattern statistics from.
        mutation_rate: Maximum uniform perturbation applied to each weight
                       dimension after each evolution step (adds diversity).
        min_outcomes:  Minimum number of patterns with at least one lookup
                       required before evolution proceeds.
        seed:          Optional RNG seed for reproducible mutations.
    """

    def __init__(
        self,
        store: PatternStore,
        *,
        mutation_rate: float = 0.05,
        min_outcomes: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        self._store = store
        self._mutation_rate = mutation_rate
        self._min_outcomes = min_outcomes
        self._rng = random.Random(seed)
        self._scorer = _PatternScorer()

        # Static baseline — weights never change
        self._static = RetrievalStrategy(name="static", weights=_STATIC_WEIGHTS)

        # Evolved strategy — updated each generation
        self._evolved = RetrievalStrategy(
            name="evolved",
            weights=RetrievalWeights(
                recency=0.25,
                frequency=0.25,
                success_rate=0.40,
                context_match=0.10,
            ),
        )

        # Tracks which pattern_ids have already been consumed by evolution
        self._patterns_seen: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def static_strategy(self) -> RetrievalStrategy:
        """The immutable static baseline strategy."""
        return self._static

    @property
    def evolved_strategy(self) -> RetrievalStrategy:
        """The current evolved strategy (updated by :meth:`evolve`)."""
        return self._evolved

    def rank_patterns(
        self,
        patterns: list[PatternRecord],
        *,
        strategy: str = "evolved",
        query_context: Optional[dict[str, Any]] = None,
        _now: Optional[float] = None,
    ) -> list[PatternRecord]:
        """Rank *patterns* using the specified retrieval strategy.

        Args:
            patterns:      Candidate patterns to rank (e.g. returned by
                           :meth:`PatternStore.lookup`).
            strategy:      ``"evolved"`` (default) or ``"static"``.
            query_context: Optional query context for context-match scoring.
            _now:          Override for the current time (used in tests).

        Returns:
            *patterns* sorted by descending relevance score under the chosen
            strategy's weights.

        Raises:
            ValueError: If *strategy* is not ``"evolved"`` or ``"static"``.
        """
        if strategy not in ("evolved", "static"):
            raise ValueError(f"strategy must be 'evolved' or 'static', got {strategy!r}")
        weights = (
            self._evolved.weights if strategy == "evolved" else self._static.weights
        )
        return self._scorer.rank(patterns, weights, query_context, _now)

    def evolve(self) -> dict[str, Any]:
        """Run one meta-evolution step and return a summary dict.

        Reads ``all_patterns()`` from the store, identifies patterns with new
        outcome data (at least one lookup), and computes updated retrieval
        weights that amplify dimensions correlated with high-success patterns.

        Returns a dict with keys:

        - ``generation``        — new generation number
        - ``evolved``           — ``True`` if weights changed this step
        - ``outcomes_consumed`` — number of new patterns processed
        - ``previous_weights``  — weight dict before evolution (if evolved)
        - ``new_weights``       — weight dict after evolution (if evolved)
        - ``fitness_score``     — current overall hit rate from the store
        - ``reason``            — explanation when ``evolved=False``
        """
        patterns = self._store.all_patterns(limit=200)
        new_patterns = [
            p for p in patterns
            if p.pattern_id not in self._patterns_seen and p.lookup_count > 0
        ]

        if len(new_patterns) < self._min_outcomes:
            logger.debug(
                "MemEvolve: not enough new outcome data (%d < %d), skipping",
                len(new_patterns),
                self._min_outcomes,
            )
            return {
                "generation": self._evolved.generation,
                "evolved": False,
                "outcomes_consumed": len(new_patterns),
                "reason": "insufficient_data",
            }

        stats = self._store.get_stats()
        prev_weights = self._evolved.weights.model_copy()
        new_weights = self._compute_evolved_weights(new_patterns)

        self._evolved = RetrievalStrategy(
            strategy_id=self._evolved.strategy_id,
            name="evolved",
            generation=self._evolved.generation + 1,
            weights=new_weights,
            fitness_score=stats.overall_hit_rate,
            total_trials=self._evolved.total_trials,
            successful_trials=self._evolved.successful_trials,
            created_at=self._evolved.created_at,
        )

        for p in new_patterns:
            self._patterns_seen.add(p.pattern_id)

        logger.info(
            "MemEvolve: advanced to generation %d (fitness=%.3f, consumed=%d)",
            self._evolved.generation,
            stats.overall_hit_rate,
            len(new_patterns),
        )

        return {
            "generation": self._evolved.generation,
            "evolved": True,
            "outcomes_consumed": len(new_patterns),
            "previous_weights": prev_weights.model_dump(),
            "new_weights": new_weights.model_dump(),
            "fitness_score": stats.overall_hit_rate,
        }

    def record_trial(self, *, strategy: str, success: bool) -> None:
        """Record a retrieval trial outcome for A/B comparison.

        Args:
            strategy: ``"evolved"`` or ``"static"``.
            success:  Whether the retrieval led to a successful optimization.
        """
        if strategy == "evolved":
            self._evolved = self._evolved.model_copy(
                update={
                    "total_trials": self._evolved.total_trials + 1,
                    "successful_trials": self._evolved.successful_trials + (
                        1 if success else 0
                    ),
                }
            )
        elif strategy == "static":
            self._static = self._static.model_copy(
                update={
                    "total_trials": self._static.total_trials + 1,
                    "successful_trials": self._static.successful_trials + (
                        1 if success else 0
                    ),
                }
            )
        else:
            raise ValueError(f"strategy must be 'evolved' or 'static', got {strategy!r}")

    def strategy_comparison(self) -> dict[str, Any]:
        """Return a JSON-serialisable comparison of evolved vs. static performance."""
        return {
            "static": {
                "name": self._static.name,
                "hit_rate": self._static.hit_rate,
                "total_trials": self._static.total_trials,
                "successful_trials": self._static.successful_trials,
                "weights": self._static.weights.model_dump(),
            },
            "evolved": {
                "name": self._evolved.name,
                "generation": self._evolved.generation,
                "hit_rate": self._evolved.hit_rate,
                "total_trials": self._evolved.total_trials,
                "successful_trials": self._evolved.successful_trials,
                "fitness_score": self._evolved.fitness_score,
                "weights": self._evolved.weights.model_dump(),
            },
            "winner": self._leading_strategy(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_evolved_weights(
        self, patterns: list[PatternRecord]
    ) -> RetrievalWeights:
        """Derive new retrieval weights from the outcome statistics of *patterns*.

        The algorithm:
        1. Compute per-pattern success rates.
        2. Separate high-success patterns (above the mean success rate).
        3. Measure how high-success patterns differ from the overall set along
           frequency and recency dimensions.
        4. Amplify weight dimensions that are positively correlated with success.
        5. Apply a small random mutation for search diversity.
        """
        if not patterns:
            return self._mutate(self._evolved.weights)

        rates = [p.success_count / max(1, p.lookup_count) for p in patterns]
        mean_rate = sum(rates) / len(rates)

        high_success = [p for p, r in zip(patterns, rates) if r > mean_rate]

        if not high_success:
            return self._mutate(self._evolved.weights)

        # --- Frequency signal ---
        avg_lookup_hi = sum(p.lookup_count for p in high_success) / len(high_success)
        avg_lookup_all = sum(p.lookup_count for p in patterns) / len(patterns)
        # Boost frequency weight when high-success patterns are queried more often
        freq_boost = 1.0 + 0.1 * (avg_lookup_hi / max(1.0, avg_lookup_all) - 1.0)

        # --- Recency signal ---
        now = time.time()
        avg_age_hi = sum(now - p.created_at for p in high_success) / len(high_success)
        avg_age_all = sum(now - p.created_at for p in patterns) / len(patterns)
        # Boost recency weight when high-success patterns are newer
        safe_age_all = max(1.0, avg_age_all)
        recency_boost = 1.0 + 0.1 * max(0.0, 1.0 - avg_age_hi / safe_age_all)

        # --- Success-rate signal ---
        # Always slightly amplify the success_rate dimension — it is the primary
        # learning signal for meta-evolution.
        sr_boost = 1.05

        cur = self._evolved.weights
        raw = RetrievalWeights(
            recency=max(0.05, cur.recency * recency_boost),
            frequency=max(0.05, cur.frequency * freq_boost),
            success_rate=max(0.10, cur.success_rate * sr_boost),
            context_match=cur.context_match,
        )
        return self._mutate(raw)

    def _mutate(self, weights: RetrievalWeights) -> RetrievalWeights:
        """Apply small uniform-random mutations to all weight dimensions."""
        r = self._mutation_rate
        return RetrievalWeights(
            recency=max(0.01, weights.recency + self._rng.uniform(-r, r)),
            frequency=max(0.01, weights.frequency + self._rng.uniform(-r, r)),
            success_rate=max(0.05, weights.success_rate + self._rng.uniform(-r, r)),
            context_match=max(0.01, weights.context_match + self._rng.uniform(-r, r)),
        )

    def _leading_strategy(self) -> str:
        """Return the name of the strategy with the higher hit rate."""
        e_trials = self._evolved.total_trials
        s_trials = self._static.total_trials
        if e_trials == 0 and s_trials == 0:
            return "none"
        if self._evolved.hit_rate >= self._static.hit_rate:
            return "evolved"
        return "static"


# ---------------------------------------------------------------------------
# A/B test manager
# ---------------------------------------------------------------------------


class ABTestManager:
    """Deterministic A/B test assignment for evolved vs. static retrieval.

    Assigns each request to either the ``"evolved"`` or ``"static"`` strategy
    using a hash of the ``request_id`` for reproducibility.  Results are
    forwarded to the :class:`MemEvolveEngine` and aggregated for comparison.

    Args:
        engine:           The engine to record trial outcomes on.
        evolved_fraction: Fraction [0, 1] of requests routed to the evolved
                          strategy.  Default is 0.5 (50/50 split).
    """

    def __init__(
        self,
        engine: MemEvolveEngine,
        *,
        evolved_fraction: float = 0.5,
    ) -> None:
        if not 0.0 <= evolved_fraction <= 1.0:
            raise ValueError("evolved_fraction must be in [0, 1]")
        self._engine = engine
        self._evolved_fraction = evolved_fraction
        self._assignments: dict[str, str] = {}

    def assign(self, request_id: str) -> str:
        """Return the strategy variant assigned to *request_id*.

        The assignment is deterministic and cached: repeated calls with the
        same ``request_id`` always return the same variant.

        Returns:
            ``"evolved"`` or ``"static"``.
        """
        if request_id in self._assignments:
            return self._assignments[request_id]
        bucket = abs(hash(request_id)) % 1000
        variant = "evolved" if bucket < int(self._evolved_fraction * 1000) else "static"
        self._assignments[request_id] = variant
        return variant

    def record_result(self, request_id: str, *, success: bool) -> str:
        """Record the outcome for *request_id*'s assigned variant.

        Args:
            request_id: The same ID passed to :meth:`assign`.
            success:    Whether the retrieval led to a successful optimization.

        Returns:
            The variant (``"evolved"`` or ``"static"``) that was recorded.
        """
        variant = self._assignments.get(request_id) or self.assign(request_id)
        self._engine.record_trial(strategy=variant, success=success)
        return variant

    def comparison(self) -> dict[str, Any]:
        """Return A/B test comparison statistics from the underlying engine."""
        return self._engine.strategy_comparison()

    @property
    def evolved_fraction(self) -> float:
        """The configured fraction of traffic routed to the evolved strategy."""
        return self._evolved_fraction

    def reset_assignments(self) -> None:
        """Clear cached request → variant assignments.

        Does **not** reset trial counts on the engine; use this only to free
        memory or restart an experiment epoch.
        """
        self._assignments.clear()
