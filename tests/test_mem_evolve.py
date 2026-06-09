"""Tests for MemEvolve — Meta-Evolution Engine (RES-12)."""

from __future__ import annotations

import time

import pytest

from gateway.mem_evolve import (
    ABTestManager,
    MemEvolveEngine,
    RetrievalStrategy,
    RetrievalWeights,
    _PatternScorer,
)
from gateway.pattern_memory import PatternRecord, PatternStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store() -> PatternStore:
    return PatternStore(db_path=":memory:")


def _record(**kwargs: object) -> PatternRecord:
    defaults: dict[str, object] = {
        "model_id": "deepseek-v3",
        "backend_id": "rtx5050",
        "pattern_type": "latency",
    }
    defaults.update(kwargs)
    return PatternRecord(**defaults)  # type: ignore[arg-type]


def _engine(store: PatternStore, **kwargs: object) -> MemEvolveEngine:
    return MemEvolveEngine(store, seed=42, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RetrievalWeights tests
# ---------------------------------------------------------------------------


class TestRetrievalWeights:
    def test_default_weights_sum_to_one(self) -> None:
        w = RetrievalWeights()
        total = w.recency + w.frequency + w.success_rate + w.context_match
        assert total == pytest.approx(1.0)

    def test_normalised_sums_to_one(self) -> None:
        w = RetrievalWeights(recency=1.0, frequency=2.0, success_rate=3.0, context_match=4.0)
        n = w.normalised()
        total = n.recency + n.frequency + n.success_rate + n.context_match
        assert total == pytest.approx(1.0)

    def test_normalised_zero_weights_returns_default(self) -> None:
        w = RetrievalWeights(recency=0.0, frequency=0.0, success_rate=0.0, context_match=0.0)
        n = w.normalised()
        total = n.recency + n.frequency + n.success_rate + n.context_match
        assert total == pytest.approx(1.0)

    def test_negative_values_clamped_to_zero(self) -> None:
        w = RetrievalWeights(recency=-1.0, frequency=-2.0, success_rate=1.0, context_match=0.0)
        assert w.recency == pytest.approx(0.0)
        assert w.frequency == pytest.approx(0.0)

    def test_normalised_proportions_preserved(self) -> None:
        w = RetrievalWeights(recency=1.0, frequency=1.0, success_rate=2.0, context_match=0.0)
        n = w.normalised()
        # success_rate should be twice recency
        assert n.success_rate == pytest.approx(n.recency * 2)


# ---------------------------------------------------------------------------
# RetrievalStrategy tests
# ---------------------------------------------------------------------------


class TestRetrievalStrategy:
    def test_hit_rate_no_trials(self) -> None:
        s = RetrievalStrategy(name="static")
        assert s.hit_rate == pytest.approx(0.0)

    def test_hit_rate_computed(self) -> None:
        s = RetrievalStrategy(name="evolved", total_trials=10, successful_trials=7)
        assert s.hit_rate == pytest.approx(0.7)

    def test_to_dict_includes_hit_rate(self) -> None:
        s = RetrievalStrategy(name="evolved", total_trials=2, successful_trials=1)
        d = s.to_dict()
        assert "hit_rate" in d
        assert d["hit_rate"] == pytest.approx(0.5)

    def test_auto_strategy_id(self) -> None:
        s1 = RetrievalStrategy(name="a")
        s2 = RetrievalStrategy(name="b")
        assert s1.strategy_id != s2.strategy_id


# ---------------------------------------------------------------------------
# _PatternScorer tests
# ---------------------------------------------------------------------------


class TestPatternScorer:
    def _scorer(self) -> _PatternScorer:
        return _PatternScorer()

    def _rec(self, **kwargs: object) -> PatternRecord:
        return _record(**kwargs)

    def test_score_is_non_negative(self) -> None:
        scorer = self._scorer()
        rec = self._rec()
        s = scorer.score(rec, RetrievalWeights())
        assert s >= 0.0

    def test_newer_pattern_scores_higher_on_recency(self) -> None:
        scorer = self._scorer()
        now = time.time()
        new_rec = self._rec(created_at=now - 3600)    # 1 hour old
        old_rec = self._rec(created_at=now - 86400 * 10)  # 10 days old
        weights = RetrievalWeights(recency=1.0, frequency=0.0, success_rate=0.0, context_match=0.0)
        assert scorer.score(new_rec, weights, _now=now) > scorer.score(old_rec, weights, _now=now)

    def test_frequently_used_scores_higher(self) -> None:
        scorer = self._scorer()
        now = time.time()
        freq_rec = self._rec(lookup_count=50, created_at=now)
        rare_rec = self._rec(lookup_count=1, created_at=now)
        weights = RetrievalWeights(recency=0.0, frequency=1.0, success_rate=0.0, context_match=0.0)
        assert scorer.score(freq_rec, weights, _now=now) > scorer.score(rare_rec, weights, _now=now)

    def test_high_success_rate_scores_higher(self) -> None:
        scorer = self._scorer()
        now = time.time()
        good_rec = self._rec(lookup_count=10, success_count=9, created_at=now)
        bad_rec = self._rec(lookup_count=10, success_count=1, created_at=now)
        weights = RetrievalWeights(recency=0.0, frequency=0.0, success_rate=1.0, context_match=0.0)
        assert scorer.score(good_rec, weights, _now=now) > scorer.score(bad_rec, weights, _now=now)

    def test_context_match_boosts_score(self) -> None:
        scorer = self._scorer()
        now = time.time()
        ctx_rec = self._rec(context={"a": 1, "b": 2}, created_at=now)
        no_ctx_rec = self._rec(context={}, created_at=now)
        weights = RetrievalWeights(recency=0.0, frequency=0.0, success_rate=0.0, context_match=1.0)
        query_ctx = {"a": 1, "b": 2}
        assert scorer.score(ctx_rec, weights, query_ctx, _now=now) > scorer.score(
            no_ctx_rec, weights, query_ctx, _now=now
        )

    def test_rank_orders_by_descending_score(self) -> None:
        scorer = self._scorer()
        now = time.time()
        weights = RetrievalWeights(
            recency=0.0, frequency=0.0, success_rate=1.0, context_match=0.0
        )
        low = self._rec(lookup_count=10, success_count=2, created_at=now)
        mid = self._rec(lookup_count=10, success_count=5, created_at=now)
        high = self._rec(lookup_count=10, success_count=9, created_at=now)
        ranked = scorer.rank([low, mid, high], weights, _now=now)
        assert ranked[0].success_count == 9
        assert ranked[-1].success_count == 2

    def test_rank_empty_list_returns_empty(self) -> None:
        scorer = self._scorer()
        assert scorer.rank([], RetrievalWeights()) == []


# ---------------------------------------------------------------------------
# MemEvolveEngine tests
# ---------------------------------------------------------------------------


class TestMemEvolveEngine:
    def test_initial_strategies_exist(self) -> None:
        engine = _engine(_store())
        assert engine.static_strategy.name == "static"
        assert engine.evolved_strategy.name == "evolved"

    def test_initial_generation_is_zero(self) -> None:
        engine = _engine(_store())
        assert engine.evolved_strategy.generation == 0

    def test_evolve_insufficient_data_returns_not_evolved(self) -> None:
        engine = _engine(_store(), min_outcomes=5)
        result = engine.evolve()
        assert result["evolved"] is False
        assert result["reason"] == "insufficient_data"

    def test_evolve_advances_generation(self) -> None:
        ps = _store()
        engine = _engine(ps, min_outcomes=2)
        # Add patterns with lookup data
        for _ in range(5):
            rec = _record()
            ps.store(rec)
            ps.lookup()  # increments lookup_count
            ps.record_outcome(rec.pattern_id, success=True)
        result = engine.evolve()
        assert result["evolved"] is True
        assert result["generation"] == 1

    def test_evolve_updates_weights(self) -> None:
        ps = _store()
        engine = _engine(ps, min_outcomes=2, mutation_rate=0.0)
        for _ in range(5):
            rec = _record()
            ps.store(rec)
            ps.lookup()
            ps.record_outcome(rec.pattern_id, success=True)
        prev_sr = engine.evolved_strategy.weights.success_rate
        engine.evolve()
        # success_rate weight should have grown (sr_boost=1.05)
        assert engine.evolved_strategy.weights.success_rate >= prev_sr

    def test_evolve_reports_previous_and_new_weights(self) -> None:
        ps = _store()
        engine = _engine(ps, min_outcomes=2)
        for _ in range(5):
            rec = _record()
            ps.store(rec)
            ps.lookup()
            ps.record_outcome(rec.pattern_id, success=True)
        result = engine.evolve()
        assert "previous_weights" in result
        assert "new_weights" in result

    def test_evolve_twice_increments_generation_twice(self) -> None:
        ps = _store()
        engine = _engine(ps, min_outcomes=2)
        for _ in range(10):
            rec = _record()
            ps.store(rec)
            ps.lookup()
            ps.record_outcome(rec.pattern_id, success=True)
        engine.evolve()
        for _ in range(5):
            rec = _record()
            ps.store(rec)
            ps.lookup()
            ps.record_outcome(rec.pattern_id, success=True)
        engine.evolve()
        assert engine.evolved_strategy.generation == 2

    def test_static_strategy_weights_never_change(self) -> None:
        ps = _store()
        engine = _engine(ps, min_outcomes=2)
        initial_weights = engine.static_strategy.weights.model_dump()
        for _ in range(10):
            rec = _record()
            ps.store(rec)
            ps.lookup()
            ps.record_outcome(rec.pattern_id, success=True)
        engine.evolve()
        assert engine.static_strategy.weights.model_dump() == initial_weights

    def test_evolve_ignores_lookup_without_outcomes(self) -> None:
        ps = _store()
        engine = _engine(ps, min_outcomes=2)
        for _ in range(5):
            rec = _record()
            ps.store(rec)
            ps.lookup()
        result = engine.evolve()
        assert result["evolved"] is False
        assert result["reason"] == "insufficient_data"

    def test_rank_patterns_evolved_returns_list(self) -> None:
        engine = _engine(_store())
        recs = [_record() for _ in range(3)]
        ranked = engine.rank_patterns(recs, strategy="evolved")
        assert len(ranked) == 3

    def test_rank_patterns_static_returns_list(self) -> None:
        engine = _engine(_store())
        recs = [_record() for _ in range(3)]
        ranked = engine.rank_patterns(recs, strategy="static")
        assert len(ranked) == 3

    def test_rank_patterns_invalid_strategy_raises(self) -> None:
        engine = _engine(_store())
        with pytest.raises(ValueError, match="strategy must be"):
            engine.rank_patterns([], strategy="unknown")

    def test_record_trial_evolved_increments_counts(self) -> None:
        engine = _engine(_store())
        engine.record_trial(strategy="evolved", success=True)
        engine.record_trial(strategy="evolved", success=False)
        assert engine.evolved_strategy.total_trials == 2
        assert engine.evolved_strategy.successful_trials == 1

    def test_record_trial_static_increments_counts(self) -> None:
        engine = _engine(_store())
        engine.record_trial(strategy="static", success=True)
        assert engine.static_strategy.total_trials == 1
        assert engine.static_strategy.successful_trials == 1

    def test_record_trial_invalid_strategy_raises(self) -> None:
        engine = _engine(_store())
        with pytest.raises(ValueError, match="strategy must be"):
            engine.record_trial(strategy="bad", success=True)

    def test_strategy_comparison_structure(self) -> None:
        engine = _engine(_store())
        comp = engine.strategy_comparison()
        assert "static" in comp
        assert "evolved" in comp
        assert "winner" in comp

    def test_strategy_comparison_winner_none_when_no_trials(self) -> None:
        engine = _engine(_store())
        assert engine.strategy_comparison()["winner"] == "none"

    def test_strategy_comparison_winner_evolved_when_better(self) -> None:
        engine = _engine(_store())
        engine.record_trial(strategy="evolved", success=True)
        engine.record_trial(strategy="evolved", success=True)
        engine.record_trial(strategy="static", success=False)
        assert engine.strategy_comparison()["winner"] == "evolved"

    def test_strategy_comparison_winner_static_when_better(self) -> None:
        engine = _engine(_store())
        engine.record_trial(strategy="static", success=True)
        engine.record_trial(strategy="evolved", success=False)
        assert engine.strategy_comparison()["winner"] == "static"


# ---------------------------------------------------------------------------
# ABTestManager tests
# ---------------------------------------------------------------------------


class TestABTestManager:
    def _manager(self, evolved_fraction: float = 0.5) -> ABTestManager:
        engine = _engine(_store())
        return ABTestManager(engine, evolved_fraction=evolved_fraction)

    def test_assign_returns_valid_variant(self) -> None:
        ab = self._manager()
        variant = ab.assign("req-001")
        assert variant in ("evolved", "static")

    def test_assign_is_deterministic(self) -> None:
        ab = self._manager()
        v1 = ab.assign("req-42")
        v2 = ab.assign("req-42")
        assert v1 == v2

    def test_assign_is_stable_across_instances(self) -> None:
        v1 = self._manager().assign("stable-id-123")
        v2 = self._manager().assign("stable-id-123")
        assert v1 == v2

    def test_assign_fraction_zero_always_static(self) -> None:
        ab = self._manager(evolved_fraction=0.0)
        for i in range(20):
            assert ab.assign(f"req-{i}") == "static"

    def test_assign_fraction_one_always_evolved(self) -> None:
        ab = self._manager(evolved_fraction=1.0)
        for i in range(20):
            assert ab.assign(f"req-{i}") == "evolved"

    def test_assign_roughly_50_50_split(self) -> None:
        ab = self._manager(evolved_fraction=0.5)
        evolved = sum(1 for i in range(1000) if ab.assign(f"req-{i}") == "evolved")
        # With 1000 requests the split should be roughly 50% ± 5%
        assert 400 <= evolved <= 600

    def test_record_result_returns_variant(self) -> None:
        ab = self._manager()
        ab.assign("req-x")
        v = ab.record_result("req-x", success=True)
        assert v in ("evolved", "static")

    def test_record_result_updates_engine_trials(self) -> None:
        engine = _engine(_store())
        ab = ABTestManager(engine, evolved_fraction=1.0)  # always evolved
        ab.assign("req-y")
        ab.record_result("req-y", success=True)
        assert engine.evolved_strategy.total_trials == 1
        assert engine.evolved_strategy.successful_trials == 1

    def test_comparison_returns_dict_with_winner(self) -> None:
        ab = self._manager()
        comp = ab.comparison()
        assert "winner" in comp
        assert "evolved" in comp
        assert "static" in comp

    def test_reset_assignments_clears_cache(self) -> None:
        ab = self._manager()
        ab.assign("req-z")
        ab.reset_assignments()
        # After reset the assignment dict is empty; re-assigning is idempotent
        v = ab.assign("req-z")
        assert v in ("evolved", "static")

    def test_evolved_fraction_property(self) -> None:
        ab = self._manager(evolved_fraction=0.3)
        assert ab.evolved_fraction == pytest.approx(0.3)

    def test_invalid_fraction_raises(self) -> None:
        engine = _engine(_store())
        with pytest.raises(ValueError):
            ABTestManager(engine, evolved_fraction=1.5)
