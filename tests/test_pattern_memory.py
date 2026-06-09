"""Tests for MemEvolve — Pattern Memory Store (RES-12)."""

from __future__ import annotations

import time

import pytest

from gateway.pattern_memory import (
    LookupOutcome,
    PatternRecord,
    PatternStats,
    PatternStore,
)


# ---------------------------------------------------------------------------
# PatternRecord tests
# ---------------------------------------------------------------------------


class TestPatternRecord:
    def _make(self, **kwargs: object) -> PatternRecord:
        defaults: dict[str, object] = {
            "model_id": "deepseek-v3",
            "backend_id": "rtx5050",
            "pattern_type": "latency",
        }
        defaults.update(kwargs)
        return PatternRecord(**defaults)  # type: ignore[arg-type]

    def test_auto_fields_populated(self) -> None:
        r = self._make()
        assert r.pattern_id
        assert r.created_at > 0

    def test_unique_pattern_ids(self) -> None:
        r1 = self._make()
        r2 = self._make()
        assert r1.pattern_id != r2.pattern_id

    def test_model_id_blank_raises(self) -> None:
        with pytest.raises(ValueError):
            self._make(model_id="   ")

    def test_backend_id_blank_raises(self) -> None:
        with pytest.raises(ValueError):
            self._make(backend_id="")

    def test_pattern_type_blank_raises(self) -> None:
        with pytest.raises(ValueError):
            self._make(pattern_type="")

    def test_success_rate_no_lookups_is_zero(self) -> None:
        r = self._make()
        assert r.success_rate == 0.0

    def test_success_rate_computed(self) -> None:
        r = self._make(lookup_count=10, success_count=7)
        assert r.success_rate == pytest.approx(0.7)

    def test_to_dict_includes_success_rate(self) -> None:
        r = self._make(lookup_count=4, success_count=2)
        d = r.to_dict()
        assert "success_rate" in d
        assert d["success_rate"] == pytest.approx(0.5)

    def test_defaults_for_optional_fields(self) -> None:
        r = self._make()
        assert r.context == {}
        assert r.recommendation == {}
        assert r.lookup_count == 0
        assert r.success_count == 0

    def test_context_and_recommendation_stored(self) -> None:
        r = self._make(
            context={"vram_gib": 12},
            recommendation={"backend": "rtx5050"},
        )
        assert r.context["vram_gib"] == 12
        assert r.recommendation["backend"] == "rtx5050"


# ---------------------------------------------------------------------------
# LookupOutcome tests
# ---------------------------------------------------------------------------


class TestLookupOutcome:
    def test_auto_fields_populated(self) -> None:
        o = LookupOutcome(pattern_id="abc", success=True)
        assert o.outcome_id
        assert o.timestamp > 0

    def test_unique_outcome_ids(self) -> None:
        o1 = LookupOutcome(pattern_id="abc", success=True)
        o2 = LookupOutcome(pattern_id="abc", success=False)
        assert o1.outcome_id != o2.outcome_id

    def test_to_dict_structure(self) -> None:
        o = LookupOutcome(pattern_id="pid", success=True, latency_s=0.5)
        d = o.to_dict()
        assert d["pattern_id"] == "pid"
        assert d["success"] is True
        assert d["latency_s"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# PatternStore — basic CRUD
# ---------------------------------------------------------------------------


class TestPatternStoreBasic:
    def _store(self) -> PatternStore:
        return PatternStore(db_path=":memory:")

    def _record(self, **kwargs: object) -> PatternRecord:
        defaults: dict[str, object] = {
            "model_id": "deepseek-v3",
            "backend_id": "rtx5050",
            "pattern_type": "latency",
        }
        defaults.update(kwargs)
        return PatternRecord(**defaults)  # type: ignore[arg-type]

    def test_store_and_retrieve_by_id(self) -> None:
        ps = self._store()
        rec = self._record()
        ps.store(rec)
        found = ps.get_pattern(rec.pattern_id)
        assert found is not None
        assert found.pattern_id == rec.pattern_id

    def test_get_pattern_unknown_returns_none(self) -> None:
        ps = self._store()
        assert ps.get_pattern("nonexistent") is None

    def test_store_replace_updates_record(self) -> None:
        ps = self._store()
        rec = self._record()
        ps.store(rec)
        updated = PatternRecord(
            pattern_id=rec.pattern_id,
            model_id=rec.model_id,
            backend_id=rec.backend_id,
            pattern_type="routing",  # changed
            created_at=rec.created_at,
        )
        ps.store(updated)
        found = ps.get_pattern(rec.pattern_id)
        assert found is not None
        assert found.pattern_type == "routing"

    def test_store_returns_record(self) -> None:
        ps = self._store()
        rec = self._record()
        result = ps.store(rec)
        assert result.pattern_id == rec.pattern_id

    def test_close_does_not_raise(self) -> None:
        ps = self._store()
        ps.close()  # must not raise


# ---------------------------------------------------------------------------
# PatternStore — lookup
# ---------------------------------------------------------------------------


class TestPatternStoreLookup:
    def _store(self) -> PatternStore:
        return PatternStore(db_path=":memory:")

    def _record(self, **kwargs: object) -> PatternRecord:
        defaults: dict[str, object] = {
            "model_id": "deepseek-v3",
            "backend_id": "rtx5050",
            "pattern_type": "latency",
        }
        defaults.update(kwargs)
        return PatternRecord(**defaults)  # type: ignore[arg-type]

    def test_lookup_empty_store_returns_empty(self) -> None:
        ps = self._store()
        assert ps.lookup() == []

    def test_lookup_no_filter_returns_all(self) -> None:
        ps = self._store()
        for _ in range(3):
            ps.store(self._record())
        results = ps.lookup(limit=10)
        assert len(results) == 3

    def test_lookup_filter_by_model_id(self) -> None:
        ps = self._store()
        ps.store(self._record(model_id="deepseek-v3"))
        ps.store(self._record(model_id="tinyllama-1b"))
        results = ps.lookup(model_id="tinyllama-1b")
        assert all(r.model_id == "tinyllama-1b" for r in results)
        assert len(results) == 1

    def test_lookup_filter_by_backend_id(self) -> None:
        ps = self._store()
        ps.store(self._record(backend_id="rtx5050"))
        ps.store(self._record(backend_id="radeon780m"))
        results = ps.lookup(backend_id="radeon780m")
        assert len(results) == 1
        assert results[0].backend_id == "radeon780m"

    def test_lookup_filter_by_pattern_type(self) -> None:
        ps = self._store()
        ps.store(self._record(pattern_type="latency"))
        ps.store(self._record(pattern_type="routing"))
        results = ps.lookup(pattern_type="routing")
        assert len(results) == 1
        assert results[0].pattern_type == "routing"

    def test_lookup_combined_filters(self) -> None:
        ps = self._store()
        ps.store(self._record(model_id="m1", backend_id="b1", pattern_type="latency"))
        ps.store(self._record(model_id="m1", backend_id="b2", pattern_type="latency"))
        ps.store(self._record(model_id="m2", backend_id="b1", pattern_type="routing"))
        results = ps.lookup(model_id="m1", backend_id="b1", pattern_type="latency")
        assert len(results) == 1

    def test_lookup_respects_limit(self) -> None:
        ps = self._store()
        for _ in range(10):
            ps.store(self._record())
        results = ps.lookup(limit=3)
        assert len(results) == 3

    def test_lookup_increments_lookup_count(self) -> None:
        ps = self._store()
        rec = self._record()
        ps.store(rec)
        ps.lookup()
        found = ps.get_pattern(rec.pattern_id)
        assert found is not None
        assert found.lookup_count == 1

    def test_lookup_increments_on_each_call(self) -> None:
        ps = self._store()
        rec = self._record()
        ps.store(rec)
        ps.lookup()
        ps.lookup()
        found = ps.get_pattern(rec.pattern_id)
        assert found is not None
        assert found.lookup_count == 2

    def test_all_patterns_does_not_increment_lookup_count(self) -> None:
        ps = self._store()
        rec = self._record()
        ps.store(rec)
        ps.all_patterns()
        found = ps.get_pattern(rec.pattern_id)
        assert found is not None
        assert found.lookup_count == 0

    def test_all_patterns_respects_limit(self) -> None:
        ps = self._store()
        for _ in range(10):
            ps.store(self._record())
        results = ps.all_patterns(limit=4)
        assert len(results) == 4


# ---------------------------------------------------------------------------
# PatternStore — outcome recording
# ---------------------------------------------------------------------------


class TestPatternStoreOutcomes:
    def _store(self) -> PatternStore:
        return PatternStore(db_path=":memory:")

    def _record(self, **kwargs: object) -> PatternRecord:
        defaults: dict[str, object] = {
            "model_id": "m",
            "backend_id": "b",
            "pattern_type": "latency",
        }
        defaults.update(kwargs)
        return PatternRecord(**defaults)  # type: ignore[arg-type]

    def test_record_outcome_returns_lookup_outcome(self) -> None:
        ps = self._store()
        rec = self._record()
        ps.store(rec)
        outcome = ps.record_outcome(rec.pattern_id, success=True)
        assert isinstance(outcome, LookupOutcome)
        assert outcome.pattern_id == rec.pattern_id
        assert outcome.success is True

    def test_success_increments_success_count(self) -> None:
        ps = self._store()
        rec = self._record()
        ps.store(rec)
        ps.record_outcome(rec.pattern_id, success=True)
        found = ps.get_pattern(rec.pattern_id)
        assert found is not None
        assert found.success_count == 1

    def test_failure_does_not_increment_success_count(self) -> None:
        ps = self._store()
        rec = self._record()
        ps.store(rec)
        ps.record_outcome(rec.pattern_id, success=False)
        found = ps.get_pattern(rec.pattern_id)
        assert found is not None
        assert found.success_count == 0

    def test_multiple_outcomes_accumulate(self) -> None:
        ps = self._store()
        rec = self._record()
        ps.store(rec)
        ps.record_outcome(rec.pattern_id, success=True)
        ps.record_outcome(rec.pattern_id, success=True)
        ps.record_outcome(rec.pattern_id, success=False)
        found = ps.get_pattern(rec.pattern_id)
        assert found is not None
        assert found.success_count == 2

    def test_outcomes_for_pattern_returns_all(self) -> None:
        ps = self._store()
        rec = self._record()
        ps.store(rec)
        ps.record_outcome(rec.pattern_id, success=True, latency_s=0.1)
        ps.record_outcome(rec.pattern_id, success=False, latency_s=0.5)
        outcomes = ps.outcomes_for_pattern(rec.pattern_id)
        assert len(outcomes) == 2

    def test_outcomes_for_unknown_pattern_returns_empty(self) -> None:
        ps = self._store()
        assert ps.outcomes_for_pattern("nonexistent") == []

    def test_outcome_latency_stored(self) -> None:
        ps = self._store()
        rec = self._record()
        ps.store(rec)
        ps.record_outcome(rec.pattern_id, success=True, latency_s=1.23)
        outcomes = ps.outcomes_for_pattern(rec.pattern_id)
        assert outcomes[0].latency_s == pytest.approx(1.23)

    def test_outcome_context_stored(self) -> None:
        ps = self._store()
        rec = self._record()
        ps.store(rec)
        ps.record_outcome(rec.pattern_id, success=True, context={"vram": 8})
        outcomes = ps.outcomes_for_pattern(rec.pattern_id)
        assert outcomes[0].context["vram"] == 8

    def test_record_outcome_unknown_pattern_raises(self) -> None:
        ps = self._store()
        with pytest.raises(ValueError, match="Unknown pattern_id"):
            ps.record_outcome("missing-pattern", success=True)


# ---------------------------------------------------------------------------
# PatternStore — statistics
# ---------------------------------------------------------------------------


class TestPatternStoreStats:
    def _store(self) -> PatternStore:
        return PatternStore(db_path=":memory:")

    def _record(self, **kwargs: object) -> PatternRecord:
        defaults: dict[str, object] = {
            "model_id": "m",
            "backend_id": "b",
            "pattern_type": "latency",
        }
        defaults.update(kwargs)
        return PatternRecord(**defaults)  # type: ignore[arg-type]

    def test_stats_empty_store(self) -> None:
        ps = self._store()
        stats = ps.get_stats()
        assert stats.total_patterns == 0
        assert stats.total_lookups == 0
        assert stats.total_successes == 0
        assert stats.overall_hit_rate == pytest.approx(0.0)

    def test_stats_counts_patterns(self) -> None:
        ps = self._store()
        ps.store(self._record(pattern_type="latency"))
        ps.store(self._record(pattern_type="routing"))
        stats = ps.get_stats()
        assert stats.total_patterns == 2

    def test_stats_patterns_by_type(self) -> None:
        ps = self._store()
        ps.store(self._record(pattern_type="latency"))
        ps.store(self._record(pattern_type="latency"))
        ps.store(self._record(pattern_type="routing"))
        stats = ps.get_stats()
        assert stats.patterns_by_type["latency"] == 2
        assert stats.patterns_by_type["routing"] == 1

    def test_stats_hit_rate_computed(self) -> None:
        ps = self._store()
        rec = self._record()
        ps.store(rec)
        ps.lookup()  # lookup_count += 1
        ps.record_outcome(rec.pattern_id, success=True)
        stats = ps.get_stats()
        assert stats.total_lookups == 1
        assert stats.total_successes == 1
        assert stats.overall_hit_rate == pytest.approx(1.0)

    def test_stats_top_patterns_limited_to_five(self) -> None:
        ps = self._store()
        for _ in range(8):
            ps.store(self._record())
        stats = ps.get_stats()
        assert len(stats.top_patterns) <= 5

    def test_stats_to_dict_is_serialisable(self) -> None:
        ps = self._store()
        ps.store(self._record())
        d = ps.get_stats().to_dict()
        assert isinstance(d, dict)
        assert "total_patterns" in d
        assert "overall_hit_rate" in d
