"""Tests for throughput benchmarking."""

from __future__ import annotations

import time

import pytest

from gateway.benchmark import BackendStats, ThroughputBenchmark, _percentile


# ---------------------------------------------------------------------------
# _percentile helper
# ---------------------------------------------------------------------------
class TestPercentileHelper:
    def test_empty_list(self):
        assert _percentile([], 50) == 0.0

    def test_single_element(self):
        assert _percentile([5.0], 50) == pytest.approx(5.0)

    def test_p50_median(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(data, 50) == pytest.approx(3.0)

    def test_p95_upper(self):
        data = list(range(1, 101))  # 1 to 100
        result = _percentile([float(x) for x in data], 95)
        assert 94.0 <= result <= 96.0

    def test_p100_returns_max(self):
        data = [1.0, 2.0, 3.0]
        assert _percentile(data, 100) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# ThroughputBenchmark
# ---------------------------------------------------------------------------
class TestThroughputBenchmark:
    def test_empty_stats(self):
        bench = ThroughputBenchmark()
        stats = bench.stats("rtx5050")
        assert stats.total_requests == 0
        assert stats.avg_latency_s == 0.0

    def test_record_and_count(self):
        bench = ThroughputBenchmark()
        bench.record("rtx5050", 0.1, success=True)
        bench.record("rtx5050", 0.2, success=True)
        stats = bench.stats("rtx5050")
        assert stats.total_requests == 2
        assert stats.successful_requests == 2
        assert stats.failed_requests == 0

    def test_failure_counting(self):
        bench = ThroughputBenchmark()
        bench.record("rtx5050", 1.0, success=False)
        stats = bench.stats("rtx5050")
        assert stats.failed_requests == 1

    def test_average_latency(self):
        bench = ThroughputBenchmark()
        bench.record("rtx5050", 0.1, success=True)
        bench.record("rtx5050", 0.3, success=True)
        stats = bench.stats("rtx5050")
        assert stats.avg_latency_s == pytest.approx(0.2)

    def test_min_max_latency(self):
        bench = ThroughputBenchmark()
        for lat in [0.1, 0.5, 1.0]:
            bench.record("b", lat, success=True)
        stats = bench.stats("b")
        assert stats.min_latency_s == pytest.approx(0.1)
        assert stats.max_latency_s == pytest.approx(1.0)

    def test_percentile_latency(self):
        bench = ThroughputBenchmark()
        for i in range(1, 101):
            bench.record("b", float(i) / 100.0, success=True)
        stats = bench.stats("b")
        assert stats.p50_latency_s == pytest.approx(0.50, abs=0.01)
        assert stats.p95_latency_s == pytest.approx(0.95, abs=0.02)

    def test_failed_requests_excluded_from_latency(self):
        bench = ThroughputBenchmark()
        bench.record("b", 999.0, success=False)  # should not affect avg
        bench.record("b", 0.1, success=True)
        stats = bench.stats("b")
        assert stats.avg_latency_s == pytest.approx(0.1)

    def test_token_tracking(self):
        bench = ThroughputBenchmark()
        bench.record("b", 0.1, success=True, tokens=100)
        bench.record("b", 0.1, success=True, tokens=200)
        stats = bench.stats("b")
        assert stats.total_tokens == 300

    def test_all_stats_returns_all_backends(self):
        bench = ThroughputBenchmark()
        bench.record("rtx5050", 0.1)
        bench.record("radeon780m", 0.2)
        all_s = bench.all_stats()
        ids = {s.backend_id for s in all_s}
        assert "rtx5050" in ids
        assert "radeon780m" in ids

    def test_report_is_serializable(self):
        bench = ThroughputBenchmark()
        bench.record("rtx5050", 0.1, tokens=50)
        report = bench.report()
        assert isinstance(report, list)
        assert isinstance(report[0], dict)
        assert "latency" in report[0]
        assert "throughput" in report[0]

    def test_reset_specific_backend(self):
        bench = ThroughputBenchmark()
        bench.record("rtx5050", 0.1)
        bench.record("radeon780m", 0.2)
        bench.reset("rtx5050")
        assert bench.stats("rtx5050").total_requests == 0
        assert bench.stats("radeon780m").total_requests == 1

    def test_reset_all(self):
        bench = ThroughputBenchmark()
        bench.record("rtx5050", 0.1)
        bench.record("radeon780m", 0.2)
        bench.reset()
        assert bench.all_stats() == []

    def test_requests_per_second_positive(self):
        bench = ThroughputBenchmark(window_seconds=60.0)
        for _ in range(10):
            bench.record("b", 0.05, success=True)
        stats = bench.stats("b")
        assert stats.requests_per_second >= 0.0

    def test_to_dict_structure(self):
        stats = BackendStats(backend_id="rtx5050", total_requests=1)
        d = stats.to_dict()
        assert d["backend_id"] == "rtx5050"
        assert "latency" in d
        assert "throughput" in d
