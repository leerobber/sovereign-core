"""Tests for Memory Store"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory.store import MemoryStore


@pytest.fixture
async def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    s = MemoryStore(db_path)
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
class TestMemoryStore:
    async def test_initialize_creates_db(self, store):
        assert store._db is not None

    async def test_save_and_get_message(self, store):
        await store.save_message("sess1", "user", "Hello")
        await store.save_message("sess1", "assistant", "Hi!", "llama3.2:3b")
        conv = await store.get_conversation("sess1")
        assert len(conv) == 2
        assert conv[0]["role"] == "user"
        assert conv[1]["content"] == "Hi!"
        assert conv[1]["model"] == "llama3.2:3b"

    async def test_separate_sessions(self, store):
        await store.save_message("sess1", "user", "A")
        await store.save_message("sess2", "user", "B")
        c1 = await store.get_conversation("sess1")
        c2 = await store.get_conversation("sess2")
        assert len(c1) == 1
        assert len(c2) == 1
        assert c1[0]["content"] == "A"
        assert c2[0]["content"] == "B"

    async def test_conversation_limit(self, store):
        for i in range(10):
            await store.save_message("sess1", "user", f"msg{i}")
        conv = await store.get_conversation("sess1", limit=5)
        assert len(conv) == 5

    async def test_record_and_get_metrics(self, store):
        await store.record_metric("llama3.2:3b", 150.0, 10, 20, "router", "low")
        await store.record_metric("llama3.2:3b", 200.0, 15, 25, "router", "low")
        await store.record_metric("qwen2.5:7b", 300.0, 50, 100, "router", "high")
        summary = await store.get_metrics_summary()
        assert "llama3.2:3b" in summary
        assert summary["llama3.2:3b"]["total_requests"] == 2
        assert "qwen2.5:7b" in summary

    async def test_percentile_latencies(self, store):
        for lat in [100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]:
            await store.record_metric("test-model", float(lat))
        p = await store.get_percentile_latencies("test-model")
        assert p["p50"] > 0
        assert p["p95"] >= p["p50"]
        assert p["p99"] >= p["p95"]

    async def test_percentile_empty(self, store):
        p = await store.get_percentile_latencies("nonexistent")
        assert p == {"p50": 0, "p95": 0, "p99": 0}

    async def test_metric_with_fallback(self, store):
        await store.record_metric("llama3.2:3b", 100.0, is_fallback=True)
        summary = await store.get_metrics_summary()
        assert summary["llama3.2:3b"]["total_requests"] == 1
