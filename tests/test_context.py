"""Tests for the shared ChromaDB context layer (RES-06)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import gateway.context as gc
from gateway.context import (
    AgentRole,
    ContextEntry,
    SharedContextLayer,
    _text_to_embedding,
    _rows_to_entries,
    init_context_layer,
    get_context_layer,
)


# ---------------------------------------------------------------------------
# Helpers: isolated SharedContextLayer (always ephemeral / in-memory)
# ---------------------------------------------------------------------------
def _make_layer(collection_name: str = "test_context") -> SharedContextLayer:
    """Return a fresh ephemeral SharedContextLayer for isolation."""
    return SharedContextLayer(collection_name=collection_name)


# ---------------------------------------------------------------------------
# _text_to_embedding
# ---------------------------------------------------------------------------
class TestTextToEmbedding:
    def test_returns_correct_dim(self) -> None:
        vec = _text_to_embedding("hello world")
        assert len(vec) == 64

    def test_all_floats_in_range(self) -> None:
        vec = _text_to_embedding("some content from the generator model")
        assert all(-1.0 <= v <= 1.0 for v in vec)

    def test_deterministic(self) -> None:
        assert _text_to_embedding("abc") == _text_to_embedding("abc")

    def test_different_texts_differ(self) -> None:
        assert _text_to_embedding("text A") != _text_to_embedding("text B")

    def test_empty_string(self) -> None:
        vec = _text_to_embedding("")
        assert len(vec) == 64
        assert all(-1.0 <= v <= 1.0 for v in vec)


# ---------------------------------------------------------------------------
# _rows_to_entries
# ---------------------------------------------------------------------------
class TestRowsToEntries:
    def test_converts_correctly(self) -> None:
        raw: dict[str, Any] = {
            "ids": ["abc123"],
            "documents": ["the verifier said ok"],
            "metadatas": [
                {
                    "role": "verifier",
                    "backend_id": "rtx5050",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "trace_id": "tid1",
                    "extra_key": "extra_val",
                }
            ],
        }
        entries = _rows_to_entries(raw)
        assert len(entries) == 1
        e = entries[0]
        assert e.entry_id == "abc123"
        assert e.role == "verifier"
        assert e.backend_id == "rtx5050"
        assert e.document == "the verifier said ok"
        assert e.trace_id == "tid1"
        assert e.metadata == {"extra_key": "extra_val"}

    def test_empty_result(self) -> None:
        assert _rows_to_entries({"ids": [], "documents": [], "metadatas": []}) == []

    def test_missing_keys_use_defaults(self) -> None:
        raw: dict[str, Any] = {
            "ids": ["x"],
            "documents": ["doc"],
            "metadatas": [{}],
        }
        entries = _rows_to_entries(raw)
        assert entries[0].role == ""
        assert entries[0].backend_id == ""
        assert entries[0].trace_id == ""


# ---------------------------------------------------------------------------
# ContextEntry.as_dict
# ---------------------------------------------------------------------------
class TestContextEntryAsDict:
    def test_as_dict_contains_expected_keys(self) -> None:
        e = ContextEntry(
            entry_id="eid",
            role="generator",
            backend_id="rtx5050",
            document="generated text",
            metadata={"k": "v"},
            timestamp="2026-01-01T00:00:00+00:00",
            trace_id="trace123",
        )
        d = e.as_dict()
        assert d["entry_id"] == "eid"
        assert d["role"] == "generator"
        assert d["backend_id"] == "rtx5050"
        assert d["document"] == "generated text"
        assert d["metadata"] == {"k": "v"}
        assert d["timestamp"] == "2026-01-01T00:00:00+00:00"
        assert d["trace_id"] == "trace123"


# ---------------------------------------------------------------------------
# SharedContextLayer — write & count
# ---------------------------------------------------------------------------
class TestSharedContextLayerWrite:
    def test_write_returns_non_empty_id(self) -> None:
        layer = _make_layer("test_write_basic")
        eid = layer.write(AgentRole.GENERATOR, "rtx5050", "I generated something")
        assert isinstance(eid, str)
        assert len(eid) > 0

    def test_count_increments_on_write(self) -> None:
        layer = _make_layer("test_count_inc")
        assert layer.count() == 0
        layer.write(AgentRole.GENERATOR, "rtx5050", "entry one")
        assert layer.count() == 1
        layer.write(AgentRole.VERIFIER, "radeon780m", "entry two")
        assert layer.count() == 2

    def test_write_with_trace_id(self) -> None:
        layer = _make_layer("test_trace_id")
        layer.write(AgentRole.SAFETY, "ryzen7cpu", "safe", trace_id="t-abc")
        entries = layer.read_by_trace("t-abc")
        assert len(entries) == 1
        assert entries[0].trace_id == "t-abc"

    def test_write_with_extra_metadata(self) -> None:
        layer = _make_layer("test_extra_meta")
        layer.write(
            AgentRole.PLANNER,
            "rtx5050",
            "plan document",
            extra_metadata={"priority": "high"},
        )
        entries = layer.read_by_role(AgentRole.PLANNER)
        assert entries[0].metadata.get("priority") == "high"


# ---------------------------------------------------------------------------
# SharedContextLayer — read_by_role
# ---------------------------------------------------------------------------
class TestReadByRole:
    def test_filters_by_role(self) -> None:
        layer = _make_layer("test_role_filter")
        layer.write(AgentRole.GENERATOR, "rtx5050", "gen output")
        layer.write(AgentRole.VERIFIER, "rtx5050", "ver output")
        layer.write(AgentRole.SAFETY, "radeon780m", "safe output")

        gen_entries = layer.read_by_role(AgentRole.GENERATOR)
        assert len(gen_entries) == 1
        assert gen_entries[0].document == "gen output"

        ver_entries = layer.read_by_role(AgentRole.VERIFIER)
        assert len(ver_entries) == 1
        assert ver_entries[0].role == "verifier"

    def test_empty_role_returns_empty(self) -> None:
        layer = _make_layer("test_empty_role")
        layer.write(AgentRole.GENERATOR, "rtx5050", "gen")
        assert layer.read_by_role(AgentRole.PLANNER) == []

    def test_limit_respected(self) -> None:
        layer = _make_layer("test_role_limit")
        for i in range(5):
            layer.write(AgentRole.REASONER, "rtx5050", f"entry {i}")
        entries = layer.read_by_role(AgentRole.REASONER, limit=3)
        assert len(entries) <= 3


# ---------------------------------------------------------------------------
# SharedContextLayer — read_by_backend
# ---------------------------------------------------------------------------
class TestReadByBackend:
    def test_filters_by_backend(self) -> None:
        layer = _make_layer("test_backend_filter")
        layer.write(AgentRole.GENERATOR, "rtx5050", "rtx output")
        layer.write(AgentRole.REASONER, "radeon780m", "radeon output")

        rtx_entries = layer.read_by_backend("rtx5050")
        assert all(e.backend_id == "rtx5050" for e in rtx_entries)
        assert len(rtx_entries) == 1

        radeon_entries = layer.read_by_backend("radeon780m")
        assert len(radeon_entries) == 1
        assert radeon_entries[0].backend_id == "radeon780m"

    def test_unknown_backend_returns_empty(self) -> None:
        layer = _make_layer("test_unknown_backend")
        layer.write(AgentRole.GENERATOR, "rtx5050", "gen")
        assert layer.read_by_backend("nonexistent") == []


# ---------------------------------------------------------------------------
# SharedContextLayer — read_cross_gpu (cross-GPU visibility)
# ---------------------------------------------------------------------------
class TestReadCrossGpu:
    def test_excludes_requesting_backend(self) -> None:
        layer = _make_layer("test_cross_gpu")
        layer.write(AgentRole.GENERATOR, "rtx5050", "RTX generated this")
        layer.write(AgentRole.VERIFIER, "rtx5050", "RTX verified this")
        layer.write(AgentRole.REASONER, "radeon780m", "Radeon reasoned this")

        # Radeon asks what the RTX backends produced
        peer_entries = layer.read_cross_gpu("radeon780m")
        assert all(e.backend_id != "radeon780m" for e in peer_entries)
        assert len(peer_entries) == 2

    def test_rtx_sees_radeon_output(self) -> None:
        layer = _make_layer("test_cross_gpu_rtx")
        layer.write(AgentRole.REASONER, "radeon780m", "Radeon's reasoning conclusion")
        layer.write(AgentRole.GENERATOR, "rtx5050", "RTX's own output")

        peer_entries = layer.read_cross_gpu("rtx5050")
        assert len(peer_entries) == 1
        assert peer_entries[0].backend_id == "radeon780m"

    def test_empty_when_only_own_entries(self) -> None:
        layer = _make_layer("test_cross_gpu_own_only")
        layer.write(AgentRole.GENERATOR, "rtx5050", "RTX only")
        peer_entries = layer.read_cross_gpu("rtx5050")
        assert peer_entries == []

    def test_limit_respected(self) -> None:
        layer = _make_layer("test_cross_gpu_limit")
        for i in range(5):
            layer.write(AgentRole.SAFETY, "radeon780m", f"safety entry {i}")
        entries = layer.read_cross_gpu("rtx5050", limit=2)
        assert len(entries) <= 2


# ---------------------------------------------------------------------------
# SharedContextLayer — read_by_trace
# ---------------------------------------------------------------------------
class TestReadByTrace:
    def test_returns_correlated_entries(self) -> None:
        layer = _make_layer("test_trace")
        layer.write(AgentRole.GENERATOR, "rtx5050", "gen", trace_id="trace-1")
        layer.write(AgentRole.VERIFIER, "rtx5050", "ver", trace_id="trace-1")
        layer.write(AgentRole.SAFETY, "radeon780m", "safe", trace_id="trace-2")

        entries = layer.read_by_trace("trace-1")
        assert len(entries) == 2
        assert all(e.trace_id == "trace-1" for e in entries)

    def test_unknown_trace_returns_empty(self) -> None:
        layer = _make_layer("test_trace_unknown")
        layer.write(AgentRole.GENERATOR, "rtx5050", "gen", trace_id="trace-X")
        assert layer.read_by_trace("trace-NONE") == []


# ---------------------------------------------------------------------------
# SharedContextLayer — read_all
# ---------------------------------------------------------------------------
class TestReadAll:
    def test_returns_all_entries(self) -> None:
        layer = _make_layer("test_read_all")
        layer.write(AgentRole.GENERATOR, "rtx5050", "gen")
        layer.write(AgentRole.VERIFIER, "radeon780m", "ver")
        layer.write(AgentRole.PLANNER, "ryzen7cpu", "plan")
        entries = layer.read_all()
        assert len(entries) == 3

    def test_limit_respected(self) -> None:
        layer = _make_layer("test_read_all_limit")
        for i in range(10):
            layer.write(AgentRole.GENERATOR, "rtx5050", f"entry {i}")
        entries = layer.read_all(limit=4)
        assert len(entries) <= 4


# ---------------------------------------------------------------------------
# SharedContextLayer — clear
# ---------------------------------------------------------------------------
class TestClear:
    def test_clear_removes_all_entries(self) -> None:
        layer = _make_layer("test_clear")
        layer.write(AgentRole.GENERATOR, "rtx5050", "a")
        layer.write(AgentRole.VERIFIER, "radeon780m", "b")
        removed = layer.clear()
        assert removed == 2
        assert layer.count() == 0

    def test_clear_empty_store(self) -> None:
        layer = _make_layer("test_clear_empty")
        assert layer.clear() == 0

    def test_write_after_clear(self) -> None:
        layer = _make_layer("test_write_after_clear")
        layer.write(AgentRole.SAFETY, "rtx5050", "initial")
        layer.clear()
        layer.write(AgentRole.SAFETY, "rtx5050", "new entry")
        assert layer.count() == 1


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
class TestSingleton:
    def test_get_context_layer_returns_instance(self) -> None:
        # Reset singleton to test lazy init
        gc._context_layer = None
        layer = get_context_layer()
        assert isinstance(layer, SharedContextLayer)

    def test_get_context_layer_returns_same_instance(self) -> None:
        gc._context_layer = None
        a = get_context_layer()
        b = get_context_layer()
        assert a is b

    def test_init_context_layer_replaces_singleton(self) -> None:
        gc._context_layer = None
        first = init_context_layer()
        second = init_context_layer()
        assert first is not second

    def test_init_context_layer_returns_shared_context_layer(self) -> None:
        layer = init_context_layer(collection_name="singleton_test")
        assert isinstance(layer, SharedContextLayer)
        gc._context_layer = None  # restore


# ---------------------------------------------------------------------------
# FastAPI context endpoints (integration tests)
# ---------------------------------------------------------------------------
@pytest.fixture
def patched_app_with_context():
    """Return the FastAPI app with mocked gateway state + real context layer."""
    import gateway.main as gm
    from gateway.benchmark import ThroughputBenchmark
    from gateway.config import GatewaySettings
    from gateway.health import BackendStatus, HealthMonitor
    from gateway.models import ModelAssigner
    from gateway.router import GatewayRouter

    cfg = GatewaySettings(failure_threshold=1, recovery_threshold=1)
    monitor = HealthMonitor(cfg=cfg)
    for bid, state in monitor.states.items():
        state.status = BackendStatus.HEALTHY
    benchmark = ThroughputBenchmark()
    router = GatewayRouter(
        health_monitor=monitor,
        assigner=ModelAssigner(),
        benchmark=benchmark,
        cfg=cfg,
    )
    context_layer = SharedContextLayer(collection_name="endpoint_test")

    gm._health_monitor = monitor
    gm._benchmark = benchmark
    gm._router = router
    gm._context = context_layer

    original_lifespan = gm.app.router.lifespan_context

    @asynccontextmanager
    async def _noop_lifespan(app):  # type: ignore[override]
        yield

    gm.app.router.lifespan_context = _noop_lifespan
    yield gm.app
    gm.app.router.lifespan_context = original_lifespan
    # clean up context layer after test
    context_layer.clear()


class TestContextWriteEndpoint:
    def test_write_returns_200_with_entry_id(self, patched_app_with_context) -> None:
        with TestClient(patched_app_with_context, raise_server_exceptions=False) as client:
            resp = client.post(
                "/context/write",
                params={
                    "role": "generator",
                    "backend_id": "rtx5050",
                    "document": "Qwen generated this response",
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "entry_id" in body
        assert body["role"] == "generator"
        assert body["backend_id"] == "rtx5050"

    def test_write_with_trace_id(self, patched_app_with_context) -> None:
        with TestClient(patched_app_with_context, raise_server_exceptions=False) as client:
            resp = client.post(
                "/context/write",
                params={
                    "role": "verifier",
                    "backend_id": "radeon780m",
                    "document": "DeepSeek verified the output",
                    "trace_id": "test-trace-1",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["trace_id"] == "test-trace-1"

    def test_write_invalid_role_returns_422(self, patched_app_with_context) -> None:
        with TestClient(patched_app_with_context, raise_server_exceptions=False) as client:
            resp = client.post(
                "/context/write",
                params={
                    "role": "invalid_role",
                    "backend_id": "rtx5050",
                    "document": "some text",
                },
            )
        assert resp.status_code == 422


class TestContextReadEndpoint:
    def test_read_all_returns_entries(self, patched_app_with_context) -> None:
        with TestClient(patched_app_with_context, raise_server_exceptions=False) as client:
            client.post(
                "/context/write",
                params={"role": "generator", "backend_id": "rtx5050", "document": "gen output"},
            )
            client.post(
                "/context/write",
                params={"role": "verifier", "backend_id": "radeon780m", "document": "ver output"},
            )
            resp = client.get("/context/read")
        assert resp.status_code == 200
        body = resp.json()
        assert "entries" in body
        assert "count" in body
        assert body["count"] >= 2

    def test_read_filter_by_role(self, patched_app_with_context) -> None:
        with TestClient(patched_app_with_context, raise_server_exceptions=False) as client:
            client.post(
                "/context/write",
                params={"role": "safety", "backend_id": "ryzen7cpu", "document": "safe"},
            )
            resp = client.get("/context/read", params={"role": "safety"})
        assert resp.status_code == 200
        body = resp.json()
        assert all(e["role"] == "safety" for e in body["entries"])

    def test_read_filter_by_backend(self, patched_app_with_context) -> None:
        with TestClient(patched_app_with_context, raise_server_exceptions=False) as client:
            client.post(
                "/context/write",
                params={"role": "planner", "backend_id": "ryzen7cpu", "document": "plan"},
            )
            resp = client.get("/context/read", params={"backend_id": "ryzen7cpu"})
        assert resp.status_code == 200
        body = resp.json()
        assert all(e["backend_id"] == "ryzen7cpu" for e in body["entries"])

    def test_read_filter_by_trace_id(self, patched_app_with_context) -> None:
        with TestClient(patched_app_with_context, raise_server_exceptions=False) as client:
            client.post(
                "/context/write",
                params={
                    "role": "generator",
                    "backend_id": "rtx5050",
                    "document": "traced gen",
                    "trace_id": "ep-trace-99",
                },
            )
            resp = client.get("/context/read", params={"trace_id": "ep-trace-99"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 1
        assert all(e["trace_id"] == "ep-trace-99" for e in body["entries"])


class TestContextCrossGpuEndpoint:
    def test_cross_gpu_excludes_requesting_backend(self, patched_app_with_context) -> None:
        with TestClient(patched_app_with_context, raise_server_exceptions=False) as client:
            client.post(
                "/context/write",
                params={"role": "generator", "backend_id": "rtx5050", "document": "RTX output"},
            )
            client.post(
                "/context/write",
                params={
                    "role": "reasoner",
                    "backend_id": "radeon780m",
                    "document": "Radeon reasoning",
                },
            )
            # RTX asks for peer (Radeon) context
            resp = client.get("/context/cross-gpu/rtx5050")
        assert resp.status_code == 200
        body = resp.json()
        assert body["backend_id"] == "rtx5050"
        assert all(e["backend_id"] != "rtx5050" for e in body["peer_entries"])
        assert body["count"] >= 1

    def test_cross_gpu_empty_when_no_peers(self, patched_app_with_context) -> None:
        with TestClient(patched_app_with_context, raise_server_exceptions=False) as client:
            client.post(
                "/context/write",
                params={
                    "role": "generator",
                    "backend_id": "rtx5050",
                    "document": "RTX only output",
                },
            )
            resp = client.get("/context/cross-gpu/rtx5050")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


class TestContextCountEndpoint:
    def test_count_returns_integer(self, patched_app_with_context) -> None:
        with TestClient(patched_app_with_context, raise_server_exceptions=False) as client:
            resp = client.get("/context/count")
        assert resp.status_code == 200
        assert isinstance(resp.json()["count"], int)


class TestContextClearEndpoint:
    def test_clear_removes_entries(self, patched_app_with_context) -> None:
        with TestClient(patched_app_with_context, raise_server_exceptions=False) as client:
            client.post(
                "/context/write",
                params={"role": "generator", "backend_id": "rtx5050", "document": "to clear"},
            )
            resp = client.delete("/context/clear")
        assert resp.status_code == 200
        body = resp.json()
        assert "cleared" in body
        assert isinstance(body["cleared"], int)
        assert body["cleared"] >= 1
