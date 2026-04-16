"""
tests/test_status.py — Tests for gateway/status.py and gateway/ws.py
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_app():
    """Create a minimal FastAPI test app with status router attached."""
    from fastapi import FastAPI
    from gateway.status import router

    app = FastAPI()
    app.include_router(router)

    # Attach mock health monitor to app state
    mock_monitor = MagicMock()
    mock_monitor.backends = [
        MagicMock(id="rtx5050", url="http://localhost:8001", device_type="nvidia_gpu"),
        MagicMock(id="radeon780m", url="http://localhost:8002", device_type="amd_gpu"),
    ]
    mock_monitor.is_healthy = lambda bid: bid == "rtx5050"
    mock_monitor.get_latency = lambda bid: 42.0 if bid == "rtx5050" else None
    mock_monitor.last_seen = lambda bid: 1234567890.0
    app.state.health_monitor = mock_monitor

    return app


@pytest.fixture
def client(mock_app):
    return TestClient(mock_app)


# ── /status/ ─────────────────────────────────────────────────────────────────

class TestStatusEndpoint:
    def test_returns_200(self, client):
        r = client.get("/status/")
        assert r.status_code == 200

    def test_has_uptime(self, client):
        data = client.get("/status/").json()
        assert "uptime_s" in data
        assert data["uptime_s"] >= 0

    def test_has_backends(self, client):
        data = client.get("/status/").json()
        assert "backends" in data
        assert isinstance(data["backends"], list)

    def test_backend_health_reflected(self, client):
        data = client.get("/status/").json()
        backends = {b["name"]: b for b in data["backends"]}
        assert backends["rtx5050"]["healthy"] is True
        assert backends["radeon780m"]["healthy"] is False

    def test_version_present(self, client):
        data = client.get("/status/").json()
        assert "version" in data


# ── /status/backends ─────────────────────────────────────────────────────────

class TestBackendsEndpoint:
    def test_returns_200(self, client):
        r = client.get("/status/backends")
        assert r.status_code == 200

    def test_contains_backend_ids(self, client):
        data = client.get("/status/backends").json()
        assert "rtx5050" in data or "error" in data

    def test_latency_format(self, client):
        data = client.get("/status/backends").json()
        if "rtx5050" in data:
            lat = data["rtx5050"].get("latency_ms")
            assert lat is None or isinstance(lat, (int, float))


# ── /status/stream SSE ───────────────────────────────────────────────────────

class TestSSEStream:
    def test_sse_content_type(self, client):
        with client.stream("GET", "/status/stream") as r:
            assert r.status_code == 200
            # Read just the first chunk
            for chunk in r.iter_text():
                assert "data:" in chunk
                break
