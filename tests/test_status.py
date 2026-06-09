"""tests/test_status.py — Status router integration tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.fixture
def mock_app():
    """Create a test FastAPI app with mocked state."""
    from fastapi import FastAPI
    app = FastAPI()

    # Mock health monitor
    monitor = MagicMock()
    monitor.is_healthy = lambda bid: bid == "rtx5050"
    monitor.get_latency = lambda bid: 0.042 if bid == "rtx5050" else None
    monitor._states = {}

    app.state.health_monitor = monitor
    app.state.router = MagicMock()
    app.state.benchmark = MagicMock()
    app.state.boot_time = 0.0

    from gateway.status import router
    app.include_router(router)
    return app


def test_status_snapshot(mock_app):
    client = TestClient(mock_app)
    r = client.get("/status/")
    assert r.status_code == 200
    data = r.json()
    assert "backends" in data
    assert "version" in data
    assert "uptime_s" in data
    assert data["healthy_count"] >= 0


def test_backends_endpoint(mock_app):
    client = TestClient(mock_app)
    r = client.get("/status/backends")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)


def test_status_has_kairos_summary(mock_app):
    client = TestClient(mock_app)
    r = client.get("/status/")
    assert r.status_code == 200
    data = r.json()
    assert "kairos" in data


def test_status_has_websocket_info(mock_app):
    client = TestClient(mock_app)
    r = client.get("/status/")
    assert r.status_code == 200
    data = r.json()
    assert "websocket" in data
