"""tests/test_kairos_routes.py — KAIROS routes integration tests."""
from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def app():
    from fastapi import FastAPI
    from gateway.kairos_routes import router
    app = FastAPI()
    app.include_router(router)
    return app


def test_leaderboard_empty(app, tmp_path):
    with patch("gateway.kairos_routes._AGENTS_DIR", tmp_path):
        client = TestClient(app)
        r = client.get("/kairos/leaderboard")
        assert r.status_code == 200
        data = r.json()
        assert "agents" in data
        assert data["total"] == 0


def test_list_agents_empty(app, tmp_path):
    with patch("gateway.kairos_routes._AGENTS_DIR", tmp_path):
        client = TestClient(app)
        r = client.get("/kairos/agents")
        assert r.status_code == 200
        data = r.json()
        assert data["agents"] == []


def test_get_agent_404(app, tmp_path):
    with patch("gateway.kairos_routes._AGENTS_DIR", tmp_path):
        client = TestClient(app)
        r = client.get("/kairos/agents/nonexistent-agent-id")
        assert r.status_code == 404


def test_sage_heuristic_fallback(app, tmp_path):
    """SAGE endpoint should return a valid response even without GPU."""
    with patch("gateway.kairos_routes._AGENTS_DIR", tmp_path), \
         patch("gateway.kairos_routes._run_sage_real", side_effect=ImportError("no gpu")):
        client = TestClient(app)
        r = client.post("/kairos/sage", json={
            "task": "optimize VRAM allocation",
            "max_cycles": 1,
        })
        assert r.status_code == 200
        data = r.json()
        assert "agent_id" in data
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0
        assert data["verification_verdict"] in ("PASS", "PARTIAL", "FAIL")
        assert "proposals" in data


def test_evolve_runs_cycles(app, tmp_path):
    """Evolve endpoint should return results for each cycle."""
    with patch("gateway.kairos_routes._AGENTS_DIR", tmp_path), \
         patch("gateway.kairos_routes._run_sage_real", side_effect=ImportError("no gpu")):
        client = TestClient(app)
        r = client.post("/kairos/evolve", json={"cycles": 3})
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) == 3
        assert "best_score" in data
        assert "total_latency_ms" in data


def test_sage_persists_agent(app, tmp_path):
    """SAGE run should create an agent JSON file."""
    with patch("gateway.kairos_routes._AGENTS_DIR", tmp_path), \
         patch("gateway.kairos_routes._run_sage_real", side_effect=ImportError("no gpu")):
        client = TestClient(app)
        r = client.post("/kairos/sage", json={"task": "test task", "max_cycles": 1})
        assert r.status_code == 200
        agent_id = r.json()["agent_id"]

        # Check file was created
        agent_file = tmp_path / f"{agent_id}.json"
        assert agent_file.exists()
        saved = json.loads(agent_file.read_text())
        assert saved["agent_id"] == agent_id
