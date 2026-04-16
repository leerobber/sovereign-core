"""
tests/test_kairos_routes.py — Tests for gateway/kairos_routes.py
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


@pytest.fixture
def client():
    from gateway.kairos_routes import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestSAGERoute:
    def test_sage_returns_200(self, client):
        r = client.post("/kairos/sage", json={"task": "optimize latency"})
        assert r.status_code == 200

    def test_sage_has_required_fields(self, client):
        data = client.post(
            "/kairos/sage",
            json={"task": "optimize vram usage", "max_cycles": 1}
        ).json()
        assert "agent_id" in data
        assert "score" in data
        assert "verification_verdict" in data
        assert "proposals" in data
        assert isinstance(data["proposals"], list)

    def test_sage_score_in_range(self, client):
        data = client.post("/kairos/sage", json={"task": "test"}).json()
        assert 0.0 <= data["score"] <= 1.0

    def test_sage_empty_task_accepted(self, client):
        r = client.post("/kairos/sage", json={"task": ""})
        # Should not crash — empty task is handled gracefully
        assert r.status_code in (200, 422)

    def test_sage_max_cycles_validation(self, client):
        r = client.post("/kairos/sage", json={"task": "test", "max_cycles": 0})
        assert r.status_code == 422  # Below minimum of 1

    def test_sage_max_cycles_too_large(self, client):
        r = client.post("/kairos/sage", json={"task": "test", "max_cycles": 999})
        assert r.status_code == 422  # Above maximum of 20


class TestEvolveRoute:
    def test_evolve_returns_200(self, client):
        r = client.post("/kairos/evolve", json={"cycles": 1})
        assert r.status_code == 200

    def test_evolve_has_results(self, client):
        data = client.post("/kairos/evolve", json={"cycles": 2}).json()
        assert "results" in data
        assert len(data["results"]) == 2

    def test_evolve_result_fields(self, client):
        data = client.post("/kairos/evolve", json={"cycles": 1}).json()
        r = data["results"][0]
        assert "agent_id" in r
        assert "generation" in r
        assert "score" in r
        assert "verification_verdict" in r
        assert "elite_promoted" in r
        assert "latency_ms" in r

    def test_evolve_total_latency_present(self, client):
        data = client.post("/kairos/evolve", json={"cycles": 1}).json()
        assert "total_latency_ms" in data
        assert data["total_latency_ms"] > 0

    def test_evolve_cycles_validation(self, client):
        r = client.post("/kairos/evolve", json={"cycles": 0})
        assert r.status_code == 422

    def test_evolve_specific_agent(self, client):
        agent_id = "test-agent-12345"
        data = client.post(
            "/kairos/evolve",
            json={"cycles": 1, "agent_id": agent_id}
        ).json()
        assert data["results"][0]["agent_id"] == agent_id


class TestLeaderboard:
    def test_leaderboard_returns_200(self, client):
        r = client.get("/kairos/leaderboard")
        assert r.status_code == 200

    def test_leaderboard_has_agents_key(self, client):
        data = client.get("/kairos/leaderboard").json()
        assert "agents" in data
        assert "total" in data

    def test_leaderboard_limit_param(self, client):
        data = client.get("/kairos/leaderboard?limit=5").json()
        assert len(data["agents"]) <= 5
