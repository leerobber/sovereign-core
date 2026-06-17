"""Tests for expansion endpoints (repos, agents, feedback)."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.gateway import app
from src.orchestration.registry import RepoStatus


@pytest.fixture
def client():
    return TestClient(app)


class TestReposEndpoint:
    def test_list_repos(self, client):
        import src.api.gateway as gw
        mock_reg = MagicMock()
        mock_reg.probe_all = AsyncMock(return_value=[
            RepoStatus(name="gh05t3", role="agent_plane", healthy=True, detail="HTTP 200")
        ])
        gw.repo_registry = mock_reg
        r = client.get("/v1/repos")
        assert r.status_code == 200
        assert "repos" in r.json()


class TestAgentsEndpoint:
    def test_list_agents(self, client):
        r = client.get("/v1/agents")
        assert r.status_code == 200
        data = r.json()
        assert "agents" in data
        assert "model_map" in data


class TestFinetunedEndpoint:
    def test_finetuned_models(self, client):
        r = client.get("/v1/models/finetuned")
        assert r.status_code == 200
        assert "model_map" in r.json()


class TestFeedbackEndpoint:
    def test_feedback_not_found(self, client):
        import src.api.gateway as gw
        mock_logger = MagicMock()
        mock_logger.add_feedback = AsyncMock(return_value=False)
        gw.run_logger = mock_logger
        r = client.post("/v1/runs/missing-id/feedback", json={"feedback": "fix this"})
        assert r.status_code == 404
