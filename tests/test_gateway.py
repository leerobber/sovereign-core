"""Tests for API Gateway"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.gateway import app, ChatRequest, ChatMessage, ModelEntry


@pytest.fixture
def client():
    return TestClient(app)


class TestSchemas:
    def test_chat_message_creation(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_request_defaults(self):
        req = ChatRequest(messages=[ChatMessage(role="user", content="Hi")])
        assert req.model is None
        assert req.stream is False

    def test_chat_request_with_model(self):
        req = ChatRequest(
            model="llama3.2:3b",
            messages=[ChatMessage(role="user", content="Hi")],
            stream=True,
        )
        assert req.model == "llama3.2:3b"
        assert req.stream is True

    def test_model_entry(self):
        entry = ModelEntry(id="llama3.2:3b", ready=True)
        assert entry.owned_by == "sovereign-core"
        assert entry.object == "model"


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        import src.api.gateway as gw
        mock_health = AsyncMock()
        mock_health.to_dict = AsyncMock(return_value={"cpu_percent": 25.0, "ram": {}, "gpu": {}, "ollama_running": True, "uptime_seconds": 100})
        mock_mm = MagicMock()
        mock_mm.models = {"llama3.2:3b": MagicMock(status=MagicMock(value="ready"))}

        gw.health = mock_health
        gw.model_manager = mock_mm

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "system" in data
        assert "models" in data


class TestModelsEndpoint:
    def test_list_models(self, client):
        import src.api.gateway as gw
        mock_mm = MagicMock()
        mock_mm.refresh_status = AsyncMock()
        from src.models.manager import ModelInfo, ModelStatus
        mock_mm.models = {
            "llama3.2:3b": ModelInfo(name="llama3.2:3b", display_name="Llama", device="cpu", max_tokens=4096, priority=1, status=ModelStatus.READY),
            "qwen2.5:7b": ModelInfo(name="qwen2.5:7b", display_name="Qwen", device="gpu", max_tokens=8192, priority=2, status=ModelStatus.OFFLINE),
        }
        gw.model_manager = mock_mm

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        import src.api.gateway as gw
        mock_mem = AsyncMock()
        mock_mem.get_metrics_summary = AsyncMock(return_value={"llama3.2:3b": {"total_requests": 5}})
        mock_mem.get_percentile_latencies = AsyncMock(return_value={"p50": 100, "p95": 200, "p99": 300})
        mock_mm = MagicMock()
        gw.memory = mock_mem
        gw.model_manager = mock_mm

        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "percentiles" in data


class TestChatEndpoint:
    def test_chat_completions_success(self, client):
        import src.api.gateway as gw
        mock_router = AsyncMock()
        mock_router.chat = AsyncMock(return_value={
            "message": {"role": "assistant", "content": "Hello!"},
            "model": "llama3.2:3b",
            "_routing": {"model_name": "llama3.2:3b", "complexity": "low", "is_fallback": False},
            "prompt_eval_count": 5,
            "eval_count": 10,
        })
        mock_mem = AsyncMock()
        gw.router = mock_router
        gw.memory = mock_mem

        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello!"

    def test_chat_completions_no_models(self, client):
        import src.api.gateway as gw
        mock_router = AsyncMock()
        mock_router.chat = AsyncMock(return_value={"error": "No models available"})
        mock_mem = AsyncMock()
        gw.router = mock_router
        gw.memory = mock_mem

        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}]
        })
        assert response.status_code == 503


class TestGenerateEndpoint:
    def test_generate_success(self, client):
        import src.api.gateway as gw
        mock_router = AsyncMock()
        mock_router.generate = AsyncMock(return_value={
            "response": "World!",
            "_routing": {"model_name": "llama3.2:3b"},
        })
        gw.router = mock_router

        response = client.post("/v1/generate", json={"prompt": "Hello"})
        assert response.status_code == 200
        data = response.json()
        assert "response" in data or "_routing" in data
