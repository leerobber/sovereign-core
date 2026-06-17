"""Tests for Model Manager"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.manager import ModelManager, ModelStatus, ModelInfo


@pytest.fixture
def config_path(tmp_path):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("""
ollama:
  base_url: "http://localhost:11434"
  timeout: 30
  retry_attempts: 3
models:
  cpu:
    name: "llama3.2:3b"
    display_name: "Llama 3.2 3B"
    device: "cpu"
    max_tokens: 4096
    priority: 1
  gpu:
    name: "qwen2.5:7b"
    display_name: "Qwen 2.5 7B"
    device: "gpu"
    max_tokens: 8192
    priority: 2
""")
    return str(cfg)


@pytest.fixture
def manager(config_path):
    return ModelManager(config_path)


class _FakeResp:
    """Fake aiohttp response that works as async context manager."""
    def __init__(self, status=200, json_data=None, text_data=""):
        self.status = status
        self._json = json_data or {}
        self._text = text_data

    async def json(self):
        return self._json

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeSession:
    """Fake aiohttp session."""
    def __init__(self, resp):
        self._resp = resp
        self.closed = False

    def get(self, *args, **kwargs):
        return self._resp

    def post(self, *args, **kwargs):
        return self._resp

    async def close(self):
        self.closed = True


class TestModelManager:
    def test_init_loads_models(self, manager):
        assert "llama3.2:3b" in manager.models
        assert "qwen2.5:7b" in manager.models

    def test_model_info_defaults(self, manager):
        llama = manager.models["llama3.2:3b"]
        assert llama.device == "cpu"
        assert llama.status == ModelStatus.UNKNOWN
        assert llama.priority == 1

    def test_get_model(self, manager):
        m = manager.get_model("llama3.2:3b")
        assert m is not None
        assert m.display_name == "Llama 3.2 3B"

    def test_get_model_missing(self, manager):
        assert manager.get_model("nonexistent") is None

    def test_get_available_models_none_ready(self, manager):
        assert manager.get_available_models() == []

    def test_get_available_models_some_ready(self, manager):
        manager.models["llama3.2:3b"].status = ModelStatus.READY
        available = manager.get_available_models()
        assert len(available) == 1
        assert available[0].name == "llama3.2:3b"

    @pytest.mark.asyncio
    async def test_check_ollama_health_success(self, manager):
        resp = _FakeResp(status=200)
        session = _FakeSession(resp)
        with patch.object(manager, "_get_session", new=AsyncMock(return_value=session)):
            assert await manager.check_ollama_health() is True

    @pytest.mark.asyncio
    async def test_check_ollama_health_failure(self, manager):
        session = MagicMock()
        session.get.side_effect = Exception("Connection refused")
        with patch.object(manager, "_get_session", new=AsyncMock(return_value=session)):
            assert await manager.check_ollama_health() is False

    @pytest.mark.asyncio
    async def test_list_remote_models(self, manager):
        resp = _FakeResp(status=200, json_data={"models": [{"name": "llama3.2:3b"}, {"name": "qwen2.5:7b"}]})
        session = _FakeSession(resp)
        with patch.object(manager, "_get_session", new=AsyncMock(return_value=session)):
            models = await manager.list_remote_models()
            assert "llama3.2:3b" in models

    @pytest.mark.asyncio
    async def test_pull_model_success(self, manager):
        resp = _FakeResp(status=200)
        session = _FakeSession(resp)
        with patch.object(manager, "_get_session", new=AsyncMock(return_value=session)):
            result = await manager.pull_model("llama3.2:3b")
            assert result is True
            assert manager.models["llama3.2:3b"].status == ModelStatus.READY

    @pytest.mark.asyncio
    async def test_generate_success(self, manager):
        resp = _FakeResp(status=200, json_data={"response": "Hello!", "model": "llama3.2:3b"})
        session = _FakeSession(resp)
        with patch.object(manager, "_get_session", new=AsyncMock(return_value=session)):
            result = await manager.generate("llama3.2:3b", "Hello")
            assert result["response"] == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_success(self, manager):
        resp = _FakeResp(status=200, json_data={"message": {"role": "assistant", "content": "Hi!"}})
        session = _FakeSession(resp)
        with patch.object(manager, "_get_session", new=AsyncMock(return_value=session)):
            result = await manager.chat("llama3.2:3b", [{"role": "user", "content": "Hi"}])
            assert result["message"]["content"] == "Hi!"

    @pytest.mark.asyncio
    async def test_refresh_status(self, manager):
        with patch.object(manager, "list_remote_models", return_value=["llama3.2:3b"]):
            statuses = await manager.refresh_status()
            assert statuses["llama3.2:3b"] == ModelStatus.READY
            assert statuses["qwen2.5:7b"] == ModelStatus.OFFLINE

    @pytest.mark.asyncio
    async def test_close(self, manager):
        manager._session = AsyncMock()
        manager._session.closed = False
        await manager.close()
        manager._session.close.assert_called_once()
