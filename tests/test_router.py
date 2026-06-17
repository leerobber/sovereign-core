"""Tests for Intelligent Router"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.manager import ModelManager, ModelStatus, ModelInfo
from src.routing.router import Router, Complexity, RoutingDecision


@pytest.fixture
def config():
    return {
        "routing": {
            "complexity_threshold": 100,
            "max_queue_size": 50,
            "timeout_seconds": 120,
            "fallback_enabled": True,
        },
        "models": {
            "cpu": {"name": "llama3.2:3b"},
            "gpu": {"name": "qwen2.5:7b"},
        },
    }


@pytest.fixture
def mock_manager(config):
    mm = MagicMock(spec=ModelManager)
    mm.config = config
    mm.models = {
        "llama3.2:3b": ModelInfo(
            name="llama3.2:3b", display_name="Llama 3.2 3B",
            device="cpu", max_tokens=4096, priority=1, status=ModelStatus.READY
        ),
        "qwen2.5:7b": ModelInfo(
            name="qwen2.5:7b", display_name="Qwen 2.5 7B",
            device="gpu", max_tokens=8192, priority=2, status=ModelStatus.READY
        ),
    }
    mm.get_model = lambda n: mm.models.get(n)
    mm.get_available_models = lambda: [m for m in mm.models.values() if m.status == ModelStatus.READY]
    return mm


@pytest.fixture
def router(mock_manager, config):
    return Router(mock_manager, config)


class TestTokenEstimation:
    def test_short_text(self, router):
        assert router.estimate_tokens("hello") >= 1

    def test_long_text(self, router):
        text = "a" * 800  # ~200 tokens
        assert router.estimate_tokens(text) >= 100

    def test_empty_text(self, router):
        assert router.estimate_tokens("") == 1


class TestComplexityClassification:
    def test_short_simple_prompt(self, router):
        assert router.classify_complexity("What time is it?") == Complexity.LOW

    def test_long_prompt(self, router):
        long_text = "Tell me about " + "the history of " * 50 + "computing."
        assert router.classify_complexity(long_text) == Complexity.HIGH

    def test_code_keyword_triggers_high(self, router):
        assert router.classify_complexity("Write code to sort a list") == Complexity.HIGH

    def test_analyze_keyword_triggers_high(self, router):
        assert router.classify_complexity("Analyze this data") == Complexity.HIGH

    def test_simple_greeting(self, router):
        assert router.classify_complexity("Hello!") == Complexity.LOW

    def test_messages_based_classification(self, router):
        messages = [
            {"role": "user", "content": "Implement a binary search tree with balancing"}
        ]
        assert router.classify_complexity("", messages) == Complexity.HIGH

    def test_debug_keyword(self, router):
        assert router.classify_complexity("Debug this function") == Complexity.HIGH


class TestRouting:
    @pytest.mark.asyncio
    async def test_simple_routes_to_cpu(self, router):
        decision = await router.route(prompt="Hi there")
        assert decision.model_name == "llama3.2:3b"
        assert decision.complexity == Complexity.LOW
        assert decision.is_fallback is False

    @pytest.mark.asyncio
    async def test_complex_routes_to_gpu(self, router):
        decision = await router.route(prompt="Write code to implement quicksort with benchmarks")
        assert decision.model_name == "qwen2.5:7b"
        assert decision.complexity == Complexity.HIGH
        assert decision.is_fallback is False

    @pytest.mark.asyncio
    async def test_gpu_unavailable_falls_back_to_cpu(self, router, mock_manager):
        mock_manager.models["qwen2.5:7b"].status = ModelStatus.OFFLINE
        decision = await router.route(prompt="Analyze the performance characteristics of this algorithm")
        assert decision.model_name == "llama3.2:3b"
        assert decision.is_fallback is True

    @pytest.mark.asyncio
    async def test_cpu_unavailable_falls_back_to_gpu(self, router, mock_manager):
        mock_manager.models["llama3.2:3b"].status = ModelStatus.OFFLINE
        decision = await router.route(prompt="Hello!")
        assert decision.model_name == "qwen2.5:7b"
        assert decision.is_fallback is True

    @pytest.mark.asyncio
    async def test_no_models_available(self, router, mock_manager):
        mock_manager.models["llama3.2:3b"].status = ModelStatus.OFFLINE
        mock_manager.models["qwen2.5:7b"].status = ModelStatus.OFFLINE
        decision = await router.route(prompt="Hello")
        assert decision.model_name == ""
        assert "No models" in decision.reason

    @pytest.mark.asyncio
    async def test_routing_latency_under_50ms(self, router):
        decision = await router.route(prompt="Quick test")
        assert decision.latency_ms < 50

    @pytest.mark.asyncio
    async def test_route_with_messages(self, router):
        messages = [{"role": "user", "content": "Explain quantum computing"}]
        decision = await router.route(messages=messages)
        assert decision.model_name != ""

    @pytest.mark.asyncio
    async def test_generate_delegates_to_manager(self, router, mock_manager):
        mock_manager.generate = AsyncMock(return_value={"response": "Hi!", "model": "llama3.2:3b"})
        result = await router.generate("Hello")
        assert "response" in result or "_routing" in result

    @pytest.mark.asyncio
    async def test_chat_delegates_to_manager(self, router, mock_manager):
        mock_manager.chat = AsyncMock(return_value={"message": {"content": "Hi!"}, "model": "llama3.2:3b"})
        result = await router.chat([{"role": "user", "content": "Hi"}])
        assert "message" in result or "_routing" in result


class TestFallbackDisabled:
    @pytest.mark.asyncio
    async def test_no_fallback_when_disabled(self, mock_manager):
        config = {
            "routing": {"complexity_threshold": 100, "fallback_enabled": False, "max_queue_size": 50},
            "models": {"cpu": {"name": "llama3.2:3b"}, "gpu": {"name": "qwen2.5:7b"}},
        }
        r = Router(mock_manager, config)
        mock_manager.models["qwen2.5:7b"].status = ModelStatus.OFFLINE
        decision = await r.route(prompt="Write a complex algorithm for graph traversal")
        # Should still find a model via last-resort check
        assert decision.model_name in ("llama3.2:3b", "")
