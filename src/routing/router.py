"""Intelligent Request Router — KAN-17"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.models.manager import ModelManager, ModelInfo, ModelStatus
from src.agents import AGENT_MODEL_MAP

logger = logging.getLogger("sovereign.router")


class Complexity(str, Enum):
    LOW = "low"
    HIGH = "high"


@dataclass
class RoutingDecision:
    model_name: str
    reason: str
    complexity: Complexity
    latency_ms: float = 0.0
    is_fallback: bool = False


@dataclass
class QueueItem:
    prompt: str
    messages: list[dict]
    priority: int
    created_at: float = field(default_factory=time.time)


class Router:
    """Routes requests to the optimal model based on complexity and availability."""

    def __init__(self, model_manager: ModelManager, config: dict):
        self.mm = model_manager
        self.complexity_threshold = config.get("routing", {}).get(
            "complexity_threshold", 100
        )
        self.fallback_enabled = config.get("routing", {}).get("fallback_enabled", True)
        self.max_queue = config.get("routing", {}).get("max_queue_size", 50)
        self._queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=self.max_queue)
        self._active_requests: dict[str, int] = {}

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def classify_complexity(self, prompt: str, messages: Optional[list[dict]] = None) -> Complexity:
        total_text = prompt
        if messages:
            total_text = " ".join(m.get("content", "") for m in messages)
        token_count = self.estimate_tokens(total_text)
        if token_count > self.complexity_threshold:
            return Complexity.HIGH
        complex_signals = [
            "explain", "analyze", "compare", "write code", "implement",
            "refactor", "debug", "```", "function", "class ", "algorithm",
        ]
        if any(s in total_text.lower() for s in complex_signals):
            return Complexity.HIGH
        return Complexity.LOW

    def _resolve_agent_model(self, messages: Optional[list[dict]]) -> Optional[str]:
        if not messages:
            return None
        for m in messages:
            content = (m.get("content") or "").upper()
            for agent_id, model_tag in AGENT_MODEL_MAP.items():
                if agent_id in content or f"[{agent_id}]" in content:
                    return model_tag
            agent = (m.get("metadata") or {}).get("agent")
            if agent and agent.upper() in AGENT_MODEL_MAP:
                return AGENT_MODEL_MAP[agent.upper()]
        return None

    def _resolve_override(self, model_override: Optional[str]) -> Optional[RoutingDecision]:
        if not model_override:
            return None
        info = self.mm.get_model(model_override)
        if info and info.status == ModelStatus.READY:
            return RoutingDecision(
                model_name=info.name,
                reason=f"Client override → {info.display_name}",
                complexity=Complexity.LOW,
                is_fallback=False,
            )
        for m in self.mm.get_available_models():
            if m.name == model_override:
                return RoutingDecision(
                    model_name=m.name,
                    reason=f"Client override → {m.display_name}",
                    complexity=Complexity.LOW,
                    is_fallback=False,
                )
        return RoutingDecision(
            model_name="",
            reason=f"Requested model unavailable: {model_override}",
            complexity=Complexity.LOW,
            is_fallback=False,
        )

    async def route(
        self, prompt: str = "", messages: Optional[list[dict]] = None
    ) -> RoutingDecision:
        start = time.perf_counter()
        complexity = self.classify_complexity(prompt, messages)
        cpu_model = self.mm.get_model(self.mm.config["models"]["cpu"]["name"])
        gpu_model = self.mm.get_model(self.mm.config["models"]["gpu"]["name"])
        chosen: Optional[ModelInfo] = None
        reason = ""
        is_fallback = False

        if complexity == Complexity.HIGH:
            if gpu_model and gpu_model.status == ModelStatus.READY:
                chosen, reason = gpu_model, "Complex request → GPU model (Qwen)"
            elif self.fallback_enabled and cpu_model and cpu_model.status == ModelStatus.READY:
                chosen, reason, is_fallback = cpu_model, "Complex request → GPU unavailable, falling back to CPU", True
        else:
            if cpu_model and cpu_model.status == ModelStatus.READY:
                chosen, reason = cpu_model, "Simple request → CPU model (Llama, fast)"
            elif self.fallback_enabled and gpu_model and gpu_model.status == ModelStatus.READY:
                chosen, reason, is_fallback = gpu_model, "Simple request → CPU unavailable, falling back to GPU", True

        elapsed = (time.perf_counter() - start) * 1000
        if chosen is None:
            available = self.mm.get_available_models()
            if available:
                chosen = available[0]
                reason = f"Fallback to first available: {chosen.display_name}"
                is_fallback = True
            else:
                return RoutingDecision("", "No models available", complexity, elapsed, False)

        decision = RoutingDecision(chosen.name, reason, complexity, elapsed, is_fallback)
        logger.info("Routed [%s] → %s in %.1fms", complexity.value, chosen.name, elapsed)
        return decision

    async def generate(
        self, prompt: str, system: str = "", model_override: Optional[str] = None
    ) -> dict:
        decision = self._resolve_override(model_override)
        if decision is None:
            decision = await self.route(prompt=prompt)
        elif not decision.model_name:
            return {"error": decision.reason, "_routing": decision.__dict__}
        if not decision.model_name:
            return {"error": decision.reason, "_routing": decision.__dict__}
        result = await self.mm.generate(decision.model_name, prompt, system=system)
        result["_routing"] = decision.__dict__
        return result

    async def chat(
        self, messages: list[dict], model_override: Optional[str] = None
    ) -> dict:
        if not model_override:
            agent_model = self._resolve_agent_model(messages)
            if agent_model:
                model_override = agent_model
        decision = self._resolve_override(model_override)
        if decision is None:
            decision = await self.route(messages=messages)
        elif not decision.model_name:
            return {"error": decision.reason, "_routing": decision.__dict__}
        if not decision.model_name:
            return {"error": decision.reason, "_routing": decision.__dict__}
        result = await self.mm.chat(decision.model_name, messages)
        result["_routing"] = decision.__dict__
        return result
