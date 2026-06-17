"""Ollama Model Lifecycle Manager — KAN-16, KAN-11"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import aiohttp
import yaml

logger = logging.getLogger("sovereign.models")


class ModelStatus(str, Enum):
    UNKNOWN = "unknown"
    PULLING = "pulling"
    READY = "ready"
    LOADING = "loading"
    RUNNING = "running"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class ModelInfo:
    name: str
    display_name: str
    device: str
    max_tokens: int
    priority: int
    status: ModelStatus = ModelStatus.UNKNOWN
    last_error: Optional[str] = None
    load_time_ms: float = 0.0


class ModelManager:
    """Manages Ollama model lifecycle: pull, load, health, generate."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.ollama_url = self.config["ollama"]["base_url"]
        self.timeout = self.config["ollama"]["timeout"]
        self.models: dict[str, ModelInfo] = {}
        self._session: Optional[aiohttp.ClientSession] = None

        # Register configured models
        for key, mcfg in self.config["models"].items():
            self.models[mcfg["name"]] = ModelInfo(
                name=mcfg["name"],
                display_name=mcfg["display_name"],
                device=mcfg["device"],
                max_tokens=mcfg["max_tokens"],
                priority=mcfg["priority"],
            )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def check_ollama_health(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.ollama_url}/api/tags") as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def list_remote_models(self) -> list[str]:
        """List models available in Ollama."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.ollama_url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model via Ollama API."""
        if model_name in self.models:
            self.models[model_name].status = ModelStatus.PULLING
        logger.info(f"Pulling model: {model_name}")

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.ollama_url}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                if resp.status == 200:
                    if model_name in self.models:
                        self.models[model_name].status = ModelStatus.READY
                    logger.info(f"Model pulled successfully: {model_name}")
                    return True
                else:
                    error = await resp.text()
                    logger.error(f"Pull failed ({resp.status}): {error}")
                    if model_name in self.models:
                        self.models[model_name].status = ModelStatus.ERROR
                        self.models[model_name].last_error = error
                    return False
        except Exception as e:
            logger.error(f"Pull exception for {model_name}: {e}")
            if model_name in self.models:
                self.models[model_name].status = ModelStatus.ERROR
                self.models[model_name].last_error = str(e)
            return False

    async def generate(
        self, model_name: str, prompt: str, system: str = "", stream: bool = False
    ) -> dict:
        """Generate a response from a model."""
        if model_name in self.models:
            self.models[model_name].status = ModelStatus.RUNNING

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
        }
        if system:
            payload["system"] = system

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.ollama_url}/api/generate", json=payload
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    if model_name in self.models:
                        self.models[model_name].status = ModelStatus.READY
                    return result
                else:
                    error = await resp.text()
                    if model_name in self.models:
                        self.models[model_name].status = ModelStatus.ERROR
                    return {"error": error, "status": resp.status}
        except Exception as e:
            if model_name in self.models:
                self.models[model_name].status = ModelStatus.ERROR
            return {"error": str(e)}

    async def chat(
        self, model_name: str, messages: list[dict], stream: bool = False
    ) -> dict:
        """Chat completion via Ollama API."""
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": stream,
        }
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.ollama_url}/api/chat", json=payload
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {"error": await resp.text(), "status": resp.status}
        except Exception as e:
            return {"error": str(e)}

    async def refresh_status(self) -> dict[str, ModelStatus]:
        """Check which configured models are actually available."""
        remote = await self.list_remote_models()
        for name, info in self.models.items():
            if any(name in r for r in remote):
                if info.status not in (ModelStatus.RUNNING, ModelStatus.PULLING):
                    info.status = ModelStatus.READY
            else:
                info.status = ModelStatus.OFFLINE
        return {n: m.status for n, m in self.models.items()}

    def get_available_models(self) -> list[ModelInfo]:
        """Return models that are ready to serve."""
        return [m for m in self.models.values() if m.status == ModelStatus.READY]

    def get_model(self, name: str) -> Optional[ModelInfo]:
        return self.models.get(name)
