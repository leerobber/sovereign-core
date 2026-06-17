
"""GhostRecall-style memory retrieval via local-ai-mesh / NPU embed."""
from __future__ import annotations

import os
from typing import Any, Optional

import aiohttp

MESH_URL = os.environ.get("LOCAL_AI_MESH_URL", "http://localhost:8011").rstrip("/")
NPU_URL = os.environ.get("NPU_EMBED_URL", "http://localhost:8111").rstrip("/")
EMBED_STRATEGY = os.environ.get("EMBED_STRATEGY", "npu_then_ollama")


async def embed_text(text: str, timeout: float = 10.0) -> list[float]:
    """Embed text using NPU-first strategy."""
    if EMBED_STRATEGY == "npu_then_ollama":
        for url in (f"{MESH_URL}/embed", f"{NPU_URL}/embed"):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post(url, json={"text": text}, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                        if r.status == 200:
                            data = await r.json()
                            vec = data.get("embedding") or data.get("vector")
                            if vec:
                                return vec
            except Exception:
                continue
    return []


async def recall(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Query memory via local-ai-mesh recall endpoint."""
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{MESH_URL}/recall",
                json={"query": query, "top_k": top_k},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    return data.get("results", data.get("memories", []))
    except Exception:
        pass
    return []
