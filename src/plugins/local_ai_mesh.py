from __future__ import annotations
import os, aiohttp
from src.plugins.base import PluginAdapter
MESH = os.environ.get("LOCAL_AI_MESH_URL", "http://localhost:8011")
class LocalAIMeshPlugin(PluginAdapter):
    name = "local_ai_mesh"
    async def health(self):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{MESH}/health", timeout=aiohttp.ClientTimeout(total=3)) as r:
                    return {"healthy": r.status == 200}
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    async def list_tools(self):
        return [{"name": "embed", "description": "NPU-first embeddings"}]
    async def invoke_tool(self, name, args):
        if name == "embed":
            async with aiohttp.ClientSession() as s:
                async with s.post(f"{MESH}/embed", json=args) as r:
                    return await r.json()
        raise NotImplementedError(name)
