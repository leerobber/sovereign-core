from __future__ import annotations
import os
import aiohttp
from src.plugins.base import PluginAdapter

GW = os.environ.get("GH05T3_GATEWAY_URL", "http://localhost:8002")


class GH05T3Plugin(PluginAdapter):
    name = "gh05t3"

    async def health(self) -> dict:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{GW}/health", timeout=aiohttp.ClientTimeout(total=3)) as r:
                    return {"healthy": r.status == 200, "status": r.status}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def list_tools(self) -> list[dict]:
        return [
            {"name": "swarm_status", "description": "GH05T3 gateway health"},
            {"name": "ghost_chat", "description": "Chat via GH05T3 MCP path"},
        ]

    async def invoke_tool(self, name: str, args: dict):
        if name == "swarm_status":
            return await self.health()
        raise NotImplementedError(name)
