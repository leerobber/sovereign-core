from __future__ import annotations
import aiohttp
from src.plugins.base import PluginAdapter

ECO = "http://localhost:8081"


class EconomyPlugin(PluginAdapter):
    name = "agent_economy"

    async def health(self) -> dict:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{ECO}/", timeout=aiohttp.ClientTimeout(total=3)) as r:
                    return {"healthy": r.status == 200}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def list_tools(self) -> list[dict]:
        return [{"name": "credit_agent", "description": "Post task completion credit"}]

    async def invoke_tool(self, name: str, args: dict):
        if name == "credit_agent":
            async with aiohttp.ClientSession() as s:
                async with s.post(f"{ECO}/task/complete", json=args) as r:
                    return await r.json()
        raise NotImplementedError(name)
