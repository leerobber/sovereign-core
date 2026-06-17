
"""MCP aggregator — exposes sovereign-core + GH05T3 tools."""
from __future__ import annotations

import os
from typing import Any

import aiohttp
from fastapi import APIRouter, HTTPException

from src.plugins.gh05t3 import GH05T3Plugin
from src.plugins.economy import EconomyPlugin

router = APIRouter(prefix="/mcp", tags=["mcp"])

_plugins = {
    "gh05t3": GH05T3Plugin(),
    "economy": EconomyPlugin(),
}


@router.get("/tools")
async def list_tools() -> dict[str, Any]:
    tools = []
    for name, plugin in _plugins.items():
        for t in await plugin.list_tools():
            tools.append({**t, "plugin": name})
    return {"tools": tools}


@router.post("/tools/{plugin}/{tool_name}")
async def invoke_tool(plugin: str, tool_name: str, args: dict | None = None) -> dict:
    p = _plugins.get(plugin)
    if not p:
        raise HTTPException(404, f"Unknown plugin: {plugin}")
    try:
        result = await p.invoke_tool(tool_name, args or {})
        return {"ok": True, "result": result}
    except NotImplementedError as e:
        raise HTTPException(501, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/health")
async def mcp_health() -> dict:
    status = {}
    for name, plugin in _plugins.items():
        status[name] = await plugin.health()
    return {"plugins": status}
