"""Repo registry — loads config/registry.yaml and probes health."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp
import yaml
from pathlib import Path

LOG = logging.getLogger("sovereign.registry")
_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RepoStatus:
    name: str
    role: str
    healthy: bool = False
    detail: str = ""
    ports: dict = field(default_factory=dict)


class RepoRegistry:
    def __init__(self, registry_path: Optional[Path] = None):
        path = registry_path or _ROOT / "config" / "registry.yaml"
        with open(path) as f:
            self._data = yaml.safe_load(f) or {}
        self.ecosystem = self._data.get("sovereign_ecosystem", {})

    def list_repos(self) -> list[dict[str, Any]]:
        out = []
        for name, cfg in self.ecosystem.items():
            if name.startswith("_") or not isinstance(cfg, dict):
                continue
            out.append({"name": name, "role": cfg.get("role", ""), "config": cfg})
        return out

    async def probe(self, name: str) -> RepoStatus:
        cfg = self.ecosystem.get(name, {})
        role = cfg.get("role", "")
        port = cfg.get("port")
        ports = cfg.get("ports", {})
        health_url = None
        if port:
            health_url = f"http://localhost:{port}/health"
        elif name == "gh05t3" and ports:
            health_url = f"http://localhost:{ports.get('gateway_v3', 8002)}/health"
        elif name == "agent_economy":
            health_url = "http://localhost:8081/"
        elif name == "local_ai_mesh":
            health_url = f"http://localhost:{cfg.get('port', 8011)}/health"

        healthy, detail = False, "no health endpoint"
        if health_url:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=3)) as r:
                        healthy = r.status == 200
                        detail = f"HTTP {r.status}"
            except Exception as e:
                detail = str(e)[:120]
        return RepoStatus(name=name, role=role, healthy=healthy, detail=detail, ports=ports or {"port": port})

    async def probe_all(self) -> list[RepoStatus]:
        names = [n for n, c in self.ecosystem.items() if isinstance(c, dict) and c.get("role")]
        return await asyncio.gather(*[self.probe(n) for n in names])
