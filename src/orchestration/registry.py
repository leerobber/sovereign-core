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


def _windows_host() -> str:
    """Host reachable from WSL for Windows-native GH05T3 (override via GH05T3_WINDOWS_HOST)."""
    import os
    import subprocess
    if os.environ.get("GH05T3_WINDOWS_HOST"):
        return os.environ["GH05T3_WINDOWS_HOST"]
    try:
        out = subprocess.check_output(
            ["ip", "route", "show", "default"], text=True, timeout=2,
        )
        parts = out.split()
        if "via" in parts:
            return parts[parts.index("via") + 1]
    except Exception:
        pass
    return "127.0.0.1"


def _gh05t3_gateway_url(ports: dict) -> str:
    """GH05T3 gateway health base URL.

    Default: localhost when GH05T3 runs in WSL (same network namespace).
    Set GH05T3_RUNTIME=windows or GH05T3_GATEWAY_URL to target Windows-native stack.
    """
    import os
    if url := os.environ.get("GH05T3_GATEWAY_URL"):
        return url.rstrip("/")
    port = ports.get("gateway_v3", 8002)
    if os.environ.get("GH05T3_RUNTIME", "wsl").lower() == "windows":
        return f"http://{_windows_host()}:{port}"
    return f"http://localhost:{port}"


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
            health_url = f"{_gh05t3_gateway_url(ports)}/health"
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
