"""System Health Monitoring"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import psutil

logger = logging.getLogger("sovereign.health")


@dataclass
class SystemHealth:
    cpu_percent: float
    ram_total_gb: float
    ram_used_gb: float
    ram_percent: float
    gpu_available: bool
    gpu_name: str
    gpu_vram_total_mb: int
    gpu_vram_used_mb: int
    gpu_utilization_percent: float
    ollama_running: bool
    uptime_seconds: float


class HealthMonitor:
    """Monitors system resources and service health."""

    def __init__(self):
        self._start_time = time.time()

    async def get_system_health(self) -> SystemHealth:
        ram = psutil.virtual_memory()
        gpu_info = await self._check_gpu()

        return SystemHealth(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            ram_total_gb=round(ram.total / (1024**3), 1),
            ram_used_gb=round(ram.used / (1024**3), 1),
            ram_percent=ram.percent,
            gpu_available=gpu_info["available"],
            gpu_name=gpu_info["name"],
            gpu_vram_total_mb=gpu_info["vram_total"],
            gpu_vram_used_mb=gpu_info["vram_used"],
            gpu_utilization_percent=gpu_info["utilization"],
            ollama_running=self._check_ollama(),
            uptime_seconds=round(time.time() - self._start_time, 1),
        )

    async def _check_gpu(self) -> dict:
        """Check GPU status via nvidia-smi."""
        info = {
            "available": False,
            "name": "N/A",
            "vram_total": 0,
            "vram_used": 0,
            "utilization": 0.0,
        }
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            if proc.returncode == 0 and stdout:
                parts = stdout.decode().strip().split(", ")
                if len(parts) >= 4:
                    info["available"] = True
                    info["name"] = parts[0]
                    info["vram_total"] = int(parts[1])
                    info["vram_used"] = int(parts[2])
                    info["utilization"] = float(parts[3])
        except Exception as e:
            logger.debug(f"GPU check failed (expected in non-GPU env): {e}")
        return info

    @staticmethod
    def _check_ollama() -> bool:
        """Check if Ollama process is running."""
        for proc in psutil.process_iter(["name"]):
            try:
                if "ollama" in (proc.info["name"] or "").lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    async def to_dict(self) -> dict:
        h = await self.get_system_health()
        return {
            "cpu_percent": h.cpu_percent,
            "ram": {
                "total_gb": h.ram_total_gb,
                "used_gb": h.ram_used_gb,
                "percent": h.ram_percent,
            },
            "gpu": {
                "available": h.gpu_available,
                "name": h.gpu_name,
                "vram_total_mb": h.gpu_vram_total_mb,
                "vram_used_mb": h.gpu_vram_used_mb,
                "utilization_percent": h.gpu_utilization_percent,
            },
            "ollama_running": h.ollama_running,
            "uptime_seconds": h.uptime_seconds,
        }
