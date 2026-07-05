"""
Minimal stub for the retired diffusion_router prototype (RES-10 era).

The real implementation was experimental and has been removed from the
active codebase.  Tests that reference it are ignored in CI.
This stub allows `pytest` collection to succeed without --ignore.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter


class DecodeMode(str, Enum):
    GREEDY = "greedy"
    SAMPLE = "sample"
    PARALLEL = "parallel"
    AUTOREGRESSIVE = "autoregressive"


class DiffusionConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class GenerationResult:
    def __init__(self, text: str = "", tokens: int = 0, **kwargs: Any) -> None:
        self.text = text
        self.tokens = tokens
        self.__dict__.update(kwargs)


class ComparisonResult:
    def __init__(self, winner: str = "stub", score: float = 0.5, **kwargs: Any) -> None:
        self.winner = winner
        self.score = score
        self.__dict__.update(kwargs)


class TokensPerWattTracker:
    def __init__(self) -> None:
        self.total_tokens = 0
        self.total_energy_j = 0.0

    def record(self, tokens: int, energy_j: float) -> None:
        self.total_tokens += tokens
        self.total_energy_j += energy_j

    def tpw(self) -> float:
        if self.total_energy_j <= 0:
            return 0.0
        return self.total_tokens / self.total_energy_j


class DiffusionRouter:
    """Stub class so importing the module and constructing works."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.router = APIRouter(prefix="/diffusion", tags=["diffusion-stub"])
        self._enabled = False
        self.config = DiffusionConfig()

    async def route(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"stub": True, "enabled": self._enabled}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self


# Provide symbols the old test file referenced
__all__ = [
    "ComparisonResult",
    "DecodeMode",
    "DiffusionConfig",
    "DiffusionRouter",
    "GenerationResult",
    "TokensPerWattTracker",
]
