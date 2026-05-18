"""MemEvolve Routes — FastAPI endpoints for MemEvolveEngine (RES-12).

Prefix: /kairos/mem-evolve
Tags:   mem-evolve

Singletons are lazily initialised on first request so PatternStore (and its
SQLite connection) is not opened until the route is actually called.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from gateway.mem_evolve import ABTestManager, MemEvolveEngine

router = APIRouter(prefix="/kairos/mem-evolve", tags=["mem-evolve"])

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_engine: Optional[MemEvolveEngine] = None
_ab_manager: Optional[ABTestManager] = None


def _get_engine() -> MemEvolveEngine:
    global _engine
    if _engine is None:
        from gateway.pattern_memory import PatternStore
        _engine = MemEvolveEngine(PatternStore())
    return _engine


def _get_ab_manager() -> ABTestManager:
    global _ab_manager
    if _ab_manager is None:
        _ab_manager = ABTestManager(_get_engine())
    return _ab_manager


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RecordRequest(BaseModel):
    """Body for POST /kairos/mem-evolve/record."""

    request_id: str
    success: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status", summary="Evolved vs. static strategy comparison")
async def get_status() -> dict[str, Any]:
    """Return strategy comparison merged with the evolved strategy's
    current generation and fitness score."""
    engine = _get_engine()
    comparison = engine.strategy_comparison()
    return {
        **comparison,
        "generation": engine.evolved_strategy.generation,
        "fitness_score": engine.evolved_strategy.fitness_score,
    }


@router.post("/evolve", summary="Run one meta-evolution step")
async def post_evolve() -> dict[str, Any]:
    """Trigger a single MemEvolve generation and return the result dict."""
    engine = _get_engine()
    return engine.evolve()


@router.post("/record", summary="Record an A/B trial result")
async def post_record(body: RecordRequest) -> dict[str, Any]:
    """Record the outcome for *request_id*'s assigned A/B variant.

    Returns the variant that was recorded and the current comparison stats.
    """
    ab = _get_ab_manager()
    variant = ab.record_result(body.request_id, success=body.success)
    return {
        "variant": variant,
        "comparison": ab.comparison(),
    }


@router.get("/comparison", summary="A/B test comparison statistics")
async def get_comparison() -> dict[str, Any]:
    """Return current A/B test comparison statistics."""
    return _get_ab_manager().comparison()


@router.post("/reset", summary="Reset A/B assignment cache")
async def post_reset() -> dict[str, str]:
    """Clear cached request-to-variant assignments.

    Trial counts on the engine are preserved; only the assignment cache
    is cleared (use to start a new experiment epoch).
    """
    _get_ab_manager().reset_assignments()
    return {"status": "ok"}
