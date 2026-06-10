"""
Sovereign Core — Heterogeneous Compute Gateway
===============================================
FastAPI application:
  - Routes inference across RTX 5050 | Radeon 780M | Ryzen 7
  - Exposes OpenAI-compatible /v1/chat/completions (for llm_local.py + SAGE)
  - KAIROS agent economy: /kairos/sage, /evolve, /leaderboard
  - Prometheus metrics: /metrics
  - SSE health stream: /status/stream
  - WebSocket event bus: /ws/events
  - Request tracing via X-Request-ID

Environment variables (prefix GATEWAY_):
  GATEWAY_HOST, GATEWAY_PORT, GATEWAY_API_KEY, GATEWAY_CORS_ORIGINS
  GATEWAY_HEALTH_CHECK_INTERVAL, GATEWAY_BACKEND_TIMEOUT
  GATEWAY_FAILURE_THRESHOLD, GATEWAY_RECOVERY_THRESHOLD
  GATEWAY_LATENCY_EMA_ALPHA, GATEWAY_LEDGER_HMAC_SECRET
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from pathlib import Path as _Path

import dataclasses as _dataclasses

from gateway.auction import (
    Auctioneer,
    InsufficientCreditsError,
    ResourceType,
    auctioneer as _auctioneer,
)
from gateway.benchmark import ThroughputBenchmark
from gateway.config import BACKENDS, BACKEND_MAP, settings
from gateway.context import AgentRole, SharedContextLayer
from gateway.diffusion_router import DecodeMode, DiffusionConfig, DiffusionRouter
from gateway.health import HealthMonitor
from gateway.inference import InferenceRequest, InferenceResponse, route_inference
from gateway.kairos import EliteRegistry, KAIROSAgent, KAIROSEvolutionEngine
from gateway.kairos_routes import router as kairos_router
from gateway.mem_evolve import ABTestManager, MemEvolveEngine
from gateway.mem_evolve_routes import router as mem_evolve_router
from gateway.metrics import INFERENCE_COUNTER, INFERENCE_LATENCY, ACTIVE_BACKENDS, record_request, metrics_output
from gateway.models import ModelAssigner
from gateway.pattern_memory import PatternRecord, PatternStore
from gateway.router import GatewayRouter
from gateway.status import router as status_router
from gateway.v1_compat import router as v1_router
from gateway.ws import event_bus, router as ws_router
from gateway.db import get_db, log_event  # persistent SQLite layer
from gateway.auth import AuthMiddleware    # API key auth + rate limiting
from gateway.iron_dome_middleware import iron_dome_guard  # injection screening

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_health_monitor: HealthMonitor
_benchmark: ThroughputBenchmark
_router: GatewayRouter
_context: SharedContextLayer
_pattern_store: PatternStore
_mem_evolve: MemEvolveEngine
_ab_test: ABTestManager
_diffusion_router: DiffusionRouter
_kairos_engine: KAIROSEvolutionEngine
_elite_registry: EliteRegistry
_boot_time: float = time.time()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _health_monitor, _benchmark, _router, _context, _pattern_store, _mem_evolve, _ab_test, _diffusion_router, _kairos_engine, _elite_registry

    # ── Initialize persistent database ───────────────────────────────────
    try:
        _db = get_db()
        log_event("gateway_boot", "main", "Sovereign Core Gateway starting up",
                  metadata={"backends": len(BACKENDS), "port": settings.port})
        logger.info("Persistent database initialized")
    except Exception as _db_exc:
        logger.warning("Database init failed (non-fatal): %s", _db_exc)

    _health_monitor = HealthMonitor(cfg=settings)
    _benchmark = ThroughputBenchmark()
    _router = GatewayRouter(
        health_monitor=_health_monitor,
        assigner=ModelAssigner(),
        benchmark=_benchmark,
        cfg=settings,
    )
    _context = SharedContextLayer()
    _pattern_store = PatternStore()
    _mem_evolve = MemEvolveEngine(_pattern_store)
    _ab_test = ABTestManager(_mem_evolve)
    _diffusion_router = DiffusionRouter()
    _kairos_engine = KAIROSEvolutionEngine()
    _elite_registry = EliteRegistry(_kairos_engine)

    # Attach to app state so all routers can access
    app.state.health_monitor = _health_monitor
    app.state.router = _router
    app.state.benchmark = _benchmark
    app.state.boot_time = _boot_time

    await _health_monitor.start()
    await _router.start()
    asyncio.create_task(event_bus.broadcast_loop(), name="ws-broadcast")

    healthy = sum(1 for b in BACKENDS if _health_monitor.is_healthy(b.id))
    ACTIVE_BACKENDS.set(healthy)

    logger.info(
        "Sovereign Core Gateway ready — %s:%d  |  %d backends  |  "
        "OpenAI compat: /v1/chat/completions  |  SAGE: /kairos/sage",
        settings.host, settings.port, len(BACKENDS),
    )

    yield

    await _health_monitor.stop()
    await _router.stop()
    _pattern_store.close()
    logger.info("Gateway shutdown complete.")


def create_app() -> FastAPI:
    cors_origins = (
        [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
        if hasattr(settings, "cors_origins") and settings.cors_origins != "*"
        else ["*"]
    )

    app = FastAPI(
        title="Sovereign Core Gateway",
        description=(
            "Heterogeneous Compute Gateway — routes inference across RTX 5050, "
            "Radeon 780M, and Ryzen 7. OpenAI-compatible API. KAIROS agent economy."
        ),
        version="2.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request tracing + auth
    @app.middleware("http")
    async def request_tracing(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = time.time()

        # API key auth (optional)
        api_key = getattr(settings, "api_key", None)
        if api_key:
            public = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
            if request.url.path not in public:
                auth = request.headers.get("Authorization", "")
                if not auth.startswith("Bearer ") or auth[7:] != api_key:
                    return JSONResponse({"detail": "Unauthorized"}, status_code=401)

        response = await call_next(request)
        latency = time.time() - request.state.start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{latency * 1000:.2f}ms"
        record_request(request.method, request.url.path, response.status_code, latency)
        return response

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(status_router)
    app.include_router(kairos_router)
    app.include_router(mem_evolve_router)
    app.include_router(ws_router)
    app.include_router(v1_router)   # ← OpenAI-compat: /v1/chat/completions + /v1/models

    # ── Core endpoints ────────────────────────────────────────────────────────

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard() -> HTMLResponse:
        """Real-time command interface — live backend health, KAIROS, latency graphs."""
        html_path = _Path(__file__).parent / "dashboard.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(), status_code=200)
        return HTMLResponse(content="<h1>Dashboard file not found</h1>", status_code=404)

    @app.get("/health", tags=["core"])
    async def health() -> dict:
        healthy = sum(1 for b in BACKENDS if _health_monitor.is_healthy(b.id))
        ACTIVE_BACKENDS.set(healthy)
        return {
            "status": "ok" if healthy > 0 else "degraded",
            "backends": {b.id: _health_monitor.is_healthy(b.id) for b in BACKENDS},
            "healthy_backends": healthy,
            "total_backends": len(BACKENDS),
            "uptime_s": round(time.time() - _boot_time, 2),
            "version": "2.1.0",
        }

    @app.get("/metrics", tags=["core"])
    async def metrics() -> dict:
        return {
            "latency_ema_s": _router._latency.all_latencies(),
            "benchmark": _benchmark.report(),
        }

    @app.get("/benchmark", tags=["benchmark"])
    async def benchmark_stats() -> dict:
        return {
            "backends": _benchmark.report(),
        }

    @app.post("/benchmark/reset", tags=["benchmark"])
    async def benchmark_reset(backend_id: str | None = None) -> dict:
        if backend_id is not None and backend_id not in BACKEND_MAP:
            raise HTTPException(status_code=404, detail=f"Unknown backend: {backend_id}")
        _benchmark.reset(backend_id)
        return {"reset": True, "backend_id": backend_id}

    @app.post("/inference", response_model=InferenceResponse, tags=["inference"])
    async def inference(req: InferenceRequest, request: Request) -> InferenceResponse:
        """Route inference via Ollama format. Use /v1/chat/completions for OpenAI format."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        t0 = time.time()
        try:
            result = await route_inference(req=req, router=_router, request_id=request_id)
            latency = time.time() - t0
            INFERENCE_COUNTER.labels(backend=result.backend_id, model=result.model, status="success").inc()
            INFERENCE_LATENCY.labels(backend=result.backend_id).observe(latency)
            await event_bus.emit("inference.completed", {
                "backend_id": result.backend_id,
                "model": result.model,
                "latency_ms": round(latency * 1000, 2),
                "tokens": result.eval_count,
                "request_id": request_id,
            })
            return result
        except Exception as exc:
            INFERENCE_COUNTER.labels(backend="unknown", model=req.model, status="error").inc()
            logger.error("Inference error [%s]: %s", request_id, exc)
            raise HTTPException(status_code=503, detail=str(exc))

    @app.post("/auction/credits", tags=["auction"])
    async def auction_credits(agent_id: str, amount: int = Query(ge=1)) -> dict:
        return _auctioneer.top_up(agent_id, amount)

    @app.post("/auction/bid", tags=["auction"])
    async def auction_bid(
        agent_id: str,
        resource_type: str,
        backend_id: str,
        votes: int = Query(ge=1),
    ) -> dict:
        if backend_id not in BACKEND_MAP:
            raise HTTPException(status_code=404, detail=f"Unknown backend: {backend_id}")
        try:
            rt = ResourceType(resource_type)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Unknown resource type: {resource_type}")
        try:
            auction_id, bid = _auctioneer.place_bid(agent_id, rt, backend_id, votes)
        except InsufficientCreditsError as e:
            raise HTTPException(status_code=402, detail=str(e))
        await event_bus.emit("auction.bid", {"auction_id": auction_id, "agent_id": agent_id})
        return {
            "auction_id": auction_id,
            "votes": bid.votes,
            "credit_cost_if_winner": bid.credit_cost,
            "agent_id": agent_id,
            "resource_type": resource_type,
            "backend_id": backend_id,
        }

    @app.get("/auction/status", tags=["auction"])
    async def auction_status() -> dict:
        return _auctioneer.status()

    @app.post("/auction/settle", tags=["auction"])
    async def auction_settle(auction_id: Optional[str] = None) -> dict:
        def _result_to_dict(r) -> dict:
            return {
                "auction_id": r.auction_id,
                "resource_type": r.resource_type.value,
                "backend_id": r.backend_id,
                "winner_agent_id": r.winner_agent_id,
                "winning_votes": r.winning_votes,
                "payment_credits": r.payment_credits,
                "settled_at": r.settled_at,
            }
        if auction_id is not None:
            try:
                result = _auctioneer.settle_auction(auction_id)
            except KeyError:
                raise HTTPException(status_code=404, detail=f"Auction not found: {auction_id}")
            return {"settled": [_result_to_dict(result)]}
        results = _auctioneer.settle_all()
        return {"settled": [_result_to_dict(r) for r in results]}

    @app.get("/auction/metrics", tags=["auction"])
    async def auction_metrics() -> dict:
        m = _auctioneer.metrics()
        return _dataclasses.asdict(m)

    @app.get("/ledger/tail", tags=["ledger"])
    async def ledger_tail(n: int = 20) -> dict:
        try:
            from gateway.ledger import SemanticLedger
            ledger = SemanticLedger.instance()
            return {"entries": ledger.tail(n), "total": ledger.count()}
        except Exception as exc:
            return {"entries": [], "total": 0, "error": str(exc)}

    @app.post("/benchmark/run", tags=["benchmark"])
    async def benchmark_run(model_id: str = "default") -> dict:
        try:
            return await _benchmark.run(model_id=model_id, router=_router)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Memory (Pattern Store + MemEvolve) endpoints ──────────────────────────

    import json as _json

    @app.post("/memory/patterns", tags=["memory"], status_code=201)
    async def memory_store_pattern(
        model_id: str,
        backend_id: str,
        pattern_type: str,
        context: str = "{}",
        recommendation: str = "{}",
    ) -> dict:
        try:
            ctx = _json.loads(context)
        except Exception:
            raise HTTPException(status_code=422, detail="context must decode to a JSON object")
        if not isinstance(ctx, dict):
            raise HTTPException(status_code=422, detail="context must decode to a JSON object")
        try:
            rec = _json.loads(recommendation)
        except Exception:
            raise HTTPException(status_code=422, detail="recommendation must decode to a JSON object")
        if not isinstance(rec, dict):
            raise HTTPException(status_code=422, detail="recommendation must decode to a JSON object")
        record = PatternRecord(
            model_id=model_id,
            backend_id=backend_id,
            pattern_type=pattern_type,
            context=ctx,
            recommendation=rec,
        )
        _pattern_store.store(record)
        return record.to_dict()

    @app.get("/memory/patterns", tags=["memory"])
    async def memory_lookup_patterns(
        model_id: Optional[str] = None,
        backend_id: Optional[str] = None,
        pattern_type: Optional[str] = None,
        strategy: str = "evolved",
        context: Optional[str] = None,
        limit: int = 20,
    ) -> dict:
        if context is not None:
            try:
                ctx = _json.loads(context)
            except Exception:
                raise HTTPException(status_code=422, detail="context must decode to a JSON object")
            if not isinstance(ctx, dict):
                raise HTTPException(status_code=422, detail="context must decode to a JSON object")
        else:
            ctx = None
        patterns = _pattern_store.lookup(
            model_id=model_id,
            backend_id=backend_id,
            pattern_type=pattern_type,
            limit=limit,
        )
        ranked = _mem_evolve.rank_patterns(patterns, strategy=strategy, query_context=ctx)
        return {
            "strategy": strategy,
            "count": len(ranked),
            "patterns": [p.to_dict() for p in ranked],
        }

    @app.post("/memory/outcome", tags=["memory"])
    async def memory_record_outcome(pattern_id: str, success: bool) -> dict:
        try:
            outcome = _pattern_store.record_outcome(pattern_id, success=success)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"Pattern not found: {pattern_id}")
        return outcome.to_dict()

    @app.get("/memory/stats", tags=["memory"])
    async def memory_stats() -> dict:
        return _pattern_store.get_stats().to_dict()

    @app.get("/memory/evolve/status", tags=["memory"])
    async def memory_evolve_status() -> dict:
        return _mem_evolve.strategy_comparison()

    @app.post("/memory/ab-test/assign", tags=["memory"])
    async def memory_ab_assign(request_id: str) -> dict:
        variant = _ab_test.assign(request_id)
        return {"request_id": request_id, "variant": variant}

    @app.get("/memory/ab-test", tags=["memory"])
    async def memory_ab_report() -> dict:
        return _ab_test.comparison()

    # ── Diffusion router endpoints ────────────────────────────────────────────

    @app.get("/diffusion/metrics", tags=["diffusion"])
    async def diffusion_metrics() -> dict:
        return _diffusion_router.metrics()

    @app.post("/diffusion/generate", tags=["diffusion"])
    async def diffusion_generate(
        num_tokens: int = 256,
        mode: str = "parallel",
        device: Optional[str] = None,
    ) -> dict:
        router = DiffusionRouter(DiffusionConfig(device=device)) if device else _diffusion_router
        try:
            decode_mode = DecodeMode(mode)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid mode: {mode}")
        try:
            return router.generate(num_tokens, decode_mode).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/diffusion/compare", tags=["diffusion"])
    async def diffusion_compare(
        num_tokens: int = 256,
        device: Optional[str] = None,
    ) -> dict:
        router = DiffusionRouter(DiffusionConfig(device=device)) if device else _diffusion_router
        return router.compare(num_tokens).to_dict()

    # ── KAIROS agent evolution endpoints ──────────────────────────────────────

    def _agent_to_dict(agent: KAIROSAgent) -> dict:
        d = _dataclasses.asdict(agent)
        d["fitness_score"] = agent.fitness_score
        return d

    @app.get("/kairos/elites", tags=["kairos"])
    async def kairos_list_elites() -> dict:
        elites = _elite_registry.list_elites()
        return {"agents": [_agent_to_dict(a) for a in elites]}

    @app.get("/kairos/agent/{agent_id}", tags=["kairos"])
    async def kairos_get_agent(agent_id: str) -> dict:
        try:
            agent = _elite_registry.get(agent_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        return _agent_to_dict(agent)

    @app.post("/kairos/evolve/{agent_id}", tags=["kairos"])
    async def kairos_evolve_agent(agent_id: str) -> dict:
        try:
            agent = _elite_registry.get(agent_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        evolved = _kairos_engine.evolve_agent(agent)
        _elite_registry.register(evolved)
        return _agent_to_dict(evolved)

    @app.post("/kairos/reconstruct/{agent_id}", tags=["kairos"])
    async def kairos_reconstruct_agent(agent_id: str) -> dict:
        try:
            new_agent = _kairos_engine.reconstruct_from_archive(agent_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Ancestor not found in archive: {agent_id}")
        _elite_registry.register(new_agent)
        return _agent_to_dict(new_agent)

    @app.get("/kairos/metrics", tags=["kairos"])
    async def kairos_metrics() -> dict:
        return _elite_registry.metrics()

    # ── Shared context layer endpoints ────────────────────────────────────────

    @app.post("/context/write", tags=["context"])
    async def context_write(
        role: AgentRole,
        backend_id: str,
        document: str,
        trace_id: Optional[str] = None,
    ) -> dict:
        entry_id = _context.write(role, backend_id, document, trace_id=trace_id)
        return {
            "entry_id": entry_id,
            "role": role.value,
            "backend_id": backend_id,
            "document": document,
            "trace_id": trace_id or "",
        }

    @app.get("/context/read", tags=["context"])
    async def context_read(
        role: Optional[str] = None,
        backend_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        limit: int = 100,
    ) -> dict:
        if trace_id is not None:
            entries = _context.read_by_trace(trace_id)
        elif backend_id is not None:
            entries = _context.read_by_backend(backend_id, limit=limit)
        elif role is not None:
            try:
                r = AgentRole(role)
            except ValueError:
                raise HTTPException(status_code=422, detail=f"Unknown role: {role}")
            entries = _context.read_by_role(r, limit=limit)
        else:
            entries = _context.read_all(limit=limit)
        return {"entries": [e.as_dict() for e in entries], "count": len(entries)}

    @app.get("/context/cross-gpu/{backend_id}", tags=["context"])
    async def context_cross_gpu(backend_id: str, limit: int = 100) -> dict:
        entries = _context.read_cross_gpu(backend_id, limit=limit)
        return {
            "backend_id": backend_id,
            "peer_entries": [e.as_dict() for e in entries],
            "count": len(entries),
        }

    @app.get("/context/count", tags=["context"])
    async def context_count() -> dict:
        return {"count": _context.count()}

    @app.delete("/context/clear", tags=["context"])
    async def context_clear() -> dict:
        cleared = _context.clear()
        return {"cleared": cleared}

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "gateway.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        loop="asyncio",
        log_level="info",
    )
