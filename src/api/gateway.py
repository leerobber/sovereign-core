"""FastAPI Gateway — KAN-18"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.memory.store import MemoryStore
from src.models.manager import ModelManager, ModelStatus
from src.routing.router import Router
from src.utils.health import HealthMonitor

from src.logging.run_logger import RunLogger
from src.orchestration.registry import RepoRegistry
from src.agents import AGENT_MODEL_MAP, SWARM_AGENTS
from src.plugins.gh05t3 import GH05T3Plugin
from src.mcp.server import router as mcp_router
import aiohttp


logger = logging.getLogger("sovereign.api")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    model: Optional[str] = Field(None, description="Model override (optional, router decides if omitted)")
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    stream: bool = Field(False, description="Enable streaming")
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    session_id: Optional[str] = Field(None, description="Persist multi-turn conversation")


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: dict = {}
    routing: dict = {}
    session_id: Optional[str] = None


class ModelEntry(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "sovereign-core"
    ready: bool


class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    system: str = ""
    stream: bool = False


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


model_manager: Optional[ModelManager] = None
router: Optional[Router] = None
memory: Optional[MemoryStore] = None
health: Optional[HealthMonitor] = None
run_logger: Optional[RunLogger] = None
repo_registry: Optional[RepoRegistry] = None
gh05t3_plugin: Optional[GH05T3Plugin] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, router, memory, health, run_logger, repo_registry, gh05t3_plugin

    config = load_config()
    model_manager = ModelManager()
    router = Router(model_manager, config)
    memory = MemoryStore(config.get("memory", {}).get("database_path", "data/sovereign.db"))
    health = HealthMonitor()
    run_logger = RunLogger()
    repo_registry = RepoRegistry()
    gh05t3_plugin = GH05T3Plugin()

    await memory.initialize()
    await model_manager.refresh_status()

    logger.info("Sovereign Core API started")
    yield

    await memory.close()
    await model_manager.close()
    logger.info("Sovereign Core API stopped")


app = FastAPI(
    title="Sovereign Core",
    description="Local AI inference gateway with intelligent routing",
    version="1.0.0",
    lifespan=lifespan,
)

_STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

app.include_router(mcp_router)


@app.get("/chat")
async def chat_ui():
    chat_html = _STATIC_DIR / "chat.html"
    if not chat_html.is_file():
        raise HTTPException(status_code=404, detail="chat.html not found")
    return FileResponse(str(chat_html))


@app.get("/health")
async def health_check():
    assert health is not None and model_manager is not None

    system_health = await health.to_dict()
    model_status = {n: m.status.value for n, m in model_manager.models.items()}
    ready_count = sum(
        1 for m in model_manager.models.values()
        if getattr(m.status, "value", m.status) == "ready"
    )
    total = len(model_manager.models)
    if ready_count == 0:
        status = "degraded"
    elif ready_count < total:
        status = "degraded"
    else:
        status = "healthy"

    return {
        "status": status,
        "system": system_health,
        "models": model_status,
        "ready_models": ready_count,
    }


@app.get("/v1/models")
async def list_models():
    assert model_manager is not None
    await model_manager.refresh_status()
    entries = [
        ModelEntry(id=m.name, ready=(m.status.value == "ready"))
        for m in model_manager.models.values()
    ]
    return {"object": "list", "data": [e.model_dump() for e in entries]}


@app.get("/metrics")
async def get_metrics():
    assert memory is not None and model_manager is not None
    summary = await memory.get_metrics_summary()
    percentiles = {}
    for model_name in summary:
        percentiles[model_name] = await memory.get_percentile_latencies(model_name)
    return {"models": summary, "percentiles": percentiles}


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    assert router is not None and memory is not None

    start = time.perf_counter()
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    session_id = request.session_id or str(uuid.uuid4())

    for msg in messages:
        await memory.save_message(session_id, msg["role"], msg["content"])

    result = await router.chat(messages, model_override=request.model)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    routing_info = result.pop("_routing", {})
    assistant_content = result.get("message", {}).get("content", "")
    model_used = result.get("model", routing_info.get("model_name", "unknown"))

    await memory.save_message(session_id, "assistant", assistant_content, model_used)
    await memory.record_metric(
        model=model_used,
        latency_ms=elapsed_ms,
        routed_by="router",
        complexity=routing_info.get("complexity", ""),
        is_fallback=routing_info.get("is_fallback", False),
    )

    if run_logger is not None:
        try:
            await run_logger.log_chat(
                source="sovereign-core",
                messages=messages,
                output=assistant_content,
                routing=routing_info,
                session_id=session_id,
                model=model_used,
                latency_ms=elapsed_ms,
            )
        except Exception as exc:
            logger.warning("Run log write failed: %s", exc)

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model_used,
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=assistant_content))],
        usage={
            "prompt_tokens": result.get("prompt_eval_count", 0),
            "completion_tokens": result.get("eval_count", 0),
            "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
        },
        routing=routing_info,
        session_id=session_id,
    )


@app.post("/v1/generate")
async def generate(request: GenerateRequest):
    assert router is not None

    start = time.perf_counter()
    result = await router.generate(
        request.prompt, system=request.system, model_override=request.model
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if "error" in result and "_routing" not in result:
        raise HTTPException(status_code=503, detail=result["error"])

    result["latency_ms"] = round(elapsed_ms, 2)
    return result


@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    assert router is not None
    await ws.accept()
    session_id = str(uuid.uuid4())
    logger.info("WebSocket session started: %s", session_id)

    try:
        while True:
            data = await ws.receive_json()
            messages = data.get("messages", [])
            if not messages:
                await ws.send_json({"error": "No messages provided"})
                continue

            result = await router.chat(messages, model_override=data.get("model"))
            routing_info = result.pop("_routing", {})
            content = result.get("message", {}).get("content", "")

            await ws.send_json({
                "session_id": session_id,
                "content": content,
                "model": result.get("model", ""),
                "routing": routing_info,
            })
    except WebSocketDisconnect:
        logger.info("WebSocket session ended: %s", session_id)

class FeedbackRequest(BaseModel):
    feedback: str = Field(..., description="Human correction for training")


class DelegateRequest(BaseModel):
    task: str = Field(..., description="Task to delegate to agent")
    context: Optional[str] = None


class AgentRegisterRequest(BaseModel):
    id: str
    role: str
    model: Optional[str] = None
    channels: list[str] = Field(default_factory=list)


@app.get("/v1/repos")
async def list_repos():
    assert repo_registry is not None
    statuses = await repo_registry.probe_all()
    return {
        "repos": [
            {"name": s.name, "role": s.role, "healthy": s.healthy, "detail": s.detail, "ports": s.ports}
            for s in statuses
        ]
    }


@app.get("/v1/agents")
async def list_agents():
    return {"agents": SWARM_AGENTS, "model_map": AGENT_MODEL_MAP}


@app.post("/v1/agents/register")
async def register_agent(req: AgentRegisterRequest):
    entry = {"id": req.id.upper(), "role": req.role, "channels": req.channels}
    if req.id.upper() not in [a["id"] for a in SWARM_AGENTS]:
        SWARM_AGENTS.append(entry)
    if req.model:
        AGENT_MODEL_MAP[req.id.upper()] = req.model
    return {"ok": True, "agent": entry}


@app.post("/v1/agents/{agent_id}/delegate")
async def delegate_to_agent(agent_id: str, request: DelegateRequest):
    assert router is not None and run_logger is not None
    agent_key = agent_id.upper()
    model = AGENT_MODEL_MAP.get(agent_key)
    system = f"You are {agent_key}, a sovereign swarm agent."
    if request.context:
        system += f" Context: {request.context}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": request.task},
    ]
    result = await router.chat(messages, model_override=model)
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    content = result.get("message", {}).get("content", "")
    routing = result.pop("_routing", {})
    await run_logger.log_chat(
        source="sovereign-core",
        messages=messages,
        output=content,
        agent_name=agent_key,
        routing=routing,
        model=result.get("model"),
    )
    return {"agent": agent_key, "model": result.get("model"), "output": content, "routing": routing}


@app.post("/v1/runs/{task_id}/feedback")
async def add_run_feedback(task_id: str, request: FeedbackRequest):
    assert run_logger is not None
    ok = await run_logger.add_feedback(task_id, request.feedback)
    if not ok:
        raise HTTPException(status_code=404, detail="task_id not found in logs")
    return {"ok": True, "task_id": task_id}


@app.get("/v1/models/finetuned")
async def list_finetuned_models():
    manifests_dir = Path("/home/leer4/sovereign-project/datasets/manifests")
    versions = []
    if manifests_dir.is_dir():
        for p in sorted(manifests_dir.glob("*.json")):
            try:
                import json
                versions.append({"file": p.name, "data": json.loads(p.read_text())})
            except Exception:
                versions.append({"file": p.name, "error": "parse_failed"})
    return {"model_map": AGENT_MODEL_MAP, "manifests": versions}

