"""
gateway/v1_compat.py — OpenAI-compatible /v1/chat/completions shim

This is the critical missing link. llm_local.py and every HyperAgents
component calls:
    POST /v1/chat/completions

But the gateway only had /inference (Ollama format).
This shim:
  1. Accepts OpenAI-format requests
  2. Converts to Ollama format
  3. Routes through GatewayRouter (RTX5050 → Radeon → Ryzen7)
  4. Returns OpenAI-format response

So llm_local.py gets a real response from the actual GPU cluster.
No mock. No stub. The real SAGE loop runs on real hardware.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
try:
    from gateway.iron_dome_middleware import iron_dome_guard as _iron_dome
    _IRON_DOME_ACTIVE = True
except ImportError:
    _IRON_DOME_ACTIVE = False
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compat"])


# ── OpenAI-format models ──────────────────────────────────────────────────────

class OAIMessage(BaseModel):
    role: str
    content: str


class OAIChatRequest(BaseModel):
    model: str = "auto"
    messages: List[OAIMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    stream: bool = False
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


class OAIChoice(BaseModel):
    index: int
    message: OAIMessage
    finish_reason: str = "stop"


class OAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OAIChoice]
    usage: OAIUsage


# ── Route handler ─────────────────────────────────────────────────────────────

@router.post("/chat/completions", response_model=OAIChatResponse)
async def chat_completions(req: OAIChatRequest, request: Request) -> OAIChatResponse:
    """
    OpenAI-compatible chat completions endpoint.
    Routes through the Sovereign Core GPU mesh.
    Called by: llm_local.py, contentai-pro LLM adapter, Termux agent.
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    t0 = time.time()

    # ── Iron Dome screening ──────────────────────────────────────────────
    if _IRON_DOME_ACTIVE:
        full_prompt = " ".join(m.content for m in req.messages)
        _allowed, _reason = _iron_dome.screen(full_prompt, req.model, "v1_compat")
        if not _allowed:
            raise HTTPException(status_code=400, detail=_reason)

    # Get router from app state
    try:
        gateway_router = request.app.state.router
    except AttributeError:
        raise HTTPException(status_code=503, detail="Gateway router not initialized")

    # Convert OpenAI messages → Ollama chat format
    from gateway.inference import InferenceRequest, InferenceOptions, ChatMessage, route_inference

    chat_messages = [
        ChatMessage(role=m.role, content=m.content)
        for m in req.messages
    ]

    # Extract system message if present
    system_content = None
    non_system = []
    for m in chat_messages:
        if m.role == "system":
            system_content = m.content
        else:
            non_system.append(m)

    # Build inference request
    infer_req = InferenceRequest(
        model=req.model,
        messages=non_system if non_system else chat_messages,
        options=InferenceOptions(
            temperature=req.temperature,
            top_p=req.top_p,
            num_predict=req.max_tokens,
            stop=req.stop or [],
            seed=req.seed,
        ),
        stream=False,
    )

    # If there's a system message, prepend it as the first user message context
    # (Ollama handles system via the model's system field)
    if system_content and non_system:
        # Prepend system context to first user message
        first = non_system[0]
        infer_req.messages[0] = ChatMessage(
            role="user",
            content=f"[System]: {system_content}\n\n[User]: {first.content}"
        )

    try:
        result = await route_inference(
            req=infer_req,
            router=gateway_router,
            request_id=request_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("v1/chat/completions error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    latency = time.time() - t0

    # Estimate tokens (Ollama gives us real counts)
    prompt_tokens = result.prompt_eval_count or _estimate_tokens(req.messages)
    completion_tokens = result.eval_count or _estimate_tokens_str(result.response)

    return OAIChatResponse(
        id=request_id,
        created=int(t0),
        model=result.model,
        choices=[OAIChoice(
            index=0,
            message=OAIMessage(role="assistant", content=result.response),
            finish_reason="stop",
        )],
        usage=OAIUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@router.get("/models")
async def list_models(request: Request) -> dict:
    """Return available models from connected backends."""
    from gateway.config import BACKENDS
    models = []
    for b in BACKENDS:
        # Common models per backend type
        if "nvidia" in b.device_type.value:
            model_ids = ["qwen2.5-32b-awq", "nemotron-3-nano", "qwen2.5:14b"]
        elif "amd" in b.device_type.value:
            model_ids = ["deepseek-coder-33b", "deepseek-coder:6.7b"]
        else:
            model_ids = ["llama3.2:3b", "mistral-7b"]

        for mid in model_ids:
            models.append({
                "id": mid,
                "object": "model",
                "owned_by": f"sovereign-{b.id}",
                "backend": b.id,
                "device": b.device_type.value,
            })

    # Add auto-routing virtual model
    models.insert(0, {
        "id": "auto",
        "object": "model",
        "owned_by": "sovereign-gateway",
        "backend": "auto-routed",
        "device": "heterogeneous",
    })
    return {"object": "list", "data": models}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_tokens(messages: List[OAIMessage]) -> int:
    """Rough token estimate: ~4 chars per token."""
    total_chars = sum(len(m.content) for m in messages)
    return max(1, total_chars // 4)


def _estimate_tokens_str(text: str) -> int:
    return max(1, len(text) // 4)
