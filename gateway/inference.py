"""
gateway/inference.py — Inference Endpoint Models & Routing Logic

Provides the /inference POST endpoint data models and the route_inference()
function that selects and calls the best available backend.

Supports:
  - Ollama-compatible API (generate + chat modes)
  - Model override (force a specific backend)
  - Streaming passthrough (SSE)
  - Per-request timeout override
  - Full response including token counts and backend metadata
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Request / Response models ─────────────────────────────────────────────────

class InferenceOptions(BaseModel):
    """Ollama-compatible generation options."""
    num_predict: int = Field(default=1024, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0)
    repeat_penalty: float = Field(default=1.1, ge=0.0)
    seed: Optional[int] = None
    stop: List[str] = Field(default_factory=list)
    num_ctx: int = Field(default=4096, ge=128)


class ChatMessage(BaseModel):
    role: str  # system | user | assistant
    content: str


class InferenceRequest(BaseModel):
    """Unified inference request — supports both generate and chat modes."""
    # Generate mode (Ollama /api/generate)
    model: str = Field(default="auto", description="Model ID or 'auto' for gateway selection")
    prompt: Optional[str] = Field(default=None, description="Prompt for generate mode")

    # Chat mode (Ollama /api/chat)
    messages: Optional[List[ChatMessage]] = Field(
        default=None, description="Messages for chat mode"
    )

    # Options
    options: InferenceOptions = Field(default_factory=InferenceOptions)
    stream: bool = Field(default=False)
    timeout: Optional[float] = Field(
        default=None, description="Per-request timeout override in seconds"
    )

    # Routing hints
    prefer_backend: Optional[str] = Field(
        default=None,
        description="Preferred backend ID (rtx5050 | radeon780m | ryzen7cpu)",
    )
    require_gpu: bool = Field(
        default=False, description="Reject CPU-only backends"
    )


class InferenceResponse(BaseModel):
    """Response from the inference endpoint."""
    request_id: str
    model: str
    backend_id: str
    backend_label: str
    response: str
    done: bool = True

    # Token counts (from Ollama response)
    prompt_eval_count: int = 0
    eval_count: int = 0
    eval_duration_ns: int = 0

    # Gateway metadata
    latency_ms: float
    routed_at: float = Field(default_factory=time.time)


# ── Routing logic ─────────────────────────────────────────────────────────────

async def route_inference(
    req: InferenceRequest,
    router,
    request_id: str = "",
) -> InferenceResponse:
    """
    Select the best backend and forward the inference request.

    Selection priority:
      1. prefer_backend (if specified and healthy)
      2. Model affinity from ModelAssigner
      3. Lowest EMA latency within each device tier
      4. GPU-only filter if require_gpu=True
    """
    from gateway.config import BACKENDS, BACKEND_MAP, GPU_DEVICE_TYPES, DeviceType

    if not request_id:
        request_id = str(uuid.uuid4())

    t0 = time.time()

    # Build candidate list
    candidates = await router.get_ordered_backends(
        model=req.model,
        prefer=req.prefer_backend,
    )

    if req.require_gpu:
        candidates = [
            c for c in candidates
            if BACKEND_MAP.get(c, None) and
               BACKEND_MAP[c].device_type in GPU_DEVICE_TYPES
        ]

    if not candidates:
        raise HTTPException(
            status_code=503,
            detail="No healthy backends available" + (
                " (GPU required but none healthy)" if req.require_gpu else ""
            ),
        )

    # Build Ollama-compatible payload
    payload = _build_payload(req)

    last_error: Optional[Exception] = None

    for backend_id in candidates:
        backend = BACKEND_MAP.get(backend_id)
        if not backend:
            continue

        try:
            result = await _call_backend(
                backend_url=backend.url,
                payload=payload,
                timeout=req.timeout or 30.0,
                use_chat_mode=(req.messages is not None),
            )

            latency_ms = (time.time() - t0) * 1000
            # Update router's EMA latency
            await router.record_latency(backend_id, latency_ms / 1000)

            return InferenceResponse(
                request_id=request_id,
                model=result.get("model", req.model),
                backend_id=backend_id,
                backend_label=backend.label,
                response=result.get("response") or _extract_chat_response(result),
                done=result.get("done", True),
                prompt_eval_count=result.get("prompt_eval_count", 0),
                eval_count=result.get("eval_count", 0),
                eval_duration_ns=result.get("eval_duration", 0),
                latency_ms=round(latency_ms, 2),
            )

        except asyncio.TimeoutError:
            last_error = asyncio.TimeoutError(
                f"Backend {backend_id} timed out after {req.timeout or 30}s"
            )
            logger.warning("Timeout on %s — trying next backend", backend_id)
        except aiohttp.ClientError as e:
            last_error = e
            logger.warning("Backend %s error: %s — trying next", backend_id, e)
        except Exception as e:
            last_error = e
            logger.warning("Unexpected error on %s: %s — trying next", backend_id, e)

    raise HTTPException(
        status_code=503,
        detail=f"All backends exhausted. Last error: {last_error}",
    )


def _build_payload(req: InferenceRequest) -> dict:
    """Convert InferenceRequest to Ollama API payload."""
    opts = {
        "num_predict": req.options.num_predict,
        "temperature": req.options.temperature,
        "top_p": req.options.top_p,
        "top_k": req.options.top_k,
        "repeat_penalty": req.options.repeat_penalty,
        "num_ctx": req.options.num_ctx,
    }
    if req.options.seed is not None:
        opts["seed"] = req.options.seed
    if req.options.stop:
        opts["stop"] = req.options.stop

    if req.messages is not None:
        # Chat mode
        return {
            "model": req.model if req.model != "auto" else "llama3",
            "messages": [m.model_dump() for m in req.messages],
            "options": opts,
            "stream": False,
        }
    else:
        # Generate mode
        return {
            "model": req.model if req.model != "auto" else "llama3",
            "prompt": req.prompt or "",
            "options": opts,
            "stream": False,
        }


async def _call_backend(
    backend_url: str,
    payload: dict,
    timeout: float,
    use_chat_mode: bool,
) -> dict:
    """Make the actual HTTP call to an Ollama backend."""
    endpoint = "/api/chat" if use_chat_mode else "/api/generate"
    url = f"{backend_url}{endpoint}"

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info,
                    history=resp.history,
                    status=resp.status,
                    message=body[:200],
                )
            return await resp.json()


def _extract_chat_response(data: dict) -> str:
    """Extract text from Ollama chat response format."""
    msg = data.get("message", {})
    if isinstance(msg, dict):
        return msg.get("content", "")
    return ""
