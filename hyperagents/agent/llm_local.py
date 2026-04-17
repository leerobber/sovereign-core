# Copyright (c) Meta Platforms, Inc. and affiliates.
# Sovereign Core adaptation — wired to Terry's actual installed models
# RTX 5050: gemma3:12b (primary brain, 8.1GB)
# Radeon 780M: qwen2.5:7b (verifier/coder, 4.7GB)
# Ryzen 7 CPU: llama3.2:3b (fast/light tasks, 2.0GB)
# Fallback: dolphin-llama3:8b or dolphin-phi

import backoff
import json
import logging
import os
import time
from typing import Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MAX_TOKENS = 8192  # Conservative — all models support this
MAX_TOKENS_LARGE = 16384  # For gemma3:12b only

# ── Model identifiers (Terry's actual Ollama models) ─────────────────────────
# Primary brain — gemma3:12b on RTX 5050 (8.1GB, most capable)
LOCAL_PRIMARY_MODEL = "gemma3:12b"

# Verifier / coder — qwen2.5:7b on Radeon 780M (4.7GB, strong at reasoning)
LOCAL_CODER_MODEL = "qwen2.5:7b"

# CPU fallback — llama3.2:3b on Ryzen 7 (2.0GB, fast light tasks)
LOCAL_CPU_MODEL = "llama3.2:3b"

# Dolphin variants (uncensored, good for adversarial critique)
LOCAL_CRITIC_MODEL = "dolphin-llama3:8b"   # Radeon 780M — adversarial critic
LOCAL_NANO_MODEL   = "dolphin-phi:latest"  # CPU — ultra-light tasks

# Default model for SAGE loop
DEFAULT_MODEL = LOCAL_PRIMARY_MODEL

# ── Gateway config ────────────────────────────────────────────────────────────
# Gateway auto-routes to the right hardware based on model name
# Port 8080 = new Windows-compatible port (8000 blocked by WinError 10013)
GATEWAY_URL = os.environ.get("SOVEREIGN_GATEWAY_URL", "http://localhost:8080")
GATEWAY_CHAT_PATH = "/v1/chat/completions"

# ── SAGE role → model mapping ─────────────────────────────────────────────────
SAGE_ROLE_MODELS = {
    "proposer":   LOCAL_PRIMARY_MODEL,  # gemma3:12b → RTX 5050
    "critic":     LOCAL_CRITIC_MODEL,   # dolphin-llama3:8b → Radeon 780M (adversarial)
    "verifier":   LOCAL_CODER_MODEL,    # qwen2.5:7b → Radeon 780M (logical check)
    "meta_agent": LOCAL_PRIMARY_MODEL,  # gemma3:12b → RTX 5050 (rule rewriter)
}


def _build_payload(
    messages: list,
    model: str,
    temperature: float,
    max_tokens: int,
    stream: bool = False,
) -> dict:
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, json.JSONDecodeError, KeyError),
    max_time=600,
    max_value=60,
    on_backoff=lambda details: logger.warning(
        "Gateway retry %d after %.1fs — %s",
        details["tries"], details["wait"], details.get("exception", "")
    ),
)
def get_response_from_llm(
    msg: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = MAX_TOKENS,
    msg_history: Optional[list] = None,
    system_prompt: Optional[str] = None,
    timeout: int = 300,
) -> Tuple[str, list, dict]:
    """
    Drop-in replacement for HyperAgents get_response_from_llm.

    Routes all inference through the Sovereign Core gateway which dispatches
    to the best available local backend:
      gemma3:12b       → RTX 5050  (8GB NVIDIA, primary brain)
      qwen2.5:7b       → Radeon 780M (4GB AMD, verifier/coder)
      dolphin-llama3   → Radeon 780M (adversarial critic)
      llama3.2:3b      → Ryzen 7 CPU (fast light tasks)
      dolphin-phi      → CPU (ultra-light, nano tasks)

    Gateway port: 8080 (Windows-compatible, no admin required)

    Signature matches Meta's upstream implementation for full compatibility.
    """
    if msg_history is None:
        msg_history = []

    # Normalise history — upstream uses 'text' key, OpenAI uses 'content'
    normalised_history = []
    for m in msg_history:
        m = dict(m)
        if "text" in m and "content" not in m:
            m["content"] = m.pop("text")
        normalised_history.append(m)

    # Build message list
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(normalised_history)
    messages.append({"role": "user", "content": msg})

    payload = _build_payload(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    url = f"{GATEWAY_URL}{GATEWAY_CHAT_PATH}"
    t0 = time.time()

    resp = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    latency_ms = (time.time() - t0) * 1000
    response_text: str = data["choices"][0]["message"]["content"]

    # Build updated history
    new_history = messages + [{"role": "assistant", "content": response_text}]

    # Convert back to 'text' key format for upstream Meta API compat
    out_history = []
    for m in new_history:
        m = dict(m)
        if "content" in m:
            m["text"] = m.pop("content")
        out_history.append(m)

    meta = {
        "model": data.get("model", model),
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
        "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
        "backend": data.get("backend_id", "unknown"),
    }

    logger.debug(
        "LLM call complete | model=%s latency=%.0fms tokens=%d",
        model, latency_ms, meta["completion_tokens"]
    )

    return response_text, out_history, meta


def get_sage_model(role: str) -> str:
    """Return the correct model for a given SAGE role."""
    return SAGE_ROLE_MODELS.get(role, DEFAULT_MODEL)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print("Testing Sovereign Core LLM adapter...")
    print(f"  Gateway: {GATEWAY_URL}")
    print(f"  Primary: {LOCAL_PRIMARY_MODEL}")
    print(f"  Verifier: {LOCAL_CODER_MODEL}")
    print(f"  CPU: {LOCAL_CPU_MODEL}")
    print()

    test_msg = "Reply with exactly: SOVEREIGN ONLINE"
    for role, model in SAGE_ROLE_MODELS.items():
        try:
            text, _, meta = get_response_from_llm(test_msg, model=model, timeout=30)
            print(f"  ✅ {role:<12} [{model:<22}] → {text[:60].strip()}")
        except Exception as e:
            print(f"  ✗  {role:<12} [{model:<22}] → OFFLINE: {e}")
    print()
    print("Test complete.")
