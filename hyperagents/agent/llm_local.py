# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted for Sovereign Core — rewired to local inference gateway (port 8001)
# Strips all OpenAI/Anthropic/Gemini API calls; routes to local Qwen2.5-32B-AWQ.

import backoff
import json
import os
from typing import Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

MAX_TOKENS = 16384

# ---------------------------------------------------------------------------
# Local model identifiers (served via Sovereign Core gateway on port 8001)
# ---------------------------------------------------------------------------
# Primary brain — always routed to RTX 5050 via gateway capability-aware routing
LOCAL_PRIMARY_MODEL = "qwen2.5-32b-awq"          # Qwen2.5-32B-AWQ on RTX 5050
LOCAL_CODER_MODEL   = "deepseek-coder-33b"        # DeepSeek-Coder on Radeon 780M
LOCAL_NANO_MODEL    = "nemotron-3-nano"            # Nemotron-3-Nano (primary brain candidate)
LOCAL_CPU_MODEL     = "mistral-7b"                # CPU fallback (Ryzen 7)

# Default: use primary brain for all HyperAgents generation
DEFAULT_MODEL = LOCAL_PRIMARY_MODEL

# ---------------------------------------------------------------------------
# Gateway endpoint — routes to best available backend automatically
# ---------------------------------------------------------------------------
GATEWAY_URL = os.environ.get("SOVEREIGN_GATEWAY_URL", "http://localhost:8000")
GATEWAY_CHAT_PATH = "/v1/chat/completions"


def _build_payload(
    messages: list,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, json.JSONDecodeError, KeyError),
    max_time=600,
    max_value=60,
)
def get_response_from_llm(
    msg: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    msg_history=None,
) -> Tuple[str, list, dict]:
    """
    Drop-in replacement for the original HyperAgents get_response_from_llm.

    Routes all requests to the Sovereign Core gateway (localhost:8000/v1/...)
    which transparently dispatches to the best available local backend:
      - qwen2.5-32b-awq  → RTX 5050  (primary brain)
      - deepseek-coder-33b → Radeon 780M (verification)
      - nemotron-3-nano   → RTX 5050  (primary brain candidate)
      - mistral-7b        → Ryzen 7 CPU (fallback)

    Signature is identical to the upstream Meta implementation so all
    HyperAgents callers (generate_loop.py, baselines, etc.) work unchanged.
    """
    if msg_history is None:
        msg_history = []

    # Normalise history: upstream uses 'text' key, OpenAI-compatible uses 'content'
    normalised_history = [
        {**m, "content": m.pop("text")} if "text" in m else dict(m)
        for m in [dict(x) for x in msg_history]
    ]

    new_msg_history = normalised_history + [{"role": "user", "content": msg}]

    payload = _build_payload(
        messages=new_msg_history,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    url = f"{GATEWAY_URL}{GATEWAY_CHAT_PATH}"
    resp = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()

    response_text: str = data["choices"][0]["message"]["content"]
    new_msg_history.append({"role": "assistant", "content": response_text})

    # Convert back to 'text' key format expected by upstream MetaGen API
    out_history = [
        {**m, "text": m.pop("content")} if "content" in m else m
        for m in new_msg_history
    ]

    return response_text, out_history, {}


if __name__ == "__main__":
    print("Testing Sovereign Core local LLM adapter...")
    test_msg = "Say 'OK' and nothing else."
    try:
        text, history, _ = get_response_from_llm(test_msg, model=DEFAULT_MODEL)
        print(f"Response: {text[:200]}")
        print("✓ Local gateway adapter working")
    except Exception as e:
        print(f"✗ Gateway not reachable: {e}")
        print(f"  Make sure gateway is running: python -m gateway.main")
