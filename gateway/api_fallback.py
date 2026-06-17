"""
GH05T3 FREE API FALLBACK SYSTEM
================================
Wraps OpenRouter, Mistral, Groq, Cerebras as fallback inference providers.
Called by the gateway when all local GPU backends are down.

Verified working (April 2026):
  OpenRouter  — Nemotron-120B + Gemma-4-26B (free tier)
  Mistral     — mistral-small-latest (1B tok/month free)
  Groq        — llama-3.3-70b (needs email verify at console.groq.com)
  Cerebras    — llama3.3-70b (needs email verify at cloud.cerebras.ai)
"""
import os, json, time, urllib.request, urllib.error
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

PROVIDERS = [
    {
        "name": "openrouter_nemotron",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "key_env": "OPENROUTER_API_KEY",
        "extra_headers": {
            "HTTP-Referer": "https://gh05t3.local",
            "X-Title": "GH05T3 Sovereign Gateway",
        },
    },
    {
        "name": "openrouter_gemma",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "google/gemma-4-26b-a4b-it:free",
        "key_env": "OPENROUTER_API_KEY",
        "extra_headers": {
            "HTTP-Referer": "https://gh05t3.local",
            "X-Title": "GH05T3 Sovereign Gateway",
        },
    },
    {
        "name": "mistral",
        "url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-small-latest",
        "key_env": "MISTRAL_API_KEY",
        "extra_headers": {},
    },
    {
        "name": "groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
        "extra_headers": {},
    },
    {
        "name": "cerebras",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "llama3.3-70b",
        "key_env": "CEREBRAS_API_KEY",
        "extra_headers": {},
    },
]

def call_provider(provider: dict, messages: list, max_tokens: int = 1024) -> Optional[str]:
    """Call a single API provider. Returns response text or None on failure."""
    key = os.getenv(provider["key_env"], "")
    if not key:
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        **provider.get("extra_headers", {}),
    }

    payload = json.dumps({
        "model": provider["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(provider["url"], data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        print(f"  [{provider['name']}] HTTP {e.code}")
        return None
    except Exception as e:
        print(f"  [{provider['name']}] Error: {str(e)[:60]}")
        return None


def fallback_inference(messages: list, max_tokens: int = 1024) -> Optional[dict]:
    """Try all API providers in order. Returns OpenAI-format response or None."""
    for provider in PROVIDERS:
        key = os.getenv(provider["key_env"], "")
        if not key:
            continue
        result = call_provider(provider, messages, max_tokens)
        if result:
            return {
                "id": f"fallback-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": provider["model"],
                "gh05t3_backend": provider["name"],
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
    return None


def status_check() -> dict:
    """Check which providers have keys configured and are reachable."""
    results = {}
    for p in PROVIDERS:
        key = os.getenv(p["key_env"], "")
        results[p["name"]] = {
            "key_set": bool(key),
            "model": p["model"],
            "provider_url": p["url"],
        }
    return results


if __name__ == "__main__":
    print("\n" + "="*54)
    print("  GH05T3 API FALLBACK STATUS")
    print("="*54)
    for name, info in status_check().items():
        icon = "OK " if info["key_set"] else "-- "
        print(f"  [{icon}] {name:<28} {info['model']}")
    print("="*54)

    # Test first available provider
    print("\nTesting first available provider...")
    result = fallback_inference([
        {"role": "user", "content": "Say: GH05T3 API fallback online."}
    ], max_tokens=30)
    if result:
        print(f"  Backend: {result['gh05t3_backend']}")
        print(f"  Reply:   {result['choices'][0]['message']['content']}")
    else:
        print("  No API keys configured. Add keys to .env file.")
    print()
