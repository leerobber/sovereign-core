"""
gateway/auth.py — API key authentication + rate limiting middleware.

Enforces GATEWAY_API_KEY if set in .env.
Rate limiting: token-bucket, 100 req/min per client IP (configurable).
Without this, the gateway is wide open — not safe for SovereignNation customers.

Usage in main.py:
    from gateway.auth import AuthMiddleware
    app.add_middleware(AuthMiddleware)
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Callable, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
_API_KEY: Optional[str] = os.environ.get("GATEWAY_API_KEY") or None

# Token bucket config — per client IP
_RATE_LIMIT_RPS: float = float(os.environ.get("GATEWAY_RATE_LIMIT_RPS", "100"))  # req/min
_RATE_LIMIT_BURST: float = float(os.environ.get("GATEWAY_RATE_LIMIT_BURST", "20"))  # max burst

# Routes that never require auth (health checks, metrics, dashboard)
_OPEN_ROUTES = {
    "/health", "/metrics", "/dashboard", "/ws/events",
    "/status", "/docs", "/openapi.json", "/redoc",
}

# Routes that are rate-limited but not auth-gated even with API key set
_RATE_ONLY_ROUTES = {"/health", "/metrics", "/dashboard"}


# ── Token Bucket Rate Limiter ─────────────────────────────────────────────────

class TokenBucket:
    """Per-client token bucket rate limiter."""

    def __init__(self, rate: float, burst: float) -> None:
        self._rate = rate / 60.0   # convert req/min → req/sec
        self._burst = burst
        self._buckets: dict[str, tuple[float, float]] = defaultdict(
            lambda: (burst, time.monotonic())
        )

    def consume(self, client_id: str) -> tuple[bool, float]:
        """
        Attempt to consume one token for client_id.
        Returns (allowed, retry_after_seconds).
        """
        tokens, last_time = self._buckets[client_id]
        now = time.monotonic()
        elapsed = now - last_time

        # Refill tokens
        tokens = min(self._burst, tokens + elapsed * self._rate)
        last_time = now

        if tokens >= 1.0:
            tokens -= 1.0
            self._buckets[client_id] = (tokens, last_time)
            return True, 0.0
        else:
            self._buckets[client_id] = (tokens, last_time)
            retry_after = (1.0 - tokens) / self._rate
            return False, round(retry_after, 2)

    def cleanup(self) -> None:
        """Remove stale buckets (older than 5 min)."""
        cutoff = time.monotonic() - 300
        stale = [k for k, (_, t) in self._buckets.items() if t < cutoff]
        for k in stale:
            del self._buckets[k]


_rate_limiter = TokenBucket(rate=_RATE_LIMIT_RPS, burst=_RATE_LIMIT_BURST)


# ── Middleware ────────────────────────────────────────────────────────────────

class AuthMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that enforces:
    1. API key authentication (if GATEWAY_API_KEY is set)
    2. Token-bucket rate limiting per client IP
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"

        # ── Rate limiting (always active) ──────────────────────────────────
        allowed, retry_after = _rate_limiter.consume(client_ip)
        if not allowed:
            logger.warning("Rate limit exceeded: client=%s path=%s", client_ip, path)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Too many requests. Retry after {retry_after}s.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        # ── API key auth (only if key is configured) ───────────────────────
        if _API_KEY and path not in _OPEN_ROUTES:
            # Accept key in Authorization header OR X-API-Key header
            auth_header = request.headers.get("Authorization", "")
            api_key_header = request.headers.get("X-API-Key", "")

            provided_key = ""
            if auth_header.startswith("Bearer "):
                provided_key = auth_header[7:]
            elif api_key_header:
                provided_key = api_key_header

            if provided_key != _API_KEY:
                logger.warning("Auth failed: client=%s path=%s", client_ip, path)
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "unauthorized",
                        "message": "Valid API key required. Set Authorization: Bearer <key> or X-API-Key: <key>",
                    },
                )

        # ── Add security headers to all responses ──────────────────────────
        response = await call_next(request)
        response.headers["X-Sovereign-Version"] = "2.0"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response
