"""
gateway/iron_dome_middleware.py — Iron Dome injection detection on live traffic.

Every inference request runs through Iron Dome before processing.
Every response is logged to GhostRecall after completion.

Integrates:
  - memory_palace/iron_dome.py  → prompt injection detection
  - memory_palace/ghost_recall.py → persistent episodic memory of requests
  - gateway/db.py               → logs blocked events to system_events table
"""
from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Iron Dome integration ─────────────────────────────────────────────────────

def get_iron_dome():
    """Return IronDome instance or None if unavailable."""
    try:
        import sys, os
        sys.path.insert(0, os.getcwd())
        from memory_palace.iron_dome import IronDome
        return IronDome()
    except Exception as e:
        logger.warning("IronDome unavailable: %s", e)
        return None


def get_ghost_recall():
    """Return GhostRecall instance or None if unavailable."""
    try:
        import sys, os
        sys.path.insert(0, os.getcwd())
        from memory_palace.ghost_recall import GhostRecall
        return GhostRecall()
    except Exception as e:
        logger.warning("GhostRecall unavailable: %s", e)
        return None


# ── Per-request screening ─────────────────────────────────────────────────────

class IronDomeGuard:
    """
    Stateless guard — call screen() before each inference request.
    Returns (allowed: bool, reason: str).
    """

    def __init__(self) -> None:
        self._dome = None
        self._recall = None
        self._initialized = False

    def _lazy_init(self) -> None:
        if not self._initialized:
            self._dome = get_iron_dome()
            self._recall = get_ghost_recall()
            self._initialized = True
            if self._dome:
                logger.info("IronDome guard active — injection screening ENABLED")
            else:
                logger.warning("IronDome unavailable — injection screening DISABLED")

    def screen(self, prompt: str, model: str = "", backend: str = "") -> tuple[bool, str]:
        """
        Screen a prompt through Iron Dome.
        Returns (True, "ok") if safe, (False, reason) if blocked.
        """
        self._lazy_init()

        if self._dome is None:
            return True, "ok"  # fail open if dome unavailable

        try:
            # Iron Dome validate call
            result = self._dome.validate_input(prompt)

            # result is dict with 'approved', 'threat_level', 'reason'
            approved = result.get("approved", True)
            reason = result.get("reason", "")
            threat = result.get("threat_level", 0.0)

            if not approved:
                logger.warning(
                    "IronDome BLOCKED request | model=%s backend=%s threat=%.2f reason=%s",
                    model, backend, threat, reason
                )
                # Log to DB
                try:
                    from gateway.db import log_event
                    log_event("iron_dome_block", "iron_dome_guard",
                              f"Blocked: {reason[:200]}",
                              severity="warning",
                              metadata={"model": model, "backend": backend,
                                        "threat_level": threat, "prompt_len": len(prompt)})
                except Exception:
                    pass
                return False, f"Request blocked by Iron Dome: {reason}"

            return True, "ok"

        except Exception as e:
            logger.warning("IronDome screen error: %s", e)
            return True, "ok"  # fail open

    def recall(self, prompt: str, response: str, model: str,
               backend: str, latency_ms: float) -> None:
        """Log a completed inference to GhostRecall episodic memory."""
        self._lazy_init()

        if self._recall is None:
            return

        try:
            self._recall.encode(
                content=f"Q: {prompt[:300]}\nA: {response[:500]}",
                source=f"{model}@{backend}",
                importance=0.6,
                tags=["inference", model, backend],
            )
        except Exception as e:
            logger.debug("GhostRecall encode error: %s", e)


# ── Module-level singleton ────────────────────────────────────────────────────
iron_dome_guard = IronDomeGuard()
