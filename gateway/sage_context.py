"""
gateway/sage_context.py — Wire ChromaDB context layer into SAGE loop.

Every SAGE cycle:
  - Proposer reads prior verified proposals from ChromaDB before generating
  - Critic writes adversarial conclusions to ChromaDB
  - Verifier reads Critic output + writes verification verdict
  - Meta-Agent reads all prior rounds, rewrites rules

This gives KAIROS true cross-cycle memory — agents learn from history.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

# ── ChromaDB context helpers ───────────────────────────────────────────────────

def get_context_layer():
    """Return SharedContextLayer instance, or None if ChromaDB unavailable."""
    try:
        from gateway.context import SharedContextLayer, AgentRole
        return SharedContextLayer(), AgentRole
    except Exception as e:
        logger.warning("ChromaDB context unavailable: %s", e)
        return None, None


def sage_write_context(role_name: str, backend_id: str, text: str,
                        trace_id: str = "", task: str = "") -> bool:
    """Write a SAGE agent output to the shared ChromaDB context."""
    ctx, AgentRole = get_context_layer()
    if ctx is None:
        return False
    try:
        role_map = {
            "proposer":   AgentRole.GENERATOR,
            "critic":     AgentRole.SAFETY,
            "verifier":   AgentRole.VERIFIER,
            "meta_agent": AgentRole.PLANNER,
        }
        role = role_map.get(role_name, AgentRole.GENERATOR)
        ctx.write(
            role=role,
            backend_id=backend_id,
            document=f"[TASK: {task[:100]}]\n{text}",
            trace_id=trace_id,
        )
        logger.debug("SAGE context written: role=%s trace=%s", role_name, trace_id)
        return True
    except Exception as e:
        logger.warning("Context write failed: %s", e)
        return False


def sage_read_prior_context(role_name: str, task: str, n: int = 3) -> str:
    """Read prior relevant context for a SAGE role before it generates."""
    ctx, AgentRole = get_context_layer()
    if ctx is None:
        return ""
    try:
        role_map = {
            "proposer":   AgentRole.VERIFIER,   # Proposer reads prior verifications
            "critic":     AgentRole.GENERATOR,  # Critic reads prior proposals
            "verifier":   AgentRole.SAFETY,     # Verifier reads prior critic output
            "meta_agent": AgentRole.PLANNER,    # Meta reads all prior planning
        }
        read_role = role_map.get(role_name, AgentRole.GENERATOR)
        results = ctx.read_by_role(role=read_role, limit=n)
        if not results or not results.get("documents"):
            return ""
        docs = results["documents"][0] if results["documents"] else []
        if not docs:
            return ""
        context_text = "\n---\n".join(docs[-n:])
        logger.debug("SAGE prior context loaded: role=%s docs=%d", role_name, len(docs))
        return f"\n[PRIOR CONTEXT FROM PREVIOUS CYCLES]\n{context_text}\n[END PRIOR CONTEXT]\n"
    except Exception as e:
        logger.warning("Context read failed: %s", e)
        return ""


def sage_cross_gpu_context(task: str, n: int = 5) -> str:
    """Read cross-GPU context — what all backends know about this task domain."""
    ctx, _ = get_context_layer()
    if ctx is None:
        return ""
    try:
        results = ctx.read_cross_gpu(limit=n)
        if not results or not results.get("documents"):
            return ""
        docs = results["documents"][0] if results["documents"] else []
        if not docs:
            return ""
        return f"\n[CROSS-GPU KNOWLEDGE]\n" + "\n---\n".join(docs[-n:]) + "\n[END]\n"
    except Exception as e:
        logger.warning("Cross-GPU context read failed: %s", e)
        return ""
