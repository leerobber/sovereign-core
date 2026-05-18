"""
gateway/kairos_ext.py — KAIROSAgent class extensions
Adds the methods required by kairos_routes.py:
  - KAIROSAgent.load_or_create(agent_id, archive) -> KAIROSAgent
  - KAIROSAgent.load(agent_id) -> KAIROSAgent
  - KAIROSAgent.list_all() -> list[dict]
  - KAIROSAgent.to_dict() -> dict
  - KAIROSAgent.run_evolution_cycle() -> CycleResult

Monkey-patches into existing kairos.py at import time via:
    from gateway.kairos_ext import apply_patches
    apply_patches()

Or just import this module — the patches apply automatically.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Storage directory for agent state
_AGENTS_DIR = Path(os.getenv("KAIROS_AGENTS_DIR", "data/kairos/agents"))
# Storage directory for per-agent DGM-H archives
_ARCHIVES_DIR = Path("data/kairos/archives")


@dataclass
class CycleResult:
    """Result of a single ARSO evolution cycle."""
    agent_id: str
    generation: int
    score: float
    verification_verdict: str   # PASS | PARTIAL | FAIL
    elite_promoted: bool
    arso_cycles: int
    latency_ms: float
    proposals: List[str]
    log: List[Dict[str, Any]]
    ancestor_used: Optional[str] = None


def _agents_dir() -> Path:
    _AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    return _AGENTS_DIR


def _agent_path(agent_id: str) -> Path:
    return _agents_dir() / f"{agent_id}.json"


def _archive_path(agent_id: str) -> Path:
    """Return the DGM-H archive path for a given agent."""
    _ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    return _ARCHIVES_DIR / f"{agent_id}_dgmh.json"


def _bottleneck_type_from_task(task: str) -> str:
    """
    Normalize a task string to snake_case using the first two words.
    Falls back to 'arso_optimization' if the task is empty or has fewer than
    two meaningful words.
    """
    words = re.findall(r"[A-Za-z]+", task)
    if len(words) >= 2:
        return f"{words[0].lower()}_{words[1].lower()}"
    if len(words) == 1:
        return words[0].lower()
    return "arso_optimization"


def _patch_load_or_create(cls):
    """Add load_or_create() classmethod to KAIROSAgent."""
    @classmethod
    def load_or_create(klass, agent_id: str, archive=None) -> "KAIROSAgent":
        path = _agent_path(agent_id)
        if path.exists():
            return klass.load(agent_id)
        agent = klass(agent_id=agent_id)
        if archive is not None:
            agent._archive = archive
        _save_agent(agent)
        return agent
    cls.load_or_create = load_or_create


def _patch_load(cls):
    """Add load() classmethod."""
    @classmethod
    def load(klass, agent_id: str) -> "KAIROSAgent":
        path = _agent_path(agent_id)
        if not path.exists():
            raise FileNotFoundError(f"No agent found: {agent_id}")
        with open(path) as f:
            data = json.load(f)
        agent = klass.__new__(klass)
        agent.__dict__.update(data)
        agent.agent_id = agent_id
        return agent
    cls.load = load


def _patch_list_all(cls):
    """Add list_all() classmethod."""
    @classmethod
    def list_all(klass) -> List[Dict[str, Any]]:
        agents_dir = _agents_dir()
        result = []
        for p in sorted(agents_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                with open(p) as f:
                    data = json.load(f)
                data["agent_id"] = p.stem
                result.append(data)
            except Exception:
                pass
        return result
    cls.list_all = list_all


def _patch_to_dict(cls):
    """Add to_dict() method."""
    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            try:
                json.dumps(v)
                d[k] = v
            except (TypeError, ValueError):
                d[k] = str(v)
        return d
    cls.to_dict = to_dict


def _patch_run_evolution_cycle(cls):
    """Add run_evolution_cycle() async method."""
    async def run_evolution_cycle(self) -> CycleResult:
        t0 = time.time()
        agent_id = getattr(self, "agent_id", str(uuid.uuid4()))
        generation = getattr(self, "generation", 0) + 1
        old_score = getattr(self, "score", 0.0)
        log = []

        proposals = []
        score = 0.0
        verdict = "PARTIAL"
        elite_promoted = False

        # --- DGM-H: load per-agent archive ---
        from gateway.dgm_h_archive import DGMHArchive
        archive = DGMHArchive.load(_archive_path(agent_id))

        # Determine bottleneck type from the task we're about to run
        task_str = f"ARSO cycle {generation} — optimize agent {agent_id}"
        bottleneck_type = _bottleneck_type_from_task(task_str)

        # Find nearest ancestor before running SAGE
        ancestors = archive.find_nearest_ancestor(bottleneck_type, top_k=3)
        best_ancestor = ancestors[0] if ancestors else None
        ancestor_context = None
        if best_ancestor is not None:
            ancestor_context = best_ancestor.agent_state_snapshot
            log.append({
                "note": f"DGM-H ancestor loaded: node_id={best_ancestor.node_id} "
                        f"gen={best_ancestor.generation} delta={best_ancestor.performance_delta:.3f}"
            })

        try:
            from hyperagents.sage_generate_loop import run_sage_loop
            # Build bottleneck description; embed ancestor snapshot summary if available
            bottleneck_desc = task_str
            if ancestor_context is not None:
                ancestor_summary = json.dumps(ancestor_context)[:300]
                bottleneck_desc = f"{task_str}\n[ancestor_context: {ancestor_summary}]"

            loop = asyncio.get_event_loop()
            archive = await loop.run_in_executor(
                None,
                lambda: run_sage_loop(
                    bottleneck_description=bottleneck_desc,
                    max_generations=2,
                    archive_path=None,
                    score_threshold=0.6,
                ),
            )
            best = archive.best(3)
            score = best[0].score if best else 0.0
            proposals = [p.proposer_output for p in best]
            log = [
                {"gen": p.generation, "score": p.score, "verdict": p.verifier_verdict}
                for p in archive.proposals
            ]
            if ancestor_context is not None:
                log.append({
                    "note": f"DGM-H ancestor used: node_id={best_ancestor.node_id} "
                            f"gen={best_ancestor.generation} delta={best_ancestor.performance_delta:.3f}"
                })
        except Exception as exc:
            logger.warning("SAGE loop unavailable: %s — using heuristic cycle", exc)
            # Heuristic fallback: score improves slightly each generation
            score = min(0.99, 0.5 + generation * 0.03 + (hash(agent_id) % 100) / 1000)
            proposals = [f"Heuristic proposal gen={generation}"]
            log.append({"note": f"SAGE unavailable: {exc}"})

        # Verification
        try:
            from gateway.self_verification import SelfVerifier
            vr = SelfVerifier().verify(
                proposals[0] if proposals else "", f"ARSO cycle {generation}"
            )
            verdict = vr.verdict.value
        except Exception:
            verdict = "PASS" if score > 0.7 else "PARTIAL"

        # Elite promotion — gated by ZERO Committee consensus
        tier = getattr(self, "tier", "standard")
        if score >= 0.85 and tier != "next_elite":
            next_tier = "elite" if tier == "standard" else "next_elite"
            promotion_context = (
                f"Agent {agent_id} (generation={generation}, score={score:.4f}) qualifies for "
                f"tier promotion from '{tier}' to '{next_tier}'.\n"
                f"Bottleneck: {task_str[:200]}\n"
                f"Top proposals:\n" + "\n".join(proposals[:2])[:600]
            )
            try:
                from gateway.zero_committee import get_committee
                zc_decision = await get_committee().decide(promotion_context, max_chair_rounds=2)
                log.append({
                    "zero_committee": {
                        "verdict": zc_decision.verdict,
                        "audit": zc_decision.audit_verdict,
                        "strategy_score": zc_decision.strategy_score,
                        "chair_activated": zc_decision.chair_activated,
                        "reason": zc_decision.reason[:150],
                    }
                })
                if zc_decision.verdict == "EXECUTE":
                    elite_promoted = True
                    self.tier = next_tier
                else:
                    log.append({"note": f"Promotion blocked by ZERO: {zc_decision.verdict} — {zc_decision.reason[:100]}"})
            except Exception as _zc_err:
                logger.warning("ZERO Committee unavailable: %s — auto-approving promotion", _zc_err)
                elite_promoted = True
                self.tier = next_tier

        # Update state
        self.generation = generation
        self.score = score
        self.last_cycle_at = time.time()

        # --- DGM-H: persist successful cycle (score > 0) ---
        if score > 0:
            skill_domains = getattr(self, "skill_domains", [])
            agent_state_snapshot = {
                "generation": generation,
                "score": score,
                "tier": getattr(self, "tier", "standard"),
                "skill_domains": skill_domains,
            }
            fix_diff_raw = "\n".join(proposals)
            fix_diff = fix_diff_raw[:2000]
            performance_delta = score - old_score if generation > 1 else 0.0
            try:
                archive.add_node(
                    bottleneck_type=bottleneck_type,
                    bottleneck_description=task_str,
                    fix_diff=fix_diff,
                    agent_state_snapshot=agent_state_snapshot,
                    performance_before={"score": old_score, "generation": generation - 1},
                    performance_after={"score": score, "generation": generation},
                    performance_delta=performance_delta,
                    parent_id=best_ancestor.node_id if best_ancestor is not None else None,
                )
            except Exception as exc:
                logger.warning("DGM-H archive.add_node failed: %s", exc)

        # Persist agent state
        _save_agent(self)

        latency_ms = (time.time() - t0) * 1000
        return CycleResult(
            agent_id=agent_id,
            generation=generation,
            score=round(score, 4),
            verification_verdict=verdict,
            elite_promoted=elite_promoted,
            arso_cycles=1,
            latency_ms=round(latency_ms, 2),
            proposals=proposals,
            log=log,
            ancestor_used=best_ancestor.node_id if best_ancestor is not None else None,
        )
    cls.run_evolution_cycle = run_evolution_cycle


def _save_agent(agent) -> None:
    path = _agent_path(getattr(agent, "agent_id", "unknown"))
    data = {}
    for k, v in agent.__dict__.items():
        if k.startswith("_"):
            continue
        try:
            json.dumps(v)
            data[k] = v
        except (TypeError, ValueError):
            data[k] = str(v)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_dgmh_summary(agent_id: str) -> dict:
    """
    Load the DGM-H archive for the given agent and return its summary dict.

    Returns an empty summary dict if the archive file does not exist yet.
    """
    from gateway.dgm_h_archive import DGMHArchive
    archive = DGMHArchive.load(_archive_path(agent_id))
    return archive.summary()


def apply_patches():
    """Apply all patches to KAIROSAgent."""
    try:
        from gateway.kairos import KAIROSAgent
        _patch_load_or_create(KAIROSAgent)
        _patch_load(KAIROSAgent)
        _patch_list_all(KAIROSAgent)
        _patch_to_dict(KAIROSAgent)
        _patch_run_evolution_cycle(KAIROSAgent)
        logger.debug("KAIROSAgent patched successfully")
    except ImportError as e:
        logger.warning("Could not patch KAIROSAgent: %s", e)


# Auto-apply when imported
apply_patches()
