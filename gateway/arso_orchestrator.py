"""
gateway/arso_orchestrator.py — ARSO Production Cycle Controller (v1)

ARSO = Autonomous Recursive Self-Optimization

Runs a production evolution cycle across a pool of specialist agents.
Each agent targets a specific bottleneck domain. All agents evolve in
parallel via asyncio.gather; results are aggregated into a ProductionReport.

Entry points:
  HTTP:       POST /kairos/orchestrate
  Standalone: python scripts/run_arso.py --cycles 3 --agents 7

Architecture:
  Orchestrator
    └── N specialist agents (one per SkillDomain)
          └── SAGE loop via /kairos/evolve (gateway HTTP)
                └── Proposer → Critic → Verifier → Meta-Agent
                      └── ZERO Committee gate (elite promotions)
                            └── DGM-H lineage archive
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bottleneck task pool — rotated across agents each production cycle
# ---------------------------------------------------------------------------
BOTTLENECK_TASKS = [
    "Reduce average inference latency for local Ollama backends — target sub-5s for 7B models",
    "Improve SAGE Proposer output quality — reduce vague proposals, increase measurable specificity",
    "Optimize memory retrieval hit rate — improve relevant context injection at each SAGE step",
    "Reduce ZERO Committee decision latency — streamline 4-agent governance for faster promotions",
    "Improve DGM-H ancestor relevance scoring — better stepping-stone selection for new agents",
    "Increase SwarmBus throughput — reduce GH05T3 gateway bottleneck under concurrent agent load",
    "Strengthen IronDome injection detection — reduce false negatives on adversarial prompts",
    "Optimize MemEvolve weight convergence — faster retrieval weight adaptation per domain",
    "Reduce nightly evolution total wall-clock time — parallelize SAGE sub-steps where safe",
    "Improve elite retention across sessions — better serialization of top agent state",
    "Increase Critic adversarial depth — catch implementation gaps before Verifier stage",
    "Reduce pattern-memory cold-start overhead — faster first-cycle context bootstrap",
]

# Agent specialist roles → bottleneck affinity
SPECIALIST_AGENTS = [
    {"name": "Architect",  "domains": ["memory_retrieval", "dgm_h_ancestor", "pattern_memory"]},
    {"name": "Latency",    "domains": ["inference_latency", "swarmbus_throughput", "sage_proposer"]},
    {"name": "Security",   "domains": ["iron_dome_injection", "zero_committee", "mem_evolve"]},
    {"name": "Evolver",    "domains": ["elite_retention", "nightly_evolution", "critic_adversarial"]},
    {"name": "Analyst",    "domains": ["sage_proposer", "mem_evolve", "latency"]},
    {"name": "Distiller",  "domains": ["pattern_memory", "memory_retrieval", "dgm_h_ancestor"]},
    {"name": "Monitor",    "domains": ["iron_dome_injection", "swarmbus_throughput", "elite_retention"]},
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    agent_id: str
    specialist_name: str
    generation: int
    score: float
    verification_verdict: str
    elite_promoted: bool
    cycles_run: int
    latency_ms: float
    error: Optional[str] = None


@dataclass
class ProductionReport:
    run_id: str
    started_at: str
    completed_at: str
    wall_clock_s: float
    total_agents: int
    total_cycles: int
    elite_count: int
    best_score: float
    best_agent_id: str
    agent_results: List[Dict[str, Any]]
    status: str  # "complete" | "partial" | "failed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"ARSO Production Cycle — {self.run_id}",
            f"  Duration:     {self.wall_clock_s:.1f}s",
            f"  Agents:       {self.total_agents}",
            f"  SAGE cycles:  {self.total_cycles}",
            f"  Elites:       {self.elite_count}/{self.total_agents}",
            f"  Best score:   {self.best_score:.4f}",
            f"  Best agent:   {self.best_agent_id}",
            f"  Status:       {self.status}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ARSOOrchestrator:
    """
    Production cycle controller for ARSO evolution.

    Usage:
        orch = ARSOOrchestrator(gateway_url="http://localhost:9000")
        report = await orch.run_production_cycle(cycles_per_agent=3)
    """

    def __init__(
        self,
        gateway_url: str = "http://localhost:9000",
        timeout_s: float = 600.0,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout_s = timeout_s

    async def _evolve_agent(
        self,
        specialist: Dict[str, Any],
        agent_id: str,
        task: str,
        cycles: int,
        score_threshold: float,
    ) -> AgentResult:
        """Run one specialist agent for N cycles via the gateway HTTP API."""
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(
                    f"{self.gateway_url}/kairos/evolve",
                    json={
                        "cycles": cycles,
                        "agent_id": agent_id,
                        "task_hint": task,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            best_result = max(data.get("results", [{}]), key=lambda r: r.get("score", 0.0))
            return AgentResult(
                agent_id=agent_id,
                specialist_name=specialist["name"],
                generation=best_result.get("generation", 0),
                score=data.get("best_score", 0.0),
                verification_verdict=best_result.get("verification_verdict", "UNKNOWN"),
                elite_promoted=data.get("elite_count", 0) > 0,
                cycles_run=data.get("results", []),
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as exc:
            logger.warning(
                "Agent %s (%s) failed: %s", agent_id, specialist["name"], exc
            )
            return AgentResult(
                agent_id=agent_id,
                specialist_name=specialist["name"],
                generation=0,
                score=0.0,
                verification_verdict="ERROR",
                elite_promoted=False,
                cycles_run=0,
                latency_ms=(time.time() - t0) * 1000,
                error=str(exc)[:200],
            )

    async def run_production_cycle(
        self,
        cycles_per_agent: int = 2,
        score_threshold: float = 0.6,
        agent_ids: Optional[List[str]] = None,
        time_budget_s: Optional[float] = None,
    ) -> ProductionReport:
        """
        Run all specialist agents in parallel for cycles_per_agent SAGE cycles each.

        Args:
            cycles_per_agent: SAGE cycles per agent (1–10)
            score_threshold: minimum score to count as successful optimization
            agent_ids: reuse existing agent IDs for continuity; generates new if None
            time_budget_s: wall-clock deadline; agents still running are cancelled
        """
        run_id = str(uuid.uuid4())[:8]
        started_at = datetime.utcnow().isoformat()
        t0 = time.time()

        # Assign agent IDs and task bottlenecks
        n = len(SPECIALIST_AGENTS)
        ids = agent_ids or [str(uuid.uuid4()) for _ in range(n)]
        tasks = [BOTTLENECK_TASKS[i % len(BOTTLENECK_TASKS)] for i in range(n)]

        logger.info(
            "ARSO production cycle %s — %d agents × %d cycles (budget=%ss)",
            run_id, n, cycles_per_agent,
            f"{time_budget_s:.0f}" if time_budget_s else "∞",
        )

        coroutines = [
            self._evolve_agent(SPECIALIST_AGENTS[i], ids[i], tasks[i], cycles_per_agent, score_threshold)
            for i in range(n)
        ]

        agent_tasks = [asyncio.create_task(c) for c in coroutines]

        if time_budget_s:
            _, pending = await asyncio.wait(agent_tasks, timeout=time_budget_s)
            for t in pending:
                t.cancel()
            if pending:
                logger.warning("ARSO cycle %s: %d/%d agents cancelled at %.0fs budget",
                               run_id, len(pending), len(agent_tasks), time_budget_s)
        else:
            await asyncio.wait(agent_tasks)

        results: List[AgentResult] = []
        for t in agent_tasks:
            if t.done() and not t.cancelled():
                exc = t.exception()
                if exc:
                    logger.warning("Agent task exception: %s", exc)
                else:
                    results.append(t.result())

        any_cancelled = any(t.cancelled() for t in agent_tasks)
        status = "partial" if any_cancelled else ("complete" if results else "failed")

        elapsed = time.time() - t0

        elite_count = sum(1 for r in results if r.elite_promoted)
        best = max(results, key=lambda r: r.score) if results else None

        report = ProductionReport(
            run_id=run_id,
            started_at=started_at,
            completed_at=datetime.utcnow().isoformat(),
            wall_clock_s=round(elapsed, 2),
            total_agents=len(results),
            total_cycles=sum(
                (r.cycles_run if isinstance(r.cycles_run, int) else len(r.cycles_run))
                for r in results
            ),
            elite_count=elite_count,
            best_score=round(best.score, 4) if best else 0.0,
            best_agent_id=best.agent_id if best else "",
            agent_results=[asdict(r) for r in results],
            status=status,
        )

        logger.info("ARSO cycle %s done — %s", run_id, report.summary().replace("\n", " | "))
        return report


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_orchestrator: Optional[ARSOOrchestrator] = None


def get_orchestrator(gateway_url: str = "http://localhost:9000") -> ARSOOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ARSOOrchestrator(gateway_url=gateway_url)
    return _orchestrator
