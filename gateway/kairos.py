"""KAIROS — nextElites Agent Economy.

KAIROS (Knowledge-Augmented Iterative Recursive Optimization System) is the
next evolutionary phase of Sovereign Core's agent economy.  It produces
"Elites" — agents that have survived multiple evolutionary cycles and proven
their optimization capabilities.

Legitimate capabilities implemented
────────────────────────────────────
1. Self-evolving agents  — HyperAgents-style recursive self-improvement.
2. Archive-aware reconstruction — rebuilds successful ancestor agents (DGM-H).
3. Cross-domain transfer — optimization skills generalise across bottleneck types.
4. Memory-evolving — agents evolve their own retrieval strategies (MemEvolve).

Explicitly excluded
────────────────────
"Undercover Mode" (persona management / identity obfuscation for stealth
open-source contribution) is NOT implemented.  See docs/KAIROS-architecture.md
and PR #31 for the ethical rationale.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SkillDomain(str, Enum):
    """Domains of optimization expertise an agent can acquire."""

    VRAM_OPTIMIZATION = "vram_optimization"
    LATENCY_REDUCTION = "latency_reduction"
    THROUGHPUT_SCALING = "throughput_scaling"
    ENERGY_EFFICIENCY = "energy_efficiency"


class AgentTier(str, Enum):
    """Evolutionary tier of a KAIROS agent."""

    STANDARD = "standard"
    ELITE = "elite"
    NEXT_ELITE = "next_elite"  # KAIROS tier


# ---------------------------------------------------------------------------
# RetrievalWeights
# ---------------------------------------------------------------------------


@dataclass
class RetrievalWeights:
    """Weights governing an agent's memory retrieval strategy (MemEvolve)."""

    recency: float = 0.4
    relevance: float = 0.4
    frequency: float = 0.2

    def normalise(self) -> RetrievalWeights:
        """Return a new RetrievalWeights normalised so weights sum to 1.0.

        Raises ZeroDivisionError if all weights are zero.
        """
        total = self.recency + self.relevance + self.frequency
        if total == 0.0:
            raise ZeroDivisionError("Cannot normalise weights that all sum to zero.")
        return RetrievalWeights(
            recency=self.recency / total,
            relevance=self.relevance / total,
            frequency=self.frequency / total,
        )


# ---------------------------------------------------------------------------
# KAIROSAgent
# ---------------------------------------------------------------------------


@dataclass
class KAIROSAgent:
    """An agent participating in the KAIROS evolutionary economy."""

    agent_id: str
    generation: int = 0
    tier: AgentTier = AgentTier.STANDARD
    optimizations_successful: int = 0
    auction_wins: int = 0
    auction_participations: int = 0
    ancestor_id: Optional[str] = None
    skill_domains: list[SkillDomain] = field(default_factory=list)
    retrieval_weights: RetrievalWeights = field(default_factory=RetrievalWeights)
    skill_transfer_successes: dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_evolved_at: Optional[float] = None

    @property
    def auction_win_rate(self) -> float:
        """Fraction of auctions this agent has won."""
        return self.auction_wins / max(1, self.auction_participations)

    @property
    def optimization_rate(self) -> float:
        """Fraction of attempts that resulted in a successful optimization."""
        return self.optimizations_successful / max(
            1, self.optimizations_successful + self.auction_participations
        )

    @property
    def fitness_score(self) -> float:
        """Composite fitness: 0.6 * optimization_rate + 0.4 * auction_win_rate."""
        return (0.6 * self.optimization_rate) + (0.4 * self.auction_win_rate)


# ---------------------------------------------------------------------------
# KAIROSEvolutionEngine
# ---------------------------------------------------------------------------


class KAIROSEvolutionEngine:
    """Core evolution engine implementing HyperAgents, DGM-H, SIMA, and MemEvolve."""

    ELITE_THRESHOLD: int = 5
    NEXT_ELITE_THRESHOLD: int = 10
    FITNESS_ELITE: float = 0.6
    FITNESS_NEXT_ELITE: float = 0.8
    MIN_WEIGHT: float = 0.05
    MAX_WEIGHT: float = 0.9

    def __init__(self) -> None:
        """Initialise with an empty agent archive."""
        self._archive: dict[str, KAIROSAgent] = {}

    def evolve_agent(self, agent: KAIROSAgent) -> KAIROSAgent:
        """Apply HyperAgents-style self-evolution: increment generation, evolve memory, evaluate tier."""
        new_weights = self.evolve_memory_strategy(agent)
        new_agent = dataclasses.replace(
            agent,
            generation=agent.generation + 1,
            retrieval_weights=new_weights,
            last_evolved_at=time.time(),
        )
        self._archive[agent.agent_id] = dataclasses.replace(new_agent)
        new_tier = self._compute_tier(new_agent)
        new_agent = dataclasses.replace(new_agent, tier=new_tier)
        logger.info(
            "Agent %s evolved to generation %d tier=%s fitness=%.4f",
            new_agent.agent_id,
            new_agent.generation,
            new_agent.tier.value,
            new_agent.fitness_score,
        )
        return new_agent

    def reconstruct_from_archive(self, ancestor_id: str) -> KAIROSAgent:
        """Rebuild a new agent from a successful archived ancestor (DGM-H pattern).

        Raises KeyError if ancestor_id is not in the archive.
        The new agent gets a fresh agent_id, generation reset to 0, and inherits
        skill_domains and retrieval_weights from the ancestor.
        """
        if ancestor_id not in self._archive:
            raise KeyError(f"Ancestor not in archive: {ancestor_id!r}")
        ancestor = self._archive[ancestor_id]
        new_agent = KAIROSAgent(
            agent_id=str(uuid.uuid4()),
            generation=0,
            tier=AgentTier.STANDARD,
            ancestor_id=ancestor_id,
            skill_domains=list(ancestor.skill_domains),
            retrieval_weights=dataclasses.replace(ancestor.retrieval_weights),
        )
        logger.info(
            "Reconstructed agent %s from ancestor %s",
            new_agent.agent_id,
            ancestor_id,
        )
        return new_agent

    def transfer_skills(
        self,
        source: KAIROSAgent,
        target: KAIROSAgent,
        domain: SkillDomain,
    ) -> KAIROSAgent:
        """Transfer a skill domain from source to target (SIMA cross-domain transfer).

        Adds domain to target skill_domains if not already present.
        Increments target skill_transfer_successes[domain.value].
        Returns the updated target agent.
        """
        new_domains = list(target.skill_domains)
        if domain not in new_domains:
            new_domains.append(domain)

        new_successes = dict(target.skill_transfer_successes)
        new_successes[domain.value] = new_successes.get(domain.value, 0) + 1

        updated = dataclasses.replace(
            target,
            skill_domains=new_domains,
            skill_transfer_successes=new_successes,
        )
        logger.info(
            "Transferred skill %s from agent %s to %s",
            domain.value,
            source.agent_id,
            target.agent_id,
        )
        return updated

    def evolve_memory_strategy(self, agent: KAIROSAgent) -> RetrievalWeights:
        """Meta-evolve the agent's retrieval weights (MemEvolve pattern).

        High-fitness agents (>0.7) shift weight toward recency (they adapt quickly).
        Low-fitness agents (<=0.3) shift weight toward relevance (past successes).
        Returns new normalised RetrievalWeights.
        """
        fitness = agent.fitness_score
        recency = agent.retrieval_weights.recency
        relevance = agent.retrieval_weights.relevance
        frequency = agent.retrieval_weights.frequency

        if fitness > 0.7:
            recency += 0.05
            frequency -= 0.05
        elif fitness <= 0.3:
            relevance += 0.05
            recency -= 0.05

        recency = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, recency))
        relevance = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, relevance))
        frequency = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, frequency))
        # Clamping prevents degenerate weights even if prior evolutions pushed a
        # weight close to zero before this cycle's increment/decrement was applied.

        return RetrievalWeights(
            recency=recency,
            relevance=relevance,
            frequency=frequency,
        ).normalise()

    def evaluate_fitness(self, agent: KAIROSAgent) -> float:
        """Return agent.fitness_score (convenience wrapper for external callers)."""
        return agent.fitness_score

    def _compute_tier(self, agent: KAIROSAgent) -> AgentTier:
        """Return new tier based on generation and fitness thresholds."""
        fitness = agent.fitness_score
        if agent.generation >= self.NEXT_ELITE_THRESHOLD and fitness >= self.FITNESS_NEXT_ELITE:
            return AgentTier.NEXT_ELITE
        if agent.generation >= self.ELITE_THRESHOLD and fitness >= self.FITNESS_ELITE:
            return AgentTier.ELITE
        return AgentTier.STANDARD


# ---------------------------------------------------------------------------
# EliteRegistry
# ---------------------------------------------------------------------------


class EliteRegistry:
    """Registry tracking all agents in the KAIROS ecosystem."""

    def __init__(self, engine: KAIROSEvolutionEngine) -> None:
        """Initialise with an empty agent store backed by the given engine."""
        self._agents: dict[str, KAIROSAgent] = {}
        self._engine = engine

    def register(self, agent: KAIROSAgent) -> None:
        """Add or update an agent in the registry."""
        self._agents[agent.agent_id] = agent

    def get(self, agent_id: str) -> KAIROSAgent:
        """Return agent by ID.

        Raises KeyError if not found.
        """
        if agent_id not in self._agents:
            raise KeyError(f"Agent not found: {agent_id!r}")
        return self._agents[agent_id]

    def promote(self, agent_id: str) -> KAIROSAgent:
        """Trigger one evolution cycle for the agent and update the registry.

        Raises KeyError if agent is not found.
        """
        agent = self.get(agent_id)
        evolved = self._engine.evolve_agent(agent)
        self._agents[agent_id] = evolved
        return evolved

    def list_elites(self) -> list[KAIROSAgent]:
        """Return agents with tier ELITE or NEXT_ELITE, sorted by fitness descending."""
        elites = [
            a
            for a in self._agents.values()
            if a.tier in (AgentTier.ELITE, AgentTier.NEXT_ELITE)
        ]
        return sorted(elites, key=lambda a: a.fitness_score, reverse=True)

    def list_all(self) -> list[KAIROSAgent]:
        """Return all agents, sorted by fitness descending."""
        return sorted(self._agents.values(), key=lambda a: a.fitness_score, reverse=True)

    def metrics(self) -> dict[str, Any]:
        """Return population stats: total, elite_count, next_elite_count, avg_fitness."""
        agents = list(self._agents.values())
        total = len(agents)
        elite_count = sum(1 for a in agents if a.tier == AgentTier.ELITE)
        next_elite_count = sum(1 for a in agents if a.tier == AgentTier.NEXT_ELITE)
        avg_fitness = sum(a.fitness_score for a in agents) / total if total else 0.0
        return {
            "total": total,
            "elite_count": elite_count,
            "next_elite_count": next_elite_count,
            "avg_fitness": round(avg_fitness, 4),
        }


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_kairos_engine = KAIROSEvolutionEngine()
elite_registry = EliteRegistry(_kairos_engine)
