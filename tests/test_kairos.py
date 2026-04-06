"""Comprehensive tests for the KAIROS agent evolution system."""

from __future__ import annotations

import dataclasses
from contextlib import asynccontextmanager
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from gateway.kairos import (
    AgentTier,
    EliteRegistry,
    KAIROSAgent,
    KAIROSEvolutionEngine,
    RetrievalWeights,
    SkillDomain,
    elite_registry,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_agent(**kwargs: object) -> KAIROSAgent:
    """Create a KAIROSAgent with sensible defaults, overriding with kwargs."""
    import uuid

    defaults: dict[str, object] = {"agent_id": str(uuid.uuid4())}
    defaults.update(kwargs)
    return KAIROSAgent(**defaults)  # type: ignore[arg-type]


def _high_fitness_agent() -> KAIROSAgent:
    """Return an agent with fitness > 0.8 (many wins, many optimizations)."""
    return _make_agent(
        optimizations_successful=90,
        auction_wins=80,
        auction_participations=100,
    )


def _low_fitness_agent() -> KAIROSAgent:
    """Return an agent with fitness <= 0.3 (no wins, no optimizations)."""
    return _make_agent(
        optimizations_successful=0,
        auction_wins=0,
        auction_participations=10,
    )


# ===========================================================================
# TestRetrievalWeights
# ===========================================================================


class TestRetrievalWeights:
    """Tests for the RetrievalWeights dataclass."""

    def test_default_weights_sum_to_1(self) -> None:
        """Default weights should already sum to 1.0."""
        w = RetrievalWeights()
        assert abs(w.recency + w.relevance + w.frequency - 1.0) < 1e-9

    def test_normalise_returns_correct_ratios(self) -> None:
        """normalise() should maintain proportions after scaling."""
        w = RetrievalWeights(recency=2.0, relevance=2.0, frequency=1.0)
        n = w.normalise()
        assert abs(n.recency + n.relevance + n.frequency - 1.0) < 1e-9
        assert abs(n.recency - 0.4) < 1e-9
        assert abs(n.relevance - 0.4) < 1e-9
        assert abs(n.frequency - 0.2) < 1e-9

    def test_normalise_with_equal_weights(self) -> None:
        """Equal weights normalise to 1/3 each."""
        w = RetrievalWeights(recency=1.0, relevance=1.0, frequency=1.0)
        n = w.normalise()
        assert abs(n.recency - 1 / 3) < 1e-9
        assert abs(n.relevance - 1 / 3) < 1e-9
        assert abs(n.frequency - 1 / 3) < 1e-9

    def test_normalise_all_zero_raises(self) -> None:
        """normalise() raises ZeroDivisionError when all weights are zero."""
        w = RetrievalWeights(recency=0.0, relevance=0.0, frequency=0.0)
        with pytest.raises(ZeroDivisionError):
            w.normalise()

    def test_normalise_returns_new_instance(self) -> None:
        """normalise() must not mutate the original instance."""
        w = RetrievalWeights(recency=2.0, relevance=2.0, frequency=1.0)
        n = w.normalise()
        assert n is not w
        assert w.recency == 2.0  # original unchanged

    def test_default_recency_value(self) -> None:
        assert RetrievalWeights().recency == 0.4

    def test_default_relevance_value(self) -> None:
        assert RetrievalWeights().relevance == 0.4

    def test_default_frequency_value(self) -> None:
        assert RetrievalWeights().frequency == 0.2


# ===========================================================================
# TestKAIROSAgent
# ===========================================================================


class TestKAIROSAgent:
    """Tests for the KAIROSAgent dataclass and its computed properties."""

    def test_default_tier_is_standard(self) -> None:
        a = _make_agent()
        assert a.tier == AgentTier.STANDARD

    def test_fitness_score_zero_activity(self) -> None:
        """With no wins or optimizations, fitness should be 0."""
        a = _make_agent(optimizations_successful=0, auction_wins=0, auction_participations=0)
        assert a.fitness_score == 0.0

    def test_fitness_score_with_wins(self) -> None:
        """Check formula: 0.6 * opt_rate + 0.4 * win_rate."""
        a = _make_agent(optimizations_successful=0, auction_wins=5, auction_participations=10)
        opt_rate = 0 / max(1, 0 + 10)
        win_rate = 5 / max(1, 10)
        expected = 0.6 * opt_rate + 0.4 * win_rate
        assert abs(a.fitness_score - expected) < 1e-9

    def test_fitness_score_only_optimizations(self) -> None:
        """Fitness with only optimizations and no auction wins."""
        a = _make_agent(optimizations_successful=8, auction_wins=0, auction_participations=2)
        opt_rate = 8 / max(1, 8 + 2)
        win_rate = 0 / max(1, 2)
        expected = 0.6 * opt_rate + 0.4 * win_rate
        assert abs(a.fitness_score - expected) < 1e-9

    def test_auction_win_rate_zero_participations(self) -> None:
        """Win rate should be 0 when agent has not participated."""
        a = _make_agent(auction_wins=0, auction_participations=0)
        assert a.auction_win_rate == 0.0

    def test_auction_win_rate_with_data(self) -> None:
        a = _make_agent(auction_wins=3, auction_participations=10)
        assert abs(a.auction_win_rate - 0.3) < 1e-9

    def test_optimization_rate_zero_activity(self) -> None:
        a = _make_agent(optimizations_successful=0, auction_participations=0)
        assert a.optimization_rate == 0.0

    def test_optimization_rate_with_data(self) -> None:
        a = _make_agent(optimizations_successful=4, auction_participations=6)
        # opt_rate = 4 / max(1, 4 + 6) = 0.4
        assert abs(a.optimization_rate - 0.4) < 1e-9

    def test_agent_id_uniqueness(self) -> None:
        """Two independently created agents should have different IDs."""
        a1 = _make_agent()
        a2 = _make_agent()
        assert a1.agent_id != a2.agent_id

    def test_skill_domains_default_empty(self) -> None:
        a = _make_agent()
        assert a.skill_domains == []

    def test_ancestor_id_default_none(self) -> None:
        a = _make_agent()
        assert a.ancestor_id is None

    def test_generation_default_zero(self) -> None:
        a = _make_agent()
        assert a.generation == 0

    def test_skill_transfer_successes_default_empty(self) -> None:
        a = _make_agent()
        assert a.skill_transfer_successes == {}


# ===========================================================================
# TestKAIROSEvolutionEngine
# ===========================================================================


class TestKAIROSEvolutionEngine:
    """Tests for KAIROSEvolutionEngine methods."""

    def setup_method(self) -> None:
        """Fresh engine for every test."""
        self.engine = KAIROSEvolutionEngine()

    def test_evolve_increments_generation(self) -> None:
        agent = _make_agent(generation=3)
        evolved = self.engine.evolve_agent(agent)
        assert evolved.generation == 4

    def test_evolve_updates_last_evolved_at(self) -> None:
        agent = _make_agent()
        assert agent.last_evolved_at is None
        evolved = self.engine.evolve_agent(agent)
        assert evolved.last_evolved_at is not None

    def test_evolve_archives_agent(self) -> None:
        agent = _make_agent()
        self.engine.evolve_agent(agent)
        assert agent.agent_id in self.engine._archive

    def test_evolve_promotes_to_elite_when_threshold_met(self) -> None:
        """Agent reaching generation>=5 with fitness>=0.6 should become ELITE."""
        agent = _make_agent(
            generation=4,
            optimizations_successful=100,
            auction_wins=100,
            auction_participations=100,
        )
        evolved = self.engine.evolve_agent(agent)
        assert evolved.generation == 5
        assert evolved.tier == AgentTier.ELITE

    def test_evolve_promotes_to_next_elite_when_threshold_met(self) -> None:
        """Agent reaching generation>=10 with fitness>=0.8 should become NEXT_ELITE."""
        agent = _make_agent(
            generation=9,
            optimizations_successful=90,
            auction_wins=10,
            auction_participations=10,
        )
        evolved = self.engine.evolve_agent(agent)
        assert evolved.generation == 10
        assert evolved.tier == AgentTier.NEXT_ELITE

    def test_reconstruct_from_archive_returns_new_agent(self) -> None:
        agent = _make_agent()
        self.engine.evolve_agent(agent)
        new_agent = self.engine.reconstruct_from_archive(agent.agent_id)
        assert new_agent.agent_id != agent.agent_id

    def test_reconstruct_sets_ancestor_id(self) -> None:
        agent = _make_agent()
        self.engine.evolve_agent(agent)
        new_agent = self.engine.reconstruct_from_archive(agent.agent_id)
        assert new_agent.ancestor_id == agent.agent_id

    def test_reconstruct_inherits_skill_domains(self) -> None:
        agent = _make_agent(skill_domains=[SkillDomain.VRAM_OPTIMIZATION])
        self.engine.evolve_agent(agent)
        new_agent = self.engine.reconstruct_from_archive(agent.agent_id)
        assert SkillDomain.VRAM_OPTIMIZATION in new_agent.skill_domains

    def test_reconstruct_resets_generation(self) -> None:
        agent = _make_agent(generation=7)
        self.engine.evolve_agent(agent)
        new_agent = self.engine.reconstruct_from_archive(agent.agent_id)
        assert new_agent.generation == 0

    def test_reconstruct_unknown_ancestor_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.engine.reconstruct_from_archive("nonexistent-id")

    def test_transfer_skills_adds_domain(self) -> None:
        source = _make_agent(skill_domains=[SkillDomain.LATENCY_REDUCTION])
        target = _make_agent()
        updated = self.engine.transfer_skills(source, target, SkillDomain.LATENCY_REDUCTION)
        assert SkillDomain.LATENCY_REDUCTION in updated.skill_domains

    def test_transfer_skills_increments_counter(self) -> None:
        source = _make_agent()
        target = _make_agent()
        updated = self.engine.transfer_skills(source, target, SkillDomain.THROUGHPUT_SCALING)
        assert updated.skill_transfer_successes[SkillDomain.THROUGHPUT_SCALING.value] == 1

    def test_transfer_skills_domain_already_present_no_duplicate(self) -> None:
        source = _make_agent()
        target = _make_agent(skill_domains=[SkillDomain.ENERGY_EFFICIENCY])
        updated = self.engine.transfer_skills(source, target, SkillDomain.ENERGY_EFFICIENCY)
        assert updated.skill_domains.count(SkillDomain.ENERGY_EFFICIENCY) == 1

    def test_transfer_skills_updates_correct_agent(self) -> None:
        """transfer_skills should return the target, not the source."""
        source = _make_agent()
        target = _make_agent()
        updated = self.engine.transfer_skills(source, target, SkillDomain.VRAM_OPTIMIZATION)
        assert updated.agent_id == target.agent_id

    def test_evolve_memory_high_fitness_shifts_recency(self) -> None:
        """High-fitness agents increase recency weight."""
        agent = _high_fitness_agent()
        original_recency = agent.retrieval_weights.recency
        new_weights = self.engine.evolve_memory_strategy(agent)
        # After adding 0.05 to recency and normalising, recency proportion should increase
        # relative to the default; verify it's above original proportional share
        assert new_weights.recency > 0.0  # basic sanity
        # Compare to a mid-fitness agent's recency — high fitness gets more recency
        mid_agent = _make_agent(
            optimizations_successful=5, auction_wins=5, auction_participations=10
        )
        mid_weights = self.engine.evolve_memory_strategy(mid_agent)
        assert new_weights.recency >= mid_weights.recency

    def test_evolve_memory_low_fitness_shifts_relevance(self) -> None:
        """Low-fitness agents increase relevance weight."""
        agent = _low_fitness_agent()
        new_weights = self.engine.evolve_memory_strategy(agent)
        mid_agent = _make_agent(
            optimizations_successful=5, auction_wins=5, auction_participations=10
        )
        mid_weights = self.engine.evolve_memory_strategy(mid_agent)
        assert new_weights.relevance >= mid_weights.relevance

    def test_evolve_memory_mid_fitness_no_shift(self) -> None:
        """Mid-range fitness (0.3 < f <= 0.7) does not shift recency or relevance."""
        # fitness ~= 0.5: optimizations=5/(5+10)=0.33 -> opt_rate; wins=5/10=0.5 -> win_rate
        # fitness = 0.6*0.33 + 0.4*0.5 = ~0.4
        agent = _make_agent(
            optimizations_successful=5,
            auction_wins=5,
            auction_participations=10,
        )
        original = agent.retrieval_weights
        new_w = self.engine.evolve_memory_strategy(agent)
        # No branch taken; weights after normalise should match original (already summing to 1)
        assert abs(new_w.recency - original.recency) < 1e-9
        assert abs(new_w.relevance - original.relevance) < 1e-9

    def test_evolve_memory_weights_normalised(self) -> None:
        """Resulting weights should always sum to 1.0."""
        for agent in [_high_fitness_agent(), _low_fitness_agent(), _make_agent()]:
            w = self.engine.evolve_memory_strategy(agent)
            assert abs(w.recency + w.relevance + w.frequency - 1.0) < 1e-9

    def test_evaluate_fitness_matches_property(self) -> None:
        agent = _high_fitness_agent()
        assert self.engine.evaluate_fitness(agent) == agent.fitness_score

    def test_compute_tier_standard_below_threshold(self) -> None:
        agent = _make_agent(generation=2)
        assert self.engine._compute_tier(agent) == AgentTier.STANDARD

    def test_compute_tier_elite(self) -> None:
        agent = _make_agent(
            generation=5,
            optimizations_successful=100,
            auction_wins=100,
            auction_participations=100,
        )
        assert self.engine._compute_tier(agent) == AgentTier.ELITE

    def test_compute_tier_next_elite(self) -> None:
        agent = _make_agent(
            generation=10,
            optimizations_successful=90,
            auction_wins=10,
            auction_participations=10,
        )
        assert self.engine._compute_tier(agent) == AgentTier.NEXT_ELITE


# ===========================================================================
# TestEliteRegistry
# ===========================================================================


class TestEliteRegistry:
    """Tests for the EliteRegistry class."""

    def setup_method(self) -> None:
        engine = KAIROSEvolutionEngine()
        self.engine = engine
        self.registry = EliteRegistry(engine)

    def test_register_and_get(self) -> None:
        agent = _make_agent()
        self.registry.register(agent)
        retrieved = self.registry.get(agent.agent_id)
        assert retrieved.agent_id == agent.agent_id

    def test_get_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.registry.get("does-not-exist")

    def test_promote_increments_generation(self) -> None:
        agent = _make_agent(generation=0)
        self.registry.register(agent)
        evolved = self.registry.promote(agent.agent_id)
        assert evolved.generation == 1

    def test_promote_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.registry.promote("ghost-agent")

    def test_list_elites_returns_only_elite_tiers(self) -> None:
        std = _make_agent()
        elite = _make_agent(
            tier=AgentTier.ELITE,
            optimizations_successful=60,
            auction_wins=60,
            auction_participations=100,
        )
        self.registry.register(std)
        self.registry.register(elite)
        elites = self.registry.list_elites()
        ids = [a.agent_id for a in elites]
        assert elite.agent_id in ids
        assert std.agent_id not in ids

    def test_list_elites_sorted_by_fitness_descending(self) -> None:
        a1 = _make_agent(tier=AgentTier.ELITE, auction_wins=3, auction_participations=10)
        a2 = _make_agent(
            tier=AgentTier.ELITE,
            optimizations_successful=80,
            auction_wins=80,
            auction_participations=100,
        )
        self.registry.register(a1)
        self.registry.register(a2)
        elites = self.registry.list_elites()
        assert elites[0].fitness_score >= elites[-1].fitness_score

    def test_list_all_returns_all_agents(self) -> None:
        agents = [_make_agent() for _ in range(5)]
        for a in agents:
            self.registry.register(a)
        all_agents = self.registry.list_all()
        assert len(all_agents) == 5

    def test_list_all_sorted_by_fitness_descending(self) -> None:
        low = _make_agent(auction_wins=0, auction_participations=10)
        high = _make_agent(
            optimizations_successful=90, auction_wins=80, auction_participations=100
        )
        self.registry.register(low)
        self.registry.register(high)
        all_agents = self.registry.list_all()
        assert all_agents[0].fitness_score >= all_agents[-1].fitness_score

    def test_metrics_empty_registry(self) -> None:
        m = self.registry.metrics()
        assert m["total"] == 0
        assert m["avg_fitness"] == 0.0

    def test_metrics_counts_tiers_correctly(self) -> None:
        self.registry.register(_make_agent(tier=AgentTier.STANDARD))
        self.registry.register(_make_agent(tier=AgentTier.ELITE))
        self.registry.register(_make_agent(tier=AgentTier.NEXT_ELITE))
        m = self.registry.metrics()
        assert m["total"] == 3
        assert m["elite_count"] == 1
        assert m["next_elite_count"] == 1

    def test_metrics_avg_fitness(self) -> None:
        a1 = _make_agent(auction_wins=0, auction_participations=0)
        a2 = _make_agent(
            optimizations_successful=0, auction_wins=10, auction_participations=10
        )
        self.registry.register(a1)
        self.registry.register(a2)
        m = self.registry.metrics()
        expected = (a1.fitness_score + a2.fitness_score) / 2
        assert abs(m["avg_fitness"] - round(expected, 4)) < 1e-9

    def test_register_overwrites_existing_agent(self) -> None:
        agent = _make_agent(generation=0)
        self.registry.register(agent)
        updated = dataclasses.replace(agent, generation=5)
        self.registry.register(updated)
        retrieved = self.registry.get(agent.agent_id)
        assert retrieved.generation == 5

    def test_list_elites_next_elite_included(self) -> None:
        ne = _make_agent(tier=AgentTier.NEXT_ELITE)
        self.registry.register(ne)
        elites = self.registry.list_elites()
        assert any(a.agent_id == ne.agent_id for a in elites)

    def test_promote_updates_registry(self) -> None:
        agent = _make_agent()
        self.registry.register(agent)
        self.registry.promote(agent.agent_id)
        stored = self.registry.get(agent.agent_id)
        assert stored.generation == 1


# ===========================================================================
# TestEnums
# ===========================================================================


class TestEnums:
    """Tests for SkillDomain and AgentTier enums."""

    def test_skill_domain_values(self) -> None:
        assert SkillDomain.VRAM_OPTIMIZATION.value == "vram_optimization"
        assert SkillDomain.LATENCY_REDUCTION.value == "latency_reduction"
        assert SkillDomain.THROUGHPUT_SCALING.value == "throughput_scaling"
        assert SkillDomain.ENERGY_EFFICIENCY.value == "energy_efficiency"

    def test_agent_tier_values(self) -> None:
        assert AgentTier.STANDARD.value == "standard"
        assert AgentTier.ELITE.value == "elite"
        assert AgentTier.NEXT_ELITE.value == "next_elite"

    def test_skill_domain_is_str_enum(self) -> None:
        """SkillDomain inherits from str so str(domain) works."""
        domain = SkillDomain.VRAM_OPTIMIZATION
        assert isinstance(domain, str)
        assert domain == "vram_optimization"

    def test_agent_tier_is_str_enum(self) -> None:
        tier = AgentTier.ELITE
        assert isinstance(tier, str)
        assert tier == "elite"

    def test_skill_domain_all_members(self) -> None:
        assert len(SkillDomain) == 4

    def test_agent_tier_all_members(self) -> None:
        assert len(AgentTier) == 3


# ===========================================================================
# TestKAIROSEndpoints
# ===========================================================================


@pytest.fixture()
def kairos_client() -> Iterator[tuple[TestClient, EliteRegistry, KAIROSEvolutionEngine]]:
    """Provide a TestClient backed by a fresh KAIROS state."""
    import gateway.main as gm
    from gateway.kairos import EliteRegistry, KAIROSEvolutionEngine
    from gateway.benchmark import ThroughputBenchmark
    from gateway.config import GatewaySettings
    from gateway.health import HealthMonitor
    from gateway.models import ModelAssigner
    from gateway.router import GatewayRouter

    fresh_engine = KAIROSEvolutionEngine()
    fresh_registry = EliteRegistry(fresh_engine)
    gm._kairos_engine = fresh_engine
    gm._elite_registry = fresh_registry

    cfg = GatewaySettings(failure_threshold=1, recovery_threshold=1)
    gm._health_monitor = HealthMonitor(cfg=cfg)
    gm._benchmark = ThroughputBenchmark()
    gm._router = GatewayRouter(
        health_monitor=gm._health_monitor,
        assigner=ModelAssigner(),
        benchmark=gm._benchmark,
        cfg=cfg,
    )

    original_lifespan = gm.app.router.lifespan_context

    @asynccontextmanager
    async def _noop_lifespan(app):  # type: ignore[override]
        yield

    gm.app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(gm.app, raise_server_exceptions=True) as client:
            yield client, fresh_registry, fresh_engine
    finally:
        gm.app.router.lifespan_context = original_lifespan


class TestKAIROSEndpoints:
    """Integration tests for KAIROS FastAPI endpoints."""

    def test_list_elites_empty(self, kairos_client: tuple) -> None:
        client, _registry, _engine = kairos_client
        resp = client.get("/kairos/elites")
        assert resp.status_code == 200
        assert resp.json() == {"agents": []}

    def test_list_elites_with_elite_agents(self, kairos_client: tuple) -> None:
        client, registry, _ = kairos_client
        elite = _make_agent(tier=AgentTier.ELITE)
        registry.register(elite)
        resp = client.get("/kairos/elites")
        assert resp.status_code == 200
        agents = resp.json()["agents"]
        assert len(agents) == 1
        assert agents[0]["agent_id"] == elite.agent_id

    def test_get_agent_success(self, kairos_client: tuple) -> None:
        client, registry, _ = kairos_client
        agent = _make_agent()
        registry.register(agent)
        resp = client.get(f"/kairos/agent/{agent.agent_id}")
        assert resp.status_code == 200
        assert resp.json()["agent_id"] == agent.agent_id

    def test_get_agent_not_found(self, kairos_client: tuple) -> None:
        client, _registry, _engine = kairos_client
        resp = client.get("/kairos/agent/nonexistent-id")
        assert resp.status_code == 404

    def test_evolve_agent_success(self, kairos_client: tuple) -> None:
        client, registry, _ = kairos_client
        agent = _make_agent()
        registry.register(agent)
        resp = client.post(f"/kairos/evolve/{agent.agent_id}")
        assert resp.status_code == 200
        assert resp.json()["generation"] == 1

    def test_evolve_agent_not_found(self, kairos_client: tuple) -> None:
        client, _registry, _engine = kairos_client
        resp = client.post("/kairos/evolve/ghost-agent")
        assert resp.status_code == 404

    def test_reconstruct_success(self, kairos_client: tuple) -> None:
        client, registry, engine = kairos_client
        agent = _make_agent()
        registry.register(agent)
        # First evolve so it gets archived
        client.post(f"/kairos/evolve/{agent.agent_id}")
        resp = client.post(f"/kairos/reconstruct/{agent.agent_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ancestor_id"] == agent.agent_id
        assert data["generation"] == 0

    def test_reconstruct_not_found(self, kairos_client: tuple) -> None:
        client, _registry, _engine = kairos_client
        resp = client.post("/kairos/reconstruct/unknown-ancestor")
        assert resp.status_code == 404

    def test_metrics_empty(self, kairos_client: tuple) -> None:
        client, _registry, _engine = kairos_client
        resp = client.get("/kairos/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["avg_fitness"] == 0.0

    def test_metrics_with_agents(self, kairos_client: tuple) -> None:
        client, registry, _ = kairos_client
        registry.register(_make_agent())
        registry.register(_make_agent(tier=AgentTier.ELITE))
        resp = client.get("/kairos/metrics")
        data = resp.json()
        assert data["total"] == 2
        assert data["elite_count"] == 1

    def test_list_elites_response_structure(self, kairos_client: tuple) -> None:
        client, registry, _ = kairos_client
        registry.register(_make_agent(tier=AgentTier.NEXT_ELITE))
        resp = client.get("/kairos/elites")
        assert "agents" in resp.json()

    def test_get_agent_response_fields(self, kairos_client: tuple) -> None:
        client, registry, _ = kairos_client
        agent = _make_agent()
        registry.register(agent)
        data = client.get(f"/kairos/agent/{agent.agent_id}").json()
        for field in (
            "agent_id",
            "generation",
            "tier",
            "fitness_score",
            "skill_domains",
            "retrieval_weights",
        ):
            assert field in data, f"Missing field: {field}"

    def test_evolve_response_has_generation(self, kairos_client: tuple) -> None:
        client, registry, _ = kairos_client
        agent = _make_agent()
        registry.register(agent)
        data = client.post(f"/kairos/evolve/{agent.agent_id}").json()
        assert "generation" in data

    def test_reconstruct_sets_ancestor_id_in_response(self, kairos_client: tuple) -> None:
        client, registry, engine = kairos_client
        agent = _make_agent()
        registry.register(agent)
        client.post(f"/kairos/evolve/{agent.agent_id}")
        data = client.post(f"/kairos/reconstruct/{agent.agent_id}").json()
        assert data["ancestor_id"] == agent.agent_id

    def test_metrics_next_elite_count(self, kairos_client: tuple) -> None:
        client, registry, _ = kairos_client
        registry.register(_make_agent(tier=AgentTier.NEXT_ELITE))
        data = client.get("/kairos/metrics").json()
        assert data["next_elite_count"] == 1

    def test_reconstruct_registers_in_registry(self, kairos_client: tuple) -> None:
        """Reconstructed agent should be retrievable via GET."""
        client, registry, engine = kairos_client
        agent = _make_agent()
        registry.register(agent)
        client.post(f"/kairos/evolve/{agent.agent_id}")
        recon_data = client.post(f"/kairos/reconstruct/{agent.agent_id}").json()
        new_id = recon_data["agent_id"]
        get_resp = client.get(f"/kairos/agent/{new_id}")
        assert get_resp.status_code == 200
