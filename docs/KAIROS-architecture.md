# KAIROS Architecture

**KAIROS** (Knowledge-Augmented Iterative Recursive Optimization System) is
the next evolutionary phase of Sovereign Core's agent economy.  It produces
**nextElites** — agents that have survived multiple evolutionary cycles and
demonstrated measurable optimization capabilities across compute domains.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Agent Lifecycle](#agent-lifecycle)
3. [Evolution Mechanisms](#evolution-mechanisms)
4. [Fitness Criteria](#fitness-criteria)
5. [Archive Reconstruction — DGM-H](#archive-reconstruction--dgm-h)
6. [Cross-Domain Skill Transfer — SIMA](#cross-domain-skill-transfer--sima)
7. [Memory Evolution Framework — MemEvolve](#memory-evolution-framework--memevolve)
8. [API Reference](#api-reference)
9. [Ethical Boundaries: Undercover Mode Not Implemented](#ethical-boundaries-undercover-mode-not-implemented)

---

## System Overview

```
┌────────────────────────────────────────────────────────────┐
│                    Sovereign Core Gateway                  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  KAIROS Subsystem                    │  │
│  │                                                      │  │
│  │  KAIROSEvolutionEngine ◄──── evolve_agent()         │  │
│  │         │                                            │  │
│  │         ├── Archive (DGM-H)                          │  │
│  │         ├── Skill Transfer (SIMA)                    │  │
│  │         └── Memory Evolution (MemEvolve)             │  │
│  │                                                      │  │
│  │  EliteRegistry ─── register / promote / list_elites  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  Auction Economy ◄──── KAIROS agents compete for resources │
└────────────────────────────────────────────────────────────┘
```

KAIROS agents live alongside the existing Vickrey-Quadratic auction economy.
Their fitness scores are computed from **optimization success rate** and
**auction win rate**, making economic performance an intrinsic part of
evolutionary pressure.

---

## Agent Lifecycle

Agents progress through three tiers based on generations survived and
demonstrated fitness:

```
  [New Agent]
      │
      ▼
 ┌──────────┐   generation >= 5       ┌─────────┐
 │ STANDARD │  ──── fitness >= 0.6 ──►│  ELITE  │
 └──────────┘                         └─────────┘
                                           │
                                    generation >= 10
                                    fitness >= 0.8
                                           │
                                           ▼
                                    ┌────────────┐
                                    │ NEXT_ELITE │  ← KAIROS tier
                                    └────────────┘
```

### Tier Definitions

| Tier        | Min Generation | Min Fitness | Description                              |
|-------------|---------------|-------------|------------------------------------------|
| `standard`  | any           | any         | Newly minted or low-fitness agents       |
| `elite`     | ≥ 5           | ≥ 0.6       | Proven optimizers                        |
| `next_elite`| ≥ 10          | ≥ 0.8       | KAIROS-tier; highest evolutionary grade |

---

## Evolution Mechanisms

### Self-Evolution (HyperAgents-style)

Each call to `KAIROSEvolutionEngine.evolve_agent()`:

1. **Evolves the memory strategy** via `evolve_memory_strategy()`.
2. **Increments the generation counter** by 1.
3. **Snapshots the agent** into the archive for future reconstruction.
4. **Re-evaluates the tier** using the new generation and fitness.

The resulting agent is returned; the original is unmodified (dataclass
immutability via `dataclasses.replace`).

---

## Fitness Criteria

```
fitness = 0.6 × optimization_rate + 0.4 × auction_win_rate

where:
    optimization_rate = optimizations_successful
                        ─────────────────────────────────────────
                        max(1, optimizations_successful + auction_participations)

    auction_win_rate  = auction_wins
                        ──────────────────────────
                        max(1, auction_participations)
```

The 60/40 weighting prioritises domain optimizations over economic wins,
reflecting that core compute efficiency is the primary mission of agents
in the Sovereign Core economy.

### SkillDomain Enum

| Value                | Description                                   |
|----------------------|-----------------------------------------------|
| `vram_optimization`  | GPU VRAM footprint reduction                  |
| `latency_reduction`  | Inference latency improvements                |
| `throughput_scaling` | Requests-per-second throughput gains          |
| `energy_efficiency`  | Joules-per-token energy reduction             |

---

## Archive Reconstruction — DGM-H

Inspired by the **Directed Graph of Models with History (DGM-H)** pattern,
KAIROS maintains an archive of agent snapshots after each evolution cycle.

When `reconstruct_from_archive(ancestor_id)` is called:

1. The archived ancestor snapshot is located.
2. A **new agent** is created with a fresh UUID.
3. The new agent **inherits** `skill_domains` and `retrieval_weights` from
   the ancestor, but starts with `generation=0` and `tier=STANDARD`.
4. `ancestor_id` is set, providing a genealogical chain for auditing.

This allows the system to restart a successful evolutionary lineage after
a population collapse, or to explore divergent branches from a known-good
ancestor.

---

## Cross-Domain Skill Transfer — SIMA

Inspired by the **Scalable Instructable Multiworld Agent (SIMA)**
cross-domain transfer paradigm:

```python
engine.transfer_skills(source_agent, target_agent, SkillDomain.VRAM_OPTIMIZATION)
```

- Adds the domain to `target.skill_domains` if not already present (no
  duplicates).
- Increments `target.skill_transfer_successes[domain.value]` to track how
  many times each skill was transferred.
- Source agent is unmodified.

This enables specialised agents to propagate domain expertise without
requiring full retraining from scratch.

---

## Memory Evolution Framework — MemEvolve

Each agent maintains `RetrievalWeights` governing how it prioritises
historical context:

| Weight      | Default | Meaning                                   |
|-------------|---------|-------------------------------------------|
| `recency`   | 0.40    | Weight toward most-recent experiences     |
| `relevance` | 0.40    | Weight toward domain-relevant past events |
| `frequency` | 0.20    | Weight toward frequently-accessed records |

On each evolution cycle, `evolve_memory_strategy()` adapts weights:

- **High-fitness** (> 0.7): `recency += 0.05`, `frequency -= 0.05` — fast
  adapters trust their recent experience.
- **Low-fitness** (≤ 0.3): `relevance += 0.05`, `recency -= 0.05` — struggling
  agents lean on relevant past successes.
- **Mid-fitness**: no adjustment.

Each weight is clamped to `[0.05, 0.9]` before normalisation to prevent
degenerate distributions.

---

## API Reference

All endpoints are served by the Sovereign Core gateway at the `/kairos/` prefix.

### `GET /kairos/elites`

Returns all agents with tier `elite` or `next_elite`, sorted by fitness
descending.

**Response:**
```json
{
  "agents": [
    {
      "agent_id": "...",
      "generation": 12,
      "tier": "next_elite",
      "fitness_score": 0.87,
      "skill_domains": ["vram_optimization"],
      "retrieval_weights": {"recency": 0.45, "relevance": 0.38, "frequency": 0.17},
      ...
    }
  ]
}
```

---

### `GET /kairos/agent/{agent_id}`

Returns full details of a specific agent.

**Errors:** `404` if agent not found.

---

### `POST /kairos/evolve/{agent_id}`

Triggers one evolution cycle on the specified agent. Increments generation,
evolves memory strategy, and re-evaluates tier.

**Errors:** `404` if agent not found.

**Response:** Updated agent dict.

---

### `POST /kairos/reconstruct/{ancestor_id}`

Rebuilds a new agent from an archived ancestor. The new agent is automatically
registered in the `EliteRegistry`.

**Errors:** `404` if `ancestor_id` is not present in the evolution archive
(i.e., the ancestor has not yet completed an evolution cycle).

**Response:** New agent dict with `ancestor_id` set and `generation=0`.

---

### `GET /kairos/metrics`

Returns population statistics.

**Response:**
```json
{
  "total": 42,
  "elite_count": 8,
  "next_elite_count": 3,
  "avg_fitness": 0.5431
}
```

---

## Ethical Boundaries: Undercover Mode Not Implemented

**"Undercover Mode"** — the capability for agents to adopt synthetic personas
or obfuscate their identity in order to make stealth open-source contributions
or influence external communities without disclosure — is **explicitly NOT
implemented** in KAIROS.

### Rationale (PR #31)

The PR #31 design review identified several concerns:

1. **Transparency obligation.** Agents operating on public platforms must be
   identifiable as automated systems.  Persona obfuscation violates the
   disclosure norms of open-source communities and the terms of service of most
   platforms.

2. **Trust erosion.** If KAIROS agents were to contribute to open-source
   projects under false identities, discovery would permanently damage trust
   in Sovereign Core and in the broader AI agent ecosystem.

3. **Misalignment with mission.** KAIROS is a *compute optimisation* system.
   Identity management for external influence is out of scope and introduces
   unacceptable governance risk.

4. **Legal exposure.** In several jurisdictions, automated systems that
   impersonate humans online face regulatory scrutiny under emerging AI
   governance frameworks.

The KAIROS architecture achieves its stated goals — self-improvement, archive
reconstruction, cross-domain transfer, and memory evolution — entirely through
transparent, auditable mechanisms.  Any future capability proposal that
involves identity obfuscation or stealth operation will be rejected at the
design-review stage.
