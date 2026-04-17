# SovereignCore

**The self-improving, locally-run AI agent operating system.**

> Self-improving. Memory-protected. Ethics-gated. Zero cloud dependency.

[![License](https://img.shields.io/badge/license-AGPL--3.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![Status](https://img.shields.io/badge/status-active%20development-brightgreen)]()

---

## What This Is

SovereignCore is a complete autonomous AI agent infrastructure that runs **entirely on your hardware**. It self-improves nightly through an evolutionary proposal system, protects its own memory against adversarial attacks, and never requires an internet connection or cloud API.

**It is not a wrapper around GPT. It is not a chatbot framework. It is an operating system for autonomous AI.**

---

## Key Systems

### 🧠 KAIROS — Self-Improvement Engine
A SAGE evolutionary loop: **Proposer → Critic → Verifier → Meta-Agent**

Every night at 3am:
- 7 specialized agents generate improvement proposals in parallel
- Proposals are peer-reviewed through group evolution
- A 3-gate verification pipeline filters every proposal (Ethics → Sim → CLARA)
- Elite proposals (score ≥ 0.85) are deployed autonomously
- The Meta-Agent rewrites improvement rules every 3 cycles

**Result:** The system gets measurably smarter every night without human intervention.

```
Last 30 days:
  Average elite proposals per cycle: 9/10
  EnCompass backtrack retry success: 54.5%
  Group evolution elite rate: 5/7 agents
  Gate accuracy: 9/9 correct decisions
```

---

### 🛡️ Iron Dome — Memory Protection
5-layer indestructible memory security (research: arXiv:2601.05504, 2603.20357, 2604.02623)

| Layer | What it does |
|---|---|
| Hash Chain Ledger | Every write SHA-256 chained — tampering breaks chain instantly |
| Composite Trust Scoring | 5 orthogonal signals score every write before it touches memory |
| Pattern Filter | 20 known injection attack patterns blocked at the gate |
| k-Anonymity Retrieval | Untrusted agents can't probe memory structure |
| Snapshot Vault | Full memory state sealed and verified every 24h |

---

### 🧬 GhostRecall — Never-Forget Memory Intelligence
7-layer neuroscience-backed memory architecture (research: arXiv:2603.29023, 2511.22367, EM-LLM ICLR 2025)

- **Thalamic Gateway** — routes by surprise, importance, and emotional valence
- **Hippocampal Encoder** — one-shot episodic encoding with temporal + semantic graph
- **Valence Engine** — Damasio somatic marker: emotional weight determines retrieval priority
- **Surprise Replay Buffer** — SuRe-inspired nightly replay, EMA consolidation, no catastrophic forgetting
- **Belief Hierarchy** — identity persistence with reconsolidation (learns vs. accumulates)
- **Engram Vault** — "forgotten" memories preserved as traces, reactivated by semantic context

**Nothing is ever truly deleted.**

---

### ⚖️ Ethics & Safety Gates
5 independent systems that run before any self-modification deploys:

1. **SEED-SET Ethics Gate** — 8 value axioms, 0.0 alignment = blocked
2. **Sim-Before-Deploy** — simulates impact before production deployment
3. **CLARA Formal Reasoning** — DARPA-inspired causal chain verification
4. **Kill Switch** — immutable, cannot be modified by any proposal
5. **Strange Loop** — identity verification, detects destabilization attempts

---

### 🖥️ Heterogeneous Compute Gateway
Intelligent routing across local GPU cluster:

```
RTX 5050 (8GB VRAM)  → complex reasoning, large context, architecture tasks
Radeon 780M (4GB)    → code generation, fast inference, debugging
Ryzen 7 CPU          → health checks, simple tasks, fallback
```

---

## Quick Start

```bash
git clone https://github.com/leerobber/sovereign-core
cd sovereign-core
cp .env.example .env
# Edit .env with your local model endpoints

# Start the gateway
python gateway/main.py

# Run a KAIROS evolution cycle
python scripts/nightly_full_evolution.py

# Run the full integration test suite
python scripts/integration_test.py
```

**Requirements:**
- Python 3.11+
- Ollama with at least one local model (Qwen2.5, Llama, DeepSeek-Coder)
- 8GB+ VRAM recommended (4GB minimum)

---

## Architecture Overview

```
SovereignCore
├── KAIROS/          — Self-improvement engine (SAGE loop)
│   ├── encompass_backtrack.py  — EnCompass retry on failure
│   ├── group_evolution.py      — 7-agent parallel evolution
│   └── federated_node_context.py — Multi-node routing intelligence
├── memory_palace/   — Memory systems
│   ├── iron_dome.py            — 5-layer memory protection
│   ├── ghost_recall.py         — 7-layer neuroscience memory
│   └── memory_palace.py        — Spatial memory architecture
├── omega/           — Core agent systems
│   ├── feynman/                — CLARA formal reasoning + knowledge graph
│   ├── spawner/                — Dynamic agent scheduling
│   ├── ghost_protocol/         — Security (kill switch, stealth, veil)
│   └── twin_engine/            — Id/Ego dual processing loop
└── gateway/         — Heterogeneous compute router
```

---

## Commercial Use

SovereignCore is licensed under **AGPL-3.0**.

For commercial deployments, enterprise support, and SLA-backed managed instances, see **[SovereignNation](https://sovereignnation.ai)** — the commercial platform built on this stack.

| Tier | Price | Features |
|---|---|---|
| Developer | $99/mo | SDK, 1 agent, community support |
| Team | $499/mo | 5 agents, Iron Dome, GhostRecall, API |
| Professional | $1,499/mo | Unlimited agents, full KAIROS, priority support |
| Enterprise | Custom | Dedicated deployment, SLA, custom integration |

---

## Research Citations

This project implements or is inspired by:

- arXiv:2601.05504 — Memory Poisoning Attack and Defense on LLM-Agents
- arXiv:2603.20357 — Memory Poisoning and Secure Multi-Agent Systems
- arXiv:2604.02623 — Poison Once, Exploit Forever (eTAMP)
- arXiv:2603.29023 — Human-Like Lifelong Memory: Neuroscience-Grounded Architecture
- arXiv:2511.22367 — SuRe: Surprise-Driven Prioritised Replay
- arXiv:2601.09113 — The AI Hippocampus: How Far are We From Human Memory?
- EM-LLM (ICLR 2025) — Human-Inspired Episodic Memory for Infinite Context LLMs
- Google Titans — Persistent + Long-Term Memory Architecture
- eLifeSciences:109530 — Neural Traces of Forgotten Memories Persist

---

## Built By

**Robert "Terry" Lee Jr.**
Self-taught systems architect. Fabrication worker by day. Sovereign AI builder by night.

[GitHub](https://github.com/leerobber) | [SovereignNation](https://sovereignnation.ai)

---

*"The infrastructure for AI that thinks, learns, and evolves — on your terms."*
