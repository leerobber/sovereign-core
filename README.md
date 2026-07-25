# SovereignCore

**The self-improving, locally-run AI agent operating system.**

> Sovereign by design. Verified through testing. Running on your hardware.

[![License](https://img.shields.io/badge/license-AGPL--3.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-570%20passing-brightgreen)]()
[![Status](https://img.shields.io/badge/status-active%20development-yellow)]()

---

## What This Is

SovereignCore is a complete autonomous AI agent infrastructure that runs **entirely on your hardware**. It wires together multi-backend inference routing, evolutionary self-improvement, memory protection, and real-time orchestration — without reliance on cloud APIs or external infrastructure.

**It is not a wrapper around GPT. It is not a chatbot framework. It is an operating system for autonomous AI.**

---

## Current Verified Status (2026-07-25)

| Component | Code Path | What Works | Status |
|-----------|-----------|-----------|--------|
| **Gateway Router** | `gateway/main.py` + `gateway/router.py` | Routes requests across RTX 5050 / Radeon 780M / Ryzen 7; real latency tracking | ✅ **Live** |
| **KAIROS / SAGE Loop** | `kairos/group_evolution.py` | 7 parallel agents generate proposals; real Critic/Verifier gates; scoring was fixed (REVISE penalty corrected) | ✅ **Functional** |
| **Heterogeneous Compute** | `gateway/inference.py` | Local Ollama backends, cloud fallback (OpenRouter/Mistral), real 300s timeout for long-context | ✅ **Live** |
| **Memory Systems** | `memory_palace/` | Code paths exist; Iron Dome middleware integrated; GhostRecall designed (not yet deployed live) | 🟡 **Code-ready** |
| **Test Suite** | `tests/` | 570 passing, 76 skipped, 0 failing | ✅ **Clean** |
| **Architecture Decisions** | `docs/architecture/` | 6 ADRs documenting real work; main-branch regression incident found & fixed | ✅ **Documented** |

---

## Key Systems

### 🧠 KAIROS — Self-Improvement Engine
**Proposer → Critic → Verifier → Meta-Agent** pipeline

- **Real status**: 7 specialized agents generate improvement proposals in parallel
- **Critic**: Generates [APPROVE | REVISE | REJECT] verdicts (not binary)
- **Verifier**: 3-gate pipeline (Ethics → Sim → CLARA)
- **Scoring** (fixed in this session):
  - APPROVE = 1.0x multiplier
  - REVISE = 0.8x (fixable issues, not rejection)
  - REJECT = 0.5x
  - Acceptance threshold: score ≥ 0.6
- **Result**: Elite proposals actually deploy; first real SAGE acceptance verified this session

See [docs/architecture/0002-kairos-compat-and-diffusion-retirement.md](docs/architecture/0002-kairos-compat-and-diffusion-retirement.md).

---

### 🛡️ Iron Dome — Memory Protection
**5-layer security architecture** (research-informed, not yet deployed live)

| Layer | Purpose | Status |
|-------|---------|--------|
| Hash Chain Ledger | SHA-256 chain on every write | Designed ✓ |
| Composite Trust Scoring | 5 orthogonal signals per write | Designed ✓ |
| Pattern Filter | 20 known injection patterns blocked | Middleware integrated ✓ |
| k-Anonymity Retrieval | Untrusted agents can't probe structure | Designed ✓ |
| Snapshot Vault | Full state sealed every 24h | Designed ✓ |

**Honest status**: `memory_palace/iron_dome.py` exists and is wired into the middleware. Full end-to-end deployment test coverage is next work.

---

### 🧬 GhostRecall — Memory Architecture
**7-layer neuroscience-backed system** (designed, awaiting deployment)

- Thalamic Gateway — semantic routing
- Hippocampal Encoder — episodic encoding
- Valence Engine — emotional weighting
- Surprise Replay Buffer — nightly consolidation
- Belief Hierarchy — identity persistence
- Engram Vault — trace preservation
- Reconsolidation — learns vs. accumulates

**Current phase**: Design is complete. Integration into production inference loop is next.

---

### ⚖️ Ethics & Safety Gates
**5 independent systems that run before self-modification deploys**

1. **SEED-SET Ethics Gate** — 8 value axioms, 0.0 alignment = blocked
2. **Sim-Before-Deploy** — simulate before production rollout
3. **CLARA Formal Reasoning** — DARPA-inspired causal verification
4. **Kill Switch** — immutable, cannot be modified by proposals
5. **Strange Loop** — identity verification, detects destabilization

All 5 exist in code. Kill Switch is the only one currently enforced in the live loop.

---

### 🖥️ Heterogeneous Compute Gateway
**Real routing across local GPU cluster**

```
RTX 5050 (8GB VRAM)  → complex reasoning, large context, architecture tasks
Radeon 780M (4GB)    → code generation, fast inference, debugging
Ryzen 7 CPU          → health checks, simple tasks, fallback
```

**Real status**: All three backends detected and routed live. Latency tracking real. Cloud fallback (OpenRouter/Mistral) auto-engages on local backend timeout.

---

## Quick Start

```bash
git clone https://github.com/ITFactorTech/sovereign-core
cd sovereign-core
cp .env.example .env
# Edit .env with your local model endpoints

# Start the gateway
python gateway/main.py

# Run a KAIROS evolution cycle (manual)
python scripts/nightly_full_evolution.py

# Run the full test suite
pytest tests/ -v
```

**Requirements:**
- Python 3.11+
- Ollama with at least one local model (Qwen2.5, Llama, DeepSeek-Coder)
- 8GB+ VRAM recommended (4GB minimum with smaller models)

---

## Architecture Overview

```
SovereignCore
├── gateway/           — HTTP router + inference + KAIROS endpoints
│   ├── main.py       — FastAPI app + backend health routing
│   ├── inference.py  — /v1/chat/completions with fallback chain
│   ├── kairos_routes.py — KAIROS/SAGE/MemEvolve endpoints
│   └── context.py    — ChromaDB shared cross-agent context
├── KAIROS/           — Self-improvement engine (SAGE loop)
│   ├── group_evolution.py — 7-agent parallel evolution
│   ├── encompass_backtrack.py — EnCompass retry on failure
│   └── federated_node_context.py — Multi-node federation
├── memory_palace/    — Memory systems
│   ├── iron_dome.py  — Memory protection middleware
│   ├── ghost_recall.py ��� Neuroscience-backed retrieval
│   └── memory_palace.py — Spatial encoding
├── omega/            — Core agent & security systems
│   ├── feynman/ — CLARA formal reasoning
│   ├── spawner/ — Dynamic agent scheduling
│   ├── ghost_protocol/ — Kill switch & security
│   └── twin_engine/ — Id/Ego dual processing
└── tests/            — 570 passing tests
```

**Key decisions documented**: See [docs/architecture/README.md](docs/architecture/README.md).

---

## Known Limitations

- **GhostRecall**: Designed and integrated in code, not yet deployed live in inference loop
- **Multi-node federation**: HTTP mesh (ADR #4) is partially implemented; full Kubernetes orchestration is future work
- **Memory protection**: Full end-to-end test coverage not yet complete; gradual hardening path planned
- **Pricing tiers**: Not applicable to open-source AGPL-3.0 licensed code. Commercial deployments are separate.

---

## Next Steps (Q3 2026)

**Phase A: Stabilization** (2 weeks)
- [ ] Merge remaining GH05T3-Sovereign integration PRs
- [ ] Full integration test suite across all 5 repos
- [ ] Document which memory gates are actively enforced vs. designed-only

**Phase B: Memory Hardening** (3 weeks)
- [ ] Deploy Iron Dome full end-to-end test
- [ ] Real tamper detection + audit logging
- [ ] Production-ready documentation

**Phase C: Multi-Repo Syncing** (4 weeks)
- [ ] Unified CI/CD across sovereign-core + GH05T3 + GH05T3-Sovereign
- [ ] Shared ledger implementation
- [ ] Real fitness signals from aetherflux-zero

See **[ROADMAP.md](ROADMAP.md)** for full details.

---

## Community & Contributing

Contributions welcome. Please:

1. Read [docs/architecture/](docs/architecture/README.md) first — understand what was decided and why
2. Run `pytest tests/ -v` locally before pushing
3. Include an ADR if your change is structural (new layer, new routing rule, etc.)

For commercial deployments or SLA-backed support, see [SovereignNation](https://sovereignnation.ai) (separate entity).

---

## Built By

**Robert "Terry" Lee Jr.**  
Self-taught systems architect. Fabrication worker by day. Sovereign AI builder by night.

[GitHub](https://github.com/leerobber) | [SovereignNation](https://sovereignnation.ai)

---

## License

**AGPL-3.0** — see [LICENSE](LICENSE)

---

*"The infrastructure for AI that thinks, learns, and evolves — on your terms."*
