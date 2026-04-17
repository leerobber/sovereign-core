# Sovereign Core

> *One person. One machine. A system that evolves itself every night.*

[![CI](https://github.com/leerobber/sovereign-core/actions/workflows/ci.yml/badge.svg)](https://github.com/leerobber/sovereign-core/actions/workflows/ci.yml)
[![KAIROS](https://img.shields.io/badge/KAIROS-evolving_nightly-7c3aed?style=flat-square)](https://github.com/leerobber/sovereign-core)
[![Local GPU](https://img.shields.io/badge/RTX_5050_+_Radeon_780M-local_cluster-76b900?style=flat-square)](https://github.com/leerobber/sovereign-core)
[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)

---

## What This Is

Sovereign Core is the central hub of a **fully local, self-improving AI platform** — built by one self-taught developer on a home GPU cluster, with zero cloud dependency.

It routes AI inference across a heterogeneous compute cluster (RTX 5050 + Radeon 780M + Ryzen 7 CPU), runs a nightly self-improvement loop that evolves its own architecture, and coordinates a multi-agent system across 5 interconnected repositories.

**This is not a demo. This is not a prototype. This runs nightly on real hardware.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              SOVEREIGN CORE GATEWAY  :8000                  │
│         Heterogeneous Compute Router (FastAPI)              │
├─────────────────┬───────────────────┬───────────────────────┤
│   RTX 5050      │   Radeon 780M     │     Ryzen 7 CPU       │
│   8GB VRAM      │   4GB VRAM        │     System RAM        │
│   Qwen2.5       │   DeepSeek-Coder  │     Llama/Mistral     │
│   :8001         │   :8002           │     :8003             │
│   Primary       │   Verifier        │     Fallback          │
└─────────────────┴───────────────────┴───────────────────────┘
```

**Routing logic:** RTX 5050 first → Radeon 780M if unavailable → Ryzen CPU as last resort. The system never cold-fails. Failover is silent and automatic.

---

## KAIROS — The Self-Improvement Engine

**KAIROS** = Knowledge-Augmented Iterative Recursive Optimization System

Every night at 3:00 AM, KAIROS runs 10 live ARSO cycles on real GPU hardware:

```
Proposer ──► generates improvement proposals (Qwen2.5 / RTX 5050)
    │
Critic ───► reviews each proposal: APPROVE / REJECT / REVISE
    │
Verifier ─► validates logic and correctness (DeepSeek-Coder / Radeon)
    │
Meta-Agent ► every 3 cycles, rewrites the rules the others operate under
    │
Archive ──► stepping stones stored. score ≥ 0.85 → Elite status
```

**The compounding effect:** The Meta-Agent improves *how improvement happens* — not just what gets improved. Each generation learns faster than the last.

- 10 cycles/night × 365 nights = **3,650+ live improvement cycles per year**
- Every stepping stone builds on the last
- Plateau detection triggers automatic rule rewrite

---

## The SAGE Loop (4-Agent Co-Evolution)

| Agent | Role | Hardware |
|-------|------|----------|
| **Proposer** | Generates improvement proposals | RTX 5050 / Qwen2.5 |
| **Critic** | Reviews and scores each proposal | Separate model (anti-capture) |
| **Verifier** | Validates logic and correctness | Radeon 780M / DeepSeek-Coder |
| **Meta-Agent** | Rewrites the operating rules every 3 cycles | RTX 5050 |

> **Critic Capture prevention:** The Proposer and Critic are always separate models on separate hardware. A system that reviews its own work will always approve it.

---

## Connected Repositories

Sovereign Core is the hub. These are the nodes:

| Repo | Role | Connection |
|------|------|-----------|
| [DGM](https://github.com/leerobber/DGM) | Darwin Gödel Machine — self-improving coding agent | Stepping stones feed KAIROS archive |
| [HyperAgents](https://github.com/leerobber/HyperAgents) | Self-referential agent swarm | Routes inference through gateway |
| [Honcho](https://github.com/leerobber/Honcho) | Mission control dashboard | WebSocket live connection to gateway |
| [contentai-pro](https://github.com/leerobber/contentai-pro) | Multi-agent content engine | Priority-0 sovereign provider |
| [Termux-Intelligent-Assistant](https://github.com/leerobber/Termux-Intelligent-Assistant) | Android mobile agent | Connects via WiFi, works offline |

---

## Quickstart

```bash
# 1. Start AI model servers on each GPU
bash scripts/ollama_start_all.sh

# 2. Pull the inference models
bash scripts/pull_models.sh

# 3. Start the gateway router
bash scripts/start_gateway.sh

# 4. Verify all endpoints healthy
bash scripts/smoke_test.sh
```

**Endpoints:**

| Service | Port | Purpose |
|---------|------|---------|
| Gateway | 8000 | Main router — all inference requests |
| RTX 5050 | 8001 | Primary inference (Qwen2.5) |
| Radeon 780M | 8002 | Verifier inference (DeepSeek-Coder) |
| Ryzen CPU | 8003 | Fallback inference (Llama/Mistral) |
| WebSocket | 8000/ws | Real-time event bus for Honcho |

---

## OpenAI Compatibility

The gateway exposes a `/v1/chat/completions` endpoint fully compatible with the OpenAI API spec. Any tool built for ChatGPT works with sovereign-core out of the box — no modification needed.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sovereign"  # any string — local auth
)

response = client.chat.completions.create(
    model="qwen2.5",
    messages=[{"role": "user", "content": "What ran in KAIROS last night?"}]
)
```

---

## Environment

```bash
cp .env.example .env
```

```env
SOVEREIGN_GATEWAY_URL=http://localhost:8000
RTX_ENDPOINT=http://localhost:8001
RADEON_ENDPOINT=http://localhost:8002
CPU_ENDPOINT=http://localhost:8003
KAIROS_CYCLES=10
KAIROS_ELITE_THRESHOLD=0.85
KAIROS_ARCHIVE_THRESHOLD=0.70
```

---

## Philosophy

Most AI infrastructure depends on cloud APIs, rate limits, and someone else's servers. 

This doesn't.

> *"Sovereign infrastructure means the compute is yours, the data is yours, the evolution is yours. No API keys. No monthly bills. No terms of service that change without warning."*

Built on two ideas:
- **Darwin** — keep the good solutions, build on them, improve generation by generation
- **Gödel** — a system powerful enough can reason about itself and rewrite itself to be better

The goal isn't the smartest system today. It's the one that gets smarter fastest over time.

---

## Built By

**Terry Lee** — Douglasville, GA  
Self-taught systems architect. Fabrication worker by day. AI infrastructure builder by night.  
No institution. No team. Just architecture.

*Self-taught. Self-funded. Self-improving — just like the systems I build.*

---

<div align="center">

**[Profile](https://github.com/leerobber)** · **[DGM](https://github.com/leerobber/DGM)** · **[HyperAgents](https://github.com/leerobber/HyperAgents)** · **[Honcho](https://github.com/leerobber/Honcho)** · **[contentai-pro](https://github.com/leerobber/contentai-pro)**

</div>
