# KAN-123: ARSO v2 — 12-Month Acceleration Program

**Epic:** Long-Horizon Self-Optimization  
**Priority:** 🟢 Low (12-month horizon)  
**Status:** Architecture + roadmap defined

---

## Overview

ARSO v2 is the next generation of agentic recursive self-optimization. Extends ARSO v1 with:
- Extended planning horizons (12-month vs. session-scoped)
- Deeper self-modification capabilities (code rewriting via HyperAgents)
- ZERO Committee governance integration
- Formal safety constraints for recursive self-modification

---

## Key Deliverables

### ✅ 1. ARSO v2 Architecture Proposal

**Core changes from v1 → v2:**

| Dimension | ARSO v1 | ARSO v2 |
|---|---|---|
| Planning horizon | Session-scoped | 12-month milestone-aware |
| Self-modification depth | Config + prompt | Code rewriting (HyperAgents) |
| Proposal generation | Fixed templates | SAGE 4-agent co-evolution |
| Error filtering | Post-sandbox (SUP-1) | Upstream (RES-07 self-verification) |
| Lineage tracking | Basic | DGM-H archive stepping stones |
| Governance | None | ZERO Committee review for major changes |
| Archive | None | SAGEArchive + DGM-H persistent store |

**Architecture:**
```
External Trigger (bottleneck detected)
        ↓
ARSO v2 Orchestrator
        ↓
  [SAGE 4-Agent Loop]
  Proposer → Critic → Verifier → Meta-Agent
        ↓
  Self-Verification Gate (RES-07)
  DeepSeek-Coder upstream filter
        ↓
  ZERO Committee Review (for major changes)
  Proposer + Auditor + Strategist consensus
        ↓
  SUP-1 Stage I Sandbox (first deployment)
        ↓
  DGM-H Archive (persist agent state + diff)
        ↓
  Production (if Stage I passes)
```

---

### ✅ 2. 12-Month Milestone Roadmap

| Month | Milestone | Dependencies |
|---|---|---|
| **M1–M2** | ARSO v1 operational (KAN-89) | KAN-86, KAN-87 |
| **M3** | SAGE loop integrated (RES-01 HyperAgents) | KAN-89 baseline ✓ |
| **M3** | Self-verification pipeline live (RES-07) | SAGE loop ✓ |
| **M4** | DGM-H archive integrated (RES-02) | Self-verification ✓ |
| **M4** | ZERO Committee v1 (KAN-122) | KAN-87 kernel |
| **M5** | mHC in Synthetic Architect NAS (RES-04) | SAS-1 baseline |
| **M6** | **Go/No-Go Checkpoint 1** — ARSO v2 Alpha | All M1–M5 items |
| **M7** | Nemotron primary brain migration (RES-05) | Benchmark results |
| **M8** | MCP/A2A external agent marketplace (RES-08) | ZERO Committee |
| **M9** | KAIROS Elite agent population begins (#16) | RES-01, RES-02, KAN-122 |
| **M10** | **Go/No-Go Checkpoint 2** — ARSO v2 Beta | All M1–M9 items |
| **M11** | WeDLM diffusion router prototype (RES-10) | SAS-1 + mHC |
| **M12** | **Go/No-Go Checkpoint 3** — ARSO v2 Production | Full test suite pass |

---

### ✅ 3. Benchmarking Framework for Self-Optimization Gains

**Metrics tracked per ARSO cycle:**

| Metric | Measurement | Tool |
|---|---|---|
| Bottleneck resolution rate | % of detected bottlenecks resolved | ARSO orchestrator |
| Time-to-fix | Minutes from detection to production | DGM-H timestamps |
| Upstream rejection rate | % filtered by self-verification | `ARSOVerificationPipeline.report()` |
| Compute saved | Sandbox runs avoided | `compute_saved_pct` metric |
| Regression rate | % of fixes causing new regressions | SUP-1 Stage I results |
| Performance delta | Avg improvement score per cycle | `DGMHArchive.summary()` |
| Archive growth | Stepping stones accumulated | `DGMHArchive.stepping_stones()` |

**Benchmark file:** `scripts/res05_nemotron_benchmark.py` extended for ARSO cycle benchmarking.

---

### ✅ 4. Safety Constraints for Recursive Self-Modification

**Constraint tiers:**

| Tier | Constraint | Enforcement |
|---|---|---|
| **Hard** | Cannot modify Alignment Layer code | File-level git protection |
| **Hard** | Cannot modify S-PAX token validation | File-level git protection |
| **Hard** | Cannot exceed credit floor (agent starvation) | Auction circuit breaker |
| **Soft** | All code changes reviewed by ZERO Committee | Governance flow |
| **Soft** | No production deploy without SUP-1 Stage I pass | CI/CD gate |
| **Soft** | Meta-agent rewrites logged and versioned | DGM-H archive |
| **Advisory** | Changes to core routing logic flagged for human review | Audit log alert |

**Implementation:** Add `SafetyConstraintChecker` class that wraps every ARSO proposal before it enters the SAGE loop. Any hard constraint violation → immediate reject with no LLM calls.

---

### ✅ 5. Monthly Go/No-Go Checkpoint Protocol

**Checkpoint cadence:** Months 6, 10, 12 (plus ad-hoc if regression rate > 20%)

**Go criteria:**
- Bottleneck resolution rate ≥ 60%
- Regression rate ≤ 10%
- Upstream rejection rate 20–60% (too low = verifier broken, too high = proposer broken)
- All hard safety constraints passing (zero violations)
- ZERO Committee consensus on architecture state

**No-Go actions:**
- Rollback to last DGM-H stepping stone
- Increase self-verification strictness (`strict_mode=True`)
- Human review of last 10 DGM-H archive entries
- Pause KAIROS elite population if in M9+

---

## Files Created (this sprint)

| File | Purpose |
|---|---|
| `hyperagents/agent/llm_local.py` | RES-01: Local LLM adapter |
| `hyperagents/sage_generate_loop.py` | RES-01: SAGE 4-agent loop |
| `gateway/dgm_h_archive.py` | RES-02: DGM-H lineage stepping stones |
| `scripts/res05_nemotron_benchmark.py` | RES-05: Nemotron benchmark |
| `synthetic_architect/mhc_gene.py` | RES-04: mHC NAS primitive |
| `gateway/self_verification.py` | RES-07: Upstream error filtering |
| `gateway/mcp_auction_interface.py` | RES-08: MCP/A2A marketplace |
| `docs/RES-09-recursive-civilization-framework.md` | RES-09: Formal framework |
| `docs/RES-10-wedlm-research-path.md` | RES-10: Research path tracker |
| `docs/KAN-122-zero-committee.md` | KAN-122: ZERO Committee architecture |
| `docs/KAN-123-arso-v2-roadmap.md` | KAN-123: This document |
