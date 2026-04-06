# RES-10: WeDLM-8B — Diffusion Language Modeling for Micro-Models

**Source:** WeDLM-8B — Diffusion Language Modeling  
**Tier:** 3 — Frontier Tech / R&D Track  
**Priority:** Low  
**Status:** Research Path in progress

---

## What It Is

WeDLM-8B applies diffusion techniques to language modeling, enabling **parallel token decoding** instead of sequential autoregressive generation. Outperforms comparable models on math, coding, and sequential reasoning tasks.

Key properties:
- All tokens generated in parallel (not one by one)
- Iterative refinement over N denoising steps
- Faster than autoregressive at same parameter count for certain task types

---

## Sovereign Core Play

If diffusion-based LM technique scales to micro-model size (50M–500M params), the **Synthetic Architect** could evolve diffusion-based router models that are:
- Inherently faster than autoregressive transformers at the same param count
- More efficient per watt (multiplies tokens-per-watt metric)
- Complementary to mHC (RES-04) for training stability

This is a **multiplier** on the agent economy throughput if it works at micro-scale.

---

## Research Path Progress

### Step 1: Monitor WeDLM scaling research for micro-model applicability
**Status:** In Progress

Key papers and findings to track:
- WeDLM-8B (original) — parallel decoding at 8B scale, strong on coding/math
- MDLM (2024) — masked diffusion LM, shows promise at smaller scales
- Plaid (2023) — continuous diffusion for text, 1B scale feasibility demonstrated
- CDCD (2022) — classifier-free guidance for diffusion LMs

**Scaling concerns at micro-model size:**
- Diffusion models typically need many denoising steps (N=10–100) — adds latency overhead at small scales where each step is fast anyway
- Quality degrades more steeply with parameter reduction vs. autoregressive models
- mHC (RES-04) may help stabilize training of diffusion micro-models

**Watch trigger:** If WeDLM or successor demonstrates competitive quality at ≤500M params with ≤20 denoising steps → proceed to Step 2.

---

### Step 2: Prototype diffusion-based router at 50M param scale
**Status:** Pending Step 1 trigger

Prototype spec:
```python
# DiffusionRouterGene — extends MicroModelGene (synthetic_architect/mhc_gene.py)
# Architecture: masked diffusion transformer
# Params: ~50M
# Denoising steps: 10 (fast inference target)
# Training: denoising score matching loss
# mHC: apply RES-04 primitives for gradient stability during training
```

Implementation notes:
- Base on MDLM architecture (simpler than continuous diffusion, easier to scale down)
- Use `MicroModelGene` from `synthetic_architect/mhc_gene.py` as the gene encoding
- Add `diffusion_steps: int` and `noise_schedule: str` fields to gene
- NAS search space: `diffusion_steps ∈ [5, 10, 20, 50]`, `noise_schedule ∈ ["cosine", "linear"]`

---

### Step 3: Compare tokens-per-watt vs autoregressive baseline
**Status:** Pending Step 2

Benchmark design:
| Metric | Autoregressive (baseline) | Diffusion Router |
|---|---|---|
| Tokens/second | TBD | TBD |
| Watts (GPU power) | TBD | TBD |
| Tokens/watt | TBD | TBD |
| Quality (BLEU/task accuracy) | TBD | TBD |
| Denoising steps needed | N/A | TBD |

Decision threshold: diffusion router must achieve **>1.3x tokens/watt** at **>90% quality** of autoregressive baseline to be worth integrating into Synthetic Architect NAS.

---

## Dependencies

- **KAN-88 SAS-1 (Synthetic Architect)** — gene encoding and NAS infrastructure
- **RES-04 mHC** — gradient stability for small diffusion model training ✓ (completed)

---

## Timeline

| Milestone | Target | Condition |
|---|---|---|
| Step 1 watch trigger | Ongoing | WeDLM ≤500M paper published |
| Step 2 prototype | +4 weeks after trigger | Step 1 trigger met |
| Step 3 benchmark | +2 weeks after prototype | Step 2 complete |
| Integration decision | +1 week after benchmark | Go/No-Go on tokens/watt threshold |
