# RES-11: Google DeepMind SIMA - Cross-Environment Skill Transfer

Source: Google DeepMind SIMA (600+ skills across 9 commercial game engines). Target: teach Sovereign agents to generalize optimization behaviors across resource bottlenecks so a VRAM-fix skill transfers to RAM/CPU/network pressure without per-environment retraining.

- **Dependencies:** KAN-89 ARSO Orchestrator, RES-01 HyperAgents
- **Skill abstraction:** resource rebalancing (shed, compress, reroute, cache, prefetch)

## Objective

Create a transferable "resource rebalancing" skill: agents detect a bottleneck, choose a shared primitive, and apply it to the substrate (VRAM, RAM, CPU, network) with minimal fine-tuning.

## Bottleneck Taxonomy & Shared Primitives

| Bottleneck | Signals | Shared primitives | Actuators |
|---|---|---|---|
| VRAM saturation | alloc failure rate, GPU mem %, batch eviction counts | downscale/quantize, shard to CPU, stream weights, cache hot tensors | model loader, GPU memory manager, offload scheduler |
| RAM pressure | OOM kills, swap %, page faults | spill to disk, compress buffers, streaming IO | OS-level cgroups, memory manager hooks |
| CPU bound | run-queue depth, latency spikes | coalesce kernels, move to GPU, trim orchestration overhead | scheduler, placement engine |
| Network limited | queue depth, p95 latency, retransmits | prefetch/cache, batch requests, reroute topology | gateway/router, client batching |

## Training Approach

1. **Environment matrix:** use ARSO orchestrator to simulate/induce VRAM, RAM, CPU, and network pressure profiles; expose signals + actuation hooks.
2. **Skill policy:** single policy head over shared primitives with substrate-specific adapters; emphasize latency/throughput reward + penalty for instability.
3. **Curriculum:** start with VRAM-only tasks, then introduce RAM/CPU/network perturbations; mix scripted heuristics as behavior cloning seeds.
4. **Transfer tests:** hold out one bottleneck during training, evaluate zero-shot and few-shot adaptation on the holdout.
5. **Instrumentation:** log reward curves, intervention types, and stability (error rate, rollback count) per environment.

## Evaluation Plan

- **Baselines:** static heuristics and per-bottleneck specialist agents.
- **Metrics:** recovery time to healthy state, throughput delta (%), tail latency delta (p95/p99), intervention cost (retries/rollbacks), success rate.
- **Transfer protocols:** train on VRAM, test on RAM/CPU/network; rotate holdouts; add noisy/novel traces to measure robustness.

## Work Plan (to-do)

- [ ] Finalize bottleneck taxonomy + primitive/action mapping with ARSO hooks.
- [ ] Wire ARSO telemetry for VRAM/RAM/CPU/network traces and replay harness.
- [ ] Build curricula (VRAM-first, mixed, and holdout transfer splits).
- [ ] Train HyperAgents on resource-rebalancing skill; capture policy checkpoints and logs.
- [ ] Run zero-shot transfer evals vs baselines; quantify metrics above.
- [ ] Apply few-shot adapter tuning for failed transfer cases.
- [ ] Package deployable policy + selector for gateway/runtime integration.
- [ ] Publish report + playbook for reuse across bottleneck types.
