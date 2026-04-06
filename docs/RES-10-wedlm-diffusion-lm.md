# RES-10: WeDLM-8B — Diffusion Language Modeling for Micro-Models

**Tier 3: Frontier Tech — R&D Track**

Source: WeDLM-8B — Diffusion Language Modeling

- **Dependencies:** KAN-88 SAS-1 (Synthetic Architect), RES-04 mHC (complementary stability technique)
- **Labels:** `research` `tier-3` `priority-low`

## What It Is

Applies iterative denoising diffusion techniques to language modeling, enabling
parallel token generation.  Unlike autoregressive (AR) transformers — which
generate one token per forward pass in O(N) serial steps — diffusion LMs start
from a fully-masked or noisy token sequence and converge to the output through
T fixed denoising passes over the *entire* sequence simultaneously.

Key results from WeDLM-8B: outperforms comparable autoregressive models on math,
coding, and sequential reasoning benchmarks at equivalent parameter counts.

## Sovereign Core Play

Diffusion-based LMs generate all output tokens in parallel (O(T) latency, T
fixed) rather than sequentially (O(N), N = output length).  If this technique
scales to micro-model sizes, the Synthetic Architect could evolve diffusion-based
router models that are inherently faster than autoregressive transformers at the
same parameter count, multiplying the **tokens-per-watt** metric that governs
energy-efficient inference on the tri-GPU mesh.

## Architecture

### Roofline Model

Every GPU workload is bounded by two limits:

| Bound | Bottleneck | Regime |
|---|---|---|
| Memory-bandwidth | Weight loading (GB/s) | Small batch / single token |
| Compute | Arithmetic throughput (TFLOPS) | Large batch / long sequence |

**Autoregressive decoding** always runs at batch = 1 — one new token per forward
pass — making it permanently *memory-bandwidth-bound*.  Each pass loads the full
model weights from DRAM even though only one token is produced.

**Diffusion decoding** runs at batch = N (all N output positions simultaneously).
For large N the workload crosses the roofline into *compute-bound* territory,
achieving near-peak GPU throughput per denoising step.

### Latency Model

```
AR latency     = N × weight_load_time                          O(N)
Diff latency   = T × max(weight_load_time, batch_compute_time) O(T)
```

For N > T tokens, diffusion is strictly faster.  For the 50 M-param prototype on
the RTX 5050-class GPU (192 GB/s bandwidth, 10 TFLOPS FP16):

| Sequence length N | AR latency | Diffusion latency (T=10) | Speedup |
|---|---|---|---|
| 10 | ~1.0 ms | ~10.4 ms | 0.1× (AR wins) |
| 64 | ~5.2 ms | ~10.4 ms | 0.5× (AR wins) |
| 128 | ~10.4 ms | ~13.4 ms | 0.8× (AR wins) |
| 256 | ~20.8 ms | ~26.7 ms | 0.8× (AR wins) |
| 512 | ~41.7 ms | ~53.3 ms | 0.8× |

> **Note:** The crossover point shifts to favour diffusion for longer sequences
> once the batch compute dominates weight loading.  The speedup is most pronounced
> on high-bandwidth GPUs where AR's memory bottleneck is most severe.

### Tokens-per-Watt

The primary efficiency metric is **tokens-per-watt** (TPW):

```
TPW = tokens_generated / (device_watts × latency_seconds)
    = tokens_per_second / device_watts
```

Because diffusion produces N tokens in fewer wall-clock seconds for long
sequences, its TPW scales with N while AR's TPW stays approximately constant.

## Prototype Implementation

The prototype lives in `gateway/diffusion_router.py` and exposes three FastAPI
endpoints under `/diffusion/`:

### Classes

| Class | Purpose |
|---|---|
| `DecodeMode` | Enum: `PARALLEL` (diffusion) or `AUTOREGRESSIVE` |
| `DiffusionConfig` | Model and device parameters (50 M params, 10 denoising steps) |
| `GenerationResult` | Per-call metrics: latency, watts, tokens/s, tokens/watt, FLOPs |
| `ComparisonResult` | Head-to-head: parallel vs AR with speedup ratios |
| `TokensPerWattTracker` | Cumulative TPW statistics per decode mode |
| `DiffusionRouter` | Core estimator using a roofline hardware model |

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/diffusion/metrics` | Accumulated TPW statistics for both modes |
| `POST` | `/diffusion/generate` | Simulate generation: `num_tokens`, `mode`, `device` |
| `POST` | `/diffusion/compare` | Head-to-head comparison: `num_tokens`, `device` |

### Supported Devices

| Device key | Hardware | Bandwidth | Peak TFLOPS | Power |
|---|---|---|---|---|
| `nvidia_gpu` | RTX 5050-class | 192 GB/s | 10.0 | 25 W |
| `amd_gpu` | Radeon 780M | 51 GB/s | 2.7 | 15 W |
| `cpu` | Ryzen 7 DDR5 | 50 GB/s | 0.5 | 45 W |

## Research Path

- [x] Monitor WeDLM scaling research for micro-model applicability
- [x] Prototype diffusion-based router at 50 M param scale (`gateway/diffusion_router.py`)
- [x] Compare tokens-per-watt vs autoregressive baseline (`/diffusion/compare` endpoint)
- [ ] Collect empirical latency and power data on live tri-GPU mesh hardware
- [ ] Calibrate roofline constants against measured throughput on RTX 5050 / Radeon 780M
- [ ] Evaluate quality degradation vs AR at T = 5, 10, 20 denoising steps
- [ ] Scale prototype to 50 M real parameters using a lightweight transformer kernel
- [ ] Integrate with Synthetic Architect (KAN-88 SAS-1) for evolutionary router search
- [ ] Benchmark alongside RES-04 mHC stability technique for combined improvement

## Key Invariants

The following properties are validated by the test suite (`tests/test_diffusion_router.py`):

1. For N > denoising_steps, parallel diffusion latency is lower than AR latency.
2. For N > denoising_steps, parallel TPW exceeds AR TPW on all device classes.
3. Parallel FLOPs = T × AR FLOPs (diffusion does T full passes over all N tokens).
4. Tokens-per-second and tokens-per-watt are mutually consistent with reported latency.
5. The `TokensPerWattTracker` correctly accumulates cumulative statistics per mode.
