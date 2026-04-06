# RES-05: Nvidia Nemotron-3-Nano as Primary Brain Candidate

## Overview

`nemotron-3-nano` is designated the **Primary Brain** for the Sovereign Core agent economy. Despite being a nano-scale model (classified as `ModelSize.SMALL` in the gateway model registry), it is always routed to the **RTX 5050** (NVIDIA GPU, 8 GiB VRAM) to maximise inference throughput and reliability.

## Motivation

Nano-scale models are typically mapped to CPU backends in the heterogeneous compute gateway, since their small VRAM footprint makes them CPU-capable. However, the Primary Brain must sustain low-latency, high-throughput inference to drive the agent economy reliably. Routing it to the most capable hardware (RTX 5050) ensures:

- **Minimum response latency** — GPU-accelerated token generation vs. CPU.
- **Reliability** — the RTX 5050 is the highest-weight backend with proven throughput.
- **Consistency** — the agent economy always uses the same hardware path for its core reasoning loop.

## Implementation

### Model Registry (`gateway/models.py`)

`nemotron-3-nano` is added to `_MODEL_REGISTRY` as `ModelSize.SMALL`:

```python
_MODEL_REGISTRY: dict[str, ModelSize] = {
    ...
    "nemotron-3-nano": ModelSize.SMALL,  # Primary Brain candidate (RES-05)
    ...
}
```

The `SMALL` classification is accurate from a VRAM perspective; the special routing is handled separately.

### Primary Brain Override (`gateway/models.py`)

`ModelAssigner.assign` checks whether the requested model id starts with the Primary Brain prefix (`nemotron-3-nano`) and, if so, promotes `NVIDIA_GPU` to the head of the device preference list before applying any explicit `device_hint`:

```python
_PRIMARY_BRAIN_MODEL = "nemotron-3-nano"

# In ModelAssigner.assign:
if model_id and model_id.lower().startswith(_PRIMARY_BRAIN_MODEL):
    device_preference = [DeviceType.NVIDIA_GPU] + [
        d for d in device_preference if d != DeviceType.NVIDIA_GPU
    ]
```

This ensures that any `nemotron-3-nano` variant (e.g. `nemotron-3-nano-instruct`) also benefits from the override.

### Routing Priority

| Request model id | Size bucket | First device |
|---|---|---|
| `tinyllama-1b` | SMALL | CPU |
| `nemotron-3-nano` | SMALL | **NVIDIA_GPU** (Primary Brain override) |
| `nemotron-3-nano-instruct` | SMALL | **NVIDIA_GPU** (Primary Brain override) |
| `deepseek-coder-33b` | LARGE | NVIDIA_GPU |

An explicit `device_hint` parameter in `ModelAssigner.assign` can still override the Primary Brain preference if required for testing or maintenance purposes.

## Testing

New tests in `tests/test_models.py` under `TestNemotronPrimaryBrain` cover:

- Registry lookup returns `ModelSize.SMALL`.
- Case-insensitive and suffix-variant lookups.
- `ModelAssigner.assign` always places `NVIDIA_GPU` first for nemotron-3-nano.
- All backends are included in the fallback list (no duplicates).
- An explicit `device_hint` can still override the Primary Brain preference.
- nemotron-3-nano routing differs from other SMALL models (CPU-first).
