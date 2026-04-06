# sovereign-core

Central hub for Sovereign Core — autonomous AI agent platform architecture, system-level epics, and cross-subsystem coordination.

---

## Research Tracks

- [RES-11: Google DeepMind SIMA — Cross-Environment Skill Transfer](docs/RES-11-cross-environment-skill-transfer.md) — transferable resource rebalancing skill across VRAM, RAM, CPU, and network bottlenecks.

## Heterogeneous Compute Gateway (KAN-86)

A latency-aware, capability-aware HTTP gateway that orchestrates the tri-GPU inference mesh.

## ContentAIOS Kernel (KAN-87)

Master kernel and sensory interface for autonomous content coordination.

- Priority-scheduled event loop with structured `KernelEvent` payloads.
- Sensory ingestion adapters: `PushSensoryInput` (webhooks/SDK pushes) and `PollingSensoryInput`
  (external pollers).
- Lightweight pub/sub bus for inter-subsystem communication.
- Bounded audit log for ingestion, dispatch, and handler outcomes.

See `docs/KAN-87-contentaios.md` for the architecture and usage example.

### Architecture

| Device | Endpoint | Role |
|---|---|---|
| RTX 5050 | `localhost:8001` | Primary GPU inference (NVIDIA, 8 GiB VRAM) |
| Radeon 780M | `localhost:8002` | Secondary GPU inference (AMD, 4 GiB VRAM) |
| Ryzen 7 CPU | `localhost:8003` | CPU fallback |
| **Gateway** | **`localhost:8000`** | **Request routing & load balancing** |

### Features

- **Latency-aware routing** — Exponential Moving Average (EMA) per backend; lower-latency backends are preferred within each device-type tier.
- **Capability-aware routing** — model size / VRAM requirements are matched to the most suitable device (large models → NVIDIA GPU → AMD GPU → CPU).
- **Health check & failover** — periodic `/health` probes with configurable failure/recovery thresholds (circuit-breaker pattern).
- **Transparent failover** — if the preferred backend is unreachable, the request is automatically retried on the next candidate.
- **Throughput benchmarking** — per-backend request counts, avg/p50/p95/p99 latency, and requests-per-second over a rolling window.
- **VAULT-ready** — model-to-device assignment is hot-reloadable via the `ModelAssigner` API.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Gateway + backend health summary |
| `GET` | `/metrics` | EMA latency readings + benchmark snapshot |
| `GET` | `/benchmark` | Full per-backend throughput report |
| `POST` | `/benchmark/reset` | Clear benchmark counters |
| `*` | `/v1/{path}` | Proxy inference request to best backend |

Query parameters on `/v1/*`:

- `model_id` — model identifier (e.g. `deepseek-coder-33b`, `mistral-7b`) for capability-based routing.
- `vram_gib` — explicit minimum VRAM requirement in GiB.

### Quick Start

```bash
pip install -r requirements.txt
python -m gateway.main          # starts on localhost:8000
```

Or via the installed script:

```bash
pip install -e .
sovereign-gateway
```

### Configuration

All settings are overridable via environment variables prefixed with `GATEWAY_`:

| Variable | Default | Description |
|---|---|---|
| `GATEWAY_HOST` | `0.0.0.0` | Bind host |
| `GATEWAY_PORT` | `8000` | Bind port |
| `GATEWAY_HEALTH_CHECK_INTERVAL` | `5.0` | Seconds between health probes |
| `GATEWAY_BACKEND_TIMEOUT` | `30.0` | Per-request forwarding timeout (s) |
| `GATEWAY_FAILURE_THRESHOLD` | `3` | Failures before marking backend unhealthy |
| `GATEWAY_RECOVERY_THRESHOLD` | `2` | Successes needed to restore a backend |
| `GATEWAY_LATENCY_EMA_ALPHA` | `0.2` | EMA smoothing factor (0–1) |

### Running Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```
