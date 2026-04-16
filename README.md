# Sovereign Core

> Autonomous AI agent platform — heterogeneous compute gateway, self-evolving KAIROS agent economy, and Aegis-Vault semantic ledger.

[![CI](https://github.com/leerobber/sovereign-core/actions/workflows/ci.yml/badge.svg)](https://github.com/leerobber/sovereign-core/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │         Sovereign Core Gateway           │
                        │              localhost:8000              │
                        │                                          │
    ┌──────────┐        │  ┌──────────┐  ┌──────────┐  ┌───────┐ │
    │  Honcho  │───────▶│  │  Router  │  │  KAIROS  │  │Auction│ │
    │(React UI)│        │  │(EMA lat.)│  │(SAGE loop│  │(VQ)   │ │
    └──────────┘        │  └────┬─────┘  └──────────┘  └───────┘ │
                        │       │                                  │
    ┌──────────┐        │  ┌────▼──────────────────────────────┐  │
    │contentai │───────▶│  │   Backend Mesh (Latency-Aware)    │  │
    │   -pro   │        │  ├───────────┬───────────┬───────────┤  │
    └──────────┘        │  │ RTX 5050  │Radeon 780M│ Ryzen 7   │  │
                        │  │:8001 GPU  │:8002 GPU  │:8003 CPU  │  │
    ┌──────────┐        │  │ 8GB VRAM  │ 4GB VRAM  │ Fallback  │  │
    │  Termux  │───────▶│  └───────────┴───────────┴───────────┘  │
    │Assistant │        │                                          │
    └──────────┘        │  ┌──────────┐  ┌──────────┐  ┌───────┐ │
                        │  │MemEvolve │  │Aegis-Vault│  │Pattern│ │
                        │  │(RES-12)  │  │  Ledger   │  │Memory │ │
                        └──┴──────────┴──┴──────────┴──┴───────┘─┘
```

### Components

| Component | File | Description |
|-----------|------|-------------|
| **Gateway Router** | `gateway/router.py` | Latency-EMA + capability-aware backend selection |
| **Health Monitor** | `gateway/health.py` | Async health probes with circuit-breaker |
| **KAIROS** | `gateway/kairos.py` | nextElites agent economy — self-evolving ARSO cycles |
| **SAGE Loop** | `hyperagents/sage_generate_loop.py` | 4-agent co-evolution: Proposer→Critic→Verifier→Meta |
| **DGM-H Archive** | `gateway/dgm_h_archive.py` | Lineage archive for ancestor reconstruction |
| **Auction** | `gateway/auction.py` | Vickrey-Quadratic resource allocation |
| **Ledger** | `gateway/ledger.py` | HMAC-SHA256 Aegis-Vault semantic ledger |
| **MemEvolve** | `gateway/mem_evolve.py` | Meta-evolution of memory retrieval strategies |
| **Self-Verification** | `gateway/self_verification.py` | DeepSeek-Coder proposal verification |
| **ContentAIOS** | `contentaios/kernel.py` | Priority-scheduled content coordination kernel |
| **mHC Gene** | `synthetic_architect/mhc_gene.py` | Manifold-constrained hyper-connections (NAS) |

---

## Quick Start

### Option 1 — Docker Compose (recommended)

```bash
# Clone
git clone https://github.com/leerobber/sovereign-core
cd sovereign-core

# Configure
cp .env.example .env
# Edit .env — set GATEWAY_API_KEY if desired

# Start everything (gateway + 3 Ollama backends + ChromaDB + Prometheus + Grafana)
docker compose up -d

# Pull a model on the NVIDIA backend
docker exec sovereign-nvidia ollama pull qwen2.5:14b
docker exec sovereign-amd ollama pull deepseek-coder:6.7b
docker exec sovereign-cpu ollama pull llama3.2:3b

# Verify
curl http://localhost:8000/health
```

### Option 2 — Local (bare metal, recommended for GPU passthrough)

```bash
# Prerequisites: Python 3.11+, Ollama running on :8001/:8002/:8003

git clone https://github.com/leerobber/sovereign-core
cd sovereign-core
pip install -r requirements.txt

# Start gateway
python -m uvicorn gateway.main:app --host 0.0.0.0 --port 8000 --reload

# Verify
python scripts/sovereign.py status
```

---

## CLI

```bash
# System status
python scripts/sovereign.py status

# GPU backend details
python scripts/sovereign.py backends

# Run KAIROS evolution cycles
python scripts/sovereign.py evolve --cycles 10

# Tail Aegis-Vault ledger
python scripts/sovereign.py ledger tail --n 25

# Benchmark a model
python scripts/sovereign.py bench --model qwen2.5:14b

# Override gateway URL
python scripts/sovereign.py --gateway http://192.168.1.100:8000 status
```

---

## API Reference

### Core

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe — returns healthy/degraded |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Interactive Swagger UI |

### Inference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/inference` | Route text generation across GPU mesh |

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:14b",
    "prompt": "Explain KAIROS in one paragraph.",
    "options": {"num_predict": 256, "temperature": 0.7}
  }'
```

### KAIROS Agent Economy

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/kairos/sage` | Route task through SAGE 4-agent loop |
| `POST` | `/kairos/evolve` | Run N ARSO evolution cycles |
| `GET` | `/kairos/leaderboard` | Top agents by score |
| `GET` | `/status/kairos/{agent_id}` | Agent details + lineage |

### Observability

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/status/` | Full system snapshot |
| `GET` | `/status/backends` | Per-backend health + latency |
| `GET` | `/status/stream` | SSE real-time health stream |
| `WS` | `/ws/events` | WebSocket event bus |

### Auction & Ledger

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/auction/bid` | Submit Vickrey-Quadratic bid |
| `GET` | `/ledger/tail` | Last N Aegis-Vault entries |
| `POST` | `/benchmark/run` | Throughput benchmark |

---

## WebSocket Events

Connect to `ws://localhost:8000/ws/events` to receive real-time events:

```js
const ws = new WebSocket('ws://localhost:8000/ws/events');
ws.onmessage = ({ data }) => {
  const { type, ts, data: payload } = JSON.parse(data);
  // type: backend.health_changed | auction.completed | kairos.cycle_complete
  //       kairos.elite_promoted | ledger.entry_written | inference.completed
};
ws.send('ping');  // → { type: "pong", ts: 1234567890 }
```

---

## Research Tracks

| ID | Title | Status |
|----|-------|--------|
| RES-05 | Nemotron-3-Nano as primary brain (RTX 5050) | 🟡 In progress |
| RES-10 | WEDLM diffusion language model | 🟡 In progress |
| RES-11 | SIMA cross-environment skill transfer | 🟡 In progress |
| RES-12 | MemEvolve meta-evolution of retrieval | ✅ Implemented |

---

## Development

```bash
# Install dev deps
pip install -r requirements.txt pytest pytest-asyncio pytest-cov httpx ruff mypy

# Lint
ruff check . && ruff format --check .

# Tests
pytest tests/ --tb=short -v --cov=gateway

# Type check
mypy gateway/ --ignore-missing-imports
```

---

## Integration

Sovereign Core is designed to be the backbone of your entire AI stack:

- **[Honcho](https://github.com/leerobber/Honcho)** — React frontend, uses `useSovereignCore` hook + `SovereignPanel`
- **[contentai-pro](https://github.com/leerobber/contentai-pro)** — routes inference through `llm_sovereign.py` adapter
- **[Termux-Intelligent-Assistant](https://github.com/leerobber/Termux-Intelligent-Assistant)** — stdlib-only `sovereign_client.py`, zero deps

Set `SOVEREIGN_GATEWAY_URL=http://<your-machine>:8000` in each project to connect.

---

## License

MIT — see [LICENSE](LICENSE)
