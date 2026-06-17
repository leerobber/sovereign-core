# Sovereign Core — Local AI System

## Project Overview
Sovereign Core is a fully autonomous, local-first AI inference stack for the Lenovo LOQ 15AHP10.
Zero cloud dependency. Zero API costs. Full sovereignty over your AI infrastructure.

### Hardware Target
- **CPU**: AMD Ryzen 7 8000-series
- **GPU**: NVIDIA GeForce RTX 5050 (8GB GDDR6, Blackwell architecture)
- **iGPU**: AMD Radeon 780M
- **RAM**: 16GB DDR5
- **OS**: Ubuntu 24.04 (WSL2 on Windows 11)

### Tech Stack
- **Model Serving**: Ollama (CPU + GPU)
- **Language**: Python 3.12, async throughout
- **API Framework**: FastAPI + Uvicorn
- **Database**: SQLite (aiosqlite) for logs/metrics, future ChromaDB for RAG
- **Search**: DuckDuckGo (no API key)
- **UI**: Rich terminal dashboard

## Architecture

```
┌─────────────────────────────────────────────┐
│              API Gateway (FastAPI)           │
│              Port 8000                       │
├─────────────────────────────────────────────┤
│           Intelligent Router                │
│    ┌──────────┐  ┌──────────┐              │
│    │ CPU Pool  │  │ GPU Pool  │              │
│    │ Llama-3.2 │  │ Qwen-2.5 │              │
│    │ 3B        │  │ 7B        │              │
│    └──────────┘  └──────────┘              │
├─────────────────────────────────────────────┤
│              Ollama Runtime                  │
│              Port 11434                      │
├─────────────────────────────────────────────┤
│         Memory / Metrics (SQLite)           │
└─────────────────────────────────────────────┘
```

## Directory Structure
```
sovereign-core/
├── CLAUDE.md              # This file — project context
├── config/
│   └── settings.yaml      # Runtime configuration
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── manager.py     # Ollama model lifecycle management
│   ├── routing/
│   │   ├── __init__.py
│   │   └── router.py      # Intelligent request routing
│   ├── api/
│   │   ├── __init__.py
│   │   └── gateway.py     # FastAPI gateway
│   ├── memory/
│   │   ├── __init__.py
│   │   └── store.py       # SQLite conversation/metrics store
│   └── utils/
│       ├── __init__.py
│       └── health.py      # System health monitoring
├── scripts/
│   ├── install_ollama.sh  # Ollama installation
│   ├── install_cuda.sh    # CUDA toolkit installation
│   └── start.sh           # Launch script
├── tests/
│   ├── __init__.py
│   ├── test_models.py     # Model manager tests
│   ├── test_router.py     # Router tests
│   ├── test_gateway.py    # API gateway tests
│   ├── test_memory.py     # Memory store tests
│   └── test_health.py     # Health monitor tests
└── sovereign-env/         # Python virtual environment
```

## KAN Board (Task Tracker)

### KAN-10: Install CUDA Toolkit
- **Priority**: High
- **Status**: TODO
- **Description**: Install NVIDIA CUDA toolkit for RTX 5050 GPU acceleration
- **Acceptance Criteria**:
  - `nvidia-smi` shows RTX 5050 with driver loaded
  - `nvcc --version` returns CUDA 12.x
  - Ollama detects GPU: `ollama run --verbose` shows CUDA backend
- **Steps**:
  1. Install NVIDIA container toolkit keys and repo
  2. Install cuda-toolkit-12 package
  3. Set PATH and LD_LIBRARY_PATH in ~/.bashrc
  4. Verify nvidia-smi output
  5. Restart Ollama to pick up GPU

### KAN-11: Deploy Qwen-2.5-7B on RTX 5050
- **Priority**: High
- **Status**: TODO (depends on KAN-10)
- **Description**: Pull and deploy Qwen-2.5-7B-Instruct via Ollama on GPU
- **Acceptance Criteria**:
  - `ollama list` shows qwen2.5:7b
  - Model loads into GPU VRAM (confirm via nvidia-smi ~5.5GB usage)
  - Response latency < 2s for short prompts
  - `python -c "import requests; print(requests.get('http://localhost:11434/api/tags').json())"` shows model
- **VRAM Budget**: ~5.5GB of 8GB (leaves headroom)

### KAN-16: Deploy Llama-3.2-3B on CPU (Quick Win)
- **Priority**: Critical — DO THIS FIRST
- **Status**: TODO
- **Description**: Install Ollama and pull Llama-3.2-3B for CPU inference
- **Acceptance Criteria**:
  - Ollama service running on port 11434
  - `ollama list` shows llama3.2:3b
  - `curl http://localhost:11434/api/generate -d '{"model":"llama3.2:3b","prompt":"hello","stream":false}'` returns valid JSON response
  - Response completes within 30s on CPU
- **Steps**:
  1. Run install_ollama.sh
  2. Start ollama serve (or verify systemd service)
  3. `ollama pull llama3.2:3b`
  4. Test with curl

### KAN-17: Intelligent Request Router
- **Priority**: Medium
- **Status**: TODO (depends on KAN-11, KAN-16)
- **Description**: Build routing logic that dispatches requests to optimal model
- **Routing Rules**:
  - Short/simple prompts (< 100 tokens) → Llama-3.2-3B (CPU, fast)
  - Complex/long prompts (> 100 tokens) → Qwen-2.5-7B (GPU, powerful)
  - If GPU model busy → fallback to CPU model
  - If both busy → queue with priority
- **Acceptance Criteria**:
  - Router correctly classifies prompts by complexity
  - Fallback works when GPU model unavailable
  - Queue handles concurrent requests without dropping
  - Latency overhead < 50ms for routing decision
  - All test_router.py tests pass

### KAN-18: API Gateway
- **Priority**: Medium
- **Status**: TODO (depends on KAN-17)
- **Description**: FastAPI gateway exposing unified /v1/chat/completions endpoint
- **Endpoints**:
  - `POST /v1/chat/completions` — OpenAI-compatible chat endpoint
  - `GET /v1/models` — List available models
  - `GET /health` — System health + model status
  - `GET /metrics` — Request counts, latencies, GPU utilization
  - `WebSocket /ws/chat` — Streaming chat interface
- **Acceptance Criteria**:
  - OpenAI-compatible request/response format
  - Streaming support via SSE and WebSocket
  - Health endpoint returns all model statuses
  - Metrics track p50/p95/p99 latencies
  - All test_gateway.py tests pass

## Conventions
- All async — use `async def` everywhere, `aiohttp`/`aiosqlite` for I/O
- Type hints on all function signatures
- Pydantic models for all API request/response schemas
- Structured logging with timestamps
- Config from settings.yaml, never hardcoded
- Tests use pytest + pytest-asyncio
