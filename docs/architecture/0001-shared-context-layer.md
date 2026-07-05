# 0001: ChromaDB-backed shared cross-agent context layer

**Status:** Accepted, live in `gateway/context.py`

## Context

Multiple models/agents run across different backends in this mesh
(Qwen2.5 on the RTX 5050 as generator, DeepSeek-Coder as verifier, the
Radeon 780M / Ryzen 7 CPU router, an adversarial-debate safety checker).
Each needs visibility into what the others concluded before producing its
own output, without requiring an external embedding-model server.

## Decision

`SharedContextLayer`: one ChromaDB collection (`sovereign_context`) shared
by all agent roles (`AgentRole`: generator/verifier/safety/reasoner/planner).
Every entry carries `role`, `backend_id`, `trace_id`, `timestamp` metadata.
Embeddings are derived deterministically from the document text via SHA-256
(repeated digest interpreted as IEEE-754 floats, clamped to `[-1, 1]`,
NaN/Inf handled) — this avoids needing an external embedding model at
runtime while still supporting similarity queries.

Key read path: `read_cross_gpu(requesting_backend_id)` — a backend calls
this to see what its *peers* concluded, excluding its own entries. Exposed
at `/context/{write,read,cross-gpu/{backend_id},count,clear}`, mounted in
`gateway/main.py`.

## Consequences

- This code existed in the repo before this decision was recorded (as an
  unwired module with no HTTP router) — see
  [0005](0005-main-branch-regression-incident.md) for how that state was
  found and fixed. The router itself is what this decision adds.
- `get_context_layer()`/`init_context_layer()` singleton pattern: lazy
  ephemeral (in-memory) client by default, replaced with a persistent one
  by the gateway lifespan. Tests can monkeypatch `gateway.main._context`
  directly (see `_get_layer()`'s support for that pattern).
