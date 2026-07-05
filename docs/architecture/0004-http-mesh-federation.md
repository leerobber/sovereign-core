# 0004: HTTP mesh federation with GH05T3 / GH05T3-Sovereign

**Status:** Accepted, live (partial — not every ecosystem repo is registered yet)

## Context

This repo's own README describes it as "the hub that everything connects
to," and its `main` branch already treats several other repos as health-checkable
"silos" (`gateway/main.py`'s `/silos` registration of GH05T3/Jarvis/MYTHOS/
openclaw/verelene_v5/economy — added in `af224e2`, though that specific
54-line addition was later lost in the regression described in
[0005](0005-main-branch-regression-incident.md) and hasn't been restored).
GH05T3 itself is a real, separately-deployed process with its own dual
runtime story (WSL vs. Windows-native).

## Decision

**Federate over HTTP, don't import code across repos.** `src/orchestration/registry.py`
+ `config/registry.yaml`: a `RepoRegistry` that loads ecosystem entries
from YAML and probes each one's `/health` (or a dual-runtime-aware URL for
`gh05t3` specifically — `_gh05t3_gateway_url()` finds the WSL default
gateway IP when `GH05T3_RUNTIME=windows`). Exposed at `GET /v1/repos` on
this gateway.

Registered entries as of this writing:

| Entry | What | Probe |
|---|---|---|
| `sovereign_core` | this repo's own gateway | `:8000/health` |
| `gh05t3` | GH05T3's `gateway_v3.py`, dual-runtime aware | `:8002/health` (WSL) or Windows-default-gateway IP |
| `gh05t3_sovereign` | GH05T3-Sovereign's `gateway_v3.py` + `/oss/genome/*` | `:8002/health` — **same default port as `gh05t3`**, set `GATEWAY_PORT` on one of them if running both |
| `agent_economy` | Agent credit economy | `:8081/health` |
| `local_ai_mesh` | Multi-model inference mesh | `:8011/health` |

This mirrors a change already made independently elsewhere in this repo's
own history: `nightly_full_evolution.py` was switched to drive its mesh
via HTTP instead of importing local KAIROS modules directly — "federate,
not import" predates this specific registry addition.

## Consequences

- Live-verified: `RepoRegistry.probe_all()` was run against a real running
  GH05T3-Sovereign `gateway_v3.py` on port 8002 — returned `healthy=True`.
  The same run also demonstrated the real port-collision risk in the table
  above: with only GH05T3-Sovereign's gateway up, the `gh05t3` entry's
  probe *also* reported healthy, because both targets currently resolve to
  the same port.
- Not yet registered here: `DGM`, `contentai-pro`, `Termux-Intelligent-Assistant`,
  `Honcho`, `sovereign-core-rs`/`sovereign-gpu` — real repos in the same
  ecosystem, not yet wired into this registry.
- The real `/silos` + ARSO-orchestrator-over-HTTP feature from `af224e2`
  is currently gone from `main` (see [0005](0005-main-branch-regression-incident.md))
  and would need to be deliberately re-added, not assumed still present.
