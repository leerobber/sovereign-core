# 0006: Git/data hygiene cleanup

**Status:** Resolved

## Context

Two local clones of this repo existed: the canonical one
(`/home/leer4/sovereign-core`, WSL-native filesystem) and a stale one
(`/mnt/c/Users/leer4/sovereign-core`, Windows-mounted NTFS). The Windows
one was several commits behind and showed 83 "modified" files — checked
and confirmed to be pure CRLF line-ending noise (every line of e.g.
`gateway/main.py` rewritten with no real content change), not real
divergent work.

## Decision

- **`.gitignore`** gained real, previously-missing entries: `llama.cpp/`
  (a *nested git clone* of the llama.cpp project living untracked inside
  this repo's working tree — has its own `.git`, must never become plain
  tracked files here), `sovereign-env/` (an alternate venv, distinct from
  the already-ignored `.venv`), `data/` (contains `sovereign.db`, runtime
  state), `memory_palace/belief_hierarchy.json` (regenerated identity
  state — was tracked in an earlier commit, `999f2d1`, then apparently
  de-tracked later; the ignore rule documents that it should stay that
  way), `.claude/settings.local.json` (personal tool-permission config,
  not shared config), and stray `gateway.pid`/`gateway_test.log.err`/
  `ghost_protocol/ghost_sessions.json` runtime files.
- Two genuinely useful, real scripts existed **only** in the stale
  Windows-side clone and nowhere else: `scripts/kill_gateway_port.bat`
  (kills a stuck python/uvicorn process on a given port) and
  `scripts/remove_wsl_portproxy.bat` (clears a stale WSL→Windows port
  proxy). Ported into the canonical clone before the stale clone was
  removed — real, small, and directly relevant to the exact port-collision
  problem in [0004](0004-http-mesh-federation.md).
- `scripts/start_mesh_windows.bat` (new): starts GH05T3's backend/gateway
  on Windows if not already responding on `:8001`/`:8002`, matching the
  same health-check-before-start pattern the ops scripts above use.
- The stale Windows-side clone was deleted after the above (confirmed no
  unique content remained) — reclaimed 437MB.

## Consequences

None — this is pure cleanup. The two ported scripts and the new one are
tracked now, so this class of "only exists in one physical clone"
problem doesn't recur for these three files specifically.
