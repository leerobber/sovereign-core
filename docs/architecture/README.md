# Architecture Decisions

Lightweight ADRs (Architecture Decision Record: Status / Context /
Decision / Consequences) for real, tested decisions made in this repo.

This folder doesn't replace the existing feature/research docs elsewhere
in `docs/` (`KAIROS-architecture.md`, `isa_spec.md`, the `RES-*`/`KAN-*`
docs, etc.) — those are specs and research reports for specific
initiatives. This folder is the durable, permanent decision log: what was
decided, why, and what broke and got fixed along the way. Written from
real code and real test runs, not aspiration.

| # | Decision | Status |
|---|---|---|
| [0001](0001-shared-context-layer.md) | ChromaDB-backed shared cross-agent context layer | Accepted, live |
| [0002](0002-kairos-compat-and-diffusion-retirement.md) | KAIROS test-compat endpoints; retire the diffusion_router prototype | Accepted, live |
| [0003](0003-gateway-port-standardization.md) | Gateway port standardized to 8080 | Accepted, live |
| [0004](0004-http-mesh-federation.md) | HTTP mesh federation with GH05T3 / GH05T3-Sovereign | Accepted, live, partial |
| [0005](0005-main-branch-regression-incident.md) | `main` branch regression incident (104 failing tests) — found and fixed | Resolved |
| [0006](0006-repo-hygiene.md) | Git/data hygiene cleanup | Resolved |

## How to add a new one

Copy the format of any existing entry, number it sequentially, and add a
row to the table above. Prefer documenting a real decision after it's been
implemented and tested over speculating about one in advance.
