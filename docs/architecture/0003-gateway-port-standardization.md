# 0003: Gateway port standardized to 8080

**Status:** Accepted, live

## Context

`gateway/zero_committee.py` and `hyperagents/agent/llm_local.py` both
pointed at port 9000 with a comment attributing it to
`launch_gateway.py`. That file doesn't exist anywhere in this repo.
Meanwhile `.env.example` documents `GATEWAY_PORT=8080` and
`SOVEREIGN_GATEWAY_URL=http://localhost:8080` as the real, current
convention.

## Decision

Corrected both to `http://localhost:8080`, matching `.env.example`.

Confirmed via a real test, not just the doc: `tests/test_sovereign_stack.py::TestLLMLocalAdapter::test_gateway_port_is_8080`
— a test with that exact name — was failing against the pre-fix `9000`
value. This wasn't a style preference; the codebase's own test suite
already expected 8080.

Separately, `scripts/sovereign.py`'s CLI had a real bug in the same area:
`GATEWAY = args.gateway` ran unconditionally, so every invocation without
an explicit `--gateway` flag overwrote the module-level `GATEWAY` default
with `None`. Fixed to `if getattr(args, "gateway", None): GATEWAY =
args.gateway`.

## Consequences

None beyond the fix itself — this closes a real, currently-failing test
rather than opening a new tradeoff.
