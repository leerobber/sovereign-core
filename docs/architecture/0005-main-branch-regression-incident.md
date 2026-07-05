# 0005: `main` branch regression incident — found and fixed

**Status:** Resolved (merge commit `781cdc1`)

## What happened

A prior merge on this repo's `main`, `f469f1e` ("Merge remote-tracking
branch origin/main; resolve conflicts accepting upstream gateway
changes"), resolved a conflict by accepting "upstream" wholesale. That
silently:

- Reverted `contentaios/kernel.py`, `contentaios/sensory.py`, and
  `gateway/context.py` to older versions, while `contentaios/__init__.py`
  kept importing names (`FileAuditSink`) that only existed in the newer
  version — breaking test collection outright.
- Deleted a real 54-line feature (`af224e2`'s `/silos` registration +
  ARSO-orchestrator-over-HTTP wiring in `kairos_routes.py`) that had been
  added in the commit immediately before it.
- Left several tests/config values stale relative to the code they were
  meant to check: `gateway/main.py`'s `/metrics` returns real Prometheus
  plain-text, but `tests/test_main.py` asserted `.json()` on it; `/health`
  returns `"healthy"`/`"degraded"`, but a test asserted `== "ok"`; the
  gateway port drift described in [0003](0003-gateway-port-standardization.md).

This meant `origin/main` was quietly broken while a separate local clone
(`/home/leer4/sovereign-core`, which had never pulled the bad merge)
carried an uncommitted diff that happened to fix most of it — reviewed and
committed for real as `1e2dee5` (see [0001](0001-shared-context-layer.md)
and [0002](0002-kairos-compat-and-diffusion-retirement.md)).

## How it was actually confirmed (not assumed)

`git worktree add` a scratch checkout of `origin/main`, ran its own test
suite as-is: **523 passed, 104 failed, 1 collection error.** One of the
104: `test_gateway_port_is_8080`, failing because `main` still had `9000`
— direct proof the fix was correct, not a style preference.

Every file `1e2dee5` and `origin/main` both touched was diffed with line
endings normalized first (the working tree had CRLF, `origin`'s blobs are
LF, which made raw diffs 100% noise) to check whether the two sides were a
real conflict or one was a strict superset of the other. Result: `1e2dee5`'s
side was a backward-compatible superset in every case — same base,
additive changes only — confirmed by reading the actual diffs, not by
assuming size differences meant a real disagreement.

## Fix

A real two-parent merge commit (`781cdc1`), not a force-push or a
silent overwrite: `git merge --no-commit --no-ff 1e2dee5` onto a fresh
`main` tracking `origin/main`, 4 real conflicts resolved by taking the
verified-correct side (`.gitignore`, `gateway/kairos_routes.py`,
`gateway/main.py`, `scripts/sovereign.py`), the rest merged cleanly via
3-way merge. Full suite re-run against the actual merged tree afterward:
**570 passed, 76 skipped, 0 failed.**

## Consequences / lesson

"Accept upstream" during a merge conflict is not automatically safe — it
can silently regress the side being discarded even when that side was
correct. Before resolving a conflict that way again, diff the two sides
directly (with line endings normalized) rather than trusting whichever
side happens to be "upstream" at that moment.
