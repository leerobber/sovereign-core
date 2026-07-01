"""S-ISA — Sovereign Instruction Set Architecture opcodes."""
from enum import IntEnum


class Opcode(IntEnum):
    # ── Control ──────────────────────────────────────────────────
    SPAWN_AGENT   = 1
    KILL_AGENT    = 2
    ROUTE_MESSAGE = 3
    BROADCAST     = 4
    SYNC          = 5

    # ── Model ────────────────────────────────────────────────────
    RUN_MODEL     = 10
    EVAL_MODEL    = 11
    LOAD_MODEL    = 12
    UNLOAD_MODEL  = 13

    # ── Memory ───────────────────────────────────────────────────
    WRITE_MEMORY  = 20
    READ_MEMORY   = 21
    SUMMARIZE_MEMORY = 22
    SEARCH_MEMORY = 23

    # ── Reasoning ────────────────────────────────────────────────
    PLAN          = 30
    REFLECT       = 31
    CRITIQUE      = 32
    VOTE          = 33
    EMBED         = 34
    SEARCH        = 35
    LINK_CONTEXT  = 36

    # ── Graph (DGM) ──────────────────────────────────────────────
    BUILD_GRAPH   = 40
    QUERY_GRAPH   = 41
    UPDATE_GRAPH  = 42

    # ── Workflow ─────────────────────────────────────────────────
    RUN_WORKFLOW  = 50
    CHECKPOINT    = 51
    EMIT_RESULT   = 52
