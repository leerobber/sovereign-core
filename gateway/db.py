"""
gateway/db.py — Persistent SQLite database layer for Sovereign Core.

Provides a single shared DB connection with:
  - Auto-migration (creates tables if they don't exist)
  - Thread-safe access via connection pool
  - Used by PatternStore, SemanticLedger, and KAIROS archive
  - Survives gateway restarts — all learning is persistent

Usage:
    from gateway.db import get_db, DB_PATH
    conn = get_db()
    conn.execute("SELECT * FROM patterns")
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH = Path(os.environ.get("SOVEREIGN_DB_PATH", "data/sovereign.db"))

# Thread-local storage for per-thread connections
_local = threading.local()
_init_lock = threading.Lock()
_initialized = False


# ── Schema ─────────────────────────────────────────────────────────────────────
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Pattern Memory (MemEvolve / RES-12)
CREATE TABLE IF NOT EXISTS patterns (
    pattern_id      TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    model_id        TEXT NOT NULL,
    backend_id      TEXT NOT NULL,
    pattern_type    TEXT NOT NULL,
    context_json    TEXT NOT NULL DEFAULT '{}',
    recommendation_json TEXT NOT NULL DEFAULT '{}',
    lookup_count    INTEGER NOT NULL DEFAULT 0,
    success_count   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_patterns_model_backend
    ON patterns(model_id, backend_id);

CREATE INDEX IF NOT EXISTS idx_patterns_type
    ON patterns(pattern_type);

-- Pattern lookup outcomes (for MemEvolve meta-evolution)
CREATE TABLE IF NOT EXISTS pattern_outcomes (
    outcome_id      TEXT PRIMARY KEY,
    pattern_id      TEXT NOT NULL REFERENCES patterns(pattern_id) ON DELETE CASCADE,
    timestamp       REAL NOT NULL,
    success         INTEGER NOT NULL,  -- 1=success, 0=failure
    latency_ms      REAL,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_outcomes_pattern
    ON pattern_outcomes(pattern_id);

-- Semantic Ledger (Aegis-Vault)
CREATE TABLE IF NOT EXISTS ledger_entries (
    entry_id        TEXT PRIMARY KEY,
    timestamp       REAL NOT NULL,
    operation       TEXT NOT NULL,
    backend_id      TEXT NOT NULL,
    model_id        TEXT,
    parent_id       TEXT,
    content_hash    TEXT NOT NULL DEFAULT '',
    integrity_tag   TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_ledger_backend
    ON ledger_entries(backend_id);

CREATE INDEX IF NOT EXISTS idx_ledger_timestamp
    ON ledger_entries(timestamp);

CREATE INDEX IF NOT EXISTS idx_ledger_operation
    ON ledger_entries(operation);

-- KAIROS agent archive (evolutionary stepping stones)
CREATE TABLE IF NOT EXISTS kairos_agents (
    agent_id        TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    generation      INTEGER NOT NULL DEFAULT 0,
    tier            TEXT NOT NULL DEFAULT 'standard',
    score           REAL NOT NULL DEFAULT 0.0,
    optimizations_successful INTEGER NOT NULL DEFAULT 0,
    auction_wins    INTEGER NOT NULL DEFAULT 0,
    ancestor_id     TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);

-- KAIROS proposals (each SAGE cycle outcome)
CREATE TABLE IF NOT EXISTS kairos_proposals (
    proposal_id     TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL REFERENCES kairos_agents(agent_id) ON DELETE CASCADE,
    created_at      REAL NOT NULL,
    task            TEXT NOT NULL,
    proposal_text   TEXT NOT NULL,
    score           REAL NOT NULL DEFAULT 0.0,
    verdict         TEXT NOT NULL DEFAULT 'unknown',  -- PASS/PARTIAL/FAIL
    cycle           INTEGER NOT NULL DEFAULT 0,
    retried         INTEGER NOT NULL DEFAULT 0,       -- 1 if EnCompass retried
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_proposals_agent
    ON kairos_proposals(agent_id);

CREATE INDEX IF NOT EXISTS idx_proposals_score
    ON kairos_proposals(score DESC);

-- MemEvolve retrieval strategy evolution history
CREATE TABLE IF NOT EXISTS retrieval_strategies (
    strategy_id     TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    name            TEXT NOT NULL,
    recency_weight  REAL NOT NULL DEFAULT 0.4,
    relevance_weight REAL NOT NULL DEFAULT 0.4,
    frequency_weight REAL NOT NULL DEFAULT 0.2,
    hit_rate        REAL NOT NULL DEFAULT 0.0,
    generation      INTEGER NOT NULL DEFAULT 0,
    active          INTEGER NOT NULL DEFAULT 0   -- 1 = currently active strategy
);

-- System events (health changes, backend status, boot events)
CREATE TABLE IF NOT EXISTS system_events (
    event_id        TEXT PRIMARY KEY,
    timestamp       REAL NOT NULL,
    event_type      TEXT NOT NULL,  -- backend_health, gateway_boot, kairos_cycle, etc.
    severity        TEXT NOT NULL DEFAULT 'info',  -- info/warning/error/critical
    source          TEXT NOT NULL,
    message         TEXT NOT NULL,
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp
    ON system_events(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_events_type
    ON system_events(event_type);
"""


def _initialize_db(conn: sqlite3.Connection) -> None:
    """Run schema migrations on a fresh connection."""
    conn.executescript(SCHEMA)
    conn.commit()
    logger.info("Database initialized at %s", DB_PATH)


def get_db() -> sqlite3.Connection:
    """
    Return a thread-local SQLite connection.
    Creates the database and tables on first use.
    """
    global _initialized

    if not hasattr(_local, "conn") or _local.conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(DB_PATH),
            check_same_thread=False,
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for concurrent reads
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")

        with _init_lock:
            if not _initialized:
                _initialize_db(conn)
                _initialized = True

        _local.conn = conn

    return _local.conn


def close_db() -> None:
    """Close the current thread's connection."""
    if hasattr(_local, "conn") and _local.conn is not None:
        _local.conn.close()
        _local.conn = None


def log_event(
    event_type: str,
    source: str,
    message: str,
    severity: str = "info",
    metadata: Optional[dict] = None,
) -> None:
    """
    Write a system event to the database.
    Non-blocking — errors are logged but never raised.
    """
    import time, uuid, json
    try:
        conn = get_db()
        conn.execute(
            """INSERT INTO system_events
               (event_id, timestamp, event_type, severity, source, message, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                time.time(),
                event_type,
                severity,
                source,
                message,
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.warning("Failed to log system event: %s", exc)
