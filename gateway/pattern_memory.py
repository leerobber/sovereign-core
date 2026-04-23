"""MemEvolve — Pattern Memory Store (RES-12).

SQLite-backed store for optimization patterns with instrumented lookup tracking.
Records success/failure outcomes so the MemEvolve engine can evolve retrieval
strategies based on which lookups led to successful optimizations.

Components
──────────
PatternRecord  — Pydantic model representing one stored optimization pattern.
LookupOutcome  — A single recorded success/failure outcome for a pattern lookup.
PatternStats   — Aggregated statistics about the Pattern Memory store.
PatternStore   — SQLite-backed store with filtering, outcome tracking, and stats.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class PatternRecord(BaseModel):
    """An optimization pattern stored in Pattern Memory.

    Patterns capture a successful (or candidate) remediation for a given
    (model_id, backend_id, pattern_type) combination.  Lookup and success
    counters are updated in-place as agents use and report back on patterns.
    """

    pattern_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = Field(default_factory=time.time)
    model_id: str
    backend_id: str
    pattern_type: str  # e.g. "latency", "routing", "allocation", "optimization"
    context: dict[str, Any] = Field(default_factory=dict)
    recommendation: dict[str, Any] = Field(default_factory=dict)
    lookup_count: int = 0
    success_count: int = 0

    @field_validator("model_id")
    @classmethod
    def model_id_must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_id must be a non-empty string")
        return v

    @field_validator("backend_id")
    @classmethod
    def backend_id_must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("backend_id must be a non-empty string")
        return v

    @field_validator("pattern_type")
    @classmethod
    def pattern_type_must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("pattern_type must be a non-empty string")
        return v

    @property
    def success_rate(self) -> float:
        """Fraction of lookups that reported a successful optimization."""
        if self.lookup_count == 0:
            return 0.0
        return self.success_count / self.lookup_count

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation including computed fields."""
        d = self.model_dump()
        d["success_rate"] = self.success_rate
        return d


class LookupOutcome(BaseModel):
    """A recorded success/failure outcome for a single pattern lookup.

    Created by :meth:`PatternStore.record_outcome` after an agent reports
    whether a retrieved pattern led to a successful optimization.
    """

    outcome_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    pattern_id: str
    timestamp: float = Field(default_factory=time.time)
    success: bool
    latency_s: float = 0.0
    context: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class PatternStats(BaseModel):
    """Aggregated statistics about the Pattern Memory store."""

    total_patterns: int
    total_lookups: int
    total_successes: int
    overall_hit_rate: float
    patterns_by_type: dict[str, int]
    top_patterns: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Pattern Store
# ---------------------------------------------------------------------------


class PatternStore:
    """SQLite-backed Pattern Memory store with outcome instrumentation.

    Patterns are keyed by a UUID ``pattern_id`` and indexed for fast retrieval
    on ``(model_id, backend_id, pattern_type)``.  Every call to
    :meth:`lookup` increments ``lookup_count`` for each returned pattern, and
    :meth:`record_outcome` updates ``success_count`` when an optimization
    succeeds.

    Use ``db_path=":memory:"`` (the default) for testing; supply a file path
    for durable persistence.

    Args:
        db_path: Path to the SQLite database file, or ``":memory:"``.
    """
    # empty = use data/sovereign.db persistent store
    def __init__(self, db_path: str = "") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()
        logger.info("PatternStore initialised (db_path=%r)", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create tables and indices if they do not yet exist."""
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id       TEXT PRIMARY KEY,
                    created_at       REAL    NOT NULL,
                    model_id         TEXT    NOT NULL,
                    backend_id       TEXT    NOT NULL,
                    pattern_type     TEXT    NOT NULL,
                    context_json     TEXT    NOT NULL DEFAULT '{}',
                    recommendation_json TEXT NOT NULL DEFAULT '{}',
                    lookup_count     INTEGER NOT NULL DEFAULT 0,
                    success_count    INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_patterns_lookup
                    ON patterns (model_id, backend_id, pattern_type);

                CREATE TABLE IF NOT EXISTS lookup_outcomes (
                    outcome_id   TEXT PRIMARY KEY,
                    pattern_id   TEXT NOT NULL,
                    timestamp    REAL NOT NULL,
                    success      INTEGER NOT NULL,
                    latency_s    REAL NOT NULL DEFAULT 0.0,
                    context_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (pattern_id) REFERENCES patterns (pattern_id)
                );

                CREATE INDEX IF NOT EXISTS idx_outcomes_pattern
                    ON lookup_outcomes (pattern_id);
                """
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, record: PatternRecord) -> PatternRecord:
        """Persist *record* (insert or replace) and return it.

        If a pattern with the same ``pattern_id`` already exists it is fully
        replaced.
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO patterns
                    (pattern_id, created_at, model_id, backend_id, pattern_type,
                     context_json, recommendation_json, lookup_count, success_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.pattern_id,
                    record.created_at,
                    record.model_id,
                    record.backend_id,
                    record.pattern_type,
                    json.dumps(record.context),
                    json.dumps(record.recommendation),
                    record.lookup_count,
                    record.success_count,
                ),
            )
        logger.debug("PatternStore: stored pattern %s", record.pattern_id)
        return record

    def lookup(
        self,
        *,
        model_id: Optional[str] = None,
        backend_id: Optional[str] = None,
        pattern_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[PatternRecord]:
        """Retrieve patterns matching the given filter criteria.

        Each returned pattern has its ``lookup_count`` incremented to track
        retrieval frequency.  Patterns are returned ordered by recency
        (newest first) so callers can re-rank using evolved weights.

        Args:
            model_id:     Filter to a specific model.
            backend_id:   Filter to a specific backend.
            pattern_type: Filter to a specific pattern category.
            limit:        Maximum number of patterns to return.

        Returns:
            A list of matching :class:`PatternRecord` objects.
        """
        # Only hardcoded column-name literals are ever appended to conditions;
        # all user-sup
