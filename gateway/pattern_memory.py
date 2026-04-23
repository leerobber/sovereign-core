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
        # all user-supplied values flow through parameter binding (the "?" placeholders)
        # so no SQL injection is possible.
        conditions = []
        params: list[Any] = []

        if model_id is not None:
            conditions.append("model_id = ?")
            params.append(model_id)
        if backend_id is not None:
            conditions.append("backend_id = ?")
            params.append(backend_id)
        if pattern_type is not None:
            conditions.append("pattern_type = ?")
            params.append(pattern_type)

        where_clause = " AND ".join(conditions) if conditions else "1"
        query = f"""
            SELECT * FROM patterns
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        cursor = self._conn.execute(query, params)
        rows = cursor.fetchall()

        results: list[PatternRecord] = []
        for row in rows:
            rec = self._row_to_pattern(row)
            results.append(rec)
            # Increment lookup counter
            self._conn.execute(
                "UPDATE patterns SET lookup_count = lookup_count + 1 WHERE pattern_id = ?",
                (rec.pattern_id,),
            )
        self._conn.commit()

        logger.debug(
            "PatternStore: lookup returned %d patterns (model=%s, backend=%s, type=%s)",
            len(results),
            model_id,
            backend_id,
            pattern_type,
        )
        return results

    def record_outcome(
        self,
        pattern_id: str,
        success: bool,
        latency_s: float = 0.0,
        context: Optional[dict[str, Any]] = None,
    ) -> LookupOutcome:
        """Record whether an optimization based on *pattern_id* succeeded.

        Increments ``success_count`` for the referenced pattern if *success*
        is ``True``. Always inserts a new :class:`LookupOutcome` row.

        Args:
            pattern_id: The pattern whose outcome is being reported.
            success:    Whether the optimization succeeded.
            latency_s:  Optional latency measurement (e.g. time to apply fix).
            context:    Additional metadata about the outcome.

        Returns:
            The created :class:`LookupOutcome` object.

        Raises:
            ValueError: If *pattern_id* does not reference an existing pattern.
        """
        # Verify the pattern exists
        cursor = self._conn.execute(
            "SELECT 1 FROM patterns WHERE pattern_id = ?", (pattern_id,)
        )
        if cursor.fetchone() is None:
            raise ValueError(f"No pattern found with pattern_id={pattern_id!r}")

        outcome = LookupOutcome(
            pattern_id=pattern_id,
            success=success,
            latency_s=latency_s,
            context=context or {},
        )

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO lookup_outcomes
                    (outcome_id, pattern_id, timestamp, success, latency_s, context_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome.outcome_id,
                    outcome.pattern_id,
                    outcome.timestamp,
                    int(outcome.success),
                    outcome.latency_s,
                    json.dumps(outcome.context),
                ),
            )
            if success:
                self._conn.execute(
                    "UPDATE patterns SET success_count = success_count + 1 WHERE pattern_id = ?",
                    (pattern_id,),
                )

        logger.debug(
            "PatternStore: recorded outcome %s for pattern %s (success=%s)",
            outcome.outcome_id,
            pattern_id,
            success,
        )
        return outcome

    def get_stats(self) -> PatternStats:
        """Compute and return aggregated statistics about the store."""
        cursor = self._conn.execute(
            """
            SELECT
                COUNT(*) AS total_patterns,
                SUM(lookup_count) AS total_lookups,
                SUM(success_count) AS total_successes
            FROM patterns
            """
        )
        row = cursor.fetchone()
        total_patterns = row["total_patterns"] or 0
        total_lookups = row["total_lookups"] or 0
        total_successes = row["total_successes"] or 0

        overall_hit_rate = 0.0
        if total_lookups > 0:
            overall_hit_rate = total_successes / total_lookups

        # Breakdown by pattern_type
        cursor = self._conn.execute(
            """
            SELECT pattern_type, COUNT(*) AS count
            FROM patterns
            GROUP BY pattern_type
            ORDER BY count DESC
            """
        )
        patterns_by_type = {row["pattern_type"]: row["count"] for row in cursor.fetchall()}

        # Top patterns by success rate (min 5 lookups to qualify)
        cursor = self._conn.execute(
            """
            SELECT * FROM patterns
            WHERE lookup_count >= 5
            ORDER BY (1.0 * success_count / lookup_count) DESC, lookup_count DESC
            LIMIT 5
            """
        )
        top_patterns = [self._row_to_pattern(r).to_dict() for r in cursor.fetchall()]

        return PatternStats(
            total_patterns=total_patterns,
            total_lookups=total_lookups,
            total_successes=total_successes,
            overall_hit_rate=overall_hit_rate,
            patterns_by_type=patterns_by_type,
            top_patterns=top_patterns,
        )

    def clear(self) -> None:
        """Delete all patterns and outcomes from the store."""
        with self._conn:
            self._conn.execute("DELETE FROM lookup_outcomes")
            self._conn.execute("DELETE FROM patterns")
        logger.info("PatternStore: cleared all patterns and outcomes")

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
        logger.info("PatternStore: connection closed")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_pattern(self, row: sqlite3.Row) -> PatternRecord:
        """Convert a SQLite row into a :class:`PatternRecord`."""
        return PatternRecord(
            pattern_id=row["pattern_id"],
            created_at=row["created_at"],
            model_id=row["model_id"],
            backend_id=row["backend_id"],
            pattern_type=row["pattern_type"],
            context=json.loads(row["context_json"]),
            recommendation=json.loads(row["recommendation_json"]),
            lookup_count=row["lookup_count"],
            success_count=row["success_count"],
        )
