"""SQLite Conversation & Metrics Store"""

import asyncio
import json
import logging
import time
from typing import Optional

import aiosqlite

logger = logging.getLogger("sovereign.memory")

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    latency_ms REAL NOT NULL,
    routed_by TEXT,
    complexity TEXT,
    is_fallback INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_metrics_model ON metrics(model);
CREATE INDEX IF NOT EXISTS idx_metrics_time ON metrics(created_at);
"""


class MemoryStore:
    """Async SQLite store for conversations and request metrics."""

    def __init__(self, db_path: str = "data/sovereign.db"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Create database and tables."""
        import os
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        logger.info(f"Memory store initialized: {self.db_path}")

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def save_message(
        self, session_id: str, role: str, content: str, model: str = ""
    ) -> int:
        assert self._db is not None
        cursor = await self._db.execute(
            "INSERT INTO conversations (session_id, role, content, model, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, model, time.time()),
        )
        await self._db.commit()
        return cursor.lastrowid or 0

    async def get_conversation(self, session_id: str, limit: int = 50) -> list[dict]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT role, content, model, created_at FROM conversations WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {"role": r[0], "content": r[1], "model": r[2], "created_at": r[3]}
            for r in reversed(rows)
        ]

    async def record_metric(
        self,
        model: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        routed_by: str = "",
        complexity: str = "",
        is_fallback: bool = False,
    ) -> None:
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO metrics (model, prompt_tokens, completion_tokens, latency_ms, routed_by, complexity, is_fallback, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (model, prompt_tokens, completion_tokens, latency_ms, routed_by, complexity, int(is_fallback), time.time()),
        )
        await self._db.commit()

    async def get_metrics_summary(self) -> dict:
        """Get aggregate metrics."""
        assert self._db is not None
        cursor = await self._db.execute(
            """SELECT model,
                      COUNT(*) as total,
                      AVG(latency_ms) as avg_latency,
                      MIN(latency_ms) as min_latency,
                      MAX(latency_ms) as max_latency,
                      SUM(prompt_tokens) as total_prompt_tokens,
                      SUM(completion_tokens) as total_completion_tokens
               FROM metrics GROUP BY model"""
        )
        rows = await cursor.fetchall()
        return {
            row[0]: {
                "total_requests": row[1],
                "avg_latency_ms": round(row[2], 2),
                "min_latency_ms": round(row[3], 2),
                "max_latency_ms": round(row[4], 2),
                "total_prompt_tokens": row[5] or 0,
                "total_completion_tokens": row[6] or 0,
            }
            for row in rows
        }

    async def get_percentile_latencies(self, model: str) -> dict:
        """Get p50/p95/p99 latencies for a model."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT latency_ms FROM metrics WHERE model = ? ORDER BY latency_ms",
            (model,),
        )
        rows = await cursor.fetchall()
        if not rows:
            return {"p50": 0, "p95": 0, "p99": 0}

        latencies = [r[0] for r in rows]
        n = len(latencies)

        def percentile(p: float) -> float:
            idx = int(n * p / 100)
            return round(latencies[min(idx, n - 1)], 2)

        return {"p50": percentile(50), "p95": percentile(95), "p99": percentile(99)}
