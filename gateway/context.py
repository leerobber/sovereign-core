"""Shared ChromaDB context layer — persistent cross-agent memory.

Each model/agent in the Sovereign Core cognitive mesh reads from and writes
to this single shared collection so that:

- Qwen2.5 (generator, RTX 5050) sees what DeepSeek-Coder (verifier) concluded.
- The Radeon 780M sees what the RTX 5050 produced and vice versa.
- The Adversarial Debate safety checker can read all prior context before ruling.
- The Llama-3.2 reasoner/router can consult prior verification results.

Design
------
One ChromaDB collection (``sovereign_context``) is shared by all agent roles.
Every entry carries standard metadata fields:

  - ``role``       : AgentRole value (generator, verifier, safety, reasoner, planner)
  - ``backend_id`` : Backend that produced this entry (rtx5050, radeon780m, ryzen7cpu)
  - ``trace_id``   : Optional request-correlation identifier
  - ``timestamp``  : ISO-8601 wall-clock time of the write

Embeddings
----------
A deterministic hash-based embedding is derived from the document text so that
no external embedding model server is required at runtime.  Metadata-filtered
lookups (``read_by_role``, ``read_cross_gpu``) use ChromaDB ``get()`` with
``where`` clauses; similarity-ranked ``query()`` calls are also supported by
providing a query embedding derived the same way.
"""

from __future__ import annotations

import hashlib
import logging
import struct
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Sequence, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import GetResult, Where

logger = logging.getLogger(__name__)

# Embedding dimensionality used for hash-based vectors
_EMBED_DIM = 64

# ChromaDB collection name shared across all agent roles
_COLLECTION_NAME = "sovereign_context"


# ---------------------------------------------------------------------------
# Agent role taxonomy
# ---------------------------------------------------------------------------
class AgentRole(str, Enum):
    """Cognitive roles assigned to agents in the Sovereign Core mesh.

    Each role maps to a specific model and compute backend:

    - GENERATOR  → Qwen2.5 (RTX 5050, primary GPU)
    - VERIFIER   → DeepSeek-Coder (RTX 5050 or Radeon 780M)
    - SAFETY     → Adversarial Debate (any backend)
    - REASONER   → Llama-3.2 (Radeon 780M or CPU router)
    - PLANNER    → Planning component (any backend)
    """

    GENERATOR = "generator"
    VERIFIER = "verifier"
    SAFETY = "safety"
    REASONER = "reasoner"
    PLANNER = "planner"


# ---------------------------------------------------------------------------
# Deterministic hash-based embedding
# ---------------------------------------------------------------------------
def _text_to_embedding(text: str) -> list[float]:
    """Derive a deterministic fixed-dim float vector from *text* via SHA-256.

    This approach avoids any external model download while providing a stable,
    reproducible embedding that allows vector-similarity queries within the
    ChromaDB collection.

    The 64-float vector is constructed by interpreting successive 4-byte
    windows of the repeated SHA-256 digest as IEEE 754 single-precision
    floats, then clamping to ``[-1.0, 1.0]`` to handle degenerate bit
    patterns (NaN / Inf).
    """
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
    # Repeat the 32-byte digest to cover _EMBED_DIM * 4 bytes
    needed = _EMBED_DIM * 4
    repeated = (digest * (needed // len(digest) + 1))[:needed]
    floats: list[float] = []
    for i in range(_EMBED_DIM):
        raw: float = struct.unpack_from("f", repeated, i * 4)[0]
        # Clamp and handle NaN (NaN != NaN is True)
        if raw != raw or raw == float("inf") or raw == float("-inf"):
            floats.append(0.0)
        else:
            floats.append(max(-1.0, min(1.0, raw)))
    return floats


# ---------------------------------------------------------------------------
# Context entry data class
# ---------------------------------------------------------------------------
class ContextEntry:
    """A single context record returned from :class:`SharedContextLayer`."""

    __slots__ = (
        "entry_id",
        "role",
        "backend_id",
        "document",
        "metadata",
        "timestamp",
        "trace_id",
    )

    def __init__(
        self,
        entry_id: str,
        role: str,
        backend_id: str,
        document: str,
        metadata: dict[str, Any],
        timestamp: str,
        trace_id: str,
    ) -> None:
        self.entry_id = entry_id
        self.role = role
        self.backend_id = backend_id
        self.document = document
        self.metadata = metadata
        self.timestamp = timestamp
        self.trace_id = trace_id

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "entry_id": self.entry_id,
            "role": self.role,
            "backend_id": self.backend_id,
            "document": self.document,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _rows_to_entries(result: GetResult) -> list[ContextEntry]:
    """Convert a ChromaDB ``get()`` / ``query()`` result to :class:`ContextEntry` list."""
    ids: list[str] = result.get("ids") or []
    docs: list[str] = result.get("documents") or []
    raw_metas = result.get("metadatas") or []
    metas: list[dict[str, Any]] = [dict(m) for m in raw_metas]
    entries: list[ContextEntry] = []
    for entry_id, doc, meta in zip(ids, docs, metas):
        m: dict[str, Any] = meta or {}
        core_keys = {"role", "backend_id", "timestamp", "trace_id"}
        entries.append(
            ContextEntry(
                entry_id=entry_id,
                role=m.get("role", ""),
                backend_id=m.get("backend_id", ""),
                document=doc or "",
                metadata={k: v for k, v in m.items() if k not in core_keys},
                timestamp=m.get("timestamp", ""),
                trace_id=m.get("trace_id", ""),
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Shared context layer
# ---------------------------------------------------------------------------
class SharedContextLayer:
    """ChromaDB-backed shared memory for the Sovereign Core cognitive mesh.

    All models and agents read from and write to a single ChromaDB collection.
    Metadata-based ``where`` filters provide fast retrieval by role or backend
    without requiring semantic query embeddings.

    Parameters
    ----------
    persist_directory:
        File-system path for a persistent ChromaDB store.  When ``None`` an
        in-memory (ephemeral) client is used — suitable for testing and
        single-process deployments.
    collection_name:
        Override the shared collection name (primarily for isolation in tests).
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = _COLLECTION_NAME,
    ) -> None:
        if persist_directory is not None:
            self._client: ClientAPI = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,  # embeddings supplied explicitly via _text_to_embedding
        )
        logger.info(
            "SharedContextLayer ready (collection=%r, persist=%r)",
            collection_name,
            persist_directory,
        )

    # ------------------------------------------------------------------
    # Write protocol
    # ------------------------------------------------------------------
    def write(
        self,
        role: AgentRole,
        backend_id: str,
        document: str,
        *,
        trace_id: Optional[str] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Persist an agent conclusion to the shared context store.

        Parameters
        ----------
        role:
            Cognitive role of the writing agent.
        backend_id:
            Compute backend that produced this result (e.g. ``"rtx5050"``).
        document:
            Free-text conclusion, summary, or structured output from the agent.
        trace_id:
            Optional correlation ID linking this entry to a specific request.
        extra_metadata:
            Additional key/value pairs stored alongside the standard fields.
            Values must be strings, ints, floats, or booleans (ChromaDB
            metadata constraints).

        Returns
        -------
        str
            The unique entry ID assigned to the new context record.
        """
        entry_id = uuid.uuid4().hex
        ts = datetime.now(tz=timezone.utc).isoformat()
        embedding = _text_to_embedding(document)

        metadata: dict[str, Any] = {
            "role": role.value,
            "backend_id": backend_id,
            "timestamp": ts,
            "trace_id": trace_id or "",
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        self._collection.add(
            ids=[entry_id],
            embeddings=cast(list[Sequence[float]], [embedding]),
            documents=[document],
            metadatas=[metadata],
        )
        logger.debug(
            "Context write: role=%s backend=%s trace_id=%s entry_id=%s",
            role.value,
            backend_id,
            trace_id,
            entry_id,
        )
        return entry_id

    # ------------------------------------------------------------------
    # Read protocols
    # ------------------------------------------------------------------
    def read_by_role(self, role: AgentRole, *, limit: int = 20) -> list[ContextEntry]:
        """Return the most recent entries for a specific agent role.

        Parameters
        ----------
        role:
            The role to filter on.
        limit:
            Maximum number of entries to return.
        """
        result = self._collection.get(
            where={"role": role.value},
            limit=limit,
        )
        return _rows_to_entries(result)

    def read_by_backend(self, backend_id: str, *, limit: int = 20) -> list[ContextEntry]:
        """Return recent entries produced by a specific backend.

        Parameters
        ----------
        backend_id:
            The backend identifier to filter on (e.g. ``"radeon780m"``).
        limit:
            Maximum number of entries to return.
        """
        result = self._collection.get(
            where={"backend_id": backend_id},
            limit=limit,
        )
        return _rows_to_entries(result)

    def read_cross_gpu(
        self,
        requesting_backend_id: str,
        *,
        limit: int = 20,
    ) -> list[ContextEntry]:
        """Return context produced by *other* backends — cross-GPU visibility.

        This is the primary cross-GPU call: a backend invokes this method to
        see what its peers concluded before generating its own output.  For
        example, the Radeon 780M reasoner calls this to read what the RTX 5050
        generator/verifier produced.

        Parameters
        ----------
        requesting_backend_id:
            The backend making the request.  Entries from this backend are
            excluded so only peer conclusions are returned.
        limit:
            Maximum number of entries to return.
        """
        where: Where = cast(Where, {"backend_id": {"$ne": requesting_backend_id}})
        result = self._collection.get(
            where=where,
            limit=limit,
        )
        return _rows_to_entries(result)

    def read_by_trace(self, trace_id: str) -> list[ContextEntry]:
        """Return all entries correlated with a single request trace ID.

        Parameters
        ----------
        trace_id:
            The trace/correlation ID to look up.
        """
        result = self._collection.get(
            where={"trace_id": trace_id},
        )
        return _rows_to_entries(result)

    def read_all(self, *, limit: int = 100) -> list[ContextEntry]:
        """Return the most recent entries across all roles and backends.

        Parameters
        ----------
        limit:
            Maximum number of entries to return.
        """
        result = self._collection.get(limit=limit)
        return _rows_to_entries(result)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def clear(self) -> int:
        """Delete all entries from the context store.

        Returns
        -------
        int
            Number of entries removed.
        """
        result = self._collection.get()
        ids: list[str] = result.get("ids") or []
        count = len(ids)
        if ids:
            self._collection.delete(ids=ids)
        logger.info("SharedContextLayer cleared %d entries", count)
        return count

    def count(self) -> int:
        """Return the total number of stored context entries."""
        return self._collection.count()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_context_layer: Optional[SharedContextLayer] = None


def get_context_layer() -> SharedContextLayer:
    """Return the module-level :class:`SharedContextLayer` singleton.

    The singleton is initialised lazily on first access with an ephemeral
    (in-memory) client.  The gateway lifespan replaces it with a properly
    configured instance via :func:`init_context_layer`.
    """
    global _context_layer
    if _context_layer is None:
        _context_layer = SharedContextLayer()
    return _context_layer


def init_context_layer(
    persist_directory: Optional[str] = None,
    collection_name: str = _COLLECTION_NAME,
) -> SharedContextLayer:
    """Initialise (or re-initialise) the module-level singleton.

    Call this from the gateway lifespan to configure persistence and
    collection name before serving any requests.

    Parameters
    ----------
    persist_directory:
        Optional file-system path for persistent storage.  ``None`` → ephemeral.
    collection_name:
        ChromaDB collection name (override primarily for tests).
    """
    global _context_layer
    _context_layer = SharedContextLayer(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    return _context_layer
