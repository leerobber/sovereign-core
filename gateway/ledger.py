"""Aegis-Vault — Decentralized Semantic Ledger.

Provides trust scoring, provenance tracking, and integrity verification
across all Sovereign Core operations.

Components
──────────
LedgerEntry       — Immutable record written to the ledger.
ProvenanceChain   — Ordered chain of entries tracing content origin.
TrustScorer       — Computes a normalised [0, 1] trust score per backend.
IntegrityVerifier — HMAC-SHA256 verification of individual entries and chains.
SemanticLedger    — Façade that wires the above together.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import statistics
import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# HMAC digest algorithm used for all integrity tags
_DIGEST = "sha256"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class LedgerEntry(BaseModel):
    """Immutable record representing one operation in the Sovereign Core.

    Each entry captures the operation type, the backend that handled it,
    an optional reference to a parent entry (enabling provenance chains),
    and a SHA-256 content hash of the payload produced by the operation.
    An ``integrity_tag`` (HMAC-SHA256) may be attached via
    :meth:`IntegrityVerifier.sign`.
    """

    entry_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = Field(default_factory=time.time)
    operation: str
    backend_id: str
    model_id: Optional[str] = None
    parent_id: Optional[str] = None
    content_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    integrity_tag: str = ""

    model_config = {"frozen": True}

    @field_validator("operation")
    @classmethod
    def operation_must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("operation must be a non-empty string")
        return v

    @field_validator("backend_id")
    @classmethod
    def backend_id_must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("backend_id must be a non-empty string")
        return v

    def canonical_bytes(self) -> bytes:
        """Return a stable JSON byte-string used for hashing and signing.

        The ``integrity_tag`` is excluded so the tag can be computed over
        all other fields without circularity.
        """
        payload: dict[str, Any] = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "backend_id": self.backend_id,
            "model_id": self.model_id,
            "parent_id": self.parent_id,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------


class IntegrityVerifier:
    """HMAC-SHA256 integrity verification for ledger entries and chains.

    All methods are intentionally stateless — they operate only on the
    arguments supplied.

    Args:
        secret: Shared secret used for HMAC computation.  Must be kept
            confidential; it is never stored in any ledger entry.
    """

    def __init__(self, secret: str) -> None:
        if not secret:
            raise ValueError("secret must be a non-empty string")
        self._secret = secret.encode()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sign(self, entry: LedgerEntry) -> LedgerEntry:
        """Return a copy of *entry* with its ``integrity_tag`` populated.

        The tag is an HMAC-SHA256 digest of :meth:`LedgerEntry.canonical_bytes`.
        """
        tag = self._compute_tag(entry)
        return entry.model_copy(update={"integrity_tag": tag})

    def verify(self, entry: LedgerEntry) -> bool:
        """Return ``True`` when *entry*'s ``integrity_tag`` is valid."""
        if not entry.integrity_tag:
            return False
        expected = self._compute_tag(entry)
        return hmac.compare_digest(expected, entry.integrity_tag)

    def verify_chain(self, entries: list[LedgerEntry]) -> bool:
        """Verify all entries in a chain and check parent-link consistency.

        Each entry must have a valid ``integrity_tag``.  For every entry
        after the first, its ``parent_id`` must equal the previous entry's
        ``entry_id``.

        Returns:
            ``True`` only when every entry passes verification *and* the
            chain linkage is intact.
        """
        for i, entry in enumerate(entries):
            if not self.verify(entry):
                logger.warning(
                    "Chain verification failed at position %d (entry_id=%s)",
                    i,
                    entry.entry_id,
                )
                return False
            if i > 0 and entry.parent_id != entries[i - 1].entry_id:
                logger.warning(
                    "Chain linkage broken at position %d: parent_id=%s expected %s",
                    i,
                    entry.parent_id,
                    entries[i - 1].entry_id,
                )
                return False
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_tag(self, entry: LedgerEntry) -> str:
        return hmac.new(self._secret, entry.canonical_bytes(), _DIGEST).hexdigest()

    @staticmethod
    def content_hash(data: bytes) -> str:
        """Return a SHA-256 hex digest of *data* for use as ``content_hash``."""
        return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Provenance chain
# ---------------------------------------------------------------------------


class ProvenanceChain:
    """Ordered sequence of ledger entries tracing the origin of a content item.

    The chain is built by appending entries in chronological order.  Each
    entry (except the first) must declare the ``entry_id`` of its predecessor
    as its ``parent_id``.

    Provenance chains are *append-only*; no modification or deletion is
    supported.
    """

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []
        self._index: dict[str, LedgerEntry] = {}

    def append(self, entry: LedgerEntry) -> None:
        """Append *entry* to the chain.

        Raises:
            ValueError: If ``entry.entry_id`` is already in the chain, or if
                the chain is non-empty and ``entry.parent_id`` does not match
                the current head entry's ``entry_id``.
        """
        if entry.entry_id in self._index:
            raise ValueError(f"Duplicate entry_id: {entry.entry_id}")
        if self._entries:
            head_id = self._entries[-1].entry_id
            if entry.parent_id != head_id:
                raise ValueError(
                    f"parent_id mismatch: expected {head_id!r}, "
                    f"got {entry.parent_id!r}"
                )
        self._entries.append(entry)
        self._index[entry.entry_id] = entry
        logger.debug("ProvenanceChain: appended entry %s", entry.entry_id)

    def head(self) -> Optional[LedgerEntry]:
        """Return the most recently appended entry, or ``None`` if empty."""
        return self._entries[-1] if self._entries else None

    def trace(self, entry_id: str) -> list[LedgerEntry]:
        """Return all entries from the chain origin up to and including *entry_id*.

        Returns an empty list if *entry_id* is not in the chain.
        """
        if entry_id not in self._index:
            return []
        target_idx = next(
            (i for i, e in enumerate(self._entries) if e.entry_id == entry_id), None
        )
        if target_idx is None:
            return []
        return list(self._entries[: target_idx + 1])

    def all_entries(self) -> list[LedgerEntry]:
        """Return an ordered copy of all entries in the chain."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Trust scoring
# ---------------------------------------------------------------------------


class TrustScorer:
    """Computes a normalised [0, 1] trust score for each backend.

    The score is a weighted combination of three sub-scores:

    success_rate (weight 0.5)
        Fraction of recorded operations that completed successfully.
    integrity_rate (weight 0.3)
        Fraction of recorded entries whose ``integrity_tag`` was verified
        as valid (requires an :class:`IntegrityVerifier` to be supplied).
        Defaults to 1.0 when no verifier is provided.
    latency_consistency (weight 0.2)
        Inverse of the coefficient of variation of recorded latencies.
        A backend with highly variable latency scores lower.  Clamped to
        [0, 1].

    The combined score is clamped to [0.0, 1.0].
    """

    _W_SUCCESS: float = 0.5
    _W_INTEGRITY: float = 0.3
    _W_CONSISTENCY: float = 0.2

    def __init__(self, verifier: Optional[IntegrityVerifier] = None) -> None:
        self._verifier = verifier
        # Per-backend telemetry
        self._successes: dict[str, int] = {}
        self._failures: dict[str, int] = {}
        self._latencies: dict[str, list[float]] = {}
        self._entries: dict[str, list[LedgerEntry]] = {}

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record(
        self,
        backend_id: str,
        *,
        success: bool,
        latency_s: float,
        entry: Optional[LedgerEntry] = None,
    ) -> None:
        """Record one operation for *backend_id*.

        Args:
            backend_id: The backend that handled the operation.
            success: Whether the operation completed without error.
            latency_s: Observed round-trip latency in seconds.
            entry: Optional :class:`LedgerEntry` to track for integrity scoring.
        """
        if success:
            self._successes[backend_id] = self._successes.get(backend_id, 0) + 1
        else:
            self._failures[backend_id] = self._failures.get(backend_id, 0) + 1
        self._latencies.setdefault(backend_id, []).append(latency_s)
        if entry is not None:
            self._entries.setdefault(backend_id, []).append(entry)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, backend_id: str) -> float:
        """Return the current trust score for *backend_id* in [0.0, 1.0].

        Returns 0.5 (neutral) for backends with no recorded observations.
        """
        successes = self._successes.get(backend_id, 0)
        failures = self._failures.get(backend_id, 0)
        total = successes + failures
        if total == 0:
            return 0.5  # no data → neutral

        success_rate = successes / total
        integrity_rate = self._integrity_rate(backend_id)
        latency_consistency = self._latency_consistency(backend_id)

        raw = (
            self._W_SUCCESS * success_rate
            + self._W_INTEGRITY * integrity_rate
            + self._W_CONSISTENCY * latency_consistency
        )
        return max(0.0, min(1.0, raw))

    def all_scores(self) -> dict[str, float]:
        """Return trust scores for every backend that has been recorded."""
        all_ids = set(self._successes) | set(self._failures)
        return {bid: self.score(bid) for bid in all_ids}

    def report(self) -> list[dict[str, Any]]:
        """Return a JSON-serialisable trust report."""
        return [
            {
                "backend_id": bid,
                "score": self.score(bid),
                "total_operations": (
                    self._successes.get(bid, 0) + self._failures.get(bid, 0)
                ),
                "successes": self._successes.get(bid, 0),
                "failures": self._failures.get(bid, 0),
            }
            for bid in sorted(set(self._successes) | set(self._failures))
        ]

    # ------------------------------------------------------------------
    # Sub-score helpers
    # ------------------------------------------------------------------

    def _integrity_rate(self, backend_id: str) -> float:
        entries = self._entries.get(backend_id)
        if not entries or self._verifier is None:
            return 1.0
        verified = sum(1 for e in entries if self._verifier.verify(e))
        return verified / len(entries)

    def _latency_consistency(self, backend_id: str) -> float:
        latencies = self._latencies.get(backend_id, [])
        if len(latencies) < 2:
            return 1.0  # insufficient data → assume consistent
        mean = statistics.mean(latencies)
        if mean == 0.0:
            return 1.0
        stdev = statistics.stdev(latencies)
        cv = stdev / mean  # coefficient of variation
        # Map CV to a [0, 1] score: CV=0 → 1.0, CV≥2 → 0.0
        return max(0.0, 1.0 - cv / 2.0)


# ---------------------------------------------------------------------------
# Semantic ledger façade
# ---------------------------------------------------------------------------


class SemanticLedger:
    """Decentralised semantic ledger for Sovereign Core operations.

    Combines :class:`ProvenanceChain`, :class:`TrustScorer`, and
    :class:`IntegrityVerifier` into a single, easy-to-use façade.

    Operations recorded here are:

    1. Signed (HMAC-SHA256) and appended to a per-session provenance chain.
    2. Forwarded to the :class:`TrustScorer` to update per-backend scores.
    3. Stored in an ordered audit log for later inspection.

    Args:
        secret: Shared secret for :class:`IntegrityVerifier`.  Required for
            signing and verification.
    """

    def __init__(self, secret: str) -> None:
        self._verifier = IntegrityVerifier(secret)
        self._scorer = TrustScorer(verifier=self._verifier)
        self._chains: dict[str, ProvenanceChain] = {}
        self._audit_log: list[LedgerEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        operation: str,
        backend_id: str,
        session_id: str = "default",
        model_id: Optional[str] = None,
        content: bytes = b"",
        success: bool = True,
        latency_s: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> LedgerEntry:
        """Record one operation and return the resulting signed entry.

        Args:
            operation: Human-readable operation name (e.g. ``"inference"``).
            backend_id: Backend that handled the operation.
            session_id: Session/provenance-chain identifier.  Entries sharing
                the same ``session_id`` form a single provenance chain.
            model_id: Optional model identifier.
            content: Raw response content used to compute ``content_hash``.
            success: Whether the operation completed without error.
            latency_s: Round-trip latency in seconds.
            metadata: Arbitrary key-value metadata to attach.

        Returns:
            The signed :class:`LedgerEntry` that was appended to the ledger.
        """
        chain = self._chains.setdefault(session_id, ProvenanceChain())
        head = chain.head()
        entry = LedgerEntry(
            operation=operation,
            backend_id=backend_id,
            model_id=model_id,
            parent_id=head.entry_id if head is not None else None,
            content_hash=IntegrityVerifier.content_hash(content),
            metadata=metadata or {},
        )
        signed = self._verifier.sign(entry)
        chain.append(signed)
        self._audit_log.append(signed)
        self._scorer.record(
            backend_id, success=success, latency_s=latency_s, entry=signed
        )
        logger.info(
            "Ledger: recorded operation=%r backend=%s session=%s entry=%s",
            operation,
            backend_id,
            session_id,
            signed.entry_id,
        )
        return signed

    def verify_session(self, session_id: str) -> bool:
        """Verify the integrity of all entries in *session_id*'s chain.

        Returns:
            ``True`` if the chain is intact and all entries are valid.
            ``False`` if the session does not exist or verification fails.
        """
        chain = self._chains.get(session_id)
        if chain is None:
            return False
        return self._verifier.verify_chain(chain.all_entries())

    def trust_score(self, backend_id: str) -> float:
        """Return the current trust score for *backend_id* in [0.0, 1.0]."""
        return self._scorer.score(backend_id)

    def trust_report(self) -> list[dict[str, Any]]:
        """Return a JSON-serialisable trust report for all observed backends."""
        return self._scorer.report()

    def provenance(self, session_id: str) -> list[LedgerEntry]:
        """Return the full provenance chain for *session_id*."""
        chain = self._chains.get(session_id)
        return chain.all_entries() if chain is not None else []

    def audit_log(self) -> list[LedgerEntry]:
        """Return an ordered copy of all entries ever recorded."""
        return list(self._audit_log)
