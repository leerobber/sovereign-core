"""Tests for Aegis-Vault — Decentralized Semantic Ledger."""

from __future__ import annotations

import datetime

import pytest
from pydantic import ValidationError

from gateway.ledger import (
    IntegrityVerifier,
    LedgerEntry,
    ProvenanceChain,
    SemanticLedger,
    TrustScorer,
)


# ---------------------------------------------------------------------------
# LedgerEntry tests
# ---------------------------------------------------------------------------


class TestLedgerEntry:
    def _make_entry(self, **kwargs: object) -> LedgerEntry:
        defaults: dict[str, object] = {
            "operation": "inference",
            "backend_id": "rtx5050",
        }
        defaults.update(kwargs)
        return LedgerEntry(**defaults)  # type: ignore[arg-type]

    def test_auto_fields_populated(self) -> None:
        entry = self._make_entry()
        assert entry.entry_id
        assert entry.timestamp > 0

    def test_unique_entry_ids(self) -> None:
        e1 = self._make_entry()
        e2 = self._make_entry()
        assert e1.entry_id != e2.entry_id

    def test_canonical_bytes_excludes_integrity_tag(self) -> None:
        entry = self._make_entry()
        b1 = entry.canonical_bytes()
        signed = entry.model_copy(update={"integrity_tag": "abc123"})
        b2 = signed.canonical_bytes()
        assert b1 == b2

    def test_canonical_bytes_is_deterministic(self) -> None:
        entry = self._make_entry()
        assert entry.canonical_bytes() == entry.canonical_bytes()

    def test_operation_must_not_be_blank(self) -> None:
        with pytest.raises(ValueError):
            self._make_entry(operation="   ")

    def test_backend_id_must_not_be_blank(self) -> None:
        with pytest.raises(ValueError):
            self._make_entry(backend_id="")

    def test_to_dict_contains_key_fields(self) -> None:
        entry = self._make_entry(model_id="deepseek")
        d = entry.to_dict()
        assert d["operation"] == "inference"
        assert d["backend_id"] == "rtx5050"
        assert d["model_id"] == "deepseek"

    def test_frozen_model_rejects_mutation(self) -> None:
        entry = self._make_entry()
        with pytest.raises(ValidationError):
            entry.operation = "changed"  # type: ignore[misc]

    def test_optional_fields_default_to_none_or_empty(self) -> None:
        entry = self._make_entry()
        assert entry.model_id is None
        assert entry.parent_id is None
        assert entry.content_hash == ""
        assert entry.integrity_tag == ""
        assert entry.metadata == {}

    def test_metadata_stored_correctly(self) -> None:
        entry = self._make_entry(metadata={"tokens": 42, "source": "test"})
        assert entry.metadata["tokens"] == 42
        assert entry.metadata["source"] == "test"

    def test_canonical_bytes_with_non_json_serializable_metadata(self) -> None:
        """datetime values in metadata must not raise TypeError (Bug 1)."""
        entry = self._make_entry(
            metadata={"ts": datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)}
        )
        # canonical_bytes() uses model_dump(mode="json") which coerces datetime → str
        result = entry.canonical_bytes()
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_canonical_bytes_metadata_is_deterministic_with_datetime(self) -> None:
        """Same datetime metadata must produce the same canonical bytes."""
        dt = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
        e1 = self._make_entry(metadata={"created": dt})
        e2 = self._make_entry(
            entry_id=e1.entry_id,
            timestamp=e1.timestamp,
            metadata={"created": dt},
        )
        assert e1.canonical_bytes() == e2.canonical_bytes()


# ---------------------------------------------------------------------------
# IntegrityVerifier tests
# ---------------------------------------------------------------------------


class TestIntegrityVerifier:
    _SECRET = "test-secret-key"

    def _make_verifier(self) -> IntegrityVerifier:
        return IntegrityVerifier(self._SECRET)

    def _make_entry(self, **kwargs: object) -> LedgerEntry:
        defaults: dict[str, object] = {
            "operation": "inference",
            "backend_id": "rtx5050",
        }
        defaults.update(kwargs)
        return LedgerEntry(**defaults)  # type: ignore[arg-type]

    def test_empty_secret_raises(self) -> None:
        with pytest.raises(ValueError):
            IntegrityVerifier("")

    def test_sign_sets_integrity_tag(self) -> None:
        verifier = self._make_verifier()
        entry = self._make_entry()
        signed = verifier.sign(entry)
        assert signed.integrity_tag != ""

    def test_sign_does_not_mutate_original(self) -> None:
        verifier = self._make_verifier()
        entry = self._make_entry()
        verifier.sign(entry)
        assert entry.integrity_tag == ""

    def test_verify_signed_entry_returns_true(self) -> None:
        verifier = self._make_verifier()
        signed = verifier.sign(self._make_entry())
        assert verifier.verify(signed)

    def test_verify_unsigned_entry_returns_false(self) -> None:
        verifier = self._make_verifier()
        entry = self._make_entry()
        assert not verifier.verify(entry)

    def test_verify_tampered_field_returns_false(self) -> None:
        verifier = self._make_verifier()
        signed = verifier.sign(self._make_entry())
        # Change a field without re-signing
        tampered = signed.model_copy(update={"operation": "tampered"})
        assert not verifier.verify(tampered)

    def test_wrong_secret_fails_verification(self) -> None:
        verifier1 = self._make_verifier()
        verifier2 = IntegrityVerifier("different-secret")
        signed = verifier1.sign(self._make_entry())
        assert not verifier2.verify(signed)

    def test_content_hash_is_deterministic(self) -> None:
        h1 = IntegrityVerifier.content_hash(b"hello world")
        h2 = IntegrityVerifier.content_hash(b"hello world")
        assert h1 == h2

    def test_content_hash_differs_for_different_inputs(self) -> None:
        h1 = IntegrityVerifier.content_hash(b"hello")
        h2 = IntegrityVerifier.content_hash(b"world")
        assert h1 != h2

    def test_content_hash_empty_bytes(self) -> None:
        h = IntegrityVerifier.content_hash(b"")
        assert len(h) == 64  # SHA-256 hex digest length

    def test_verify_chain_single_entry(self) -> None:
        verifier = self._make_verifier()
        e1 = verifier.sign(self._make_entry())
        assert verifier.verify_chain([e1])

    def test_verify_chain_empty_returns_true(self) -> None:
        verifier = self._make_verifier()
        assert verifier.verify_chain([])

    def test_verify_chain_multi_entry_valid(self) -> None:
        verifier = self._make_verifier()
        e1 = verifier.sign(self._make_entry())
        e2 = verifier.sign(self._make_entry(parent_id=e1.entry_id))
        e3 = verifier.sign(self._make_entry(parent_id=e2.entry_id))
        assert verifier.verify_chain([e1, e2, e3])

    def test_verify_chain_unsigned_entry_fails(self) -> None:
        verifier = self._make_verifier()
        e1 = self._make_entry()  # not signed
        assert not verifier.verify_chain([e1])

    def test_verify_chain_broken_parent_link_fails(self) -> None:
        verifier = self._make_verifier()
        e1 = verifier.sign(self._make_entry())
        e2 = verifier.sign(self._make_entry(parent_id="wrong-id"))
        assert not verifier.verify_chain([e1, e2])

    def test_verify_chain_tampered_middle_entry_fails(self) -> None:
        verifier = self._make_verifier()
        e1 = verifier.sign(self._make_entry())
        e2 = verifier.sign(self._make_entry(parent_id=e1.entry_id))
        e3 = verifier.sign(self._make_entry(parent_id=e2.entry_id))
        # Tamper with the middle entry
        tampered = e2.model_copy(update={"operation": "tampered"})
        assert not verifier.verify_chain([e1, tampered, e3])

    def test_verify_chain_root_with_non_none_parent_id_fails(self) -> None:
        """Root entry (position 0) with parent_id set must fail chain verification (Bug 3)."""
        verifier = self._make_verifier()
        e1 = verifier.sign(self._make_entry(parent_id="some-external-id"))
        assert not verifier.verify_chain([e1])


# ---------------------------------------------------------------------------
# ProvenanceChain tests
# ---------------------------------------------------------------------------


class TestProvenanceChain:
    def _make_entry(self, **kwargs: object) -> LedgerEntry:
        defaults: dict[str, object] = {
            "operation": "inference",
            "backend_id": "rtx5050",
        }
        defaults.update(kwargs)
        return LedgerEntry(**defaults)  # type: ignore[arg-type]

    def test_empty_chain_head_is_none(self) -> None:
        chain = ProvenanceChain()
        assert chain.head() is None

    def test_empty_chain_length_is_zero(self) -> None:
        chain = ProvenanceChain()
        assert len(chain) == 0

    def test_append_single_entry(self) -> None:
        chain = ProvenanceChain()
        entry = self._make_entry()
        chain.append(entry)
        assert chain.head() == entry
        assert len(chain) == 1

    def test_append_linked_entries(self) -> None:
        chain = ProvenanceChain()
        e1 = self._make_entry()
        e2 = self._make_entry(parent_id=e1.entry_id)
        chain.append(e1)
        chain.append(e2)
        assert chain.head() == e2
        assert len(chain) == 2

    def test_append_wrong_parent_raises(self) -> None:
        chain = ProvenanceChain()
        e1 = self._make_entry()
        e2 = self._make_entry(parent_id="wrong-id")
        chain.append(e1)
        with pytest.raises(ValueError, match="parent_id mismatch"):
            chain.append(e2)

    def test_append_duplicate_entry_id_raises(self) -> None:
        chain = ProvenanceChain()
        e = self._make_entry()
        chain.append(e)
        with pytest.raises(ValueError, match="Duplicate entry_id"):
            chain.append(e)

    def test_first_entry_may_have_no_parent(self) -> None:
        chain = ProvenanceChain()
        e = self._make_entry()
        chain.append(e)  # parent_id is None — must succeed
        assert len(chain) == 1

    def test_trace_returns_entries_up_to_target(self) -> None:
        chain = ProvenanceChain()
        e1 = self._make_entry()
        e2 = self._make_entry(parent_id=e1.entry_id)
        e3 = self._make_entry(parent_id=e2.entry_id)
        chain.append(e1)
        chain.append(e2)
        chain.append(e3)
        assert chain.trace(e2.entry_id) == [e1, e2]

    def test_trace_full_chain(self) -> None:
        chain = ProvenanceChain()
        e1 = self._make_entry()
        e2 = self._make_entry(parent_id=e1.entry_id)
        e3 = self._make_entry(parent_id=e2.entry_id)
        chain.append(e1)
        chain.append(e2)
        chain.append(e3)
        assert chain.trace(e3.entry_id) == [e1, e2, e3]

    def test_trace_unknown_id_returns_empty(self) -> None:
        chain = ProvenanceChain()
        assert chain.trace("nonexistent") == []

    def test_all_entries_returns_ordered_copy(self) -> None:
        chain = ProvenanceChain()
        e1 = self._make_entry()
        e2 = self._make_entry(parent_id=e1.entry_id)
        chain.append(e1)
        chain.append(e2)
        assert chain.all_entries() == [e1, e2]

    def test_all_entries_returns_copy_not_reference(self) -> None:
        chain = ProvenanceChain()
        e1 = self._make_entry()
        chain.append(e1)
        copy = chain.all_entries()
        copy.clear()
        assert len(chain) == 1  # original chain unaffected

    def test_append_root_with_non_none_parent_id_raises(self) -> None:
        """First entry with a non-None parent_id must be rejected (Bug 3)."""
        chain = ProvenanceChain()
        e = self._make_entry(parent_id="external-parent")
        with pytest.raises(ValueError, match="Root entry must have parent_id=None"):
            chain.append(e)

    def test_trace_uses_position_index(self) -> None:
        """trace() must return correct slice regardless of chain length (O(1) fix)."""
        chain = ProvenanceChain()
        entries = []
        prev_id = None
        for i in range(5):
            e = self._make_entry(parent_id=prev_id) if i > 0 else self._make_entry()
            chain.append(e)
            entries.append(e)
            prev_id = e.entry_id
        assert chain.trace(entries[2].entry_id) == entries[:3]


# ---------------------------------------------------------------------------
# TrustScorer tests
# ---------------------------------------------------------------------------


class TestTrustScorer:
    def _make_scorer(self) -> TrustScorer:
        return TrustScorer()

    def test_score_no_data_returns_neutral(self) -> None:
        scorer = self._make_scorer()
        assert scorer.score("unknown-backend") == pytest.approx(0.5)

    def test_score_all_successes_is_high(self) -> None:
        scorer = self._make_scorer()
        for _ in range(10):
            scorer.record("b1", success=True, latency_s=0.1)
        assert scorer.score("b1") > 0.8

    def test_score_all_failures_is_low(self) -> None:
        scorer = self._make_scorer()
        for _ in range(10):
            scorer.record("b1", success=False, latency_s=0.1)
        # success_rate=0 → score ≤ 0.5 (neutral); integrity and consistency
        # sub-scores keep the total at exactly 0.5 when no verifier is set.
        assert scorer.score("b1") <= 0.5

    def test_score_clamped_to_unit_interval(self) -> None:
        scorer = self._make_scorer()
        for _ in range(10):
            scorer.record("b1", success=True, latency_s=0.01)
        s = scorer.score("b1")
        assert 0.0 <= s <= 1.0

    def test_score_mixed_outcomes(self) -> None:
        scorer = self._make_scorer()
        for _ in range(7):
            scorer.record("b1", success=True, latency_s=0.1)
        for _ in range(3):
            scorer.record("b1", success=False, latency_s=0.1)
        score = scorer.score("b1")
        assert 0.5 < score < 1.0  # 70% success → above neutral

    def test_all_scores_returns_all_recorded_backends(self) -> None:
        scorer = self._make_scorer()
        scorer.record("b1", success=True, latency_s=0.1)
        scorer.record("b2", success=False, latency_s=0.5)
        scores = scorer.all_scores()
        assert set(scores.keys()) == {"b1", "b2"}

    def test_report_structure(self) -> None:
        scorer = self._make_scorer()
        scorer.record("b1", success=True, latency_s=0.1)
        report = scorer.report()
        assert len(report) == 1
        row = report[0]
        assert "backend_id" in row
        assert "score" in row
        assert "total_operations" in row
        assert "successes" in row
        assert "failures" in row

    def test_report_empty_when_no_records(self) -> None:
        scorer = self._make_scorer()
        assert scorer.report() == []

    def test_latency_consistency_single_observation(self) -> None:
        scorer = self._make_scorer()
        scorer.record("b1", success=True, latency_s=0.5)
        # Single observation → consistency = 1.0, full weight
        score = scorer.score("b1")
        assert score > 0.5

    def test_integrity_rate_uses_verifier(self) -> None:
        verifier = IntegrityVerifier("secret")
        scorer = TrustScorer(verifier=verifier)
        signed = verifier.sign(LedgerEntry(operation="op", backend_id="b1"))
        unsigned = LedgerEntry(operation="op", backend_id="b1")
        scorer.record("b1", success=True, latency_s=0.1, entry=signed)
        scorer.record("b1", success=True, latency_s=0.1, entry=unsigned)
        score_with_bad = scorer.score("b1")

        scorer2 = TrustScorer(verifier=verifier)
        scorer2.record("b1", success=True, latency_s=0.1, entry=signed)
        scorer2.record("b1", success=True, latency_s=0.1, entry=signed)
        score_all_good = scorer2.score("b1")

        # Both signed → higher score than one unsigned
        assert score_all_good > score_with_bad

    def test_no_verifier_integrity_rate_defaults_to_one(self) -> None:
        scorer = TrustScorer(verifier=None)
        unsigned = LedgerEntry(operation="op", backend_id="b1")
        scorer.record("b1", success=True, latency_s=0.1, entry=unsigned)
        # Without a verifier integrity_rate = 1.0 → does not penalise score
        assert scorer.score("b1") > 0.8

    def test_record_rejects_nan_latency(self) -> None:
        """NaN latency_s must raise ValueError (Copilot fix)."""
        scorer = self._make_scorer()
        with pytest.raises(ValueError, match="latency_s"):
            scorer.record("b1", success=True, latency_s=float("nan"))

    def test_record_rejects_inf_latency(self) -> None:
        """Infinite latency_s must raise ValueError (Copilot fix)."""
        scorer = self._make_scorer()
        with pytest.raises(ValueError, match="latency_s"):
            scorer.record("b1", success=True, latency_s=float("inf"))

    def test_record_rejects_negative_latency(self) -> None:
        """Negative latency_s must raise ValueError (Copilot fix)."""
        scorer = self._make_scorer()
        with pytest.raises(ValueError, match="latency_s"):
            scorer.record("b1", success=True, latency_s=-0.1)

    def test_record_accepts_zero_latency(self) -> None:
        """Zero is a valid latency_s (boundary value)."""
        scorer = self._make_scorer()
        scorer.record("b1", success=True, latency_s=0.0)  # must not raise


# ---------------------------------------------------------------------------
# SemanticLedger integration tests
# ---------------------------------------------------------------------------


class TestSemanticLedger:
    _SECRET = "integration-secret"

    def _make_ledger(self) -> SemanticLedger:
        return SemanticLedger(self._SECRET)

    def test_record_returns_signed_entry(self) -> None:
        ledger = self._make_ledger()
        entry = ledger.record(operation="inference", backend_id="rtx5050")
        assert entry.integrity_tag != ""

    def test_record_content_hash_stored(self) -> None:
        ledger = self._make_ledger()
        entry = ledger.record(
            operation="inference", backend_id="rtx5050", content=b"hello"
        )
        expected = IntegrityVerifier.content_hash(b"hello")
        assert entry.content_hash == expected

    def test_verify_session_valid_after_records(self) -> None:
        ledger = self._make_ledger()
        ledger.record(operation="op1", backend_id="rtx5050", session_id="s1")
        ledger.record(operation="op2", backend_id="rtx5050", session_id="s1")
        assert ledger.verify_session("s1")

    def test_verify_session_unknown_returns_false(self) -> None:
        ledger = self._make_ledger()
        assert not ledger.verify_session("nonexistent")

    def test_trust_score_updates_on_success(self) -> None:
        ledger = self._make_ledger()
        for _ in range(5):
            ledger.record(
                operation="inference",
                backend_id="rtx5050",
                success=True,
                latency_s=0.1,
            )
        assert ledger.trust_score("rtx5050") > 0.5

    def test_trust_score_neutral_for_unknown_backend(self) -> None:
        ledger = self._make_ledger()
        assert ledger.trust_score("never-seen") == pytest.approx(0.5)

    def test_trust_report_includes_recorded_backend(self) -> None:
        ledger = self._make_ledger()
        ledger.record(operation="inference", backend_id="rtx5050")
        report = ledger.trust_report()
        assert any(r["backend_id"] == "rtx5050" for r in report)

    def test_provenance_returns_chain_in_order(self) -> None:
        ledger = self._make_ledger()
        ledger.record(operation="op1", backend_id="rtx5050", session_id="s1")
        ledger.record(operation="op2", backend_id="rtx5050", session_id="s1")
        prov = ledger.provenance("s1")
        assert len(prov) == 2
        assert prov[0].operation == "op1"
        assert prov[1].operation == "op2"

    def test_provenance_parent_links_correct(self) -> None:
        ledger = self._make_ledger()
        e1 = ledger.record(operation="op1", backend_id="rtx5050", session_id="s1")
        e2 = ledger.record(operation="op2", backend_id="rtx5050", session_id="s1")
        assert e2.parent_id == e1.entry_id

    def test_provenance_unknown_session_returns_empty(self) -> None:
        ledger = self._make_ledger()
        assert ledger.provenance("nonexistent") == []

    def test_audit_log_accumulates_all_entries(self) -> None:
        ledger = self._make_ledger()
        ledger.record(operation="op1", backend_id="rtx5050", session_id="s1")
        ledger.record(operation="op2", backend_id="radeon780m", session_id="s2")
        assert len(ledger.audit_log()) == 2

    def test_audit_log_returns_copy(self) -> None:
        ledger = self._make_ledger()
        ledger.record(operation="op", backend_id="rtx5050")
        log = ledger.audit_log()
        log.clear()
        assert len(ledger.audit_log()) == 1  # original unaffected

    def test_multiple_sessions_are_independent(self) -> None:
        ledger = self._make_ledger()
        ledger.record(operation="op", backend_id="rtx5050", session_id="s1")
        ledger.record(operation="op", backend_id="rtx5050", session_id="s2")
        assert len(ledger.provenance("s1")) == 1
        assert len(ledger.provenance("s2")) == 1
        assert ledger.verify_session("s1")
        assert ledger.verify_session("s2")

    def test_model_id_propagated_to_entry(self) -> None:
        ledger = self._make_ledger()
        entry = ledger.record(
            operation="inference",
            backend_id="rtx5050",
            model_id="deepseek-r1",
        )
        assert entry.model_id == "deepseek-r1"

    def test_metadata_propagated_to_entry(self) -> None:
        ledger = self._make_ledger()
        entry = ledger.record(
            operation="inference",
            backend_id="rtx5050",
            metadata={"tokens": 128},
        )
        assert entry.metadata["tokens"] == 128

    def test_first_entry_in_session_has_no_parent(self) -> None:
        ledger = self._make_ledger()
        entry = ledger.record(operation="op", backend_id="rtx5050", session_id="new")
        assert entry.parent_id is None

    def test_mutating_provenance_entry_does_not_break_verify_session(self) -> None:
        """Mutating a returned provenance entry's metadata must not corrupt the ledger (Bug 2)."""
        ledger = self._make_ledger()
        ledger.record(operation="op", backend_id="rtx5050", session_id="s1")
        entries = ledger.provenance("s1")
        # Mutate the returned deep-copy's metadata dict
        entries[0].metadata["injected"] = "tampered"  # type: ignore[index]
        # The stored chain must be unaffected
        assert ledger.verify_session("s1")

    def test_mutating_audit_log_entry_does_not_corrupt_ledger(self) -> None:
        """Mutating a returned audit_log entry must not affect internal state (Bug 2)."""
        ledger = self._make_ledger()
        ledger.record(operation="op", backend_id="rtx5050", session_id="s1")
        log = ledger.audit_log()
        log[0].metadata["injected"] = "tampered"  # type: ignore[index]
        assert ledger.verify_session("s1")

    def test_audit_log_bounded_by_max_size(self) -> None:
        """Audit log must not grow beyond the configured maximum entries."""
        max_size = 5
        ledger = SemanticLedger(self._SECRET, _max_audit_log=max_size)
        for i in range(max_size + 3):
            ledger.record(
                operation=f"op{i}", backend_id="rtx5050", session_id=f"s{i}"
            )
        assert len(ledger.audit_log()) == max_size
