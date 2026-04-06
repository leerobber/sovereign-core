# Aegis-Vault Integrity Check Runbook

**Component:** Aegis-Vault Decentralized Semantic Ledger  
**Epic:** KAN-90 — Trust & Provenance Tracking  
**Classification:** TWC (Trusted Whole-Chain) Verification

---

## Overview

The Aegis-Vault Semantic Ledger is the trust and provenance backbone of Sovereign
Core. Every inference request, routing decision, and health-check event is recorded
as a signed `LedgerEntry`. The entries are linked into per-session `ProvenanceChain`s
and scored by the `TrustScorer`. This runbook describes how to verify ledger
integrity, diagnose trust degradation, and respond to integrity failures.

---

## Architecture

```
SemanticLedger
├── IntegrityVerifier  — HMAC-SHA256 signing & chain verification
├── ProvenanceChain[]  — Per-session append-only entry chain
├── TrustScorer        — Weighted [0, 1] trust score per backend
└── audit_log          — Ordered record of every operation
```

### Key Invariants

| Invariant | Description |
|-----------|-------------|
| **Signed entries** | Every entry carries an HMAC-SHA256 `integrity_tag` computed from all other fields. |
| **Parent linkage** | Each entry (except the first in a session) references the `entry_id` of its predecessor. |
| **Append-only** | Chains are never modified; entries are only appended. |
| **Content hash** | The SHA-256 digest of the raw response content is stored in `content_hash`. |

---

## Integrity Verification Procedure

### 1. Verify a Single Entry

```python
from gateway.ledger import IntegrityVerifier, LedgerEntry

verifier = IntegrityVerifier(secret=AEGIS_SECRET)
is_valid = verifier.verify(entry)
```

**Expected:** `True`  
**If False:** The `integrity_tag` does not match the entry's canonical bytes. The
entry may have been tampered with or was never signed.

---

### 2. Verify a Session Chain

```python
from gateway.ledger import SemanticLedger

ledger = SemanticLedger(secret=AEGIS_SECRET)
is_intact = ledger.verify_session(session_id="<SESSION_ID>")
```

**Expected:** `True`  
**If False:** One or more of the following faults has occurred:

| Symptom | Possible Cause |
|---------|---------------|
| `integrity_tag` mismatch at position N | Entry N was tampered with after signing. |
| `parent_id` mismatch at position N | Entry N was inserted out of order or from a different chain. |
| Session not found | The session ID has never been recorded, or the ledger was restarted without persistence. |

---

### 3. Inspect the Full Provenance Chain

```python
entries = ledger.provenance(session_id="<SESSION_ID>")
for entry in entries:
    print(entry.entry_id, entry.operation, entry.backend_id, entry.timestamp)
```

Each entry includes:
- `entry_id` — unique 32-character hex identifier  
- `timestamp` — Unix wall-clock time the entry was created  
- `operation` — operation type (e.g. `"inference"`, `"health_check"`)  
- `backend_id` — backend that handled the operation  
- `model_id` — model used (if applicable)  
- `parent_id` — predecessor entry ID (None for chain root)  
- `content_hash` — SHA-256 of the raw response content  
- `metadata` — arbitrary key-value pairs  
- `integrity_tag` — HMAC-SHA256 of all other fields  

---

### 4. Review the Audit Log

```python
all_entries = ledger.audit_log()
```

The audit log contains every entry ever recorded across all sessions, in
insertion order. Use this to reconstruct the full operational timeline.

---

## Trust Score Diagnostics

### Retrieving Trust Scores

```python
# Single backend
score = ledger.trust_score(backend_id="rtx5050")

# All backends
report = ledger.trust_report()
# [{"backend_id": "rtx5050", "score": 0.92, "total_operations": 150, ...}, ...]
```

### Trust Score Thresholds

| Score Range | Interpretation | Recommended Action |
|-------------|---------------|-------------------|
| `0.80 – 1.00` | ✅ Highly trusted | Normal operation |
| `0.60 – 0.79` | ⚠️ Degraded trust | Investigate recent failures |
| `0.40 – 0.59` | 🟠 Low trust | Consider reducing routing weight |
| `0.00 – 0.39` | 🔴 Untrusted | Remove from active rotation |
| `0.50` (neutral) | ℹ️ No data | Insufficient observations |

### Trust Score Components

The score is a weighted sum of three sub-scores:

| Sub-score | Weight | Description |
|-----------|--------|-------------|
| `success_rate` | 50 % | Fraction of operations that completed without error |
| `integrity_rate` | 30 % | Fraction of entries with a valid `integrity_tag` |
| `latency_consistency` | 20 % | Inverse coefficient of variation of observed latencies |

---

## Incident Response

### Scenario A — Single Entry Integrity Failure

**Symptom:** `verifier.verify(entry)` returns `False` for a specific entry.

**Steps:**
1. Identify the entry by `entry_id` and `timestamp`.
2. Check whether the secret used at verification matches the one used at signing.
3. If the secret is correct, the entry was tampered with or corrupted in transit.
4. Quarantine the affected session; do not trust its downstream content.
5. File an incident report with the `entry_id`, `session_id`, and `backend_id`.

---

### Scenario B — Chain Linkage Break

**Symptom:** `verify_chain` returns `False` with a `parent_id mismatch` log warning.

**Steps:**
1. Locate the position in the chain where the break occurred (log output includes position index).
2. Compare the `parent_id` of the offending entry against the `entry_id` of its predecessor.
3. Determine whether entries were reordered, dropped, or injected.
4. Invalidate the entire chain from the break point onwards.

---

### Scenario C — Trust Score Drops Below 0.60

**Symptom:** `ledger.trust_score("backend_id")` returns a value below 0.60.

**Steps:**
1. Check `trust_report()` for the affected backend's `successes` and `failures`.
2. If `failures` is high: cross-reference with `HealthMonitor.status_report()` for
   circuit-breaker state.
3. If `integrity_rate` is low: review signed vs unsigned entries — possible signing
   key mismatch or missing call to `IntegrityVerifier.sign()`.
4. If `latency_consistency` is low: investigate network jitter or resource contention
   on the backend device.

---

## Security Considerations

- **The HMAC secret** (`AEGIS_SECRET`) must never appear in logs, entries, or API
  responses. Store it in a secrets manager and inject at runtime.
- **Constant-time comparison** (`hmac.compare_digest`) is used for all tag
  verification to prevent timing side-channel attacks.
- **Content hashes** use SHA-256 (collision-resistant). They confirm that the
  content referenced by an entry has not changed, but do not encrypt it.
- **Ledger entries are not encrypted at rest.** Apply appropriate storage-layer
  encryption for sensitive fields in `metadata`.

---

## Integration Points

| System | Integration |
|--------|-------------|
| `GatewayRouter` | Call `ledger.record(...)` after each routed request with `latency_s` and `success` flags. |
| `HealthMonitor` | Optionally record health-check events as `operation="health_check"`. |
| `ThroughputBenchmark` | Trust scores complement benchmark latency data for backend ranking. |

---

## Reference

- Module: `gateway/ledger.py`
- Tests: `tests/test_ledger.py`
- Related issues: KAN-87 (kernel integration), KAN-59 (VAULT adapter provenance)
