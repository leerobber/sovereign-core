# SovereignCore Roadmap

**Current phase: Stabilization (Q3 2026)**

Last updated: 2026-07-25 (test status: 570 passing, 76 skipped, 0 failing)

---

## Phase A: Stabilization (2 weeks)

### Goals
- Merge remaining integration PRs from GH05T3-Sovereign
- Verify all 5 repos (GH05T3, GH05T3-Sovereign, sovereign-core, aetherflux-zero, aethyro-ntg) work together
- Document current limitations honestly

### Tasks
- [ ] **Merge GH05T3-Sovereign PRs** (2 blocked PRs)
  - Torch dependency conflict
  - aetherflux bridge integration
- [ ] **Full integration test suite** (sovereign-core ↔ GH05T3 mesh)
  - Real /v1/chat/completions round trip through mesh
  - KAIROS cycle triggering from GH05T3
  - Real fitness signal flow from aetherflux-zero
- [ ] **Document memory systems status**
  - Which gates are LIVE (Kill Switch only)
  - Which are CODE-READY (Iron Dome, GhostRecall)
  - Which are DESIGNED (CLARA, Ethics)
- [ ] **Fix hardcoded secrets**
  - Remove Slack OAuth values from code
  - Move all secrets to .env template
  - Add validation + clear error messages

### Success Criteria
- All 570 tests pass
- GH05T3-Sovereign PRs merged
- Integration test runs end-to-end
- No hardcoded secrets in code

---

## Phase B: Memory Systems Hardening (3 weeks)

### Goals
- Deploy Iron Dome (memory protection) from design → live
- Full end-to-end test coverage
- Real audit trail of all memory writes

### Tasks
- [ ] **Iron Dome end-to-end test**
  - Write → hash chain
  - Tamper detection
  - Real error on corrupted chain
  - Performance impact baseline
- [ ] **GhostRecall integration**
  - Wire into inference path
  - Test episodic memory + replay
  - Measure retrieval latency
- [ ] **Audit middleware**
  - Log every memory write
  - Persisted audit trail (daily rotation)
  - Real analysis of threat patterns detected

### Success Criteria
- Iron Dome 100% test coverage
- GhostRecall retrieves memories < 50ms
- Audit logs automatically rotated
- 0 false positives on attack pattern detection (first week)

---

## Phase C: Multi-Repo Synchronization (4 weeks)

### Goals
- Unified CI/CD across all repos
- Shared data layer (ledger, genome store)
- Real fitness signals flowing end-to-end

### Tasks
- [ ] **Unified GitHub Actions**
  - Single workflow file
  - Triggers cross-repo tests
  - Coordinated releases
- [ ] **Ledger sharing**
  - Move KAIROS ledger → shareable format (not just in-process)
  - GH05T3 can read sovereign-core evolutions
  - Real timeline of improvements across both
- [ ] **Genome registry**
  - Unified agent genome store
  - aetherflux-zero fitness signals auto-update genome ratings
  - Real selective breeding (high-fitness → promoted)
- [ ] **Integration test matrix**
  - All 5 repos in one CI run
  - Real data flows (not mocked)
  - Performance regression detection

### Success Criteria
- Single `git push` triggers all tests
- Fitness signals flow live (not batch)
- Genome registry has 50+ real agent variations
- No performance regressions from previous quarter

---

## Phase D: Production Readiness (Q4 2026, TBD)

### Goals
- Kill Switch → fully activated by default (not opt-in)
- Full memory protection deployment
- SLA-ready operational docs
- Real monitoring + alerting

### Tasks (not yet scoped)
- [ ] Runtime dashboards
- [ ] Alert rules (latency, memory, ethics gate triggers)
- [ ] Backup + recovery procedures
- [ ] Load testing under real workloads

---

## Known Blockers

| Blocker | Impact | Planned Resolution |
|---------|--------|-------------------|
| GH05T3-Sovereign torch dependency | Can't merge 2 integration PRs | Investigate version conflict (Phase A) |
| Iron Dome full test coverage | Can't ship memory protection | Write end-to-end tests (Phase B) |
| Hardcoded Slack OAuth | Credentials in code | Migrate to .env (Phase A) |
| Ledger format (in-process only) | Can't share evolution history | Design JSON/protobuf format (Phase C) |

---

## Milestones

| Milestone | Target | Status |
|-----------|--------|--------|
| **S1: Stabilization** | 2026-08-08 | 🟡 In Progress |
| **S2: Memory Hardening** | 2026-08-29 | 🔵 Planned |
| **S3: Multi-Repo Sync** | 2026-09-26 | 🔵 Planned |
| **S4: Production Ready** | 2026-12-31 | 🔵 Future |

---

## How to Contribute

Pick any item from **Phase A** (highest priority) and:

1. Reference this roadmap in your PR
2. Update this file when you complete a task
3. Link related issues/ADRs
4. Include test coverage for new work

Questions? Open an issue or see [docs/COMMUNITY_GROWTH.md](docs/COMMUNITY_GROWTH.md).
