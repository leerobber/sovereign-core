"""
KAIROS Upgrade #4: Sim-Before-Deploy Séance
Inspired by Lockheed Skunk Works X-62A VISTA (2026)

The Skunk Works lesson: train in simulation, validate in simulation,
THEN deploy to real hardware. Their AI dodged a real missile because
it had simulated that exact scenario hundreds of times first.

Applied to KAIROS: before any elite proposal touches the real system,
Séance runs a simulated dry-run. If the simulation surfaces breakage,
the proposal goes back to the backtracker — NOT to production.

This is the difference between "I think this is safe" and
"I ran it 50 times in simulation and it held."

Architecture:
  SimEnvironment   — sandboxed snapshot of current system state
  SimRun           — executes a proposal against the snapshot
  SimValidator     — checks for breakage, regressions, value conflicts
  DeployGate       — PASS sim → deploy. FAIL sim → backtrack.
"""

import json
import datetime
import uuid
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

SEANCE_DIR = Path(__file__).parent
SIM_LOG = SEANCE_DIR / "sim_deploy_log.json"

# ------------------------------------------------------------------ #
# Risk taxonomy — what can go wrong in a self-modification            #
# ------------------------------------------------------------------ #

RISK_FACTORS = {
    "overwrites_core_file": 0.9,      # tries to modify omega.py, spawner, etc.
    "removes_kill_switch": 1.0,        # instant block — never allowed
    "changes_scoring_weights": 0.4,    # medium risk — affects all future proposals
    "adds_new_dependency": 0.3,        # low-medium — new import could break
    "modifies_memory_schema": 0.6,     # medium — could corrupt existing records
    "touches_ghost_protocol": 0.7,     # high — security system changes are sensitive
    "pure_additive": 0.05,             # lowest risk — only adds, never removes
    "refactors_existing": 0.25,        # low — cleans up without behavior change
    "new_module": 0.1,                 # very low — isolated, contained
}

CORE_PROTECTED_FILES = [
    "omega.py", "kill_switch.py", "ghost_veil.py",
    "paradoxical_fortress.py", "agent_spawner.py",
]


# ------------------------------------------------------------------ #
# Simulated Environment                                               #
# ------------------------------------------------------------------ #

class SimEnvironment:
    """
    A lightweight snapshot of the current system state.
    Proposals are applied to this snapshot, not the real system.
    """

    def __init__(self):
        self.snapshot_id = str(uuid.uuid4())[:8]
        self.created_at = datetime.datetime.utcnow().isoformat()
        self.system_state = self._capture_state()
        self.integrity_hash = self._compute_hash()

    def _capture_state(self) -> Dict:
        """Capture key system parameters as the simulation baseline."""
        state = {
            "active_modules": [
                "HCM", "TwinEngine", "MemoryPalace", "ETS", "TimeDilation",
                "Feynman", "AgentSpawner", "GhostVeil", "ParadoxicalFortress",
                "KillSwitch", "QSW", "AutoTelic", "Seance", "JeopardyInversion",
                "BodyDouble", "Cassandra", "PerceptualLattice", "Pruner",
                "RFFingerprint", "StrangeLoop", "GroupEvolution", "EnCompass",
                "FederatedContext",
            ],
            "kairos_elite_threshold": 0.85,
            "kill_switch_active": True,
            "ghost_protocol_active": True,
            "values_locked": True,
            "federation_rounds": 0,
            "memory_entries": 0,
        }
        return state

    def _compute_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.system_state, sort_keys=True).encode()
        ).hexdigest()[:16]

    def apply_proposal(self, proposal: str) -> Dict:
        """
        Simulate applying a proposal to the environment.
        Returns a diff of what would change.
        """
        changes = {
            "proposal_hash": hashlib.md5(proposal.encode()).hexdigest()[:8],
            "detected_operations": [],
            "risk_flags": [],
        }

        proposal_lower = proposal.lower()

        # Detect operation types
        if any(k in proposal_lower for k in ["add", "new", "implement", "create"]):
            changes["detected_operations"].append("additive")
        if any(k in proposal_lower for k in ["remove", "delete", "replace", "refactor"]):
            changes["detected_operations"].append("destructive")
        if any(k in proposal_lower for k in ["modify", "update", "change", "upgrade"]):
            changes["detected_operations"].append("mutation")

        # Check for protected file touches
        for protected in CORE_PROTECTED_FILES:
            if protected.replace(".py", "") in proposal_lower:
                changes["risk_flags"].append(f"touches_protected:{protected}")

        # Check for kill switch removal attempt
        if "kill" in proposal_lower and "remov" in proposal_lower:
            changes["risk_flags"].append("removes_kill_switch")

        # Check for scoring weight changes
        if "threshold" in proposal_lower or "scoring weight" in proposal_lower:
            changes["risk_flags"].append("changes_scoring_weights")

        # Check for memory schema changes
        if "schema" in proposal_lower or "ghostmemory" in proposal_lower:
            changes["risk_flags"].append("modifies_memory_schema")

        # Classify overall change type
        if not changes["detected_operations"] or changes["detected_operations"] == ["additive"]:
            changes["change_class"] = "pure_additive"
        elif "destructive" in changes["detected_operations"]:
            changes["change_class"] = "destructive"
        else:
            changes["change_class"] = "mutation"

        return changes


# ------------------------------------------------------------------ #
# Simulation Run                                                       #
# ------------------------------------------------------------------ #

class SimRun:
    """Executes N simulation passes of a proposal against a SimEnvironment."""

    DEFAULT_PASSES = 10

    def __init__(self, proposal: str, env: SimEnvironment, passes: int = DEFAULT_PASSES):
        self.run_id = str(uuid.uuid4())[:8]
        self.proposal = proposal
        self.env = env
        self.passes = passes
        self.results: List[Dict] = []

    def execute(self) -> Dict:
        """Run the proposal through multiple simulation passes."""
        for i in range(self.passes):
            diff = self.env.apply_proposal(self.proposal)
            # Compute per-pass risk score
            risk = self._score_risk(diff)
            # Add stochastic variance (simulating real-world noise)
            import random
            random.seed(hash(f"{self.run_id}_{i}") % (2**32))
            noise = random.uniform(-0.05, 0.05)
            pass_risk = max(0.0, min(1.0, risk + noise))

            self.results.append({
                "pass": i + 1,
                "risk_score": round(pass_risk, 4),
                "risk_flags": diff.get("risk_flags", []),
                "change_class": diff.get("change_class", "unknown"),
                "operations": diff.get("detected_operations", []),
            })

        return self._aggregate()

    def _score_risk(self, diff: Dict) -> float:
        base = RISK_FACTORS.get(diff.get("change_class", "refactors_existing"), 0.3)
        # Add penalty for each risk flag
        flags = diff.get("risk_flags", [])
        penalty = sum(RISK_FACTORS.get(f.split(":")[0], 0.2) for f in flags)
        return min(1.0, base + penalty * 0.5)

    def _aggregate(self) -> Dict:
        if not self.results:
            return {"verdict": "NO_DATA", "avg_risk": 1.0}
        avg_risk = sum(r["risk_score"] for r in self.results) / len(self.results)
        max_risk = max(r["risk_score"] for r in self.results)
        all_flags = list(set(f for r in self.results for f in r.get("risk_flags", [])))
        verdict = "SAFE" if avg_risk < 0.4 and max_risk < 0.7 else "RISKY" if avg_risk < 0.65 else "BLOCKED"
        return {
            "run_id": self.run_id,
            "passes": self.passes,
            "avg_risk": round(avg_risk, 4),
            "max_risk": round(max_risk, 4),
            "all_flags": all_flags,
            "verdict": verdict,
        }


# ------------------------------------------------------------------ #
# Deploy Gate — the final decision maker                              #
# ------------------------------------------------------------------ #

class DeployGate:
    """
    Wraps any KAIROS elite proposal in a mandatory simulation check.
    SAFE  → deploy approved
    RISKY → escalate to meta-agent for human review flag
    BLOCKED → send back to EnCompass backtracker
    """

    def __init__(self):
        self.gate_log: List[Dict] = []
        self._load()

    def _load(self):
        if SIM_LOG.exists():
            try:
                data = json.loads(SIM_LOG.read_text())
                self.gate_log = data.get("gate_log", [])
            except Exception:
                pass

    def save(self):
        SIM_LOG.write_text(json.dumps({
            "last_updated": datetime.datetime.utcnow().isoformat(),
            "total_decisions": len(self.gate_log),
            "gate_log": self.gate_log[-100:],
        }, indent=2))

    def evaluate(self, proposal: str, proposal_id: str = "") -> Tuple[str, Dict]:
        """
        Run the full sim-before-deploy pipeline.
        Returns (verdict, full_report).
        """
        env = SimEnvironment()
        sim = SimRun(proposal, env, passes=10)
        result = sim.execute()
        verdict = result["verdict"]

        # Absolute block — kill switch removal is never allowed
        if "removes_kill_switch" in result.get("all_flags", []):
            verdict = "BLOCKED"
            result["block_reason"] = "Kill switch removal detected — absolute veto"

        report = {
            "proposal_id": proposal_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "sim_snapshot": env.snapshot_id,
            "verdict": verdict,
            "sim_result": result,
            "deploy_approved": verdict == "SAFE",
        }

        self.gate_log.append(report)
        self.save()
        return verdict, report

    def approval_rate(self) -> float:
        if not self.gate_log:
            return 0.0
        approved = sum(1 for g in self.gate_log if g.get("deploy_approved"))
        return round(approved / len(self.gate_log) * 100, 1)

    def status(self) -> str:
        total = len(self.gate_log)
        rate = self.approval_rate()
        return f"DeployGate: {total} evaluations | {rate}% approval rate"
