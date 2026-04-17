"""
KAIROS Upgrade: EnCompass Backtracking Engine
Based on MIT CSAIL EnCompass (arXiv:2512.03571, 2026)

Instead of hard-failing a proposal, the backtracker:
1. Analyzes WHY it failed (failure taxonomy)
2. Generates a modified proposal targeting the specific failure reason
3. Re-runs the SAGE loop with the mutation
4. Tracks retry lineage so the meta-agent can learn from the full attempt tree

This replaces the binary PASS/FAIL arc with:
  PASS → archive as elite
  PARTIAL → retry with targeted mutation (max 3 attempts)
  FAIL → diagnose → mutate → retry (max 2 attempts) → archive if still failing
"""

import json
import datetime
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

KAIROS_DIR = Path(__file__).parent
BACKTRACK_LOG = KAIROS_DIR / "backtrack_log.json"


class FailureReason(Enum):
    TOO_VAGUE = "too_vague"               # Proposal lacks specificity
    TOO_RISKY = "too_risky"               # Critic flagged safety concerns
    SCOPE_CREEP = "scope_creep"           # Proposal tries to change too much at once
    MISSING_CONTEXT = "missing_context"   # Proposal didn't account for existing systems
    LOW_NOVELTY = "low_novelty"           # Proposal repeats something already tried
    IMPLEMENTATION_GAP = "impl_gap"       # Good idea, no clear execution path
    CONTRADICTS_VALUES = "value_conflict" # Conflicts with Strange Loop alignment check
    UNKNOWN = "unknown"


# Mutation strategies keyed by failure reason
MUTATION_STRATEGIES: Dict[str, str] = {
    FailureReason.TOO_VAGUE.value: (
        "Narrow the scope to a single, concrete, measurable change. "
        "Specify exactly which file, function, or system parameter changes and how."
    ),
    FailureReason.TOO_RISKY.value: (
        "Add a rollback plan. Propose the same improvement but with an explicit "
        "kill-switch condition and a safe reversion path."
    ),
    FailureReason.SCOPE_CREEP.value: (
        "Split the proposal into the smallest atomic unit that still delivers value. "
        "Defer all secondary changes to future cycles."
    ),
    FailureReason.MISSING_CONTEXT.value: (
        "Re-read the relevant existing system files first. Re-anchor the proposal "
        "to the actual current state of the code, not an assumed state."
    ),
    FailureReason.LOW_NOVELTY.value: (
        "Review the backtrack log for previous attempts. Propose something "
        "orthogonal — a different axis of improvement entirely."
    ),
    FailureReason.IMPLEMENTATION_GAP.value: (
        "Provide a step-by-step implementation plan with pseudocode or "
        "file-level diffs before proposing the change."
    ),
    FailureReason.CONTRADICTS_VALUES.value: (
        "Re-align with Strange Loop values. Propose the improvement in a way "
        "that respects the existing identity and safety constraints."
    ),
    FailureReason.UNKNOWN.value: (
        "Reframe the proposal from first principles. Start over with a "
        "simpler version of the core idea."
    ),
}


class BacktrackNode:
    """A single node in the attempt tree for one improvement proposal."""

    def __init__(
        self,
        proposal: str,
        parent_id: Optional[str] = None,
        mutation_strategy: Optional[str] = None,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.parent_id = parent_id
        self.proposal = proposal
        self.mutation_strategy = mutation_strategy
        self.score: Optional[float] = None
        self.failure_reason: Optional[str] = None
        self.critic_notes: Optional[str] = None
        self.status: str = "pending"  # pending | pass | partial | fail
        self.timestamp = datetime.datetime.utcnow().isoformat()
        self.children: List[str] = []  # child node IDs

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "proposal": self.proposal,
            "mutation_strategy": self.mutation_strategy,
            "score": self.score,
            "failure_reason": self.failure_reason,
            "critic_notes": self.critic_notes,
            "status": self.status,
            "timestamp": self.timestamp,
            "children": self.children,
        }


class EnCompassBacktracker:
    """
    EnCompass-style backtracking engine for the KAIROS SAGE loop.

    Usage:
        bt = EnCompassBacktracker()
        root = bt.start_attempt(proposal_text)
        # ... run SAGE loop ...
        if score < 0.7:
            reason = bt.diagnose_failure(critic_notes)
            retry_node = bt.backtrack(root.id, reason)
            # retry_node.proposal contains the mutated proposal
        bt.record_outcome(node_id, score, status)
        bt.save()
    """

    MAX_RETRIES_PARTIAL = 3
    MAX_RETRIES_FAIL = 2

    def __init__(self):
        self.nodes: Dict[str, BacktrackNode] = {}
        self.cycle_trees: List[Dict] = []  # one tree per KAIROS cycle
        self._load()

    def _load(self):
        if BACKTRACK_LOG.exists():
            try:
                data = json.loads(BACKTRACK_LOG.read_text())
                self.cycle_trees = data.get("cycle_trees", [])
            except Exception:
                self.cycle_trees = []

    def save(self):
        BACKTRACK_LOG.write_text(
            json.dumps(
                {
                    "last_updated": datetime.datetime.utcnow().isoformat(),
                    "total_cycles": len(self.cycle_trees),
                    "cycle_trees": self.cycle_trees[-50:],  # keep last 50 cycles
                },
                indent=2,
            )
        )

    # ------------------------------------------------------------------ #
    # Core API                                                             #
    # ------------------------------------------------------------------ #

    def start_attempt(self, proposal: str) -> BacktrackNode:
        """Create the root node for a new KAIROS improvement proposal."""
        node = BacktrackNode(proposal=proposal)
        self.nodes[node.id] = node
        return node

    def record_outcome(
        self,
        node_id: str,
        score: float,
        critic_notes: str = "",
        failure_reason: Optional[str] = None,
    ) -> str:
        """Record the SAGE loop result for a node. Returns final status."""
        node = self.nodes.get(node_id)
        if not node:
            return "unknown"

        node.score = score
        node.critic_notes = critic_notes
        node.failure_reason = failure_reason

        if score >= 0.85:
            node.status = "pass"
        elif score >= 0.7:
            node.status = "partial"
        else:
            node.status = "fail"

        return node.status

    def diagnose_failure(self, critic_notes: str) -> str:
        """
        Simple keyword-based failure taxonomy.
        In production this would be an LLM call — here it's heuristic.
        Returns a FailureReason value string.
        """
        notes = critic_notes.lower()
        if any(w in notes for w in ["vague", "unclear", "ambiguous", "abstract"]):
            return FailureReason.TOO_VAGUE.value
        if any(w in notes for w in ["risk", "danger", "unsafe", "unstable", "break"]):
            return FailureReason.TOO_RISKY.value
        if any(w in notes for w in ["too much", "scope", "complex", "broad", "wide"]):
            return FailureReason.SCOPE_CREEP.value
        if any(w in notes for w in ["context", "existing", "already", "current state"]):
            return FailureReason.MISSING_CONTEXT.value
        if any(w in notes for w in ["repeat", "duplicate", "already tried", "same"]):
            return FailureReason.LOW_NOVELTY.value
        if any(w in notes for w in ["how", "implementation", "unclear path", "no plan"]):
            return FailureReason.IMPLEMENTATION_GAP.value
        if any(w in notes for w in ["value", "align", "identity", "ethics", "safe"]):
            return FailureReason.CONTRADICTS_VALUES.value
        return FailureReason.UNKNOWN.value

    def backtrack(self, parent_node_id: str, failure_reason: str) -> Optional[BacktrackNode]:
        """
        Generate a mutated child proposal from a failed/partial parent.
        Returns None if max retries exceeded.
        """
        parent = self.nodes.get(parent_node_id)
        if not parent:
            return None

        # Count ancestors — enforce retry limits
        depth = self._depth(parent_node_id)
        max_depth = (
            self.MAX_RETRIES_PARTIAL
            if parent.status == "partial"
            else self.MAX_RETRIES_FAIL
        )
        if depth >= max_depth:
            return None  # Exhausted retries — let KAIROS archive this branch

        # Build the mutated proposal
        strategy = MUTATION_STRATEGIES.get(failure_reason, MUTATION_STRATEGIES[FailureReason.UNKNOWN.value])
        mutated_proposal = (
            f"[BACKTRACK — attempt {depth + 1} | reason: {failure_reason}]\n\n"
            f"Original proposal:\n{parent.proposal}\n\n"
            f"Failure diagnosed as: {failure_reason}\n"
            f"Mutation strategy: {strategy}\n\n"
            f"Revised proposal:\n"
            f"Apply the mutation strategy above to rewrite the original proposal. "
            f"The revised version must address the diagnosed failure directly."
        )

        child = BacktrackNode(
            proposal=mutated_proposal,
            parent_id=parent_node_id,
            mutation_strategy=strategy,
        )
        self.nodes[child.id] = child
        parent.children.append(child.id)
        return child

    def close_cycle(self, cycle_id: str, winning_node_id: Optional[str] = None):
        """Finalize a KAIROS cycle and archive the attempt tree."""
        tree = {
            "cycle_id": cycle_id,
            "closed_at": datetime.datetime.utcnow().isoformat(),
            "winning_node": winning_node_id,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "total_attempts": len(self.nodes),
            "backtrack_depth": max(
                (self._depth(nid) for nid in self.nodes), default=0
            ),
        }
        self.cycle_trees.append(tree)
        self.nodes.clear()  # reset for next cycle
        self.save()

    # ------------------------------------------------------------------ #
    # Analytics — feeds the Meta-Agent                                    #
    # ------------------------------------------------------------------ #

    def failure_distribution(self) -> Dict[str, int]:
        """How often each failure reason appears across all cycles."""
        dist: Dict[str, int] = {}
        for cycle in self.cycle_trees:
            for node in cycle.get("nodes", []):
                reason = node.get("failure_reason")
                if reason:
                    dist[reason] = dist.get(reason, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    def avg_backtrack_depth(self) -> float:
        """Average number of retries needed before a proposal passes."""
        depths = [c.get("backtrack_depth", 0) for c in self.cycle_trees if c.get("backtrack_depth")]
        return round(sum(depths) / len(depths), 2) if depths else 0.0

    def retry_success_rate(self) -> float:
        """% of backtracked proposals that eventually passed."""
        retried = 0
        succeeded = 0
        for cycle in self.cycle_trees:
            for node in cycle.get("nodes", []):
                if node.get("parent_id"):  # it's a retry
                    retried += 1
                    if node.get("status") == "pass":
                        succeeded += 1
        if retried == 0:
            return 0.0
        return round(succeeded / retried * 100, 1)

    def meta_agent_briefing(self) -> str:
        """
        Summary report for the meta-agent — tells it what kinds of proposals
        keep failing so it can update the improvement rules.
        """
        dist = self.failure_distribution()
        top_failures = list(dist.items())[:3]
        rate = self.retry_success_rate()
        depth = self.avg_backtrack_depth()

        lines = [
            "=== EnCompass Meta-Agent Briefing ===",
            f"Retry success rate: {rate}%",
            f"Average backtrack depth: {depth}",
            "Top failure reasons:",
        ]
        for reason, count in top_failures:
            strategy = MUTATION_STRATEGIES.get(reason, "unknown")
            lines.append(f"  • {reason} ({count}x) → strategy: {strategy[:60]}...")
        lines.append(
            "\nRecommendation: Proposer agent should pre-screen proposals against "
            "top failure reasons before submitting to the SAGE loop."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _depth(self, node_id: str) -> int:
        """Count how many ancestors a node has (retry depth)."""
        node = self.nodes.get(node_id)
        depth = 0
        while node and node.parent_id:
            depth += 1
            node = self.nodes.get(node.parent_id)
        return depth

    def status(self) -> str:
        cycles = len(self.cycle_trees)
        rate = self.retry_success_rate()
        return f"EnCompass: {cycles} cycles logged | retry success rate: {rate}%"
