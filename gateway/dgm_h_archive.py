"""
DGM-H Archive Integration for Sovereign Core (RES-02)
Extends Aegis-Vault lineage tracking with "archive as stepping stones" concept.

Every successful ARSO cycle persists:
  - The full agent state that produced the fix
  - The bottleneck context
  - The fix diff
  - Performance delta

When a future bottleneck occurs, reconstruct the nearest ancestor agent
that solved a similar problem (DGM-H core concept).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentLineageNode:
    """A single node in the DGM-H lineage tree."""
    node_id: str
    parent_id: Optional[str]
    generation: int
    bottleneck_type: str           # e.g. "vram_oom", "latency_spike", "accuracy_drop"
    bottleneck_description: str
    fix_diff: str                  # The actual code/config change that was applied
    agent_state_snapshot: dict     # Full agent configuration at time of fix
    performance_before: dict       # Metrics before fix
    performance_after: dict        # Metrics after fix
    performance_delta: float       # Scalar improvement score
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    is_stepping_stone: bool = False  # Marked true if descendants improved from this

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def improvement_score(self) -> float:
        return self.performance_delta


@dataclass
class DGMHArchive:
    """
    DGM-H Archive — stores agent lineage as stepping stones.

    Key insight from DGM-H: don't just store the fix, store the entire agent
    state that produced it. When future bottlenecks appear, reconstruct
    the agent that solved a similar problem rather than starting from scratch.
    """
    nodes: dict[str, AgentLineageNode] = field(default_factory=dict)
    archive_path: Optional[Path] = None

    def add_node(
        self,
        bottleneck_type: str,
        bottleneck_description: str,
        fix_diff: str,
        agent_state_snapshot: dict,
        performance_before: dict,
        performance_after: dict,
        performance_delta: float,
        parent_id: Optional[str] = None,
    ) -> AgentLineageNode:
        """Record a successful ARSO cycle as a lineage node."""
        generation = 0
        if parent_id and parent_id in self.nodes:
            generation = self.nodes[parent_id].generation + 1

        node_id = hashlib.sha256(
            f"{bottleneck_type}:{fix_diff}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        node = AgentLineageNode(
            node_id=node_id,
            parent_id=parent_id,
            generation=generation,
            bottleneck_type=bottleneck_type,
            bottleneck_description=bottleneck_description,
            fix_diff=fix_diff,
            agent_state_snapshot=agent_state_snapshot,
            performance_before=performance_before,
            performance_after=performance_after,
            performance_delta=performance_delta,
        )

        self.nodes[node_id] = node

        # Mark parent as stepping stone if this child improved on it
        if parent_id and parent_id in self.nodes:
            parent = self.nodes[parent_id]
            if performance_delta > parent.performance_delta:
                parent.is_stepping_stone = True

        if self.archive_path:
            self._persist()

        logger.info(
            "DGM-H node %s added (gen=%d, bottleneck=%s, delta=%.3f)",
            node_id, generation, bottleneck_type, performance_delta,
        )
        return node

    def find_nearest_ancestor(
        self,
        bottleneck_type: str,
        top_k: int = 3,
    ) -> list[AgentLineageNode]:
        """
        Find the best ancestor nodes for a given bottleneck type.
        Prioritises: stepping stones > high performance delta > recency.

        This is the core DGM-H reconstruction mechanism — instead of starting
        fresh, we resume from the agent state that worked best before.
        """
        candidates = [
            n for n in self.nodes.values()
            if n.bottleneck_type == bottleneck_type
        ]

        if not candidates:
            # Fallback: find nodes from related bottleneck types
            candidates = list(self.nodes.values())

        # Score: stepping_stone bonus + performance_delta
        def score(n: AgentLineageNode) -> float:
            return n.performance_delta + (0.5 if n.is_stepping_stone else 0.0)

        return sorted(candidates, key=score, reverse=True)[:top_k]

    def reconstruct_agent_from_ancestor(
        self, node_id: str
    ) -> tuple[dict, list[AgentLineageNode]]:
        """
        Reconstruct the full lineage path to a given node.
        Returns (agent_state_snapshot, lineage_path).

        Used by ARSO Orchestrator to resume from a known-good agent state.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found in archive")

        node = self.nodes[node_id]
        lineage_path: list[AgentLineageNode] = [node]

        current = node
        while current.parent_id and current.parent_id in self.nodes:
            current = self.nodes[current.parent_id]
            lineage_path.insert(0, current)

        return node.agent_state_snapshot, lineage_path

    def stepping_stones(self) -> list[AgentLineageNode]:
        """Return all nodes marked as stepping stones, sorted by generation."""
        return sorted(
            [n for n in self.nodes.values() if n.is_stepping_stone],
            key=lambda n: n.generation,
        )

    def summary(self) -> dict:
        return {
            "total_nodes": len(self.nodes),
            "stepping_stones": len(self.stepping_stones()),
            "bottleneck_types": list({n.bottleneck_type for n in self.nodes.values()}),
            "max_generation": max((n.generation for n in self.nodes.values()), default=0),
            "best_delta": max((n.performance_delta for n in self.nodes.values()), default=0.0),
        }

    def _persist(self) -> None:
        assert self.archive_path is not None
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.archive_path, "w") as f:
            json.dump(
                {nid: n.to_dict() for nid, n in self.nodes.items()},
                f, indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> "DGMHArchive":
        archive = cls(archive_path=path)
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            archive.nodes = {
                nid: AgentLineageNode(**data) for nid, data in raw.items()
            }
            logger.info("Loaded DGM-H archive: %d nodes", len(archive.nodes))
        return archive
