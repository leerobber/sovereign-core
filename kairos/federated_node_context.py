"""
KAIROS Upgrade #3: Federated Node Context Layer
Inspired by NASA JPL FAME (Federated Autonomous MEasurement, 2026)

Instead of 3 compute nodes that only load-balance work, they now share
what they LEARN from each inference run. The RTX learns Qwen2.5 behavioral
patterns → broadcasts to the pool → Radeon uses it to route smarter.

Architecture:
  NodeMemory         — each node keeps a local inference journal
  FederatedContext   — aggregates all node journals into a shared knowledge base
  ContextBroadcaster — pushes new learnings to all nodes after each run
  RoutingIntelligence— uses the federated context to make smarter routing decisions

The nodes become collectively intelligent, not just collectively available.
"""

import json
import datetime
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any

KAIROS_DIR = Path(__file__).parent
FEDERATION_LOG = KAIROS_DIR / "federation_log.json"

# ------------------------------------------------------------------ #
# Node definitions — mirrors sovereign-core gateway                   #
# ------------------------------------------------------------------ #

COMPUTE_NODES = {
    "rtx5050": {
        "port": 8001,
        "vram_gb": 8,
        "primary_model": "qwen2.5:14b",
        "strengths": ["large context", "complex reasoning", "code generation"],
        "weaknesses": ["slow cold start", "high VRAM pressure at 14B"],
    },
    "radeon780m": {
        "port": 8002,
        "vram_gb": 4,
        "primary_model": "deepseek-coder:6.7b",
        "strengths": ["fast inference", "code tasks", "low latency"],
        "weaknesses": ["limited context window", "weaker at abstract reasoning"],
    },
    "ryzen7_cpu": {
        "port": 8003,
        "vram_gb": 0,
        "primary_model": "llama3.2:3b",
        "strengths": ["always available", "small tasks", "fallback reliability"],
        "weaknesses": ["slowest", "weakest model", "not for heavy lifting"],
    },
}


# ------------------------------------------------------------------ #
# Node Memory — local inference journal per node                      #
# ------------------------------------------------------------------ #

class NodeMemory:
    """
    Each compute node keeps a rolling journal of what it learned
    from its inference runs — latency patterns, failure modes,
    task types it excels at, prompts that confused it.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.node_info = COMPUTE_NODES.get(node_id, {})
        self.journal: List[Dict] = []
        self.learned_patterns: Dict[str, Any] = {
            "fast_task_types": [],
            "slow_task_types": [],
            "failure_modes": [],
            "optimal_prompt_patterns": [],
            "avg_latency_ms": 0,
            "success_rate": 1.0,
        }

    def record_inference(
        self,
        task_type: str,
        latency_ms: float,
        success: bool,
        prompt_pattern: str = "",
        notes: str = "",
    ):
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "task_type": task_type,
            "latency_ms": latency_ms,
            "success": success,
            "prompt_pattern": prompt_pattern[:100],
            "notes": notes,
        }
        self.journal.append(entry)
        self._update_patterns(entry)

    def _update_patterns(self, entry: Dict):
        # Rolling average latency
        latencies = [e["latency_ms"] for e in self.journal if e["latency_ms"] > 0]
        self.learned_patterns["avg_latency_ms"] = round(
            sum(latencies) / len(latencies), 1
        ) if latencies else 0

        # Success rate
        total = len(self.journal)
        successes = sum(1 for e in self.journal if e["success"])
        self.learned_patterns["success_rate"] = round(successes / total, 3) if total else 1.0

        # Fast/slow task types
        task = entry["task_type"]
        if entry["success"] and entry["latency_ms"] < 2000:
            if task not in self.learned_patterns["fast_task_types"]:
                self.learned_patterns["fast_task_types"].append(task)
        if not entry["success"] or entry["latency_ms"] > 8000:
            if task not in self.learned_patterns["slow_task_types"]:
                self.learned_patterns["slow_task_types"].append(task)

    def broadcast_summary(self) -> Dict:
        """What this node wants to share with the federation."""
        return {
            "node_id": self.node_id,
            "model": self.node_info.get("primary_model", "unknown"),
            "observations": len(self.journal),
            "learned_patterns": self.learned_patterns,
            "strengths_confirmed": self.node_info.get("strengths", []),
            "last_updated": datetime.datetime.utcnow().isoformat(),
        }

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "journal": self.journal[-20:],  # last 20 entries
            "learned_patterns": self.learned_patterns,
        }


# ------------------------------------------------------------------ #
# Federated Context — shared knowledge base across all nodes          #
# ------------------------------------------------------------------ #

class FederatedContext:
    """
    Aggregates learnings from all nodes into a unified routing intelligence base.
    This is the FAME equivalent — each node's observations become collective knowledge.
    """

    def __init__(self):
        self.node_summaries: Dict[str, Dict] = {}
        self.routing_table: Dict[str, str] = {}  # task_type → best_node
        self.consensus_patterns: List[Dict] = []
        self.federation_rounds: int = 0
        self._load()

    def _load(self):
        if FEDERATION_LOG.exists():
            try:
                data = json.loads(FEDERATION_LOG.read_text())
                self.node_summaries = data.get("node_summaries", {})
                self.routing_table = data.get("routing_table", {})
                self.federation_rounds = data.get("federation_rounds", 0)
            except Exception:
                pass

    def save(self):
        FEDERATION_LOG.write_text(json.dumps({
            "last_updated": datetime.datetime.utcnow().isoformat(),
            "federation_rounds": self.federation_rounds,
            "node_summaries": self.node_summaries,
            "routing_table": self.routing_table,
            "consensus_patterns": self.consensus_patterns[-50:],
        }, indent=2))

    def ingest_node_summary(self, summary: Dict):
        node_id = summary["node_id"]
        self.node_summaries[node_id] = summary

    def synthesize(self):
        """
        Build consensus routing table from all node learnings.
        For each task type observed, assign the node with highest
        success rate + lowest latency as the preferred router.
        """
        self.federation_rounds += 1
        task_node_scores: Dict[str, Dict[str, float]] = {}

        for node_id, summary in self.node_summaries.items():
            patterns = summary.get("learned_patterns", {})
            fast_tasks = patterns.get("fast_task_types", [])
            slow_tasks = patterns.get("slow_task_types", [])
            success_rate = patterns.get("success_rate", 1.0)
            avg_latency = patterns.get("avg_latency_ms", 5000)

            for task in fast_tasks:
                if task not in task_node_scores:
                    task_node_scores[task] = {}
                # Score = success_rate * (1 / normalized_latency)
                score = success_rate * (10000 / max(avg_latency, 100))
                task_node_scores[task][node_id] = round(score, 3)

            for task in slow_tasks:
                if task not in task_node_scores:
                    task_node_scores[task] = {}
                task_node_scores[task][node_id] = task_node_scores[task].get(node_id, 0) * 0.5

        # Build routing table — best node per task type
        new_table = {}
        for task, node_scores in task_node_scores.items():
            if node_scores:
                best = max(node_scores.items(), key=lambda x: x[1])
                new_table[task] = best[0]

        self.routing_table = new_table
        self.save()
        return new_table

    def best_node_for(self, task_type: str) -> str:
        """
        Returns the federally-recommended node for a given task type.
        Falls back to rtx5050 (primary) if no consensus exists yet.
        """
        return self.routing_table.get(task_type, "rtx5050")

    def federation_report(self) -> str:
        lines = [
            "=== Federated Node Context Report ===",
            f"Federation rounds: {self.federation_rounds}",
            f"Nodes reporting: {len(self.node_summaries)}/3",
            f"Routing table entries: {len(self.routing_table)}",
            "",
            "Routing table (task → best node):",
        ]
        for task, node in list(self.routing_table.items())[:10]:
            lines.append(f"  • {task} → {node}")
        if not self.routing_table:
            lines.append("  (collecting data — runs after first inference cycle)")
        return "\n".join(lines)

    def status(self) -> str:
        return f"FederatedContext: {self.federation_rounds} rounds | {len(self.routing_table)} routing rules | {len(self.node_summaries)} nodes"


# ------------------------------------------------------------------ #
# Routing Intelligence — wraps gateway routing with federated context #
# ------------------------------------------------------------------ #

class RoutingIntelligence:
    """
    Drop-in upgrade for the sovereign-core GatewayRouter.
    Uses federated context to make smarter routing decisions
    instead of pure load balancing.
    """

    def __init__(self):
        self.context = FederatedContext()
        self.node_memories: Dict[str, NodeMemory] = {
            node_id: NodeMemory(node_id) for node_id in COMPUTE_NODES
        }

    def route(self, task_type: str, prompt: str = "") -> str:
        """
        Returns the recommended node_id for this task.
        Uses federated routing table if available, else falls back to
        model-fit heuristics.
        """
        # Check federated table first
        if self.context.routing_table:
            recommended = self.context.best_node_for(task_type)
            return recommended

        # Fallback heuristics before federation has data
        if any(k in task_type.lower() for k in ["code", "debug", "function", "implement"]):
            return "radeon780m"
        if any(k in task_type.lower() for k in ["reason", "analyze", "plan", "architect"]):
            return "rtx5050"
        return "ryzen7_cpu"

    def record_result(self, node_id: str, task_type: str, latency_ms: float, success: bool):
        """Call after each inference run to feed learnings back into node memory."""
        if node_id in self.node_memories:
            self.node_memories[node_id].record_inference(task_type, latency_ms, success)

    def federate(self):
        """
        Broadcast all node memories to the federated context and synthesize.
        Called at end of each KAIROS cycle to update routing intelligence.
        """
        for node_id, mem in self.node_memories.items():
            self.context.ingest_node_summary(mem.broadcast_summary())
        routing_table = self.context.synthesize()
        return routing_table

    def status(self) -> str:
        return self.context.status()
