"""
KAIROS Upgrade #6: Dynamic Agent Priority Scoring
Inspired by Raytheon Coyote Block 3NK drone swarm defeat system (2026)

Coyote's key innovation: real-time priority scoring of targets in a swarm.
Not "which drone is closest" but "improvement_potential ÷ intercept_cost"
— value delivered per unit of resource spent.

Applied to KAIROS: the 7 SAGE agents don't run in fixed order.
Each agent task gets scored dynamically:
  priority = (improvement_potential * urgency) / (compute_cost * risk)

High-value, cheap, urgent tasks run first.
Low-yield, expensive tasks get deferred or dropped from the current cycle.

This stops the system from wasting GPU cycles on marginal proposals
when better ones are queued behind them.

Architecture:
  TaskProfile       — describes a pending agent task
  PriorityScorer    — computes dynamic priority score
  PriorityQueue     — orders all 7 agent tasks by score
  AgentScheduler    — dispatches tasks in priority order, respects resource limits
"""

import json
import datetime
import heapq
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

SPAWNER_DIR = Path(__file__).parent
PRIORITY_LOG = SPAWNER_DIR / "priority_log.json"


# ------------------------------------------------------------------ #
# Task Profile                                                         #
# ------------------------------------------------------------------ #

class TaskProfile:
    """
    Describes a single agent task pending in the KAIROS queue.
    All values are 0.0–1.0 normalized.
    """

    def __init__(
        self,
        agent_name: str,
        task_description: str,
        improvement_potential: float = 0.5,   # how much this could improve the system
        compute_cost: float = 0.5,             # GPU/CPU cost relative to max cycle
        urgency: float = 0.5,                  # time-sensitive? recent failure trigger?
        risk: float = 0.3,                     # chance of destabilizing something
        novelty: float = 0.5,                  # how different from recent proposals
    ):
        self.agent_name = agent_name
        self.task_description = task_description
        self.improvement_potential = max(0.01, min(1.0, improvement_potential))
        self.compute_cost = max(0.01, min(1.0, compute_cost))
        self.urgency = max(0.01, min(1.0, urgency))
        self.risk = max(0.01, min(1.0, risk))
        self.novelty = max(0.01, min(1.0, novelty))
        self.created_at = datetime.datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        return {
            "agent": self.agent_name,
            "task": self.task_description[:100],
            "improvement_potential": self.improvement_potential,
            "compute_cost": self.compute_cost,
            "urgency": self.urgency,
            "risk": self.risk,
            "novelty": self.novelty,
        }


# ------------------------------------------------------------------ #
# Priority Scorer — the Coyote formula                                #
# ------------------------------------------------------------------ #

class PriorityScorer:
    """
    Coyote-inspired scoring: value_delivered / resource_spent
    With urgency as a multiplier and risk as a divisor.

    Formula:
      priority = (improvement_potential * urgency * novelty) / (compute_cost * (1 + risk))

    This means:
      - High potential + low cost = runs first
      - Low potential + high cost = runs last or deferred
      - High risk = penalized regardless of potential
      - Urgency spikes = immediate priority boost
      - Novel proposals get a slight boost over repetitive ones
    """

    def score(self, task: TaskProfile) -> float:
        numerator = task.improvement_potential * task.urgency * (0.5 + 0.5 * task.novelty)
        denominator = task.compute_cost * (1.0 + task.risk)
        return round(numerator / denominator, 4)

    def score_all(self, tasks: List[TaskProfile]) -> List[Tuple[float, TaskProfile]]:
        """Returns [(score, task)] sorted highest first."""
        scored = [(self.score(t), t) for t in tasks]
        scored.sort(key=lambda x: -x[0])
        return scored

    def explain(self, task: TaskProfile) -> str:
        score = self.score(task)
        return (
            f"{task.agent_name} | score={score:.4f} | "
            f"potential={task.improvement_potential:.2f} "
            f"cost={task.compute_cost:.2f} "
            f"urgency={task.urgency:.2f} "
            f"risk={task.risk:.2f} "
            f"novelty={task.novelty:.2f}"
        )


# ------------------------------------------------------------------ #
# Priority Queue                                                        #
# ------------------------------------------------------------------ #

class PriorityQueue:
    """Min-heap based priority queue for agent tasks (inverted for max-priority)."""

    def __init__(self):
        self._heap: List[Tuple[float, int, TaskProfile]] = []
        self._counter = 0  # tiebreaker
        self.scorer = PriorityScorer()

    def push(self, task: TaskProfile):
        score = self.scorer.score(task)
        # Negate score for max-heap behavior
        heapq.heappush(self._heap, (-score, self._counter, task))
        self._counter += 1

    def pop(self) -> Optional[TaskProfile]:
        if not self._heap:
            return None
        _, _, task = heapq.heappop(self._heap)
        return task

    def peek(self) -> Optional[Tuple[float, TaskProfile]]:
        if not self._heap:
            return None
        neg_score, _, task = self._heap[0]
        return (-neg_score, task)

    def all_ranked(self) -> List[Tuple[float, TaskProfile]]:
        """Returns all tasks sorted by priority (non-destructive)."""
        items = [(-neg_s, t) for neg_s, _, t in self._heap]
        items.sort(key=lambda x: -x[0])
        return items

    def size(self) -> int:
        return len(self._heap)


# ------------------------------------------------------------------ #
# Agent Scheduler                                                       #
# ------------------------------------------------------------------ #

class AgentScheduler:
    """
    Manages the 7 SAGE agent task queue with dynamic priority scoring.
    Replaces fixed-order execution with Coyote-style value-optimized dispatch.
    """

    # Resource budget per KAIROS cycle (normalized units)
    CYCLE_COMPUTE_BUDGET = 5.0
    MIN_PRIORITY_THRESHOLD = 0.3   # tasks below this get deferred

    def __init__(self):
        self.scorer = PriorityScorer()
        self.queue = PriorityQueue()
        self.history: List[Dict] = []
        self.deferred: List[TaskProfile] = []
        self._load()

    def _load(self):
        if PRIORITY_LOG.exists():
            try:
                data = json.loads(PRIORITY_LOG.read_text())
                self.history = data.get("history", [])
            except Exception:
                pass

    def save(self):
        PRIORITY_LOG.write_text(json.dumps({
            "last_updated": datetime.datetime.utcnow().isoformat(),
            "total_scheduled": len(self.history),
            "history": self.history[-100:],
        }, indent=2))

    def enqueue(self, task: TaskProfile):
        self.queue.push(task)

    def enqueue_all(self, tasks: List[TaskProfile]):
        for t in tasks:
            self.queue.push(t)

    def schedule_cycle(self) -> Tuple[List[TaskProfile], List[TaskProfile]]:
        """
        Plan the next execution cycle within the compute budget.
        Returns (tasks_to_run, tasks_deferred).
        """
        ranked = self.queue.all_ranked()
        to_run = []
        deferred = []
        budget_used = 0.0

        for score, task in ranked:
            if score < self.MIN_PRIORITY_THRESHOLD:
                deferred.append(task)
                continue
            if budget_used + task.compute_cost > self.CYCLE_COMPUTE_BUDGET:
                deferred.append(task)
                continue
            to_run.append(task)
            budget_used += task.compute_cost

        self.deferred = deferred

        # Log the schedule decision
        schedule_record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "tasks_scheduled": len(to_run),
            "tasks_deferred": len(deferred),
            "budget_used": round(budget_used, 2),
            "schedule": [
                {"agent": t.agent_name, "score": self.scorer.score(t)}
                for t in to_run
            ],
        }
        self.history.append(schedule_record)
        self.save()

        return to_run, deferred

    def build_default_queue(self) -> "AgentScheduler":
        """
        Load the default 7 SAGE agent tasks with heuristic priority estimates.
        Called at the start of each KAIROS cycle.
        """
        from .dynamic_priority import TaskProfile  # self-reference safe

        default_tasks = [
            TaskProfile("Architect", "Design structural improvements to sovereign-core modules",
                        improvement_potential=0.75, compute_cost=0.6, urgency=0.5, risk=0.3, novelty=0.6),
            TaskProfile("Researcher", "Import latest frontier AI techniques into the stack",
                        improvement_potential=0.85, compute_cost=0.4, urgency=0.7, risk=0.2, novelty=0.9),
            TaskProfile("Coder", "Implement concrete code improvements identified by Architect",
                        improvement_potential=0.7, compute_cost=0.7, urgency=0.6, risk=0.35, novelty=0.5),
            TaskProfile("Analyst", "Identify metric bottlenecks and efficiency gaps",
                        improvement_potential=0.65, compute_cost=0.3, urgency=0.4, risk=0.1, novelty=0.5),
            TaskProfile("Monitor", "Audit system health and improve safety mechanisms",
                        improvement_potential=0.6, compute_cost=0.25, urgency=0.8, risk=0.05, novelty=0.4),
            TaskProfile("Distiller", "Compress and synthesize GhostMemory learnings",
                        improvement_potential=0.55, compute_cost=0.2, urgency=0.3, risk=0.05, novelty=0.4),
            TaskProfile("Evolver", "Improve the KAIROS loop mechanics themselves",
                        improvement_potential=0.95, compute_cost=0.8, urgency=0.5, risk=0.45, novelty=0.8),
        ]
        self.enqueue_all(default_tasks)
        return self

    def print_schedule(self):
        """Print the current priority ranking for all queued tasks."""
        ranked = self.queue.all_ranked()
        print(f"\n{'='*55}")
        print(f"  Dynamic Agent Priority Schedule ({len(ranked)} tasks)")
        print(f"  Budget: {self.CYCLE_COMPUTE_BUDGET} units")
        print(f"{'='*55}")
        budget = 0.0
        for i, (score, task) in enumerate(ranked):
            fits = budget + task.compute_cost <= self.CYCLE_COMPUTE_BUDGET
            above_threshold = score >= self.MIN_PRIORITY_THRESHOLD
            status = "✅ RUN" if fits and above_threshold else "⏸  DEFER"
            print(f"  {i+1}. {task.agent_name:<12} score={score:.4f} cost={task.compute_cost:.2f} → {status}")
            if fits and above_threshold:
                budget += task.compute_cost
        print(f"  Total compute used: {budget:.2f}/{self.CYCLE_COMPUTE_BUDGET}")
        print(f"{'='*55}\n")

    def status(self) -> str:
        total = len(self.history)
        return f"AgentScheduler: {total} cycles scheduled | queue size: {self.queue.size()}"
