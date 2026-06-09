"""RES-02: Darwin Gödel Machine — Hyper Extension (DGM-H) subsystem.

The :class:`DGMHSubsystem` implements a mocked meta-learning loop.  It
subscribes to the ``kernel.task_complete`` and ``kernel.task_failed``
meta-events emitted by the :class:`~contentaios.kernel.ContentKernel`
scheduler after every dispatched task.

On each signal the subsystem:

1. Updates a lightweight "policy" dictionary to reinforce successful
   behaviours or increase the mutation weight for failed ones.
2. Records a ``self_improvement`` action in the :class:`~contentaios.kernel.AuditLog`.
3. Periodically emits a ``kernel.optimize`` event back into the kernel
   (every *emit_every* signals) to simulate recursive self-improvement.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from contentaios.types import KernelEvent, Priority

if TYPE_CHECKING:  # pragma: no cover
    from contentaios.kernel import AuditLog, ContentKernel

logger = logging.getLogger(__name__)

_DEFAULT_POLICY: dict[str, Any] = {
    "success_weight": 1.0,
    "failure_weight": 1.0,
    "mutation_rate": 0.1,
    "optimization_threshold": 0.7,
}

# Topics emitted by the kernel scheduler that DGM-H reacts to.
TOPIC_TASK_COMPLETE = "kernel.task_complete"
TOPIC_TASK_FAILED = "kernel.task_failed"
# Topic emitted by DGM-H when an optimisation cycle is triggered.
TOPIC_OPTIMIZE = "kernel.optimize"


class DGMHSubsystem:
    """Darwin Gödel Machine — Hyper Extension subsystem.

    Parameters
    ----------
    kernel:
        The :class:`~contentaios.kernel.ContentKernel` instance to attach to.
    audit_log:
        The :class:`~contentaios.kernel.AuditLog` for recording
        ``self_improvement`` events.
    emit_every:
        How many signals (task completions + failures combined) must be
        received before an ``kernel.optimize`` event is emitted back into
        the kernel.  Defaults to 10.
    """

    def __init__(
        self,
        kernel: "ContentKernel",
        audit_log: "AuditLog",
        *,
        emit_every: int = 10,
    ) -> None:
        self._kernel = kernel
        self._audit = audit_log
        self._emit_every = max(1, emit_every)
        self._policy: dict[str, Any] = dict(_DEFAULT_POLICY)
        self._successes: int = 0
        self._failures: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def attach(self) -> None:
        """Register this subsystem with the kernel message bus."""
        self._kernel.register_subsystem(
            "dgm_h",
            [TOPIC_TASK_COMPLETE, TOPIC_TASK_FAILED],
            self._handle_signal,
        )

    @property
    def policy(self) -> dict[str, Any]:
        """A copy of the current meta-learning policy."""
        return dict(self._policy)

    @property
    def signal_count(self) -> int:
        """Total number of signals (successes + failures) observed."""
        return self._successes + self._failures

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _handle_signal(self, event: KernelEvent) -> None:
        success = event.type == TOPIC_TASK_COMPLETE
        if success:
            self._successes += 1
        else:
            self._failures += 1

        self._mutate_policy(success=success)

        if self.signal_count % self._emit_every == 0:
            await self._emit_optimization()

    def _mutate_policy(self, *, success: bool) -> None:
        """Adjust policy weights based on the observed outcome."""
        rate = self._policy["mutation_rate"]
        if success:
            self._policy["success_weight"] = min(
                2.0, self._policy["success_weight"] + rate
            )
        else:
            self._policy["failure_weight"] = min(
                2.0, self._policy["failure_weight"] + rate
            )

        self._audit.record(
            actor="dgm_h",
            action="self_improvement",
            detail={
                "policy": dict(self._policy),
                "success": success,
                "successes": self._successes,
                "failures": self._failures,
            },
        )
        logger.debug(
            "DGM-H policy mutated (success=%s): %s", success, self._policy
        )

    async def _emit_optimization(self) -> None:
        """Emit an optimised task back into the kernel."""
        self._audit.record(
            actor="dgm_h",
            action="self_improvement",
            detail={
                "event": "optimization_emitted",
                "policy": dict(self._policy),
                "signal_count": self.signal_count,
            },
        )
        await self._kernel.emit(
            TOPIC_OPTIMIZE,
            payload={
                "policy": dict(self._policy),
                "signal_count": self.signal_count,
            },
            priority=Priority.LOW,
        )
        logger.info(
            "DGM-H emitted optimization task after %d signals", self.signal_count
        )
