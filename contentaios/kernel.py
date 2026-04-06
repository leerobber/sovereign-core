from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import partial
from typing import Awaitable, Callable, Iterable, Optional

from contentaios.types import AuditRecord, KernelEvent, Priority

logger = logging.getLogger(__name__)

EventHandler = Callable[[KernelEvent], Awaitable[None]]
ScheduledFn = Callable[[], Awaitable[None]]


class AuditLog:
    """In-memory audit log with bounded retention."""

    def __init__(self, max_entries: int = 500) -> None:
        self._entries: deque[AuditRecord] = deque(maxlen=max_entries)

    def record(self, actor: str, action: str, detail: dict) -> None:
        self._entries.append(
            AuditRecord(
                timestamp=datetime.now(tz=timezone.utc),
                actor=actor,
                action=action,
                detail=dict(detail),
            )
        )

    def tail(self, count: int = 50) -> list[AuditRecord]:
        return list(self._entries)[-count:]


class MessageBus:
    """Lightweight inter-subsystem publish/subscribe bus."""

    def __init__(self, audit_log: AuditLog) -> None:
        self._subscribers: dict[str, list[tuple[str, EventHandler]]] = defaultdict(list)
        self._audit = audit_log

    def subscribe(self, topic: str, subsystem: str, handler: EventHandler) -> None:
        self._subscribers[topic].append((subsystem, handler))
        self._audit.record(
            actor="kernel",
            action="subscribe",
            detail={"topic": topic, "subsystem": subsystem},
        )

    async def publish(self, event: KernelEvent) -> None:
        targets = list(self._subscribers.get(event.type, []))
        if not targets:
            self._audit.record(
                actor="kernel",
                action="no_subscriber",
                detail={"type": event.type, "trace_id": event.trace_id},
            )
            return

        async def _deliver(subsystem: str, handler: EventHandler) -> None:
            await handler(event)
            self._audit.record(
                actor=subsystem,
                action="handled",
                detail={"type": event.type, "trace_id": event.trace_id},
            )

        results = await asyncio.gather(
            *(_deliver(subsystem, handler) for subsystem, handler in targets),
            return_exceptions=True,
        )
        for (subsystem, _), result in zip(targets, results):
            if isinstance(result, Exception):
                self._audit.record(
                    actor=subsystem,
                    action="handler_failed",
                    detail={"error": str(result), "trace_id": event.trace_id},
                )


class ContentKernel:
    """Master kernel coordinating sensory inputs and subsystems."""

    def __init__(
        self,
        sensory_inputs: Optional[Iterable["SensoryInput"]] = None,
        audit_log: Optional[AuditLog] = None,
    ) -> None:
        self._audit = audit_log or AuditLog()
        self._bus = MessageBus(self._audit)
        self._queue: asyncio.PriorityQueue[
            tuple[int, int, ScheduledFn]
        ] = asyncio.PriorityQueue()
        self._sequence = 0
        self._running = False
        self._scheduler_task: Optional[asyncio.Task[None]] = None
        self._sensory_inputs = list(sensory_inputs or [])
        self._sensor_tasks: list[asyncio.Task[None]] = []

    @property
    def audit_log(self) -> AuditLog:
        return self._audit

    def register_subsystem(self, name: str, topics: Iterable[str], handler: EventHandler) -> None:
        for topic in topics:
            self._bus.subscribe(topic, subsystem=name, handler=handler)
        self._audit.record(
            actor="kernel",
            action="register_subsystem",
            detail={"name": name, "topics": list(topics)},
        )

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler())
        for sensor in self._sensory_inputs:
            task = asyncio.create_task(sensor.start(self.ingest_event))
            self._sensor_tasks.append(task)
            self._audit.record(
                actor="kernel", action="sensor_started", detail={"sensor": sensor.name}
            )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for sensor in self._sensory_inputs:
            await sensor.stop()
            self._audit.record(
                actor="kernel", action="sensor_stopped", detail={"sensor": sensor.name}
            )

        for task in self._sensor_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._sensor_tasks.clear()

        if self._scheduler_task:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
        self._scheduler_task = None

    async def ingest_event(self, event: KernelEvent) -> None:
        """Push a sensory event into the kernel for scheduling."""
        self._audit.record(
            actor=event.source,
            action="ingest",
            detail={"type": event.type, "trace_id": event.trace_id},
        )
        await self._enqueue(partial(self._dispatch_event, event), priority=event.priority)

    async def emit(self, topic: str, payload, *, priority: Priority = Priority.NORMAL) -> None:
        """Publish an inter-subsystem message into the scheduler."""
        event = KernelEvent(source="kernel", type=topic, payload=payload, priority=priority)
        await self.ingest_event(event)

    async def schedule(self, fn: ScheduledFn, *, priority: Priority = Priority.NORMAL) -> None:
        """Schedule an arbitrary coroutine for execution."""
        await self._enqueue(fn, priority=priority)

    async def join(self) -> None:
        """Wait until all scheduled work has been processed."""
        if self._sensory_inputs:
            await asyncio.gather(*(sensor.wait_idle() for sensor in self._sensory_inputs))
        await self._queue.join()

    async def _dispatch_event(self, event: KernelEvent) -> None:
        await self._bus.publish(event)

    async def _enqueue(self, fn: ScheduledFn, *, priority: Priority) -> None:
        self._sequence += 1
        await self._queue.put((int(priority), self._sequence, fn))

    async def _scheduler(self) -> None:
        while self._running:
            priority, _, fn = await self._queue.get()
            try:
                await fn()
                self._audit.record(
                    actor="kernel",
                    action="task_complete",
                    detail={"priority": Priority(priority).name.lower()},
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Kernel task failed: %s", exc)
                self._audit.record(
                    actor="kernel",
                    action="task_failed",
                    detail={"error": str(exc), "priority": priority},
                )
            finally:
                self._queue.task_done()


# Late import to avoid circular dependency for type checking
from contentaios.sensory import SensoryInput  # noqa: E402  isort:skip
