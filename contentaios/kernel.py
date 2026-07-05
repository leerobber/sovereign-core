from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Iterable, Optional

from contentaios.types import AuditRecord, KernelEvent, Priority

if TYPE_CHECKING:
    from contentaios.kernel import VerificationMiddleware  # type: ignore[misc]

logger = logging.getLogger(__name__)

EventHandler = Callable[[KernelEvent], Awaitable[None]]
ScheduledFn = Callable[[], Awaitable[None]]


class FileAuditSink:
    """JSONL file sink for AuditLog persistence."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: AuditRecord) -> None:
        line = json.dumps({
            "timestamp": record.timestamp.isoformat(),
            "actor": record.actor,
            "action": record.action,
            "detail": record.detail,
        }, default=str)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class MetricsAuditSink:
    """In-memory metrics counter sink keyed by 'actor.action'."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)

    def record(self, actor: str, action: str, detail: dict) -> None:
        key = f"{actor}.{action}"
        self._counts[key] += 1

    def snapshot(self) -> dict[str, int]:
        return dict(self._counts)


class VerificationMiddleware:
    """Drops unsafe events (None or non-dict payload) and audits the decision."""

    def __init__(self, audit_log: "AuditLog") -> None:
        self._audit = audit_log

    def verify(self, event: KernelEvent) -> bool:
        if event.payload is None:
            self._audit.record(
                "verification",
                "verification_failed",
                {"type": event.type, "trace_id": event.trace_id, "reason": "null_payload"},
            )
            return False
        if not isinstance(event.payload, dict):
            self._audit.record(
                "verification",
                "verification_failed",
                {"type": event.type, "trace_id": event.trace_id, "reason": "non_dict_payload"},
            )
            return False
        self._audit.record(
            "verification",
            "verification_passed",
            {"type": event.type, "trace_id": event.trace_id},
        )
        return True


class AuditLog:
    """In-memory audit log with bounded retention and pluggable sinks."""

    def __init__(
        self,
        max_entries: int = 500,
        sinks: Optional[Iterable] = None,
    ) -> None:
        """
        Initialize the audit log with a bounded in-memory store of audit records.

        Parameters:
            max_entries (int): Maximum number of audit records to retain in memory.
            sinks (Optional[Iterable]): Optional list of sink objects. Sinks may implement
                .record(actor, action, detail) or .write(AuditRecord).
        """
        self._entries: deque[AuditRecord] = deque(maxlen=max_entries)
        self._sinks: list = list(sinks or [])

    def record(self, actor: str, action: str, detail: dict) -> None:
        """
        Record an audit entry in the in-memory audit log.

        Parameters:
            actor (str): Identifier of the actor responsible for the action.
            action (str): Short name of the action performed.
            detail (dict): Additional metadata for the entry; a shallow copy of this dict is stored.

        Notes:
            The stored record is timestamped with the current UTC time.
            All configured sinks are notified.
        """
        rec = AuditRecord(
            timestamp=datetime.now(tz=timezone.utc),
            actor=actor,
            action=action,
            detail=dict(detail),
        )
        self._entries.append(rec)
        for sink in self._sinks:
            try:
                if hasattr(sink, "record"):
                    sink.record(actor, action, detail)
                elif hasattr(sink, "write"):
                    sink.write(rec)
            except Exception:
                # never let a sink break the kernel
                pass

    def tail(self, count: int = 50) -> list[AuditRecord]:
        """
        Return the most recent audit records up to the specified count.
        
        Parameters:
            count (int): Maximum number of recent records to return.
        
        Returns:
            list[AuditRecord]: A list of up to `count` most recent audit records in chronological order (oldest to newest within the returned slice).
        """
        return list(self._entries)[-count:]

    def flush_to_file(self, path: Path | str) -> None:
        """Write current in-memory entries as JSONL to the given path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for rec in self._entries:
                line = json.dumps({
                    "timestamp": rec.timestamp.isoformat(),
                    "actor": rec.actor,
                    "action": rec.action,
                    "detail": rec.detail,
                }, default=str)
                f.write(line + "\n")


class MessageBus:
    """Lightweight inter-subsystem publish/subscribe bus."""

    def __init__(self, audit_log: AuditLog, verification: Optional["VerificationMiddleware"] = None) -> None:
        """
        Create a MessageBus that routes events to subscribed handlers and records audit entries.
        
        Parameters:
            audit_log (AuditLog): AuditLog instance used to record subscription, publish, delivery, and handler-failure events.
            verification (Optional[VerificationMiddleware]): If provided, every published event is
                verified before delivery. Failed verification short-circuits delivery.
        """
        self._subscribers: dict[str, list[tuple[str, EventHandler, dict]]] = defaultdict(list)
        self._audit = audit_log
        self._verification = verification

    def subscribe(
        self,
        topic: str,
        subsystem: str,
        handler: EventHandler,
        *,
        options: Optional[dict] = None,
    ) -> None:
        """
        Register a handler to receive events published to the given topic.
        
        Parameters:
        	topic (str): The topic name to subscribe to.
        	subsystem (str): Logical name of the subscribing subsystem; recorded as the actor in audit.
        	handler (EventHandler): Async callable that will be invoked with the published event.
            options: Delivery options for retries/timeouts.
        
        Notes:
        	An audit record with actor "kernel" and action "subscribe" is created containing the topic and subsystem.
        """
        self._subscribers[topic].append((subsystem, handler, options or {}))
        self._audit.record(
            actor="kernel",
            action="subscribe",
            detail={"topic": topic, "subsystem": subsystem},
        )

    async def publish(self, event: KernelEvent) -> None:
        """
        Publish a KernelEvent to all subscribers of its topic and record audit entries for delivery outcomes.
        
        Publishes the given event to every handler subscribed to event.type. If no subscribers exist, records an audit entry with action "no_subscriber". For each successful delivery records an audit entry with action "handled" (actor set to the subsystem); if a handler raises an exception records an audit entry with action "handler_failed" including the error string and the event's trace_id.
        
        Parameters:
            event (KernelEvent): The event to publish; its `type` determines recipient subscriptions and its `trace_id` is included in audit records.
        """
        if self._verification is not None:
            if not self._verification.verify(event):
                return  # blocked by verification middleware

        targets = list(self._subscribers.get(event.type, []))
        if not targets:
            self._audit.record(
                actor="kernel",
                action="no_subscriber",
                detail={"type": event.type, "trace_id": event.trace_id},
            )
            return

        async def _deliver(subsystem: str, handler: EventHandler, options: dict) -> None:
            """
            Deliver with optional timeout + retry wrapper.
            """
            timeout_s = options.get("timeout_s")
            max_retries = options.get("max_retries", 0) or 0
            backoff = options.get("retry_backoff_s", 0.0) or 0.0

            attempt = 0
            while True:
                attempt += 1
                try:
                    if timeout_s and timeout_s > 0:
                        await asyncio.wait_for(handler(event), timeout=timeout_s)
                    else:
                        await handler(event)
                    self._audit.record(
                        actor=subsystem,
                        action="handled",
                        detail={"type": event.type, "trace_id": event.trace_id},
                    )
                    if attempt > 1:
                        self._audit.record(
                            actor=subsystem,
                            action="retry_success",
                            detail={"type": event.type, "attempt": attempt, "trace_id": event.trace_id},
                        )
                    return
                except asyncio.TimeoutError:
                    self._audit.record(
                        actor=subsystem,
                        action="handler_timeout",
                        detail={"type": event.type, "attempt": attempt, "trace_id": event.trace_id},
                    )
                    if attempt > max_retries:
                        self._audit.record(
                            actor=subsystem,
                            action="handler_failed",
                            detail={"error": "timeout", "trace_id": event.trace_id},
                        )
                        return
                except Exception as exc:
                    self._audit.record(
                        actor=subsystem,
                        action="handler_failed",
                        detail={"error": str(exc), "trace_id": event.trace_id, "attempt": attempt},
                    )
                    if attempt > max_retries:
                        return
                if backoff > 0:
                    await asyncio.sleep(backoff)

        results = await asyncio.gather(
            *(_deliver(subsystem, handler, opts) for subsystem, handler, opts in targets),
            return_exceptions=True,
        )
        # Note: the wrapper already records failures/timeouts; gather exceptions would be unexpected now
        for item, result in zip(targets, results):
            if isinstance(result, Exception) and not isinstance(result, asyncio.TimeoutError):
                subsystem = item[0]
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
        verification: Optional["VerificationMiddleware"] = None,
    ) -> None:
        """
        Initialize the ContentKernel, setting up its audit log, message bus, priority queue, and sensor/task state.
        
        Parameters:
            sensory_inputs (Optional[Iterable[SensoryInput]]): Optional iterable of SensoryInput instances.
            audit_log (Optional[AuditLog]): Optional AuditLog to record kernel and subsystem events; a new AuditLog is created if not provided.
            verification (Optional[VerificationMiddleware]): Optional verification middleware for event payloads.
        """
        self._audit = audit_log or AuditLog()
        self._verification = verification
        self._bus = MessageBus(self._audit, verification=verification)
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
        """
        Access the kernel's audit log.
        
        Returns:
            The AuditLog instance used by this kernel.
        """
        return self._audit

    def register_subsystem(
        self,
        name: str,
        topics: Iterable[str],
        handler: EventHandler,
        *,
        timeout_s: Optional[float] = None,
        max_retries: int = 0,
        retry_backoff_s: float = 0.0,
    ) -> None:
        """
        Register a subsystem by subscribing its handler to each listed topic and recording the registration in the audit log.

        Extra kwargs are forwarded for resilient delivery wrappers.
        """
        options = {
            "timeout_s": timeout_s,
            "max_retries": max_retries,
            "retry_backoff_s": retry_backoff_s,
        }
        for topic in topics:
            self._bus.subscribe(topic, subsystem=name, handler=handler, options=options)
        self._audit.record(
            actor="kernel",
            action="register_subsystem",
            detail={"name": name, "topics": list(topics)},
        )

    async def start(self) -> None:
        """
        Start the kernel by launching its scheduler and all configured sensory input tasks.
        
        If the kernel is already running, this is a no-op. On start, a background scheduler task is created, each sensory input is started (their start call is scheduled as a task), and an audit record is written for each sensor indicating it was started.
        """
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
        """
        Stop the kernel and shut down its sensor and scheduler tasks.
        
        If the kernel is not running this method returns immediately. Otherwise it sets the running flag to False, stops each configured sensory input (awaiting each sensor's stop method) and records an audit entry for each stopped sensor. It then cancels any outstanding sensor tasks and awaits their completion (suppressing asyncio.CancelledError), clears the internal sensor task list, cancels the scheduler task if present and awaits it (suppressing asyncio.CancelledError), and sets the scheduler task reference to None.
        """
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
        """
        Ingests a KernelEvent into the kernel and schedules its dispatch.
        
        Records an audit entry with the event's source, type, and trace_id, then enqueues the event's dispatch using the event's priority.
        
        Parameters:
            event (KernelEvent): The event produced by a sensory input or the kernel to be scheduled for delivery to subscribers.
        """
        self._audit.record(
            actor=event.source,
            action="ingest",
            detail={"type": event.type, "trace_id": event.trace_id},
        )
        await self._enqueue(partial(self._dispatch_event, event), priority=event.priority)

    async def emit(self, topic: str, payload, *, priority: Priority = Priority.NORMAL) -> None:
        """
        Emit a kernel-originated event into the kernel's ingestion and scheduling pipeline.
        
        Parameters:
        	topic (str): Topic/type of the event to publish.
        	payload: Arbitrary payload carried with the event.
        	priority (Priority): Scheduling priority that determines ordering in the kernel's queue; higher priority values are processed before lower ones.
        """
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
        """
        Dispatches a KernelEvent to the message bus for delivery to subscribed handlers.
        
        Parameters:
            event (KernelEvent): The kernel event to publish to subscribers.
        """
        await self._bus.publish(event)

    async def _enqueue(self, fn: ScheduledFn, *, priority: Priority) -> None:
        """
        Enqueue a scheduled coroutine callable for later execution at the specified priority.
        
        Parameters:
        	fn (ScheduledFn): A zero-argument coroutine function to be executed by the scheduler.
        	priority (Priority): Priority level that determines ordering in the queue; items with the same priority are ordered by insertion time.
        """
        self._sequence += 1
        await self._queue.put((int(priority), self._sequence, fn))

    async def _scheduler(self) -> None:
        """
        Consume and execute scheduled coroutine tasks from the kernel's priority queue until the kernel is stopped.

        After successful execution of a scheduled fn we emit kernel.task_complete (so attached
        subsystems like DGM-H can react). Failures emit kernel.task_failed.
        """
        while self._running:
            priority, _, fn = await self._queue.get()
            try:
                await fn()
                self._audit.record(
                    actor="kernel",
                    action="task_complete",
                    detail={"priority": Priority(priority).name.lower()},
                )
                # Publish meta directly via bus (not re-enqueue) so DGMH/observers react
                # without risking feedback into the priority queue during this step.
                meta_event = KernelEvent(
                    source="kernel",
                    type="kernel.task_complete",
                    payload={"priority": Priority(priority).name.lower()},
                    priority=Priority.LOW,
                )
                await self._bus.publish(meta_event)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Kernel task failed: %s", exc)
                self._audit.record(
                    actor="kernel",
                    action="task_failed",
                    detail={"error": str(exc), "priority": priority},
                )
                meta_event = KernelEvent(
                    source="kernel",
                    type="kernel.task_failed",
                    payload={"error": str(exc), "priority": priority},
                    priority=Priority.LOW,
                )
                await self._bus.publish(meta_event)
            finally:
                self._queue.task_done()


# Late import to avoid circular dependency for type checking
from contentaios.sensory import SensoryInput  # noqa: E402  isort:skip
