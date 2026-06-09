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
        """
        Initialize the audit log with a bounded in-memory store of audit records.
        
        Parameters:
            max_entries (int): Maximum number of audit records to retain in memory; when the capacity is exceeded, the oldest records are discarded.
        """
        self._entries: deque[AuditRecord] = deque(maxlen=max_entries)

    def record(self, actor: str, action: str, detail: dict) -> None:
        """
        Record an audit entry in the in-memory audit log.
        
        Parameters:
            actor (str): Identifier of the actor responsible for the action.
            action (str): Short name of the action performed.
            detail (dict): Additional metadata for the entry; a shallow copy of this dict is stored.
        
        Notes:
            The stored record is timestamped with the current UTC time.
        """
        self._entries.append(
            AuditRecord(
                timestamp=datetime.now(tz=timezone.utc),
                actor=actor,
                action=action,
                detail=dict(detail),
            )
        )

    def tail(self, count: int = 50) -> list[AuditRecord]:
        """
        Return the most recent audit records up to the specified count.
        
        Parameters:
            count (int): Maximum number of recent records to return.
        
        Returns:
            list[AuditRecord]: A list of up to `count` most recent audit records in chronological order (oldest to newest within the returned slice).
        """
        return list(self._entries)[-count:]


class MessageBus:
    """Lightweight inter-subsystem publish/subscribe bus."""

    def __init__(self, audit_log: AuditLog) -> None:
        """
        Create a MessageBus that routes events to subscribed handlers and records audit entries.
        
        Parameters:
            audit_log (AuditLog): AuditLog instance used to record subscription, publish, delivery, and handler-failure events.
        """
        self._subscribers: dict[str, list[tuple[str, EventHandler]]] = defaultdict(list)
        self._audit = audit_log

    def subscribe(self, topic: str, subsystem: str, handler: EventHandler) -> None:
        """
        Register a handler to receive events published to the given topic.
        
        Parameters:
        	topic (str): The topic name to subscribe to.
        	subsystem (str): Logical name of the subscribing subsystem; recorded as the actor in audit.
        	handler (EventHandler): Async callable that will be invoked with the published event.
        
        Notes:
        	An audit record with actor "kernel" and action "subscribe" is created containing the topic and subsystem.
        """
        self._subscribers[topic].append((subsystem, handler))
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
        targets = list(self._subscribers.get(event.type, []))
        if not targets:
            self._audit.record(
                actor="kernel",
                action="no_subscriber",
                detail={"type": event.type, "trace_id": event.trace_id},
            )
            return

        async def _deliver(subsystem: str, handler: EventHandler) -> None:
            """
            Deliver the enclosing `event` to a subsystem handler and record an audit entry indicating it was handled.
            
            Parameters:
                subsystem (str): Name of the subsystem invoked as the actor in the audit record.
                handler (EventHandler): Async callable that will be awaited with the current `event`.
            """
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
        """
        Initialize the ContentKernel, setting up its audit log, message bus, priority queue, and sensor/task state.
        
        Parameters:
            sensory_inputs (Optional[Iterable[HazardousInput]]): Optional iterable of SensoryInput instances to be managed by the kernel. The iterable is copied into an internal list.
            audit_log (Optional[AuditLog]): Optional AuditLog to record kernel and subsystem events; a new AuditLog is created if not provided.
        """
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
        """
        Access the kernel's audit log.
        
        Returns:
            The AuditLog instance used by this kernel.
        """
        return self._audit

    def register_subsystem(self, name: str, topics: Iterable[str], handler: EventHandler) -> None:
        """
        Register a subsystem by subscribing its handler to each listed topic and recording the registration in the audit log.
        
        Parameters:
            name (str): Subsystem identifier used as the subscriber name in the message bus and as the audit actor.
            topics (Iterable[str]): Iterable of topic names to subscribe the handler to; the topics are saved in the audit record as a list.
            handler (EventHandler): Async callable that will be invoked for events published to the subscribed topics.
        """
        for topic in topics:
            self._bus.subscribe(topic, subsystem=name, handler=handler)
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
        
        For each dequeued task, awaits the scheduled coroutine and on success records an audit entry with action `task_complete` including the task priority name. If the task raises an exception, records an audit entry with action `task_failed` including the error string and priority. Always marks the queue item as done.
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
