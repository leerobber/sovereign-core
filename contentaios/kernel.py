from __future__ import annotations

import asyncio
import contextlib
import json as _json
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import partial
from pathlib import Path as _Path
from typing import Awaitable, Callable, Iterable, List, Optional

from contentaios.types import AuditRecord, KernelEvent, Priority

logger = logging.getLogger(__name__)

EventHandler = Callable[[KernelEvent], Awaitable[None]]
ScheduledFn = Callable[[], Awaitable[None]]

_TOPIC_TASK_COMPLETE = "kernel.task_complete"
_TOPIC_TASK_FAILED = "kernel.task_failed"


# ---------------------------------------------------------------------------
# Audit sinks
# ---------------------------------------------------------------------------

class FileAuditSink:
    """Writes each audit record as a JSON line to a file."""

    def __init__(self, path) -> None:
        self._path = _Path(path)

    def write(self, record: AuditRecord) -> None:
        entry = {
            "timestamp": record.timestamp.isoformat(),
            "actor": record.actor,
            "action": record.action,
            "detail": record.detail,
        }
        with open(self._path, "a") as f:
            f.write(_json.dumps(entry) + "\n")


class MetricsAuditSink:
    """Counts occurrences of each actor.action pair."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def write(self, record: AuditRecord) -> None:
        key = f"{record.actor}.{record.action}"
        self._counts[key] = self._counts.get(key, 0) + 1

    def snapshot(self) -> dict[str, int]:
        return dict(self._counts)


# ---------------------------------------------------------------------------
# Verification middleware
# ---------------------------------------------------------------------------

class VerificationMiddleware:
    """Rejects events whose payload is not a non-None dict."""

    def __init__(self, audit_log: "AuditLog") -> None:
        self._audit = audit_log

    def verify(self, event: KernelEvent) -> bool:
        if event.payload is None or not isinstance(event.payload, dict):
            self._audit.record(
                actor="verification",
                action="verification_failed",
                detail={"type": event.type, "trace_id": event.trace_id},
            )
            return False
        self._audit.record(
            actor="verification",
            action="verification_passed",
            detail={"type": event.type, "trace_id": event.trace_id},
        )
        return True


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

class AuditLog:
    """In-memory audit log with bounded retention and optional sinks."""

    def __init__(self, max_entries: int = 500, sinks: Optional[List] = None) -> None:
        self._entries: deque[AuditRecord] = deque(maxlen=max_entries)
        self._sinks: list = list(sinks or [])

    def record(self, actor: str, action: str, detail: dict) -> None:
        r = AuditRecord(
            timestamp=datetime.now(tz=timezone.utc),
            actor=actor,
            action=action,
            detail=dict(detail),
        )
        self._entries.append(r)
        for sink in self._sinks:
            sink.write(r)

    def tail(self, count: int = 50) -> list[AuditRecord]:
        return list(self._entries)[-count:]

    def flush_to_file(self, path) -> None:
        """Write all retained records to *path* as JSONL."""
        with open(_Path(path), "w") as f:
            for record in self._entries:
                entry = {
                    "timestamp": record.timestamp.isoformat(),
                    "actor": record.actor,
                    "action": record.action,
                    "detail": record.detail,
                }
                f.write(_json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Message bus
# ---------------------------------------------------------------------------

_SubEntry = tuple  # (subsystem, handler, timeout_s, max_retries, retry_backoff_s)


class MessageBus:
    """Lightweight inter-subsystem publish/subscribe bus with optional retry."""

    def __init__(
        self,
        audit_log: AuditLog,
        verification: Optional[VerificationMiddleware] = None,
    ) -> None:
        self._subscribers: dict[str, list[_SubEntry]] = defaultdict(list)
        self._audit = audit_log
        self._verification = verification

    def subscribe(
        self,
        topic: str,
        subsystem: str,
        handler: EventHandler,
        timeout_s: Optional[float] = None,
        max_retries: int = 0,
        retry_backoff_s: float = 0.1,
    ) -> None:
        self._subscribers[topic].append(
            (subsystem, handler, timeout_s, max_retries, retry_backoff_s)
        )
        self._audit.record(
            actor="kernel",
            action="subscribe",
            detail={"topic": topic, "subsystem": subsystem},
        )

    async def publish(self, event: KernelEvent) -> None:
        # Verification gate
        if self._verification is not None:
            if not self._verification.verify(event):
                return

        targets = list(self._subscribers.get(event.type, []))
        if not targets:
            self._audit.record(
                actor="kernel",
                action="no_subscriber",
                detail={"type": event.type, "trace_id": event.trace_id},
            )
            return

        async def _deliver(
            subsystem: str,
            handler: EventHandler,
            timeout_s: Optional[float],
            max_retries: int,
            retry_backoff_s: float,
        ) -> None:
            for attempt in range(max_retries + 1):
                try:
                    if timeout_s is not None:
                        await asyncio.wait_for(handler(event), timeout=timeout_s)
                    else:
                        await handler(event)

                    if attempt > 0:
                        self._audit.record(
                            actor=subsystem,
                            action="retry_success",
                            detail={"type": event.type, "trace_id": event.trace_id, "attempt": attempt},
                        )
                    else:
                        self._audit.record(
                            actor=subsystem,
                            action="handled",
                            detail={"type": event.type, "trace_id": event.trace_id},
                        )
                    return

                except asyncio.TimeoutError:
                    self._audit.record(
                        actor=subsystem,
                        action="handler_timeout",
                        detail={"type": event.type, "trace_id": event.trace_id, "attempt": attempt},
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(retry_backoff_s)

                except Exception as exc:
                    self._audit.record(
                        actor=subsystem,
                        action="handler_failed",
                        detail={"error": str(exc), "trace_id": event.trace_id, "attempt": attempt},
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(retry_backoff_s)

        await asyncio.gather(
            *(
                _deliver(subsystem, handler, timeout_s, max_retries, retry_backoff_s)
                for subsystem, handler, timeout_s, max_retries, retry_backoff_s in targets
            ),
            return_exceptions=True,
        )


# ---------------------------------------------------------------------------
# Content kernel
# ---------------------------------------------------------------------------

class ContentKernel:
    """Master kernel coordinating sensory inputs and subsystems."""

    def __init__(
        self,
        sensory_inputs: Optional[Iterable["SensoryInput"]] = None,
        audit_log: Optional[AuditLog] = None,
        verification: Optional[VerificationMiddleware] = None,
    ) -> None:
        self._audit = audit_log or AuditLog()
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
        return self._audit

    def register_subsystem(
        self,
        name: str,
        topics: Iterable[str],
        handler: EventHandler,
        timeout_s: Optional[float] = None,
        max_retries: int = 0,
        retry_backoff_s: float = 0.1,
    ) -> None:
        topics_list = list(topics)
        for topic in topics_list:
            self._bus.subscribe(
                topic,
                subsystem=name,
                handler=handler,
                timeout_s=timeout_s,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
            )
        self._audit.record(
            actor="kernel",
            action="register_subsystem",
            detail={"name": name, "topics": topics_list},
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
        self._audit.record(
            actor=event.source,
            action="ingest",
            detail={"type": event.type, "trace_id": event.trace_id},
        )
        await self._enqueue(partial(self._dispatch_event, event), priority=event.priority)

    async def emit(self, topic: str, payload, *, priority: Priority = Priority.NORMAL) -> None:
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
                # Publish meta-event so DGM-H and other observers can react
                meta_event = KernelEvent(
                    source="kernel",
                    type=_TOPIC_TASK_COMPLETE,
                    payload={"priority": Priority(priority).name.lower()},
                )
                try:
                    await self._bus.publish(meta_event)
                except Exception:
                    pass
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Kernel task failed: %s", exc)
                self._audit.record(
                    actor="kernel",
                    action="task_failed",
                    detail={"error": str(exc), "priority": priority},
                )
                meta_event = KernelEvent(
                    source="kernel",
                    type=_TOPIC_TASK_FAILED,
                    payload={"error": str(exc), "priority": priority},
                )
                try:
                    await self._bus.publish(meta_event)
                except Exception:
                    pass
            finally:
                self._queue.task_done()


# Late import to avoid circular dependency for type checking
from contentaios.sensory import SensoryInput  # noqa: E402  isort:skip
