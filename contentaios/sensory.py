from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Iterable, Optional

import aiohttp

from contentaios.types import KernelEvent, Priority

EventSink = Callable[[KernelEvent], Awaitable[None]]
logger = logging.getLogger(__name__)


class SensoryInput:
    """Contract for sensory inputs feeding the kernel."""

    name: str

    async def start(self, emit: EventSink) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def stop(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def wait_idle(self) -> None:  # pragma: no cover - interface
        """Optional hook allowing the kernel to await input drains."""
        return None


class PushSensoryInput(SensoryInput):
    """Queue-backed input that external sources can push into."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._queue: asyncio.Queue[KernelEvent | None] = asyncio.Queue()
        self._runner: Optional[asyncio.Task[None]] = None
        self._emit: Optional[EventSink] = None

    async def start(self, emit: EventSink) -> None:
        self._emit = emit
        self._runner = asyncio.create_task(self._run())

    async def stop(self) -> None:
        await self._queue.put(None)
        if self._runner:
            await self._runner
            self._runner = None

    async def push(
        self,
        *,
        type: str,
        payload,
        priority: Priority = Priority.NORMAL,
        source: Optional[str] = None,
    ) -> None:
        event = KernelEvent(
            source=source or self.name,
            type=type,
            payload=payload,
            priority=priority,
        )
        await self._queue.put(event)

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            if event is None:
                self._queue.task_done()
                break
            if self._emit is not None:
                await self._emit(event)
            self._queue.task_done()

    async def wait_idle(self) -> None:
        await self._queue.join()


class PollingSensoryInput(SensoryInput):
    """Polling input that periodically fetches data from an external source."""

    def __init__(
        self,
        name: str,
        fetcher: Callable[[], Awaitable[Optional[KernelEvent]]],
        *,
        interval_s: float = 1.0,
    ) -> None:
        self.name = name
        self._fetcher = fetcher
        self._interval_s = interval_s
        self._runner: Optional[asyncio.Task[None]] = None
        self._emit: Optional[EventSink] = None
        self._running = False

    async def start(self, emit: EventSink) -> None:
        self._emit = emit
        self._running = True
        self._runner = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._runner:
            await self._runner
            self._runner = None

    async def wait_idle(self) -> None:
        await asyncio.sleep(0)

    async def _run(self) -> None:
        while self._running:
            event = await self._fetcher()
            if event is not None and self._emit is not None:
                if isinstance(event, (list, tuple)):
                    for item in event:
                        await self._emit(item)
                else:
                    await self._emit(event)
            await asyncio.sleep(self._interval_s)


class HttpPollingSensoryInput(SensoryInput):
    """HTTP polling input that fetches events from an external feed."""

    def __init__(
        self,
        name: str,
        url: str,
        *,
        interval_s: float = 1.0,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self.name = name
        self._url = url
        self._interval_s = interval_s
        self._session = session
        self._owns_session = session is None
        self._runner: Optional[asyncio.Task[None]] = None
        self._emit: Optional[EventSink] = None
        self._running = False

    async def start(self, emit: EventSink) -> None:
        self._emit = emit
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        self._running = True
        self._runner = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._runner:
            await self._runner
            self._runner = None
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def _run(self) -> None:
        assert self._session is not None
        while self._running:
            try:
                async with self._session.get(self._url) as resp:
                    data = await resp.json()
                    events = self._decode(data)
                    if self._emit is not None:
                        for event in events:
                            await self._emit(event)
            except Exception as exc:
                logger.warning("HTTP polling failed for %s: %s", self._url, exc)
            await asyncio.sleep(self._interval_s)

    def _decode(self, data) -> list[KernelEvent]:
        items = data if isinstance(data, list) else [data]
        events: list[KernelEvent] = []
        for item in items:
            if item is None:
                continue
            event = KernelEvent(
                source=item.get("source", self.name),
                type=item["type"],
                payload=item.get("payload", item),
                priority=Priority(item.get("priority", Priority.NORMAL)),
            )
            events.append(event)
        return events


class WebhookPushBridge:
    """Bridge external webhook payloads into a PushSensoryInput."""

    def __init__(
        self,
        sensor: PushSensoryInput,
        *,
        type_field: str = "type",
        payload_field: str = "payload",
    ) -> None:
        self._sensor = sensor
        self._type_field = type_field
        self._payload_field = payload_field

    async def handle_payload(
        self,
        data: dict,
        *,
        priority: Optional[Priority] = None,
        source: Optional[str] = None,
    ) -> None:
        type_val = data.get(self._type_field)
        if type_val is None:
            raise ValueError(f"Missing {self._type_field} in webhook payload")
        payload = data.get(self._payload_field, data)
        prio = priority if priority is not None else Priority(data.get("priority", Priority.NORMAL))
        await self._sensor.push(type=type_val, payload=payload, priority=prio, source=source)
