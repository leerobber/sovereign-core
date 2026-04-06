from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

from contentaios.types import KernelEvent, Priority

EventSink = Callable[[KernelEvent], Awaitable[None]]


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
                await self._emit(event)
            await asyncio.sleep(self._interval_s)
