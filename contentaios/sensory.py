from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

from contentaios.types import KernelEvent, Priority

EventSink = Callable[[KernelEvent], Awaitable[None]]


class SensoryInput:
    """Contract for sensory inputs feeding the kernel."""

    name: str

    async def start(self, emit: EventSink) -> None:  # pragma: no cover - interface
        """
        Begin producing events and route them to the provided event sink.
        
        Parameters:
            emit (EventSink): Callable that accepts produced KernelEvent instances; implementations must deliver events to this sink.
        """
        raise NotImplementedError

    async def stop(self) -> None:  # pragma: no cover - interface
        """
        Request the sensory input stop producing events and release any resources.
        
        Implementations should stop any background work, ensure no further events will be emitted, and return only after the input is fully stopped. This method may be called multiple times and should be safe to call repeatedly.
        """
        raise NotImplementedError

    async def wait_idle(self) -> None:  # pragma: no cover - interface
        """
        Allow the kernel to wait until this input has finished processing any buffered events.
        
        The default implementation returns immediately; subclasses may override to block until internal queues or background tasks are idle.
        """
        return None


class PushSensoryInput(SensoryInput):
    """Queue-backed input that external sources can push into."""

    def __init__(self, name: str) -> None:
        """
        Initialize the push-driven sensory input and its internal state.
        
        Parameters:
            name (str): Identifier for the input; used as the default `source` when events are pushed.
        
        Details:
            Creates an internal asyncio queue for `KernelEvent` instances (uses `None` as a shutdown sentinel),
            and initializes the background runner task and active event sink to `None`.
        """
        self.name = name
        self._queue: asyncio.Queue[KernelEvent | None] = asyncio.Queue()
        self._runner: Optional[asyncio.Task[None]] = None
        self._emit: Optional[EventSink] = None

    async def start(self, emit: EventSink) -> None:
        """
        Start the sensory input by registering the event sink and launching its background runner.
        
        Parameters:
            emit (EventSink): Callable that will receive forwarded KernelEvent instances.
        """
        self._emit = emit
        self._runner = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """
        Signal the input's background forwarding task to terminate and wait for it to finish.
        
        This enqueues the shutdown sentinel into the internal queue and, if a runner task exists, awaits its completion before clearing the runner reference.
        """
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
        """
        Enqueues a KernelEvent for this input to be forwarded to the kernel.
        
        Constructs a KernelEvent with the provided fields and places it on the internal queue for later delivery by the input's background runner.
        
        Parameters:
        	type (str): Event type identifier.
        	payload: Event payload (content depends on event type).
        	priority (Priority): Event priority; defaults to Priority.NORMAL.
        	source (Optional[str]): Origin identifier for the event; defaults to the input's `name` when not provided.
        """
        event = KernelEvent(
            source=source or self.name,
            type=type,
            payload=payload,
            priority=priority,
        )
        await self._queue.put(event)

    async def _run(self) -> None:
        """
        Run the background queue consumer that forwards enqueued KernelEvent items to the configured EventSink.
        
        Continuously takes items from the internal queue, stops when it receives a `None` sentinel, forwards each non-`None` event to `self._emit` if set, and calls `queue.task_done()` for every fetched item.
        """
        while True:
            event = await self._queue.get()
            if event is None:
                self._queue.task_done()
                break
            if self._emit is not None:
                await self._emit(event)
            self._queue.task_done()

    async def wait_idle(self) -> None:
        """
        Wait until all events enqueued by this input have been processed by the background runner.
        
        This yields control until the internal queue is empty and every queued item has been marked done.
        """
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
        """
        Initialize the PollingSensoryInput.
        
        Parameters:
            name (str): Identifier for the input; used as the default event source.
            fetcher (Callable[[], Awaitable[Optional[KernelEvent]]]): Async callable that returns a KernelEvent to emit or `None` when there is no event.
            interval_s (float): Seconds between fetch attempts.
        """
        self.name = name
        self._fetcher = fetcher
        self._interval_s = interval_s
        self._runner: Optional[asyncio.Task[None]] = None
        self._emit: Optional[EventSink] = None
        self._running = False

    async def start(self, emit: EventSink) -> None:
        """
        Begin polling and deliver fetched events to the provided event sink.
        
        Parameters:
            emit (EventSink): Async callable that will be invoked with each fetched KernelEvent.
        """
        self._emit = emit
        self._running = True
        self._runner = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """
        Stop the polling input and wait for its background task to finish.
        
        Signals the polling loop to stop and, if a background runner task exists, awaits its completion and clears the stored runner reference.
        """
        self._running = False
        if self._runner:
            await self._runner
            self._runner = None

    async def wait_idle(self) -> None:
        """
        Yield control to the event loop to allow other pending tasks to run without waiting for any buffered work to complete.
        """
        await asyncio.sleep(0)

    async def _run(self) -> None:
        """
        Run the polling loop that repeatedly calls the configured fetcher and forwards any returned events to the active event sink.
        
        This background task continues until the instance's `_running` flag is cleared. On each iteration it awaits the async `fetcher()`; if a `KernelEvent` is returned and an `EventSink` is set, the event is forwarded to the sink, then the task sleeps for the configured `_interval_s` before repeating.
        """
        while self._running:
            event = await self._fetcher()
            if event is not None and self._emit is not None:
                await self._emit(event)
            await asyncio.sleep(self._interval_s)
