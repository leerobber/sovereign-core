import asyncio

import pytest

from contentaios import ContentKernel, KernelEvent, PollingSensoryInput, Priority, PushSensoryInput


@pytest.mark.asyncio
async def test_sensory_pipeline_routes_events():
    sensor = PushSensoryInput("webhooks")
    kernel = ContentKernel([sensor])
    received: list[str] = []

    async def handler(event: KernelEvent) -> None:
        """
        Append the "text" field from the event's payload to the surrounding test's `received` list.
        
        Parameters:
            event (KernelEvent): Event whose payload contains a `"text"` key.
        """
        received.append(event.payload["text"])

    kernel.register_subsystem("summarizer", ["text.ingest"], handler)

    await kernel.start()
    await sensor.push(type="text.ingest", payload={"text": "hello"})
    await kernel.join()
    await kernel.stop()

    assert received == ["hello"]


@pytest.mark.asyncio
async def test_priority_scheduler_prefers_high_priority_tasks():
    sensor = PushSensoryInput("collector")
    kernel = ContentKernel([sensor])
    order: list[str] = []

    async def handler(event: KernelEvent) -> None:
        """
        Append the incoming event's type to the outer-scope `order` list.
        
        Parameters:
            event (KernelEvent): The kernel event whose `type` will be recorded.
        """
        order.append(event.type)

    kernel.register_subsystem("orchestrator", ["high", "low"], handler)

    await kernel.start()
    await sensor.push(type="low", payload={}, priority=Priority.LOW)
    await sensor.push(type="high", payload={}, priority=Priority.HIGH)
    await kernel.join()
    await kernel.stop()

    assert order[0] == "high"
    assert order[-1] == "low"


@pytest.mark.asyncio
async def test_audit_log_records_ingest_and_handling():
    sensor = PushSensoryInput("sensor-a")
    kernel = ContentKernel([sensor])

    async def handler(event: KernelEvent) -> None:
        """
        Schedules a no-op coroutine on the kernel with normal priority in response to an incoming event.
        
        Parameters:
            event (KernelEvent): The incoming kernel event that triggered this handler.
        """
        await kernel.schedule(lambda: asyncio.sleep(0), priority=Priority.NORMAL)

    kernel.register_subsystem("analyzer", ["sample"], handler)

    await kernel.start()
    await sensor.push(type="sample", payload={"value": 1})
    await kernel.join()
    await kernel.stop()

    actions = [record.action for record in kernel.audit_log.tail()]
    assert "ingest" in actions
    assert "handled" in actions
    assert "task_complete" in actions


@pytest.mark.asyncio
async def test_polling_sensory_input_feeds_kernel():
    emitted: list[int] = []
    counter = 0

    async def fetcher() -> KernelEvent | None:
        """
        Produce the next heartbeat event for the poller until three events have been emitted.
        
        Returns:
            KernelEvent: An event with source "poller", type "heartbeat", and payload {"seq": n} where n is the current sequence number.
            None: When three events have already been emitted to signal the poller should stop.
        """
        nonlocal counter
        if counter >= 3:
            return None
        counter += 1
        return KernelEvent(
            source="poller",
            type="heartbeat",
            payload={"seq": counter},
        )

    sensor = PollingSensoryInput("poller", fetcher, interval_s=0.01)
    kernel = ContentKernel([sensor])

    async def handler(event: KernelEvent) -> None:
        """
        Appends the event's payload 'seq' value to the outer `emitted` list.
        
        Parameters:
            event (KernelEvent): Event whose payload contains the integer `seq` to append to `emitted`.
        """
        emitted.append(event.payload["seq"])

    kernel.register_subsystem("monitor", ["heartbeat"], handler)

    await kernel.start()
    await asyncio.sleep(0.05)
    await kernel.join()
    await kernel.stop()

    assert emitted == [1, 2, 3]
