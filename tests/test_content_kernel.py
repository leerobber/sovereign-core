import asyncio
import json
from pathlib import Path

import pytest
from aiohttp import web

from contentaios import (
    AuditLog,
    ContentKernel,
    FileAuditSink,
    HttpPollingSensoryInput,
    KernelEvent,
    MetricsAuditSink,
    PollingSensoryInput,
    Priority,
    PushSensoryInput,
    WebhookPushBridge,
)


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
    done = asyncio.Event()
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
        if len(emitted) == 3:
            done.set()

    kernel.register_subsystem("monitor", ["heartbeat"], handler)

    await kernel.start()
    await asyncio.wait_for(done.wait(), timeout=1)
    await kernel.join()
    await kernel.stop()

    assert emitted == [1, 2, 3]


@pytest.mark.asyncio
async def test_http_polling_sensory_input_reads_feed():
    seen: list[int] = []
    emitted = False

    async def feed(_: web.Request) -> web.Response:
        nonlocal emitted
        if emitted:
            return web.json_response([])
        emitted = True
        return web.json_response([{"type": "heartbeat", "payload": {"seq": 7}}])

    app = web.Application()
    app.router.add_get("/feed", feed)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}/feed"

    sensor = HttpPollingSensoryInput("http-poller", url, interval_s=0.01)
    kernel = ContentKernel([sensor])

    async def handler(event: KernelEvent) -> None:
        seen.append(event.payload["seq"])

    kernel.register_subsystem("monitor", ["heartbeat"], handler)

    await kernel.start()
    await asyncio.sleep(0.05)
    await kernel.join()
    await kernel.stop()
    await runner.cleanup()

    assert seen == [7]


def test_webhook_push_bridge_maps_payload():
    sensor = PushSensoryInput("webhooks")
    bridge = WebhookPushBridge(sensor)
    kernel = ContentKernel([sensor])
    received: list[str] = []

    async def handler(event: KernelEvent) -> None:
        received.append(event.payload["msg"])

    kernel.register_subsystem("echo", ["incoming"], handler)

    async def run() -> None:
        await kernel.start()
        await bridge.handle_payload({"type": "incoming", "payload": {"msg": "hi"}})
        await kernel.join()
        await kernel.stop()

    asyncio.get_event_loop().run_until_complete(run())
    assert received == ["hi"]


def test_audit_log_persistence_and_metrics(tmp_path: Path):
    sink_path = tmp_path / "audit.jsonl"
    metrics_sink = MetricsAuditSink()
    audit = AuditLog(sinks=[FileAuditSink(sink_path), metrics_sink])
    audit.record("actor", "action", {"a": 1})
    audit.record("actor", "action", {"a": 2})
    audit.flush_to_file(tmp_path / "flush.jsonl")

    file_lines = sink_path.read_text().strip().splitlines()
    assert len(file_lines) == 2
    for line in file_lines:
        obj = json.loads(line)
        assert obj["actor"] == "actor"
        assert obj["action"] == "action"

    metrics = metrics_sink.snapshot()
    assert metrics["actor.action"] == 2
    assert (tmp_path / "flush.jsonl").exists()


@pytest.mark.asyncio
async def test_resilient_message_bus_retries_and_timeouts():
    sensor = PushSensoryInput("collector")
    kernel = ContentKernel([sensor])
    attempts = 0

    async def flaky(event: KernelEvent) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            await asyncio.sleep(0.2)
        elif attempts == 2:
            raise RuntimeError("boom")
        else:
            return

    kernel.register_subsystem(
        "flaky",
        ["ping"],
        flaky,
        timeout_s=0.05,
        max_retries=3,
        retry_backoff_s=0.01,
    )

    await kernel.start()
    await sensor.push(type="ping", payload={})
    await kernel.join()
    await kernel.stop()

    actions = [r.action for r in kernel.audit_log.tail()]
    assert "handler_timeout" in actions
    assert "handler_failed" in actions
    assert "retry_success" in actions

# ---------------------------------------------------------------------------
# RES-07: VerificationMiddleware tests
# ---------------------------------------------------------------------------

from contentaios import DGMHSubsystem, VerificationMiddleware
from contentaios.kernel import MessageBus


@pytest.mark.asyncio
async def test_verification_middleware_blocks_none_payload():
    """Events with a None payload must be dropped and audit-logged."""
    audit = AuditLog()
    mw = VerificationMiddleware(audit)
    bus = MessageBus(audit, verification=mw)
    reached: list[bool] = []

    async def handler(event: KernelEvent) -> None:
        reached.append(True)

    bus.subscribe("test.event", "tester", handler)
    event = KernelEvent(source="test", type="test.event", payload=None)
    await bus.publish(event)

    assert reached == [], "Handler must not be called when payload is None"
    actions = [r.action for r in audit.tail()]
    assert "verification_failed" in actions


@pytest.mark.asyncio
async def test_verification_middleware_blocks_non_dict_payload():
    """Events with a non-dict payload must be dropped and audit-logged."""
    audit = AuditLog()
    mw = VerificationMiddleware(audit)
    bus = MessageBus(audit, verification=mw)
    reached: list[bool] = []

    async def handler(event: KernelEvent) -> None:
        reached.append(True)

    bus.subscribe("test.event", "tester", handler)
    event = KernelEvent(source="test", type="test.event", payload="bad-string")
    await bus.publish(event)

    assert reached == [], "Handler must not be called when payload is not a dict"
    actions = [r.action for r in audit.tail()]
    assert "verification_failed" in actions


@pytest.mark.asyncio
async def test_verification_middleware_allows_valid_dict_payload():
    """Events with a valid dict payload must reach the handler."""
    audit = AuditLog()
    mw = VerificationMiddleware(audit)
    bus = MessageBus(audit, verification=mw)
    received: list[dict] = []

    async def handler(event: KernelEvent) -> None:
        received.append(event.payload)

    bus.subscribe("test.event", "tester", handler)
    event = KernelEvent(source="test", type="test.event", payload={"key": "value"})
    await bus.publish(event)

    assert received == [{"key": "value"}]
    actions = [r.action for r in audit.tail()]
    assert "verification_passed" in actions
    assert "verification_failed" not in actions


@pytest.mark.asyncio
async def test_verification_middleware_audit_records_detail():
    """verification_failed record must include trace_id and type fields."""
    audit = AuditLog()
    mw = VerificationMiddleware(audit)
    bus = MessageBus(audit, verification=mw)
    async def _noop(event: KernelEvent) -> None:
        pass

    bus.subscribe("check", "sub", _noop)
    event = KernelEvent(source="src", type="check", payload=None)
    await bus.publish(event)

    failed = [r for r in audit.tail() if r.action == "verification_failed"]
    assert failed, "Expected at least one verification_failed record"
    detail = failed[0].detail
    assert detail["type"] == "check"
    assert "trace_id" in detail


@pytest.mark.asyncio
async def test_kernel_with_verification_middleware_integration():
    """ContentKernel with VerificationMiddleware drops bad events end-to-end."""
    audit = AuditLog()
    mw = VerificationMiddleware(audit)
    sensor = PushSensoryInput("test-sensor")
    kernel = ContentKernel([sensor], audit_log=audit, verification=mw)
    reached: list[bool] = []

    async def handler(event: KernelEvent) -> None:
        reached.append(True)

    kernel.register_subsystem("tester", ["probe"], handler)
    await kernel.start()
    # Manually ingest an event with a None payload (bypassing PushSensoryInput
    # which always supplies a payload).
    bad_event = KernelEvent(source="external", type="probe", payload=None)
    await kernel.ingest_event(bad_event)
    await kernel.join()
    await kernel.stop()

    assert reached == []
    actions = [r.action for r in audit.tail()]
    assert "verification_failed" in actions


# ---------------------------------------------------------------------------
# RES-02: DGMHSubsystem tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dgmh_subscribes_to_kernel_meta_topics():
    """DGMHSubsystem.attach() must register on the kernel bus."""
    audit = AuditLog()
    sensor = PushSensoryInput("s")
    kernel = ContentKernel([sensor], audit_log=audit)
    dgm = DGMHSubsystem(kernel, audit, emit_every=100)
    dgm.attach()

    # Subscriptions are audited with action "subscribe"
    sub_records = [r for r in audit.tail() if r.action == "subscribe"]
    topics = [r.detail["topic"] for r in sub_records]
    assert "kernel.task_complete" in topics
    assert "kernel.task_failed" in topics


@pytest.mark.asyncio
async def test_dgmh_mutates_policy_on_success():
    """Policy success_weight increases when a task_complete signal is received."""
    audit = AuditLog()
    sensor = PushSensoryInput("s")
    kernel = ContentKernel([sensor], audit_log=audit)
    dgm = DGMHSubsystem(kernel, audit, emit_every=100)
    dgm.attach()

    initial_weight = dgm.policy["success_weight"]

    await kernel.start()
    # A valid dict payload event will complete successfully and trigger DGM-H.
    await sensor.push(type="any.topic", payload={"x": 1})
    await kernel.join()
    await kernel.stop()

    assert dgm.policy["success_weight"] > initial_weight
    si_actions = [r for r in audit.tail() if r.action == "self_improvement"]
    assert si_actions, "self_improvement must be audited"


@pytest.mark.asyncio
async def test_dgmh_mutates_policy_on_failure():
    """Policy failure_weight increases when a task_failed signal is received."""
    audit = AuditLog()
    kernel = ContentKernel(audit_log=audit)
    dgm = DGMHSubsystem(kernel, audit, emit_every=100)
    dgm.attach()

    initial_weight = dgm.policy["failure_weight"]

    # Directly publish a kernel.task_failed event to the bus (simulating a
    # task that raised an exception in the scheduler).
    event = KernelEvent(
        source="kernel",
        type="kernel.task_failed",
        payload={"error": "boom", "priority": 1},
    )
    # Access the bus via the kernel's private attr for the test.
    await kernel._bus.publish(event)  # type: ignore[attr-defined]

    assert dgm.policy["failure_weight"] > initial_weight
    si_actions = [r for r in audit.tail() if r.action == "self_improvement"]
    assert si_actions


@pytest.mark.asyncio
async def test_dgmh_emits_optimization_at_threshold():
    """DGM-H emits kernel.optimize into the kernel every emit_every signals."""
    audit = AuditLog()
    sensor = PushSensoryInput("s")
    kernel = ContentKernel([sensor], audit_log=audit)
    dgm = DGMHSubsystem(kernel, audit, emit_every=2)
    dgm.attach()

    optimized: list[dict] = []

    async def capture(event: KernelEvent) -> None:
        optimized.append(event.payload)

    kernel.register_subsystem("watcher", ["kernel.optimize"], capture)

    await kernel.start()
    # Push 2 events — after 2 task_complete signals DGM-H should emit optimize.
    await sensor.push(type="noop", payload={"i": 0})
    await sensor.push(type="noop", payload={"i": 1})
    await kernel.join()
    await kernel.stop()

    assert optimized, "DGM-H must have emitted at least one kernel.optimize event"
    assert "policy" in optimized[0]
    assert "signal_count" in optimized[0]


@pytest.mark.asyncio
async def test_dgmh_signal_count_increments():
    """signal_count must track the total number of success+failure signals."""
    audit = AuditLog()
    sensor = PushSensoryInput("s")
    kernel = ContentKernel([sensor], audit_log=audit)
    dgm = DGMHSubsystem(kernel, audit, emit_every=100)
    dgm.attach()

    await kernel.start()
    for i in range(3):
        await sensor.push(type="t", payload={"i": i})
    await kernel.join()
    await kernel.stop()

    # Each dispatched event produces 1 task_complete signal.
    assert dgm.signal_count >= 3


@pytest.mark.asyncio
async def test_kernel_emits_meta_events_to_bus():
    """The kernel scheduler must publish kernel.task_complete after each task."""
    audit = AuditLog()
    sensor = PushSensoryInput("s")
    kernel = ContentKernel([sensor], audit_log=audit)
    meta_events: list[str] = []

    async def capture(event: KernelEvent) -> None:
        meta_events.append(event.type)

    kernel.register_subsystem("observer", ["kernel.task_complete"], capture)

    await kernel.start()
    await sensor.push(type="ping", payload={"x": 1})
    await kernel.join()
    await kernel.stop()

    assert "kernel.task_complete" in meta_events
