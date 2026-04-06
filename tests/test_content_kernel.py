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


`@pytest.mark.asyncio`
async def test_polling_sensory_input_feeds_kernel():
    emitted: list[int] = []
    done = asyncio.Event()
    counter = 0

    async def fetcher() -> KernelEvent | None:
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
