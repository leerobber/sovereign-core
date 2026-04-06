# KAN-87: ContentAIOS — Master Kernel & Sensory Interface

The ContentAIOS kernel coordinates sensory ingestion, task scheduling, and inter-subsystem
messaging for downstream content systems.

## Architecture

- **Sensory inputs** — Pluggable inputs push or poll external sources into the kernel as
  structured `KernelEvent` objects. Provided adapters:
  - `PushSensoryInput`: queue-backed, accepts external pushes (webhooks, SDK calls).
  - `PollingSensoryInput`: periodic fetcher hook for pulling external signals.
- **Kernel event loop** — A priority queue drives scheduling. High-priority sensory events and
  subsystem tasks are executed first; FIFO ordering within the same priority preserves
  determinism.
- **Message bus** — Lightweight pub/sub inside the kernel. Subsystems register interest in
  topics (event types) and receive events dispatched by the scheduler.
- **Audit log** — Bounded, in-memory log (`AuditLog`) capturing ingestion, dispatch, handler
  success/failure, and task lifecycle. Tail retrieval supports observability and testing.

## Operational Flow

1. Sensory input produces a `KernelEvent` (push or poll).
2. Kernel enqueues the event with its priority and traces it in the audit log.
3. Scheduler pops the next task/event, dispatches it through the message bus, and records
   completion.
4. Subsystems handle events and can schedule additional tasks via `ContentKernel.schedule` or
   emit cross-subsystem messages with `ContentKernel.emit`.

## Usage Example

```python
import asyncio
from contentaios import ContentKernel, PushSensoryInput, Priority

sensor = PushSensoryInput("webhooks")
kernel = ContentKernel([sensor])

async def summarizer(event):
    # Do work, then maybe emit a follow-up message
    await kernel.emit("summary.ready", {"trace_id": event.trace_id}, priority=Priority.HIGH)

kernel.register_subsystem("summarizer", ["text.ingest"], summarizer)

async def main():
    await kernel.start()
    await sensor.push(type="text.ingest", payload={"text": "hello world"})
    await kernel.join()   # wait until everything scheduled is processed
    await kernel.stop()

asyncio.run(main())
```

This flow satisfies the epic deliverables:

- Kernel architecture & event loop with priority scheduling.
- Sensory ingestion pipeline (push + polling adapters).
- Inter-subsystem communication via the message bus.
- Structured audit logging for every step.
