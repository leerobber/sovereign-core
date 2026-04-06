from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict


class Priority(IntEnum):
    """Scheduling priority for kernel events and tasks."""

    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclass(slots=True)
class KernelEvent:
    """Structured payload flowing through the kernel."""

    source: str
    type: str
    payload: Any
    priority: Priority = Priority.NORMAL
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass(slots=True)
class AuditRecord:
    """Structured audit entry for kernel activities."""

    timestamp: datetime
    actor: str
    action: str
    detail: dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "action": self.action,
            "detail": self.detail,
        }
