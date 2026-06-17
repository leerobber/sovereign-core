"""Structured JSONL logging for training flywheel."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

_DEFAULT_ROOT = Path(os.environ.get(
    "SOVEREIGN_PROJECT_ROOT", "/home/leer4/sovereign-project"
))
_RAW_DIR = _DEFAULT_ROOT / "datasets" / "raw"

_REDACT_PATTERNS = [
    re.compile(p) for p in [
        r"sk-[A-Za-z0-9_-]{10,}",
        r"ghp_[A-Za-z0-9]{20,}",
        r"hf_[A-Za-z0-9]{20,}",
        r"xoxb-[A-Za-z0-9-]+",
    ]
]


def _redact(text: str) -> str:
    for pat in _REDACT_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


class RunLogEntry(BaseModel):
    task_id: str
    timestamp: str
    source: Literal["sovereign-core", "gh05t3"]
    agent_name: str = "router"
    input: str = ""
    tools_used: list[str] = Field(default_factory=list)
    messages: list[dict] = Field(default_factory=list)
    intermediate_states: list[dict] = Field(default_factory=list)
    output: str = ""
    feedback: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class RunLogger:
    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or _RAW_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, task_id: str) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.raw_dir / f"run_{ts}.jsonl"

    async def log(self, entry: RunLogEntry) -> Path:
        entry.input = _redact(entry.input)
        entry.output = _redact(entry.output)
        path = self._path_for(entry.task_id)
        line = entry.model_dump_json() + '\n'
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
        return path

    async def log_chat(self, *, source, messages, output, agent_name="router", tools_used=None, routing=None, session_id=None, model=None, latency_ms=None) -> Path:
        user_input = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_input = m.get("content", "")
                break
        entry = RunLogEntry(
            task_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            agent_name=agent_name,
            input=user_input,
            tools_used=tools_used or [],
            messages=messages,
            intermediate_states=[{"routing": routing}] if routing else [],
            output=output,
            metadata={k: v for k, v in {"session_id": session_id, "model": model, "latency_ms": latency_ms}.items() if v is not None},
        )
        return await self.log(entry)

    async def add_feedback(self, task_id: str, feedback: str) -> bool:
        feedback = _redact(feedback)
        updated = False
        for path in sorted(self.raw_dir.glob("run_*.jsonl")):
            lines = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    lines.append(line)
                    continue
                if obj.get("task_id") == task_id:
                    obj["feedback"] = feedback
                    lines.append(json.dumps(obj))
                    updated = True
                else:
                    lines.append(line)
            if updated:
                path.write_text('\n'.join(lines) + '\n', encoding="utf-8")
                break
        return updated
