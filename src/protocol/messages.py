from __future__ import annotations
from dataclasses import dataclass, field
import time
import uuid
from ..semantics.semantic_word import SemanticWord


@dataclass
class Message:
    sender_id:   int
    receiver_id: int
    word:        SemanticWord
    ts:          float     = field(default_factory=time.time)
    correlation: str       = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def word_int(self) -> int:
        return self.word.encode()

    def __repr__(self) -> str:
        return (f"Message({self.sender_id}->{self.receiver_id} "
                f"{self.word!r} corr={self.correlation})")
