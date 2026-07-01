"""
64-bit SemanticWord — the atom of the Sovereign AIOS.

Bit layout (big-endian, MSB first):
  63..56  type        (8 bits)  — word category
  55..48  intent      (8 bits)  — what to do
  47..40  channel     (8 bits)  — routing channel
  39..32  priority    (8 bits)  — 0=lowest, 255=highest
  31..16  confidence  (16 bits) — 0–65535, maps to [0.0, 1.0]
  15..0   payload_ref (16 bits) — index into runtime payload table
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum


class WordType(IntEnum):
    UNKNOWN  = 0
    CONTROL  = 1
    AGENT    = 2
    MODEL    = 3
    MEMORY   = 4
    TOOL     = 5
    RESULT   = 6
    ERROR    = 7


class IntentType(IntEnum):
    NONE      = 0
    QUERY     = 1
    PLAN      = 2
    EXECUTE   = 3
    REFLECT   = 4
    CRITIQUE  = 5
    VOTE      = 6
    SUMMARIZE = 7
    ROUTE     = 8
    SPAWN     = 9
    KILL      = 10
    EMIT      = 11


class ChannelType(IntEnum):
    SYSTEM   = 0
    USER     = 1
    MODEL    = 2
    TOOL     = 3
    EXTERNAL = 4
    INTERNAL = 5


@dataclass
class SemanticWord:
    type:        int  # 0-255
    intent:      int  # 0-255
    channel:     int  # 0-255
    priority:    int  # 0-255
    confidence:  int  # 0-65535
    payload_ref: int  # 0-65535

    def encode(self) -> int:
        return (
            ((self.type        & 0xFF) << 56) |
            ((self.intent      & 0xFF) << 48) |
            ((self.channel     & 0xFF) << 40) |
            ((self.priority    & 0xFF) << 32) |
            ((self.confidence  & 0xFFFF) << 16) |
            (self.payload_ref  & 0xFFFF)
        )

    @staticmethod
    def decode(word: int) -> "SemanticWord":
        return SemanticWord(
            type        = (word >> 56) & 0xFF,
            intent      = (word >> 48) & 0xFF,
            channel     = (word >> 40) & 0xFF,
            priority    = (word >> 32) & 0xFF,
            confidence  = (word >> 16) & 0xFFFF,
            payload_ref =  word        & 0xFFFF,
        )

    @property
    def confidence_f(self) -> float:
        return self.confidence / 65535.0

    @classmethod
    def make(
        cls,
        type: WordType   = WordType.CONTROL,
        intent: IntentType = IntentType.NONE,
        channel: ChannelType = ChannelType.INTERNAL,
        priority: int    = 128,
        confidence: float = 1.0,
        payload_ref: int = 0,
    ) -> "SemanticWord":
        return cls(
            type        = int(type),
            intent      = int(intent),
            channel     = int(channel),
            priority    = max(0, min(255, priority)),
            confidence  = max(0, min(65535, int(confidence * 65535))),
            payload_ref = max(0, min(65535, payload_ref)),
        )

    def __repr__(self) -> str:
        t = WordType(self.type).name if self.type < 8 else str(self.type)
        i = IntentType(self.intent).name if self.intent < 12 else str(self.intent)
        return (f"SemanticWord({t}.{i} ch={self.channel} pri={self.priority} "
                f"conf={self.confidence_f:.2f} ref={self.payload_ref})")
