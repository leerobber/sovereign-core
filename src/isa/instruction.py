"""
Sovereign Instruction encoding.

Binary format:
  Byte 0     : opcode (1 byte)
  Byte 1     : arg count N (1 byte)
  Bytes 2..  : N × 8-byte big-endian integers (semantic words or IDs)
"""
from __future__ import annotations
import struct
from dataclasses import dataclass, field
from .opcodes import Opcode


@dataclass
class Instruction:
    opcode: Opcode
    args:   list[int] = field(default_factory=list)

    def encode(self) -> bytes:
        header = struct.pack("BB", int(self.opcode), len(self.args))
        body   = struct.pack(f">{len(self.args)}Q", *self.args) if self.args else b""
        return header + body

    @staticmethod
    def decode(data: bytes) -> "Instruction":
        if len(data) < 2:
            raise ValueError("Instruction too short")
        opcode_val, n_args = struct.unpack("BB", data[:2])
        needed = 2 + n_args * 8
        if len(data) < needed:
            raise ValueError(f"Truncated instruction: need {needed} bytes, got {len(data)}")
        args = list(struct.unpack(f">{n_args}Q", data[2:needed])) if n_args else []
        return Instruction(opcode=Opcode(opcode_val), args=args)

    def __repr__(self) -> str:
        return f"Instruction({self.opcode.name} args={self.args})"
