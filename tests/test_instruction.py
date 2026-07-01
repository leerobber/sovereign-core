"""Tests for S-ISA Instruction encode/decode."""
import pytest
from src.isa.instruction import Instruction
from src.isa.opcodes import Opcode


def test_encode_decode_no_args():
    instr = Instruction(opcode=Opcode.PLAN)
    assert Instruction.decode(instr.encode()) == instr


def test_encode_decode_with_args():
    instr = Instruction(opcode=Opcode.ROUTE_MESSAGE, args=[12345, 67890, 0xFFFFFFFFFFFFFFFF])
    decoded = Instruction.decode(instr.encode())
    assert decoded.opcode == instr.opcode
    assert decoded.args == instr.args


def test_encode_all_opcodes():
    for op in Opcode:
        instr = Instruction(opcode=op, args=[1, 2])
        decoded = Instruction.decode(instr.encode())
        assert decoded.opcode == op
        assert decoded.args == [1, 2]


def test_decode_truncated_raises():
    with pytest.raises(ValueError):
        Instruction.decode(b"\x01")  # too short


def test_decode_truncated_args_raises():
    with pytest.raises(ValueError):
        Instruction.decode(b"\x1e\x02" + b"\x00" * 8)  # says 2 args but only 1


def test_encode_size():
    instr = Instruction(opcode=Opcode.PLAN, args=[1, 2, 3])
    assert len(instr.encode()) == 2 + 3 * 8  # header + 3 × 8 bytes


def test_repr():
    instr = Instruction(opcode=Opcode.PLAN)
    assert "PLAN" in repr(instr)
