"""Tests for Agent state machine transitions."""
import pytest
from src.agents.base_agent import Agent
from src.agents.agent_state import AgentState
from src.isa.instruction import Instruction
from src.isa.opcodes import Opcode
from src.semantics.semantic_word import SemanticWord, WordType, IntentType


def make_word(**kw):
    return SemanticWord.make(**kw).encode()


def test_initial_state():
    a = Agent(id=1)
    assert a.state == AgentState.IDLE


def test_receive_transitions_to_waiting():
    a = Agent(id=1)
    a.receive(make_word())
    assert a.state == AgentState.WAITING_INPUT
    assert len(a.inbox) == 1


def test_plan_consumes_inbox_and_emits():
    a = Agent(id=1)
    a.receive(make_word(intent=IntentType.PLAN))
    emitted = a.step(Instruction(opcode=Opcode.PLAN))
    assert len(emitted) == 1
    assert a.state == AgentState.IDLE
    assert len(a.inbox) == 0


def test_reflect_no_emit():
    a = Agent(id=1)
    emitted = a.step(Instruction(opcode=Opcode.REFLECT))
    assert emitted == []
    assert a.state == AgentState.IDLE


def test_critique_emits_one_per_inbox():
    a = Agent(id=1)
    for _ in range(3):
        a.receive(make_word())
    emitted = a.step(Instruction(opcode=Opcode.CRITIQUE))
    assert len(emitted) == 3
    assert a.state == AgentState.IDLE


def test_vote_picks_highest_confidence():
    high = SemanticWord.make(confidence=0.9, payload_ref=9).encode()
    low  = SemanticWord.make(confidence=0.1, payload_ref=1).encode()
    a = Agent(id=1)
    emitted = a.step(Instruction(opcode=Opcode.VOTE, args=[high, low]))
    assert len(emitted) == 1
    winner = SemanticWord.decode(emitted[0])
    assert winner.payload_ref == 9


def test_unknown_opcode_sets_error():
    a = Agent(id=1)
    # Use a raw Instruction with an opcode that has no handler
    instr = Instruction(opcode=Opcode.RUN_MODEL)  # not handled in base Agent
    a.step(instr)
    assert a.state == AgentState.ERROR


def test_write_then_read_memory():
    a = Agent(id=1)
    a.step(Instruction(opcode=Opcode.WRITE_MEMORY, args=[42]))
    assert a.memory_ref == 42
    emitted = a.step(Instruction(opcode=Opcode.READ_MEMORY))
    assert len(emitted) == 1
    w = SemanticWord.decode(emitted[0])
    assert w.payload_ref == 42


def test_log_populated():
    a = Agent(id=1)
    a.receive(make_word())
    a.step(Instruction(opcode=Opcode.PLAN))
    assert any("PLAN" in entry for entry in a.log)
