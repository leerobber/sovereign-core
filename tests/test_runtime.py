"""Tests for the Sovereign Runtime."""
import pytest
from src.kernel.runtime import Runtime
from src.isa.instruction import Instruction
from src.isa.opcodes import Opcode
from src.semantics.semantic_word import SemanticWord, WordType, IntentType
from src.agents.agent_state import AgentState


def test_spawn_agent_returns_id():
    rt = Runtime()
    aid = rt.spawn_agent()
    assert isinstance(aid, int)
    assert aid in rt.agents


def test_spawn_increments_id():
    rt = Runtime()
    a1 = rt.spawn_agent()
    a2 = rt.spawn_agent()
    assert a2 == a1 + 1


def test_kill_agent_removes_it():
    rt = Runtime()
    aid = rt.spawn_agent()
    rt.kill_agent(aid)
    assert aid not in rt.agents


def test_dispatch_plan():
    rt = Runtime()
    aid = rt.spawn_agent()
    task = SemanticWord.make(intent=IntentType.PLAN).encode()
    rt.route_message(0, aid, task)
    emitted = rt.dispatch_instruction(aid, Instruction(opcode=Opcode.PLAN))
    assert len(emitted) == 1


def test_route_message_delivers():
    rt = Runtime()
    a1 = rt.spawn_agent()
    a2 = rt.spawn_agent()
    word = SemanticWord.make().encode()
    rt.route_message(a1, a2, word)
    assert word in rt.agents[a2].inbox


def test_route_to_missing_agent_raises():
    rt = Runtime()
    with pytest.raises(KeyError):
        rt.route_message(0, 999, SemanticWord.make().encode())


def test_broadcast_all_except_sender():
    rt = Runtime()
    ids = [rt.spawn_agent() for _ in range(3)]
    word = SemanticWord.make().encode()
    rt.broadcast(ids[0], word)
    assert word not in rt.agents[ids[0]].inbox
    assert word in rt.agents[ids[1]].inbox
    assert word in rt.agents[ids[2]].inbox


def test_payload_table():
    rt = Runtime()
    idx = rt.store_payload({"key": "value"})
    assert rt.get_payload(idx) == {"key": "value"}
    assert rt.get_payload(999) is None


def test_hooks_fired_on_spawn():
    rt = Runtime()
    fired = []
    rt.add_hook(lambda ev, **kw: fired.append(ev))
    rt.spawn_agent()
    assert "spawn" in fired


def test_spawn_instruction():
    rt = Runtime()
    aid = rt.spawn_agent()
    emitted = rt.dispatch_instruction(aid, Instruction(opcode=Opcode.SPAWN_AGENT))
    assert len(emitted) == 1
    assert len(rt.agents) == 2


def test_kill_instruction():
    rt = Runtime()
    a1 = rt.spawn_agent()
    a2 = rt.spawn_agent()
    rt.dispatch_instruction(a1, Instruction(opcode=Opcode.KILL_AGENT, args=[a2]))
    assert a2 not in rt.agents


def test_status():
    rt = Runtime()
    rt.spawn_agent()
    s = rt.status()
    assert s["agents"] == 1
    assert "agent_states" in s


def test_scheduler_demo():
    from src.kernel.scheduler import run_demo
    rt = run_demo()
    assert len(rt.agents) == 2
