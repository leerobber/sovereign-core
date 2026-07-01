"""
Sovereign Runtime — manages agents, dispatches S-ISA instructions,
routes semantic words between agents, and logs transitions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import logging

from ..agents.base_agent import Agent
from ..agents.agent_state import AgentState
from ..isa.instruction import Instruction
from ..isa.opcodes import Opcode
from ..semantics.semantic_word import SemanticWord

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    agents:        dict[int, Agent] = field(default_factory=dict)
    payload_table: list[object]     = field(default_factory=list)
    _next_id:      int              = field(default=1, repr=False)
    _hooks:        list[Callable]   = field(default_factory=list, repr=False)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def spawn_agent(self, agent_cls=None) -> int:
        agent_id = self._next_id
        self._next_id += 1
        cls = agent_cls or Agent
        agent = cls(id=agent_id)
        self.agents[agent_id] = agent
        logger.info("SPAWN agent_id=%d", agent_id)
        self._emit_event("spawn", agent_id=agent_id)
        return agent_id

    def kill_agent(self, agent_id: int) -> None:
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info("KILL agent_id=%d", agent_id)
            self._emit_event("kill", agent_id=agent_id)

    # ── Instruction dispatch ──────────────────────────────────────────────────

    def dispatch_instruction(
        self, agent_id: int, instruction: Instruction
    ) -> list[int]:
        """
        Send an instruction to an agent.
        Returns the list of emitted semantic word ints.
        Handles SPAWN_AGENT and KILL_AGENT at the runtime level.
        """
        if instruction.opcode == Opcode.SPAWN_AGENT:
            new_id = self.spawn_agent()
            result = SemanticWord.make(payload_ref=new_id & 0xFFFF)
            return [result.encode()]

        if instruction.opcode == Opcode.KILL_AGENT:
            target = instruction.args[0] if instruction.args else agent_id
            self.kill_agent(int(target) & 0xFFFFFF)
            return []

        agent = self.agents.get(agent_id)
        if agent is None:
            raise KeyError(f"No agent with id={agent_id}")

        before = agent.state
        emitted = agent.step(instruction)
        after = agent.state

        if before != after:
            logger.debug("agent=%d %s -> %s", agent_id, before.name, after.name)
            self._emit_event("transition", agent_id=agent_id,
                             from_state=before.name, to_state=after.name)

        for w in emitted:
            logger.debug("agent=%d emitted %s", agent_id, SemanticWord.decode(w))

        return emitted

    # ── Message routing ───────────────────────────────────────────────────────

    def route_message(
        self, sender_id: int, receiver_id: int, word_int: int
    ) -> None:
        receiver = self.agents.get(receiver_id)
        if receiver is None:
            raise KeyError(f"No agent with id={receiver_id}")
        receiver.receive(word_int)
        logger.debug("route %d -> %d  %s",
                     sender_id, receiver_id, SemanticWord.decode(word_int))

    def broadcast(self, sender_id: int, word_int: int) -> None:
        for aid, agent in self.agents.items():
            if aid != sender_id:
                agent.receive(word_int)

    # ── Payload table ─────────────────────────────────────────────────────────

    def store_payload(self, obj: object) -> int:
        idx = len(self.payload_table)
        self.payload_table.append(obj)
        return idx

    def get_payload(self, ref: int) -> object:
        return self.payload_table[ref] if ref < len(self.payload_table) else None

    # ── Hooks ─────────────────────────────────────────────────────────────────

    def add_hook(self, fn: Callable) -> None:
        self._hooks.append(fn)

    def _emit_event(self, event: str, **kwargs) -> None:
        for fn in self._hooks:
            try:
                fn(event, **kwargs)
            except Exception:
                pass

    # ── Introspection ─────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "agents": len(self.agents),
            "payloads": len(self.payload_table),
            "next_id": self._next_id,
            "agent_states": {
                aid: agent.state.name for aid, agent in self.agents.items()
            },
        }
