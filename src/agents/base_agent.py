"""
Base Agent — state machine driven by S-ISA instructions.

Each step() call consumes one Instruction and may emit new semantic words
(encoded as ints) that the Runtime routes to other agents.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .agent_state import AgentState
from ..isa.instruction import Instruction
from ..isa.opcodes import Opcode
from ..semantics.semantic_word import SemanticWord, WordType, IntentType, ChannelType


@dataclass
class Agent:
    id:         int
    state:      AgentState = AgentState.IDLE
    inbox:      list[int]  = field(default_factory=list)
    memory_ref: int        = 0
    log:        list[str]  = field(default_factory=list)

    def receive(self, word_int: int) -> None:
        self.inbox.append(word_int)
        if self.state == AgentState.IDLE:
            self.state = AgentState.WAITING_INPUT

    def step(self, instruction: Instruction) -> list[int]:
        """
        Process one instruction.  Returns a list of emitted semantic word ints.
        State transitions follow the S-ISA spec.
        """
        emitted: list[int] = []
        op = instruction.opcode

        if op == Opcode.PLAN:
            self.state = AgentState.PROCESSING
            consumed = list(self.inbox)
            self.inbox.clear()
            result = SemanticWord.make(
                type        = WordType.RESULT,
                intent      = IntentType.EMIT,
                channel     = ChannelType.INTERNAL,
                priority    = 200,
                confidence  = 0.9,
                payload_ref = self.id & 0xFFFF,
            )
            emitted.append(result.encode())
            self.log.append(f"PLAN consumed={len(consumed)} emitted=1")
            self.state = AgentState.IDLE

        elif op == Opcode.REFLECT:
            self.state = AgentState.REFLECTING
            self.log.append("REFLECT")
            self.state = AgentState.IDLE

        elif op == Opcode.CRITIQUE:
            self.state = AgentState.PROCESSING
            for word_int in self.inbox:
                w = SemanticWord.decode(word_int)
                critique = SemanticWord.make(
                    type        = WordType.RESULT,
                    intent      = IntentType.CRITIQUE,
                    channel     = ChannelType.INTERNAL,
                    priority    = w.priority,
                    confidence  = max(0.0, w.confidence_f - 0.1),
                    payload_ref = w.payload_ref,
                )
                emitted.append(critique.encode())
            self.inbox.clear()
            self.log.append(f"CRITIQUE emitted={len(emitted)}")
            self.state = AgentState.IDLE

        elif op == Opcode.VOTE:
            votes = [SemanticWord.decode(a) for a in instruction.args]
            if votes:
                winner = max(votes, key=lambda v: v.confidence)
                emitted.append(winner.encode())
                self.log.append(f"VOTE winner=ref:{winner.payload_ref}")
            self.state = AgentState.IDLE

        elif op == Opcode.WRITE_MEMORY:
            if instruction.args:
                self.memory_ref = instruction.args[0] & 0xFFFF
            self.log.append(f"WRITE_MEMORY ref={self.memory_ref}")

        elif op == Opcode.READ_MEMORY:
            mem_word = SemanticWord.make(
                type        = WordType.MEMORY,
                intent      = IntentType.EMIT,
                channel     = ChannelType.INTERNAL,
                priority    = 128,
                confidence  = 1.0,
                payload_ref = self.memory_ref,
            )
            emitted.append(mem_word.encode())
            self.log.append(f"READ_MEMORY ref={self.memory_ref}")

        elif op == Opcode.ROUTE_MESSAGE:
            for w in self.inbox:
                emitted.append(w)
            self.inbox.clear()
            self.log.append(f"ROUTE_MESSAGE forwarded={len(emitted)}")

        elif op == Opcode.EMIT_RESULT:
            for w in self.inbox:
                emitted.append(w)
            self.inbox.clear()
            self.state = AgentState.IDLE

        else:
            self.state = AgentState.ERROR
            self.log.append(f"UNKNOWN opcode={op}")

        return emitted

    def __repr__(self) -> str:
        return f"Agent(id={self.id} state={self.state.name} inbox={len(self.inbox)})"
