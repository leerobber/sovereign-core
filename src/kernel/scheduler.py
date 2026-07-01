"""
Sovereign Scheduler — demo loop showing the kernel end-to-end.

Run:
    python -m src.kernel.scheduler
"""
from __future__ import annotations
import logging
from .runtime import Runtime
from ..isa.instruction import Instruction
from ..isa.opcodes import Opcode
from ..semantics.semantic_word import SemanticWord, WordType, IntentType, ChannelType

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_demo() -> Runtime:
    """
    Demo:
      1. Create runtime
      2. Spawn a planner agent + a critic agent
      3. Send planner a PLAN instruction with a semantic word in its inbox
      4. Route planner output to critic for CRITIQUE
      5. Print all state transitions and emitted words
    """
    rt = Runtime()

    events = []
    rt.add_hook(lambda ev, **kw: events.append((ev, kw)))

    # Spawn two agents
    planner_id = rt.spawn_agent()
    critic_id  = rt.spawn_agent()
    logger.info("Spawned planner=%d critic=%d", planner_id, critic_id)

    # Give planner a task word
    task = SemanticWord.make(
        type       = WordType.CONTROL,
        intent     = IntentType.PLAN,
        channel    = ChannelType.USER,
        priority   = 200,
        confidence = 1.0,
        payload_ref = 0,
    )
    rt.route_message(sender_id=0, receiver_id=planner_id, word_int=task.encode())
    logger.info("Routed task -> planner: %s", task)

    # Planner executes PLAN
    plan_instr = Instruction(opcode=Opcode.PLAN)
    plan_out = rt.dispatch_instruction(planner_id, plan_instr)
    logger.info("Planner emitted %d words", len(plan_out))
    for w in plan_out:
        logger.info("  %s", SemanticWord.decode(w))

    # Route planner output to critic
    for w in plan_out:
        rt.route_message(sender_id=planner_id, receiver_id=critic_id, word_int=w)

    # Critic executes CRITIQUE
    critique_instr = Instruction(opcode=Opcode.CRITIQUE)
    critique_out = rt.dispatch_instruction(critic_id, critique_instr)
    logger.info("Critic emitted %d words", len(critique_out))
    for w in critique_out:
        logger.info("  %s", SemanticWord.decode(w))

    logger.info("Runtime status: %s", rt.status())
    logger.info("Events: %s", [e[0] for e in events])

    return rt


if __name__ == "__main__":
    run_demo()
