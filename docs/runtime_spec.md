# Sovereign Runtime Spec

## Overview
The Runtime is the kernel of Sovereign AIOS. It:
- Manages a registry of Agents
- Dispatches S-ISA Instructions to agents
- Routes SemanticWords between agents
- Maintains a shared payload table for large objects
- Fires event hooks for observability

## Key Concepts

### Agent Registry
`Runtime.agents: Dict[int, Agent]` — maps agent ID → Agent instance.
IDs are monotonically increasing integers starting at 1.

### Payload Table
`Runtime.payload_table: List[Any]` — stores arbitrary Python objects.
`payload_ref` field in SemanticWord indexes into this table, allowing agents
to pass large data (strings, dicts, numpy arrays) without encoding them in the 64-bit word.

### Dispatch Flow
```
User / orchestrator
    │
    ▼
Runtime.dispatch_instruction(agent_id, instruction)
    │
    ├─ SPAWN_AGENT / KILL_AGENT handled by Runtime itself
    │
    └─ Everything else → Agent.step(instruction)
            │
            ├─ State transition (logged)
            └─ Emitted semantic words → returned to caller
```

### Message Routing
```
Runtime.route_message(sender_id, receiver_id, word_int)
    → receiver.receive(word_int)
    → receiver.state = WAITING_INPUT (if was IDLE)

Runtime.broadcast(sender_id, word_int)
    → route to every agent except sender
```

## Event Hooks
Attach observers with `Runtime.add_hook(fn)`.
Hook signature: `fn(event: str, **kwargs)`

Events emitted:
- `spawn` — `agent_id`
- `kill`  — `agent_id`
- `transition` — `agent_id, from_state, to_state`

## Module Map
```
sovereign-core/
  src/
    semantics/semantic_word.py   ← 64-bit word encoding
    isa/opcodes.py               ← S-ISA opcode enum
    isa/instruction.py           ← instruction binary format
    agents/agent_state.py        ← state machine enum
    agents/base_agent.py         ← base Agent class
    kernel/runtime.py            ← this document
    kernel/scheduler.py          ← demo run loop
    protocol/messages.py         ← Message wrapper
    protocol/channels.py         ← ChannelType enum
```

## Integration Points

| Downstream repo    | How it uses the kernel                               |
|--------------------|------------------------------------------------------|
| GH05T3             | Expert agents (Planner, Critic, Builder) extend Agent |
| HyperAgents        | Imports Runtime, builds task graph, schedules instructions |
| aethyro-backend    | HTTP gateway — JSON → SemanticWord → Runtime → JSON  |
| Honcho             | Business agents with S-ISA workflows                 |
| DGM                | BUILD_GRAPH / QUERY_GRAPH opcodes                    |
| VAGEN              | EMBED / SEARCH opcodes                               |
| Termux             | Thin client → aethyro-backend                        |
