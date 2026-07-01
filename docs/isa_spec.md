# S-ISA — Sovereign Instruction Set Architecture

## Instruction Format
```
Byte 0     : opcode (uint8)
Byte 1     : arg count N (uint8)
Bytes 2..  : N × 8-byte big-endian unsigned integers
              (encoded SemanticWords or agent/model IDs)
```

## Opcode Table

### Control (1–9)
| Opcode | Name          | Args                    | Effect                              |
|--------|---------------|-------------------------|-------------------------------------|
| 1      | SPAWN_AGENT   | —                       | Runtime spawns a new agent          |
| 2      | KILL_AGENT    | [agent_id]              | Runtime kills the target agent      |
| 3      | ROUTE_MESSAGE | [receiver_id, word]     | Agent forwards inbox to receiver    |
| 4      | BROADCAST     | [word]                  | Send word to all agents             |
| 5      | SYNC          | —                       | Barrier — wait for all agents idle  |

### Model (10–19)
| Opcode | Name         | Args            | Effect                         |
|--------|--------------|-----------------|--------------------------------|
| 10     | RUN_MODEL    | [model_id, word]| Run model with semantic input  |
| 11     | EVAL_MODEL   | [model_id]      | Evaluate model quality         |
| 12     | LOAD_MODEL   | [model_id]      | Load model into memory         |
| 13     | UNLOAD_MODEL | [model_id]      | Unload model                   |

### Memory (20–29)
| Opcode | Name            | Args    | Effect                        |
|--------|-----------------|---------|-------------------------------|
| 20     | WRITE_MEMORY    | [ref]   | Set agent memory_ref          |
| 21     | READ_MEMORY     | —       | Emit word with memory_ref     |
| 22     | SUMMARIZE_MEMORY| —       | Compress memory               |
| 23     | SEARCH_MEMORY   | [query] | Search memory store           |

### Reasoning (30–39)
| Opcode | Name         | Args           | Effect                          |
|--------|--------------|----------------|---------------------------------|
| 30     | PLAN         | —              | Consume inbox, emit plan word   |
| 31     | REFLECT      | —              | Self-evaluate, transition state |
| 32     | CRITIQUE     | —              | Evaluate each inbox word        |
| 33     | VOTE         | [word, word..] | Pick highest confidence winner  |
| 34     | EMBED        | [word]         | Generate embedding              |
| 35     | SEARCH       | [query_word]   | Nearest-neighbor search         |
| 36     | LINK_CONTEXT | [word, word]   | Link two concepts               |

### Graph / DGM (40–49)
| Opcode      | Name         | Args         | Effect                     |
|-------------|--------------|--------------|----------------------------|
| 40          | BUILD_GRAPH  | [word..]     | Build semantic graph       |
| 41          | QUERY_GRAPH  | [query_word] | Query semantic graph       |
| 42          | UPDATE_GRAPH | [word..]     | Add nodes/edges            |

### Workflow (50–59)
| Opcode | Name         | Args         | Effect                       |
|--------|--------------|--------------|------------------------------|
| 50     | RUN_WORKFLOW | [workflow_id]| Execute workflow             |
| 51     | CHECKPOINT   | —            | Save runtime state           |
| 52     | EMIT_RESULT  | —            | Forward inbox as final output|

## Example
```python
from src.isa.instruction import Instruction
from src.isa.opcodes import Opcode
from src.semantics.semantic_word import SemanticWord, IntentType

task = SemanticWord.make(intent=IntentType.PLAN).encode()
instr = Instruction(opcode=Opcode.PLAN, args=[task])
data  = instr.encode()    # bytes
back  = Instruction.decode(data)
assert back.opcode == Opcode.PLAN
```
