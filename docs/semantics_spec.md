# SemanticWord — 64-bit Semantic Encoding

## Purpose
A SemanticWord is the fundamental unit of communication in Sovereign AIOS.
Every piece of information — tasks, results, errors, model outputs — is encoded as a 64-bit integer that can be routed between agents, stored in memory, and processed by the Runtime.

## Bit Layout

```
Bit range   Field         Bits   Range        Meaning
63..56      type          8      0–255        Word category (see WordType)
55..48      intent        8      0–255        What to do with it (see IntentType)
47..40      channel       8      0–255        Routing channel (see ChannelType)
39..32      priority      8      0–255        0=lowest, 255=highest
31..16      confidence    16     0–65535      Maps to [0.0, 1.0] via /65535
15..0       payload_ref   16     0–65535      Index into runtime payload table
```

## WordType Values
| Value | Name    | Meaning                          |
|-------|---------|----------------------------------|
| 0     | UNKNOWN | Unclassified                     |
| 1     | CONTROL | Runtime control signal           |
| 2     | AGENT   | Agent identity or directive      |
| 3     | MODEL   | Model input/output               |
| 4     | MEMORY  | Memory read/write result         |
| 5     | TOOL    | Tool call or result              |
| 6     | RESULT  | Processed output ready to emit   |
| 7     | ERROR   | Error condition                  |

## IntentType Values
| Value | Name      | Meaning                    |
|-------|-----------|----------------------------|
| 0     | NONE      | No specific intent         |
| 1     | QUERY     | Retrieve information       |
| 2     | PLAN      | Generate a plan            |
| 3     | EXECUTE   | Execute an action          |
| 4     | REFLECT   | Self-evaluate              |
| 5     | CRITIQUE  | Evaluate another agent     |
| 6     | VOTE      | Cast a vote                |
| 7     | SUMMARIZE | Compress/distill           |
| 8     | ROUTE     | Forward to another agent   |
| 9     | SPAWN     | Create a new agent         |
| 10    | KILL      | Terminate an agent         |
| 11    | EMIT      | Publish a result           |

## ChannelType Values
| Value | Name     | Meaning               |
|-------|----------|-----------------------|
| 0     | SYSTEM   | Runtime messages      |
| 1     | USER     | From a human          |
| 2     | MODEL    | From a model          |
| 3     | TOOL     | Tool results          |
| 4     | EXTERNAL | Outside the system    |
| 5     | INTERNAL | Agent-to-agent        |

## Encoding Example
```python
from src.semantics.semantic_word import SemanticWord, WordType, IntentType, ChannelType

word = SemanticWord.make(
    type       = WordType.CONTROL,
    intent     = IntentType.PLAN,
    channel    = ChannelType.USER,
    priority   = 200,
    confidence = 0.95,
    payload_ref = 7,
)
encoded = word.encode()   # → int64
decoded = SemanticWord.decode(encoded)  # roundtrip
```
