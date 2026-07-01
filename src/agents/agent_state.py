from enum import Enum


class AgentState(Enum):
    IDLE            = 0
    WAITING_INPUT   = 1
    PROCESSING      = 2
    AWAITING_MODEL  = 3
    COORDINATING    = 4
    REFLECTING      = 5
    ERROR           = 6
