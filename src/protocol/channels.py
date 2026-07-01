from enum import IntEnum


class ChannelType(IntEnum):
    SYSTEM   = 0
    USER     = 1
    MODEL    = 2
    TOOL     = 3
    EXTERNAL = 4
    INTERNAL = 5
