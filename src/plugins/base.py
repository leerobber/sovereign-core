from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class PluginAdapter(ABC):
    name: str

    @abstractmethod
    async def health(self) -> dict: ...

    @abstractmethod
    async def list_tools(self) -> list[dict]: ...

    @abstractmethod
    async def invoke_tool(self, name: str, args: dict) -> Any: ...
