"""Central configuration loader."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for name in ("settings.yaml", "training.yaml", "context.yaml"):
        path = _ROOT / "config" / name
        if path.is_file():
            with open(path) as f:
                settings[name.replace(".yaml", "")] = yaml.safe_load(f) or {}
    registry = _ROOT / "config" / "registry.yaml"
    if registry.is_file():
        with open(registry) as f:
            settings["registry"] = yaml.safe_load(f) or {}
    return settings


class Settings:
    def __init__(self) -> None:
        self._data = load_settings()

    @property
    def server(self) -> dict:
        return self._data.get("settings", {}).get("server", {})

    @property
    def training(self) -> dict:
        return self._data.get("training", {})

    @property
    def context(self) -> dict:
        return self._data.get("context", {})

    @property
    def registry(self) -> dict:
        return self._data.get("registry", {})
