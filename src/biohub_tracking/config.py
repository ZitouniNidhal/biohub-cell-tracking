"""Configuration loading with clear dotted-key access."""

from pathlib import Path
from typing import Any
import yaml


class Config:
    def __init__(self, path: str | Path = "config.yaml") -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.path}")
        data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.data: dict[str, Any] = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self.data
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value
