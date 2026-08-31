import yaml
from pathlib import Path
from typing import Any, Dict, Optional

class Config:
    """Singleton configuration loader to bridge config.yaml and the Python code."""
    _instance = None

    def __new__(cls, config_path: str = "config.yaml"):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: str):
        path = Path(config_path)
        if not path.exists():
            # Fallback for different execution directories
            # Try to find config.yaml in the project root
            root = path.parent.parent # Assuming called from src/biohub_tracking or similar
            candidate = root / "config.yaml"
            if candidate.exists():
                path = candidate
            else:
                raise FileNotFoundError(f"Configuration file not found at {config_path} or project root.")

        with open(path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get value using dot notation (e.g., 'segmentation.cellpose.diameter').

        Args:
            key_path: Dot-separated path to the config value.
            default: Value to return if the key is not found.
        """
        keys = key_path.split('.')
        val = self._config
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default

# Global config instance for easy import
cfg = Config()
