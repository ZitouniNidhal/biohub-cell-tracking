"""Small shared helpers."""

from pathlib import Path
import numpy as np


def euclidean_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)))


def volume_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def ensure_path(path: str | Path) -> Path:
    return Path(path)
