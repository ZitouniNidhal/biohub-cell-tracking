import numpy as np
from pathlib import Path
from typing import Union

def calculate_euclidean_dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two points.

    Args:
        p1: First point (z, y, x).
        p2: Second point (z, y, x).
    Returns:
        Distance as a float.
    """
    return float(np.linalg.norm(p1 - p2))

def calculate_jaccard_index(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute the Intersection over Union (IoU) / Jaccard Index for two binary masks.

    Args:
        mask1: First binary mask.
        mask2: Second binary mask.
    Returns:
        Jaccard index as a float in [0, 1].
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0

def ensure_path(path: Union[str, Path]) -> Path:
    """
    Ensure a path is a pathlib.Path object.

    Args:
        path: String or Path object.
    Returns:
        Path object.
    """
    return Path(path) if isinstance(path, str) else path

def compute_volume_ratio(v1: float, v2: float) -> float:
    """
    Compute the ratio of the smaller volume to the larger volume.

    Args:
        v1: Volume of first object.
        v2: Volume of second object.
    Returns:
        Ratio in [0, 1].
    """
    if v1 == 0 or v2 == 0:
        return 0.0
    return min(v1, v2) / max(v1, v2)
