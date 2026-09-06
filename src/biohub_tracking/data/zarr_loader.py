"""Read time-series Zarr samples without loading the whole sequence."""

from pathlib import Path
from typing import Iterator
import numpy as np
import zarr


def _array_from_group(group: zarr.Group) -> zarr.Array:
    arrays = list(group.arrays())
    if not arrays:
        raise ValueError("No array found in Zarr sample")
    return arrays[0][1]


def iter_frames(sample_path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    root = zarr.open_group(str(sample_path), mode="r")
    array = _array_from_group(root) if hasattr(root, "arrays") else root
    if array.ndim == 3:
        yield 0, np.asarray(array)
    elif array.ndim == 4:
        for frame in range(array.shape[0]):
            yield frame, np.asarray(array[frame])
    else:
        raise ValueError(f"Expected (z, y, x) or (t, z, y, x), got {array.shape}")
