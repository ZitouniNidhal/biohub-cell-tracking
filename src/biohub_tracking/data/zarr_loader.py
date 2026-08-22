"""Load BioHub competition volumes stored as Zarr v3 arrays."""

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np


def open_volume(path: str | Path):
	"""Open a sample directory and return its ``(T, Z, Y, X)`` array."""
	import zarr

	root = zarr.open(str(path), mode="r")
	array = root["0"] if "0" in root else root
	if len(array.shape) != 4:
		raise ValueError(f"Expected a 4D volume, got shape {array.shape}")
	return array


def iter_frames(path: str | Path) -> Iterator[Tuple[int, np.ndarray]]:
	"""Yield one timepoint at a time to keep memory use bounded."""
	volume = open_volume(path)
	for frame_index in range(volume.shape[0]):
		yield frame_index, np.asarray(volume[frame_index])
