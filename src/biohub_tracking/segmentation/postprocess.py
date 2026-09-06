"""Label-mask cleanup."""

import numpy as np
from skimage.segmentation import clear_border


def postprocess_labels(labels: np.ndarray, min_volume: int, max_volume: int, remove_border: bool = False) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    if remove_border:
        labels = clear_border(labels)
    cleaned = np.zeros_like(labels)
    next_id = 1
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = labels == label
        if min_volume <= int(mask.sum()) <= max_volume:
            cleaned[mask] = next_id
            next_id += 1
    return cleaned
