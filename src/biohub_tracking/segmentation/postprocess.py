"""Post-processing utilities for segmentation label maps."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def filter_by_size(
    labels: np.ndarray,
    min_volume: int = 30,
    max_volume: int = 50_000,
) -> np.ndarray:
    """Remove cells that are too small or too large.

    Args:
        labels:     Integer label array (Z, Y, X).
        min_volume: Minimum cell size in voxels.
        max_volume: Maximum cell size in voxels.

    Returns:
        Filtered label array (same shape, some labels zeroed out).
    """
    from skimage.measure import regionprops

    out = labels.copy()
    for region in regionprops(labels):
        if region.area < min_volume or region.area > max_volume:
            out[labels == region.label] = 0

    return out


def remove_border_cells(
    labels: np.ndarray,
    border_z: int = 1,
    border_xy: int = 1,
) -> np.ndarray:
    """Remove cells that touch the image border (often artefacts).

    Args:
        labels:    Integer label array (Z, Y, X).
        border_z:  Number of border slices to check in Z.
        border_xy: Number of border pixels to check in Y/X.

    Returns:
        Filtered label array.
    """
    out = labels.copy()
    border_ids = set()

    # Z borders
    for z in range(border_z):
        border_ids |= set(np.unique(labels[z]))
        border_ids |= set(np.unique(labels[-(z + 1)]))

    # Y/X borders
    border_ids |= set(np.unique(labels[:, :border_xy, :]))
    border_ids |= set(np.unique(labels[:, -border_xy:, :]))
    border_ids |= set(np.unique(labels[:, :, :border_xy]))
    border_ids |= set(np.unique(labels[:, :, -border_xy:]))

    border_ids.discard(0)

    for bid in border_ids:
        out[labels == bid] = 0

    return out


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Re-assign label IDs to be sequential starting from 1.

    Args:
        labels: Integer label array with possible gaps.

    Returns:
        Relabelled array.
    """
    from skimage.segmentation import relabel_sequential as _relabel
    out, _, _ = _relabel(labels)
    return out


def postprocess_labels(
    labels: np.ndarray,
    min_volume: int = 30,
    max_volume: int = 50_000,
    remove_border: bool = False,
    border_z: int = 1,
    border_xy: int = 1,
) -> np.ndarray:
    """Apply all post-processing steps in sequence.

    Args:
        labels:        Raw label array from segmenter.
        min_volume:    Minimum cell volume (voxels).
        max_volume:    Maximum cell volume (voxels).
        remove_border: Whether to remove border-touching cells.
        border_z:      Border thickness in Z.
        border_xy:     Border thickness in XY.

    Returns:
        Clean, sequentially labelled array.
    """
    labels = filter_by_size(labels, min_volume, max_volume)

    if remove_border:
        labels = remove_border_cells(labels, border_z, border_xy)

    labels = relabel_sequential(labels)
    logger.debug(f"postprocess: {labels.max()} cells remaining.")
    return labels
