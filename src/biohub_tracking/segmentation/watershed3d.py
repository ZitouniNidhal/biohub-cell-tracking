"""3D marker-controlled watershed segmentation."""

import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def watershed3d(
    image: np.ndarray,
    markers: Optional[np.ndarray] = None,
    threshold: float = 0.05,
    sigma_z: float = 1.0,
    sigma_xy: float = 2.0,
    min_distance_z: int = 2,
    min_distance_xy: int = 4,
) -> np.ndarray:
    """3D marker-controlled watershed for separating touching cells.

    Args:
        image:       3D fluorescence image (Z, Y, X).
        markers:     Optional pre-computed seed/marker array. If None,
                     local maxima are used as seeds.
        threshold:   Foreground threshold (fraction of max intensity).
        sigma_z:     Gaussian smoothing sigma along Z.
        sigma_xy:    Gaussian smoothing sigma along Y and X.
        min_distance_z:  Minimum distance between local maxima (Z).
        min_distance_xy: Minimum distance between local maxima (Y/X).

    Returns:
        Integer label array (Z, Y, X).
    """
    from skimage.filters import gaussian
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy.ndimage import label as nd_label

    # Normalise
    img = image.astype(float)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()

    # Smooth
    smoothed = gaussian(img, sigma=(sigma_z, sigma_xy, sigma_xy))

    # Binary foreground mask
    mask = smoothed > threshold

    if markers is None:
        # Find local maxima as seeds
        coords = peak_local_max(
            smoothed,
            min_distance=min_distance_xy,
            footprint=_anisotropic_ball(min_distance_z, min_distance_xy),
            labels=mask,
        )
        seed_img = np.zeros(smoothed.shape, dtype=bool)
        seed_img[tuple(coords.T)] = True
        markers, _ = nd_label(seed_img)
        logger.debug(f"watershed3d: found {markers.max()} seeds.")

    labels = watershed(-smoothed, markers=markers, mask=mask)
    return labels.astype(np.int32)


def _anisotropic_ball(radius_z: int, radius_xy: int) -> np.ndarray:
    """Create an ellipsoidal structuring element for anisotropic data."""
    dz = np.arange(-radius_z, radius_z + 1)
    dxy = np.arange(-radius_xy, radius_xy + 1)
    zz, yy, xx = np.meshgrid(dz, dxy, dxy, indexing="ij")
    footprint = (zz / max(radius_z, 1)) ** 2 + (yy / max(radius_xy, 1)) ** 2 + (xx / max(radius_xy, 1)) ** 2 <= 1.0
    return footprint.astype(bool)


def split_touching_cells(
    labels: np.ndarray,
    min_cell_volume: int = 50,
    sigma_z: float = 1.0,
    sigma_xy: float = 2.0,
) -> np.ndarray:
    """Apply watershed to further split potentially merged detections.

    Args:
        labels:           Input label array from initial segmentation.
        min_cell_volume:  Minimum acceptable cell volume (voxels).
        sigma_z / sigma_xy: Smoothing for distance-transform watershed.

    Returns:
        Refined label array.
    """
    from skimage.segmentation import watershed
    from skimage.morphology import remove_small_objects
    from scipy.ndimage import distance_transform_edt, label as nd_label
    from skimage.feature import peak_local_max

    binary = labels > 0
    binary = remove_small_objects(binary, min_size=min_cell_volume)

    # Distance transform
    dist = distance_transform_edt(binary)

    # Peaks in distance map → seeds
    coords = peak_local_max(dist, min_distance=3, labels=binary)
    seed_img = np.zeros(dist.shape, dtype=bool)
    seed_img[tuple(coords.T)] = True
    markers, _ = nd_label(seed_img)

    new_labels = watershed(-dist, markers=markers, mask=binary)
    return new_labels.astype(np.int32)
