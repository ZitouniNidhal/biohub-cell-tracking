"""CPU-safe 3D blob segmentation fallback."""

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def segment(image: np.ndarray, sigma: float = 1.2, threshold: float = 0.05) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    smoothed = ndi.gaussian_filter(image, sigma=sigma)
    cutoff = float(smoothed.min() + threshold * (smoothed.max() - smoothed.min()))
    mask = smoothed > cutoff
    distance = ndi.distance_transform_edt(mask)
    peaks = peak_local_max(distance, labels=mask, footprint=np.ones((3, 3, 3)), exclude_border=False)
    markers = np.zeros(mask.shape, dtype=np.int32)
    if len(peaks):
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    markers, _ = ndi.label(markers)
    return watershed(-distance, markers, mask=mask).astype(np.int32)
