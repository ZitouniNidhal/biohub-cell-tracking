"""3D instance segmenter wrapping Cellpose with a blob-detector fallback."""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

from biohub_tracking.tracking.linker import Cell
from biohub_tracking.segmentation.postprocess import postprocess_labels

logger = logging.getLogger(__name__)


def _voxel_to_um(
    centroid_vox: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
) -> np.ndarray:
    """Convert (z, y, x) voxel coordinates to µm."""
    return centroid_vox * np.array(voxel_size_um)


class CellSegmenter:
    """Segment 3D+t fluorescence images into per-frame cell instances.

    Tries Cellpose first; falls back to a Gaussian blob detector if
    Cellpose is unavailable (e.g. inside a CPU-only Kaggle notebook).
    """

    def __init__(
        self,
        method: str = "cellpose",           # "cellpose" | "blob"
        diameter: float = 12.0,             # expected cell diameter (pixels)
        do_3D: bool = True,
        anisotropy: float = 4.0,            # z vs xy resolution ratio (1.625/0.40625)
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 50,                 # min cell volume (voxels)
        max_volume: int = 50_000,           # max cell volume (voxels)
        channels: Tuple[int, int] = (0, 0), # grayscale
        voxel_size_um: Tuple[float, float, float] = (1.0, 0.347, 0.347),
        model_type: str = "cyto3",
        remove_border: bool = False,
    ):
        self.method = method
        self.diameter = diameter
        self.do_3D = do_3D
        self.anisotropy = anisotropy
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size
        self.max_volume = max_volume
        self.channels = list(channels)
        self.voxel_size_um = voxel_size_um
        self.model_type = model_type
        self.remove_border = remove_border

        self._cellpose_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_stack(
        self,
        frames: Dict[int, np.ndarray],  # frame_index -> 3D array (Z, Y, X)
    ) -> Dict[int, List[Cell]]:
        """Segment all frames and return detected cells.

        Args:
            frames: Mapping from frame index to 3D numpy image array.

        Returns:
            Mapping from frame index to list of Cell objects.
        """
        all_cells: Dict[int, List[Cell]] = {}
        for t, img in frames.items():
            logger.info(f"Segmenting frame {t} …")
            labels = self._segment_frame(img)
            all_cells[t] = self._labels_to_cells(labels, t)
        return all_cells

    def segment_frame(
        self, img: np.ndarray, frame_idx: int = 0
    ) -> Tuple[np.ndarray, List[Cell]]:
        """Segment a single 3D frame.

        Returns:
            (label_array, cells) where label_array is (Z, Y, X) integer mask.
        """
        labels = self._segment_frame(img)
        cells = self._labels_to_cells(labels, frame_idx)
        return labels, cells

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _segment_frame(self, img: np.ndarray) -> np.ndarray:
        """Return cleaned integer label array (Z, Y, X) after postprocessing."""
        if self.method == "cellpose":
            try:
                raw = self._cellpose_segment(img)
            except Exception as exc:
                logger.warning(f"Cellpose failed ({exc}), falling back to blob detector.")
                raw = self._blob_segment(img)
        else:
            raw = self._blob_segment(img)

        # Always clean up: remove debris (too small / too large) and relabel
        return postprocess_labels(
            raw,
            min_volume=self.min_size,
            max_volume=self.max_volume,
            remove_border=self.remove_border,
        )

    def _cellpose_segment(self, img: np.ndarray) -> np.ndarray:
        """Run Cellpose 3D segmentation."""
        if self._cellpose_model is None:
            from cellpose import models
            self._cellpose_model = models.Cellpose(
                gpu=False, model_type=self.model_type
            )

        masks, _, _, _ = self._cellpose_model.eval(
            img,
            diameter=self.diameter,
            channels=self.channels,
            do_3D=self.do_3D,
            anisotropy=self.anisotropy,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            min_size=self.min_size,
        )
        return masks.astype(np.int32)

    def _blob_segment(self, img: np.ndarray) -> np.ndarray:
        """Gaussian blob detector + watershed fallback segmentation."""
        from skimage.feature import blob_log
        from skimage.segmentation import watershed
        from skimage.filters import gaussian
        from skimage.morphology import ball
        from scipy.ndimage import label as nd_label

        # Normalise
        img_norm = img.astype(float)
        img_norm -= img_norm.min()
        if img_norm.max() > 0:
            img_norm /= img_norm.max()

        # Smooth
        sigma_xy = self.diameter / 4.0
        sigma_z = sigma_xy / self.anisotropy
        smoothed = gaussian(img_norm, sigma=(sigma_z, sigma_xy, sigma_xy))

        # Detect blob centres
        blobs = blob_log(
            smoothed,
            min_sigma=max(1.0, self.diameter / 8),
            max_sigma=self.diameter / 2,
            num_sigma=5,
            threshold=0.05,
        )
        if len(blobs) == 0:
            return np.zeros(img.shape, dtype=np.int32)

        # Create seed mask
        seeds = np.zeros(img.shape, dtype=np.int32)
        for i, (z, y, x, _sigma) in enumerate(blobs):
            zz, yy, xx = int(round(z)), int(round(y)), int(round(x))
            if 0 <= zz < img.shape[0] and 0 <= yy < img.shape[1] and 0 <= xx < img.shape[2]:
                seeds[zz, yy, xx] = i + 1

        # Watershed from seeds
        labels = watershed(-smoothed, seeds, mask=smoothed > 0.05)
        return labels.astype(np.int32)

    def _labels_to_cells(
        self, labels: np.ndarray, frame: int
    ) -> List[Cell]:
        """Convert a label array to a list of Cell objects."""
        from skimage.measure import regionprops

        cells = []
        for region in regionprops(labels):
            if region.area < self.min_size:
                continue

            centroid_vox = np.array(region.centroid, dtype=float)
            centroid_um = _voxel_to_um(centroid_vox, self.voxel_size_um)

            features = {
                "intensity_mean": float(region.mean_intensity)
                if hasattr(region, "mean_intensity")
                else 0.0,
                "bbox_volume": float(
                    (region.bbox[3] - region.bbox[0])
                    * (region.bbox[4] - region.bbox[1])
                    * (region.bbox[5] - region.bbox[2])
                ),
            }

            cell = Cell(
                id=int(region.label),
                frame=frame,
                centroid=centroid_vox,
                centroid_um=centroid_um,
                volume=float(region.area),
                features=features,
            )
            cells.append(cell)

        return cells
