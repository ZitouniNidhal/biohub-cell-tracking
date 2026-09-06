"""Per-frame 3D cell segmentation."""

from typing import Iterable
import numpy as np
from scipy import ndimage as ndi

from biohub_tracking.constants import DEFAULT_ANISOTROPY, DEFAULT_VOXEL_SIZE_UM
from biohub_tracking.models.cellpose_adapter import CellposeAdapter
from biohub_tracking.segmentation.postprocess import postprocess_labels
from biohub_tracking.segmentation.watershed3d import segment as watershed_segment
from biohub_tracking.tracking.linker import Cell


class CellSegmenter:
    def __init__(self, method: str = "blob", diameter: float = 12.0, anisotropy: float = DEFAULT_ANISOTROPY, flow_threshold: float = 0.4, cellprob_threshold: float = 0.0, min_size: int = 50, max_volume: int = 50_000, channels: tuple[int, int] = (0, 0), voxel_size_um: tuple[float, float, float] = DEFAULT_VOXEL_SIZE_UM, model_type: str = "cyto3", remove_border: bool = False) -> None:
        self.method, self.diameter, self.anisotropy = method, diameter, anisotropy
        self.flow_threshold, self.cellprob_threshold = flow_threshold, cellprob_threshold
        self.min_size, self.max_volume, self.channels = min_size, max_volume, channels
        self.voxel_size_um, self.model_type, self.remove_border = voxel_size_um, model_type, remove_border
        self._cellpose = None

    def segment_frame(self, image: np.ndarray, frame_idx: int) -> tuple[np.ndarray, list[Cell]]:
        if self.method == "cellpose":
            try:
                self._cellpose = self._cellpose or CellposeAdapter(self.model_type)
                labels = self._cellpose.segment(image, diameter=self.diameter, anisotropy=self.anisotropy, channels=self.channels, flow_threshold=self.flow_threshold, cellprob_threshold=self.cellprob_threshold, min_size=self.min_size)
            except (ImportError, RuntimeError):
                labels = watershed_segment(image)
        else:
            labels = watershed_segment(image)
        labels = postprocess_labels(labels, self.min_size, self.max_volume, self.remove_border)
        cells: list[Cell] = []
        for index, label in enumerate(np.unique(labels)):
            if label == 0:
                continue
            mask = labels == label
            centroid = np.asarray(ndi.center_of_mass(mask), dtype=float)
            cells.append(Cell(index, frame_idx, centroid, centroid * np.asarray(self.voxel_size_um), float(mask.sum())))
        return labels, cells

    def segment_stack(self, frames: Iterable[tuple[int, np.ndarray]]) -> dict[int, list[Cell]]:
        return {frame: self.segment_frame(image, frame)[1] for frame, image in frames}
