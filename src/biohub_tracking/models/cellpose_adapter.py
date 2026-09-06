"""Optional Cellpose integration isolated from the rest of the pipeline."""

import numpy as np


class CellposeAdapter:
    def __init__(self, model_type: str = "cyto3", gpu: bool = False) -> None:
        self.model_type = model_type
        self.gpu = gpu
        self._model = None

    def segment(self, image: np.ndarray, *, diameter: float, anisotropy: float, channels: tuple[int, int], flow_threshold: float, cellprob_threshold: float, min_size: int) -> np.ndarray:
        if self._model is None:
            from cellpose import models
            self._model = models.Cellpose(gpu=self.gpu, model_type=self.model_type)
        masks, *_ = self._model.eval(
            image, channels=list(channels), diameter=diameter, do_3D=True,
            anisotropy=anisotropy, flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold, min_size=min_size,
        )
        return np.asarray(masks, dtype=np.int32)
