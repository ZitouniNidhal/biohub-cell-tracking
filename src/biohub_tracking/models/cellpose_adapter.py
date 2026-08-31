import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from cellpose import models

logger = logging.getLogger(__name__)

class CellposeAdapter:
    """
    Adapter for the Cellpose segmentation model to provide a standardized interface
    for the CellSegmenter.
    """

    def __init__(
        self,
        model_type: str = "cyto3",
        gpu: bool = False,
    ):
        """
        Initialize the Cellpose model.

        Args:
            model_type: Cellpose model type (e.g., "cyto", "cyto2", "cyto3").
            gpu: Whether to use GPU acceleration.
        """
        self.model_type = model_type
        self.gpu = gpu
        self._model = None

    def _get_model(self):
        """Lazy initialization of the Cellpose model."""
        if self._model is None:
            logger.info(f"Initializing Cellpose model ({self.model_type}) with GPU={self.gpu}...")
            self._model = models.Cellpose(gpu=self.gpu, model_type=self.model_type)
        return self._model

    def segment(
        self,
        img: np.ndarray,
        diameter: float = 12.0,
        channels: Tuple[int, int] = (0, 0),
        do_3D: bool = True,
        anisotropy: float = 4.0,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 50,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Segment a 3D image using Cellpose.

        Args:
            img: Input image array (Z, Y, X).
            diameter: Expected cell diameter in pixels.
            channels: Channel configuration (e.g., (0, 0) for grayscale).
            do_3D: Whether to perform 3D segmentation.
            anisotropy: Z vs XY resolution ratio.
            flow_threshold: Threshold for flow consistency.
            cellprob_threshold: Probability threshold for cell detection.
            min_size: Minimum cell volume (voxels).
            **kwargs: Additional arguments passed to Cellpose eval.

        Returns:
            Integer label array (Z, Y, X).
        """
        model = self._get_model()

        # Cellpose eval returns (masks, flows, outlines, properties)
        masks, _, _, _ = model.eval(
            img,
            diameter=diameter,
            channels=channels,
            do_3D=do_3D,
            anisotropy=anisotropy,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            **kwargs
        )

        return masks.astype(np.int32)
