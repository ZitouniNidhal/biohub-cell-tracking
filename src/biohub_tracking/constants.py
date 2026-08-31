from typing import Tuple

# Physical voxel size in µm: [Z, Y, X]
DEFAULT_VOXEL_SIZE_UM: Tuple[float, float, float] = (1.625, 0.40625, 0.40625)

# True anisotropy for Cellpose = Z_voxel / XY_voxel (≈ 4.0)
DEFAULT_ANISOTROPY: float = DEFAULT_VOXEL_SIZE_UM[0] / DEFAULT_VOXEL_SIZE_UM[1]

# Kaggle Evaluation Weights
DIVISION_SCORE_WEIGHT: float = 0.1  # Score = Edge Jaccard + 0.1 * Division Jaccard
DEFAULT_IOU_THRESHOLD: float = 0.5  # Standard threshold for segmentation AP
MATCH_DIST_UM: float = 7.0           # Max matching distance (µm) for node matching
