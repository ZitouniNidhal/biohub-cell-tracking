"""Core data structures: Cell, Track, and Hungarian frame-to-frame linker."""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from biohub_tracking.utils import calculate_euclidean_dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    """Detected cell at a single time point."""

    id: int                              # unique ID within its frame
    frame: int                           # time index
    centroid: np.ndarray                 # (z, y, x) in voxels
    centroid_um: Optional[np.ndarray]    # (z, y, x) in µm — set after calibration
    volume: float                        # voxel count
    features: Dict[str, float] = field(default_factory=dict)

    def distance_to(self, other: "Cell", use_um: bool = True) -> float:
        """Euclidean distance to another cell (in µm if available, else voxels)."""
        a = self.centroid_um if use_um and self.centroid_um is not None else self.centroid
        b = other.centroid_um if use_um and other.centroid_um is not None else other.centroid
        return calculate_euclidean_dist(a, b)


@dataclass
class Track:
    """A cell lineage track spanning multiple frames."""

    track_id: int
    cell_ids: List[int] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)
    centroids: List[np.ndarray] = field(default_factory=list)
    parent_id: Optional[int] = None         # set if this track is a division daughter
    children_ids: List[int] = field(default_factory=list)  # set if this track divides

    @property
    def start_frame(self) -> int:
        return self.frames[0] if self.frames else -1

    @property
    def end_frame(self) -> int:
        return self.frames[-1] if self.frames else -1

    @property
    def length(self) -> int:
        return len(self.frames)


# ---------------------------------------------------------------------------
# Hungarian frame-to-frame linker
# ---------------------------------------------------------------------------

class HungarianLinker:
    """Greedy bipartite matching between consecutive frames using the
    Hungarian algorithm (scipy linear_sum_assignment).

    Matches cells within *max_distance_um* µm.  Cells without a match are
    treated as appearing / disappearing.
    """

    def __init__(
        self,
        max_distance: float = 7.0,        # µm  (Kaggle matching threshold)
        use_volume_cost: bool = True,
        volume_weight: float = 0.3,
    ):
        self.max_distance = max_distance
        self.use_volume_cost = use_volume_cost
        self.volume_weight = volume_weight

    # ------------------------------------------------------------------
    def link(
        self,
        cells_t: List[Cell],
        cells_t1: List[Cell],
    ) -> List[Tuple[int, int, float]]:
        """Match cells between two consecutive frames.

        Args:
            cells_t:  Cells at frame t.
            cells_t1: Cells at frame t+1.

        Returns:
            List of (cell_id_t, cell_id_t1, confidence) tuples.
            Confidence is in [0, 1], higher is better.
        """
        if not cells_t or not cells_t1:
            return []

        # Build cost matrix (shape: len(cells_t) x len(cells_t1))
        cost_matrix = self._build_cost_matrix(cells_t, cells_t1)

        # Mask infeasible pairs
        infeasible = cost_matrix >= self.max_distance
        cost_matrix_masked = cost_matrix.copy()
        cost_matrix_masked[infeasible] = self.max_distance * 10  # large sentinel

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix_masked)

        links = []
        for r, c in zip(row_ind, col_ind):
            dist = cost_matrix[r, c]
            if dist < self.max_distance:
                confidence = 1.0 - dist / self.max_distance
                links.append((cells_t[r].id, cells_t1[c].id, confidence))
            # else: no valid link — cell appears/disappears

        return links

    # ------------------------------------------------------------------
    def _build_cost_matrix(
        self, cells_t: List[Cell], cells_t1: List[Cell]
    ) -> np.ndarray:
        """Compute the (n_t x n_t1) cost matrix."""
        # Spatial term: Euclidean distance in µm
        c0 = np.array([
            c.centroid_um if c.centroid_um is not None else c.centroid
            for c in cells_t
        ])
        c1 = np.array([
            c.centroid_um if c.centroid_um is not None else c.centroid
            for c in cells_t1
        ])
        cost = cdist(c0, c1, metric="euclidean")  # (n_t, n_t1)

        if self.use_volume_cost:
            # Add volume-change penalty, normalised to [0, max_distance]
            vol_t = np.array([c.volume for c in cells_t])[:, None]
            vol_t1 = np.array([c.volume for c in cells_t1])[None, :]
            vol_ratio = np.minimum(vol_t, vol_t1) / (np.maximum(vol_t, vol_t1) + 1e-8)
            vol_cost = (1.0 - vol_ratio) * self.max_distance
            cost = (1.0 - self.volume_weight) * cost + self.volume_weight * vol_cost

        return cost
