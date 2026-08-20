"""Kalman filter tracker for smooth trajectory prediction."""

import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub_tracking.tracking.linker import Cell, Track

logger = logging.getLogger(__name__)


class KalmanFilter3D:
    """Constant-velocity Kalman filter for 3D cell motion prediction.

    State vector: [z, y, x, vz, vy, vx]
    Measurement:  [z, y, x]
    """

    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 2.0,
    ):
        dt = 1.0  # one frame step

        # State transition matrix (constant velocity model)
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process noise covariance
        self.Q = np.eye(6) * process_noise

        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise

        # State & covariance
        self.x: Optional[np.ndarray] = None   # (6,)
        self.P: Optional[np.ndarray] = None   # (6, 6)

    # ------------------------------------------------------------------
    def initialize(self, position: np.ndarray) -> None:
        """Initialise state from first detection (zero velocity assumed)."""
        self.x = np.zeros(6)
        self.x[:3] = position
        self.P = np.eye(6) * 10.0  # high initial uncertainty

    def predict(self) -> np.ndarray:
        """Predict next position and update covariance."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def update(self, measurement: np.ndarray) -> None:
        """Correct state with a new measurement."""
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy() if self.x is not None else np.zeros(3)


# ---------------------------------------------------------------------------
# Multi-object Kalman tracker
# ---------------------------------------------------------------------------

class KalmanTracker:
    """Multi-object tracker using per-track Kalman filters.

    Uses greedy nearest-neighbour matching on predicted vs. measured positions.
    """

    def __init__(
        self,
        max_distance: float = 15.0,    # µm
        max_missed_frames: int = 3,
        process_noise: float = 1.0,
        measurement_noise: float = 2.0,
    ):
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Active filters indexed by track_id
        self._filters: Dict[int, KalmanFilter3D] = {}
        self._missed: Dict[int, int] = {}       # frames missed since last update
        self._track_cells: Dict[int, List[Tuple[int, int]]] = {}  # track -> [(frame, cell_id)]
        self._next_track_id = 1

    # ------------------------------------------------------------------
    def update(
        self,
        frame: int,
        cells: List[Cell],
    ) -> List[Tuple[int, int]]:
        """Process detections for one frame.

        Args:
            frame: Current frame index.
            cells: Detected cells at this frame.

        Returns:
            List of (track_id, cell_id) assignments for this frame.
        """
        # 1. Predict next positions for all active tracks
        predictions: Dict[int, np.ndarray] = {}
        for tid, kf in self._filters.items():
            predictions[tid] = kf.predict()

        assignments: List[Tuple[int, int]] = []

        if not predictions or not cells:
            # Either no tracks or no detections — initialise tracks for all cells
            for cell in cells:
                tid = self._init_track(frame, cell)
                assignments.append((tid, cell.id))
            return assignments

        # 2. Build cost matrix between predictions and detections
        track_ids = list(predictions.keys())
        pred_pos = np.array([predictions[t] for t in track_ids])
        cell_pos = np.array([
            c.centroid_um if c.centroid_um is not None else c.centroid
            for c in cells
        ])

        from scipy.spatial.distance import cdist
        cost = cdist(pred_pos, cell_pos)

        # 3. Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_cells = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < self.max_distance:
                tid = track_ids[r]
                cell = cells[c]
                pos = cell.centroid_um if cell.centroid_um is not None else cell.centroid
                self._filters[tid].update(pos)
                self._missed[tid] = 0
                self._track_cells[tid].append((frame, cell.id))
                assignments.append((tid, cell.id))
                matched_tracks.add(tid)
                matched_cells.add(c)

        # 4. Increment missed counter for unmatched tracks; remove stale ones
        for tid in list(self._filters.keys()):
            if tid not in matched_tracks:
                self._missed[tid] = self._missed.get(tid, 0) + 1
                if self._missed[tid] > self.max_missed_frames:
                    del self._filters[tid]
                    del self._missed[tid]

        # 5. Initialise new tracks for unmatched detections
        for ci, cell in enumerate(cells):
            if ci not in matched_cells:
                tid = self._init_track(frame, cell)
                assignments.append((tid, cell.id))

        return assignments

    # ------------------------------------------------------------------
    def _init_track(self, frame: int, cell: Cell) -> int:
        """Create a new Kalman filter track for a cell."""
        tid = self._next_track_id
        self._next_track_id += 1

        kf = KalmanFilter3D(self.process_noise, self.measurement_noise)
        pos = cell.centroid_um if cell.centroid_um is not None else cell.centroid
        kf.initialize(pos)

        self._filters[tid] = kf
        self._missed[tid] = 0
        self._track_cells[tid] = [(frame, cell.id)]

        return tid

    def get_tracks(self) -> Dict[int, List[Tuple[int, int]]]:
        """Return all accumulated (frame, cell_id) sequences per track."""
        return dict(self._track_cells)
