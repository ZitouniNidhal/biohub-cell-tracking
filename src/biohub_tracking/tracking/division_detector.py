import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

from biohub_tracking.tracking.linker import Cell
from biohub_tracking.utils import calculate_euclidean_dist, compute_volume_ratio

logger = logging.getLogger(__name__)

@dataclass
class DivisionEvent:
    """Represents a cell division event."""
    parent_id: int
    parent_frame: int
    child1_id: int
    child2_id: int
    division_frame: int
    confidence: float

class DivisionDetector:
    """
    Detects cell divisions by identifying 1-to-2 splits in the tracking results.
    """

    def __init__(
        self,
        min_size_ratio: float = 0.3,
        max_size_ratio: float = 0.8,
        max_distance_um: float = 10.0,
        vol_balance_weight: float = 0.5,
        dist_score_weight: float = 0.5,
    ):
        """
        Initialize DivisionDetector.

        Args:
            min_size_ratio: Minimum volume ratio of daughter to mother.
            max_size_ratio: Maximum volume ratio of daughter to mother.
            max_distance_um: Max distance between mother and daughters.
            vol_balance_weight: Weight for volume conservation score.
            dist_score_weight: Weight for spatial proximity score.
        """
        self.min_size_ratio = min_size_ratio
        self.max_size_ratio = max_size_ratio
        self.max_distance_um = max_distance_um
        self.vol_balance_weight = vol_balance_weight
        self.dist_score_weight = dist_score_weight

    def detect(
        self,
        all_cells: Dict[int, List[Cell]],
        links: List[Tuple[int, int, int, float]],  # (frame, cell_id_t, cell_id_t1, conf)
    ) -> List[DivisionEvent]:
        """
        Detect division events.

        Args:
            all_cells: Frame index -> list of cells.
            links: List of links between consecutive frames.

        Returns:
            List of detected DivisionEvent objects.
        """
        divisions = []
        frames = sorted(all_cells.keys())

        for i in range(len(frames) - 1):
            t = frames[i]
            t1 = frames[i+1]

            cells_t = all_cells[t]
            cells_t1 = all_cells[t1]

            # 1. Find cells at t with NO outgoing links
            linked_at_t = {l[1] for l in links if l[0] == t}
            orphans_t = [c for c in cells_t if c.id not in linked_at_t]

            # 2. Find cells at t1 with NO incoming links
            linked_at_t1 = {l[2] for l in links if l[0] == t}
            orphans_t1 = [c for c in cells_t1 if c.id not in linked_at_t1]

            if not orphans_t or len(orphans_t1) < 2:
                continue

            # 3. Search for 1-to-2 split
            for mother in orphans_t:
                potential_daughters = []
                for daughter in orphans_t1:
                    dist = calculate_euclidean_dist(mother.centroid_um, daughter.centroid_um)
                    if dist <= self.max_distance_um:
                        potential_daughters.append(daughter)

                if len(potential_daughters) >= 2:
                    # Evaluate pairs of potential daughters
                    for idx1 in range(len(potential_daughters)):
                        for idx2 in range(idx1 + 1, len(potential_daughters)):
                            d1 = potential_daughters[idx1]
                            d2 = potential_daughters[idx2]

                            confidence = self._score_division(mother, d1, d2)
                            if confidence > 0.5: # Threshold for detection
                                divisions.append(DivisionEvent(
                                    parent_id=mother.id,
                                    parent_frame=t,
                                    child1_id=d1.id,
                                    child2_id=d2.id,
                                    division_frame=t1,
                                    confidence=confidence
                                ))
                                # A mother can only divide once
                                break
                        else: continue
                        break

        return divisions

    def _score_division(self, mother: Cell, d1: Cell, d2: Cell) -> float:
        """Compute a confidence score for a potential division."""
        # Volume conservation: V_mother approx V_d1 + V_d2
        vol_sum = d1.volume + d2.volume
        vol_error = abs(mother.volume - vol_sum) / (mother.volume + 1e-8)
        vol_score = 1.0 - min(1.0, vol_error)

        # Volume symmetry: Daughters should be balanced
        ratio = compute_volume_ratio(d1.volume, d2.volume)
        sym_score = 1.0 if self.min_size_ratio <= ratio <= self.max_size_ratio else 0.0

        # Spatial proximity
        dist1 = calculate_euclidean_dist(mother.centroid_um, d1.centroid_um)
        dist2 = calculate_euclidean_dist(mother.centroid_um, d2.centroid_um)
        avg_dist = (dist1 + dist2) / 2
        dist_score = 1.0 - (avg_dist / self.max_distance_um)

        # Weighted aggregate
        total_score = (self.vol_balance_weight * vol_score * sym_score) + \
                      (self.dist_score_weight * dist_score)

        return float(total_score)
