"""Cell division event detection and classification."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.spatial.distance import cdist

from biohub_tracking.tracking.linker import Cell

logger = logging.getLogger(__name__)


@dataclass
class DivisionEvent:

    parent_id: int                # cell ID of the mother at parent_frame
    parent_frame: int             # frame of the mother cell
    child1_id: int                # cell ID of daughter 1 at division_frame
    child2_id: int                # cell ID of daughter 2 at division_frame
    division_frame: int           # frame where daughters appear
    confidence: float = 1.0      # detection confidence [0, 1]


class DivisionClassifier:
    """Detects cell divisions by analysing cell size changes between frames.

    Strategy:
      For each "orphan" pair of cells at frame t+1 (cells that have no
      confident Hungarian match), check if their combined volume roughly
      equals a single cell at frame t that also has no outgoing match.
      If so, declare a division event.
    """

    def __init__(
        self,
        min_size_ratio: float = 0.3,       # daughter ≥ 30 % of mother volume
        max_size_ratio: float = 0.8,       # daughter ≤ 80 % of mother volume
        max_distance_um: float = 12.0,     # max centroid distance mother → daughter (µm)
        max_frame_offset: int = 2,         # look ahead up to N frames for daughters
    ):
        self.min_size_ratio = min_size_ratio
        self.max_size_ratio = max_size_ratio
        self.max_distance_um = max_distance_um
        self.max_frame_offset = max_frame_offset

    # ------------------------------------------------------------------
    def detect(
        self,
        all_cells: Dict[int, List[Cell]],
        linked_ids: Dict[int, List[Tuple[int, int]]],  # frame -> [(id_t, id_t1)]
    ) -> List[DivisionEvent]:
        """Detect division events.

        Args:
            all_cells:  frame -> list of Cell objects.
            linked_ids: frame -> list of (cell_id_at_t, cell_id_at_t+1) pairs
                        that are already accounted for by normal tracking links.

        Returns:
            List of detected DivisionEvent objects.
        """
        divisions: List[DivisionEvent] = []
        frames = sorted(all_cells.keys())

        for i, t in enumerate(frames[:-1]):
            cells_t = all_cells[t]
            existing_links = set(a for a, _ in linked_ids.get(t, []))

            # Mother candidates: cells at frame t without a forward link
            unlinked_mothers = [c for c in cells_t if c.id not in existing_links]

            for lookahead in range(1, self.max_frame_offset + 1):
                t_next = t + lookahead
                if t_next not in all_cells:
                    break

                cells_next = all_cells[t_next]
                # Unlinked daughters = cells at t_next with no incoming link from t_next-1
                existing_targets = set(b for _, b in linked_ids.get(t_next - 1, []))
                unlinked_daughters = [c for c in cells_next if c.id not in existing_targets]

                new_divs = self._match_divisions(
                    unlinked_mothers, unlinked_daughters, t, t_next
                )
                divisions.extend(new_divs)

        # Deduplicate: one mother → one division event
        seen_mothers: Dict[Tuple[int, int], DivisionEvent] = {}
        for div in divisions:
            key = (div.parent_frame, div.parent_id)
            if key not in seen_mothers or div.confidence > seen_mothers[key].confidence:
                seen_mothers[key] = div

        return list(seen_mothers.values())

    # ------------------------------------------------------------------
    def _match_divisions(
        self,
        mothers: List[Cell],
        daughters: List[Cell],
        t_mother: int,
        t_daughter: int,
    ) -> List[DivisionEvent]:
        """Try to match mothers to pairs of daughters."""
        if len(mothers) == 0 or len(daughters) < 2:
            return []

        events = []

        # Pairwise daughter distances to each mother
        m_pos = np.array([
            c.centroid_um if c.centroid_um is not None else c.centroid
            for c in mothers
        ])
        d_pos = np.array([
            c.centroid_um if c.centroid_um is not None else c.centroid
            for c in daughters
        ])
        dist_md = cdist(m_pos, d_pos)  # (n_mothers, n_daughters)

        used_daughters = set()

        for mi, mother in enumerate(mothers):
            # Find daughters within distance threshold
            close_mask = dist_md[mi] < self.max_distance_um
            close_idxs = [j for j in range(len(daughters))
                          if close_mask[j] and j not in used_daughters]

            if len(close_idxs) < 2:
                continue

            # Try all pairs of close daughters
            best_event: Optional[DivisionEvent] = None
            best_score = -1.0
            best_di1: int = -1
            best_di2: int = -1

            for j1 in range(len(close_idxs)):
                for j2 in range(j1 + 1, len(close_idxs)):
                    d1 = daughters[close_idxs[j1]]
                    d2 = daughters[close_idxs[j2]]

                    # Volume consistency check
                    total_daughter_vol = d1.volume + d2.volume
                    if mother.volume < 1:
                        continue
                    ratio1 = d1.volume / mother.volume
                    ratio2 = d2.volume / mother.volume

                    if not (self.min_size_ratio <= ratio1 <= self.max_size_ratio):
                        continue
                    if not (self.min_size_ratio <= ratio2 <= self.max_size_ratio):
                        continue

                    # Score: prefer equal-sized daughters and close distances
                    vol_balance = 1.0 - abs(ratio1 - ratio2)
                    dist_score = 1.0 - (
                        (dist_md[mi, close_idxs[j1]] + dist_md[mi, close_idxs[j2]])
                        / (2 * self.max_distance_um)
                    )
                    score = 0.5 * vol_balance + 0.5 * dist_score

                    if score > best_score:
                        best_score = score
                        best_di1 = close_idxs[j1]
                        best_di2 = close_idxs[j2]
                        best_event = DivisionEvent(
                            parent_id=mother.id,
                            parent_frame=t_mother,
                            child1_id=d1.id,
                            child2_id=d2.id,
                            division_frame=t_daughter,
                            confidence=score,
                        )

            if best_event is not None:
                events.append(best_event)
                used_daughters.add(best_di1)
                used_daughters.add(best_di2)

        return events
