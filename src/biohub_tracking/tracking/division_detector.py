"""Detect mother-to-two-daughter events among unmatched cells."""

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from biohub_tracking.tracking.linker import Cell
from biohub_tracking.utils import volume_ratio


@dataclass(frozen=True)
class DivisionEvent:
    parent_id: int
    parent_frame: int
    child1_id: int
    child2_id: int
    division_frame: int
    confidence: float


class DivisionDetector:
    def __init__(self, min_size_ratio: float = 0.3, max_size_ratio: float = 0.8, max_distance_um: float = 10.0) -> None:
        self.min_size_ratio, self.max_size_ratio, self.max_distance_um = min_size_ratio, max_size_ratio, max_distance_um

    def detect(self, parents: Iterable[Cell], daughters: Iterable[Cell], linked_parent_ids: set[int], linked_daughter_ids: set[int]) -> list[DivisionEvent]:
        events = []
        for parent in parents:
            if parent.id in linked_parent_ids:
                continue
            for first, second in combinations([cell for cell in daughters if cell.id not in linked_daughter_ids], 2):
                ratios = (volume_ratio(first.volume, parent.volume), volume_ratio(second.volume, parent.volume))
                if not all(self.min_size_ratio <= ratio <= self.max_size_ratio for ratio in ratios):
                    continue
                distance = max(parent.distance_to(first), parent.distance_to(second))
                if distance <= self.max_distance_um:
                    confidence = max(0.0, 1.0 - distance / self.max_distance_um)
                    events.append(DivisionEvent(parent.id, parent.frame, first.id, second.id, first.frame, confidence))
                    break
        return events
