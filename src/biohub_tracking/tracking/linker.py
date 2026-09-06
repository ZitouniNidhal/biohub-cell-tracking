"""Cell representations and frame-to-frame Hungarian matching."""

from dataclasses import dataclass, field
from typing import Iterable
import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub_tracking.utils import euclidean_distance, volume_ratio


@dataclass(frozen=True)
class Cell:
    id: int
    frame: int
    centroid: np.ndarray
    centroid_um: np.ndarray
    volume: float
    features: dict[str, float] = field(default_factory=dict)

    def distance_to(self, other: "Cell") -> float:
        return euclidean_distance(self.centroid_um, other.centroid_um)


class HungarianLinker:
    def __init__(self, max_distance: float, volume_weight: float = 0.3) -> None:
        self.max_distance = max_distance
        self.volume_weight = volume_weight

    def link(self, source: Iterable[Cell], target: Iterable[Cell]) -> list[tuple[int, int, float]]:
        left, right = list(source), list(target)
        if not left or not right:
            return []
        distances = np.array([[cell.distance_to(other) for other in right] for cell in left])
        ratios = np.array([[abs(1.0 - volume_ratio(other.volume, cell.volume)) for other in right] for cell in left])
        cost = distances + self.volume_weight * ratios * self.max_distance
        cost[distances > self.max_distance] = np.inf
        rows, columns = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e12))
        result = []
        for row, column in zip(rows, columns):
            if distances[row, column] <= self.max_distance:
                confidence = max(0.0, 1.0 - distances[row, column] / self.max_distance)
                result.append((left[row].id, right[column].id, confidence))
        return result
