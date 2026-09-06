"""Write node and edge predictions in the competition CSV layout."""

from pathlib import Path
from typing import Iterable
import csv

from biohub_tracking.tracking.division_detector import DivisionEvent
from biohub_tracking.tracking.linker import Cell

HEADER = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


class SubmissionBuilder:
    def build_rows(self, dataset: str, all_cells: dict[int, list[Cell]], links: Iterable[tuple[int, int, int, float]], divisions: Iterable[DivisionEvent] = ()) -> list[dict[str, int | str]]:
        rows: list[dict[str, int | str]] = []
        node_ids: dict[tuple[int, int], int] = {}
        next_node = 1
        for frame in sorted(all_cells):
            for cell in all_cells[frame]:
                node_ids[(frame, cell.id)] = next_node
                z, y, x = (int(round(value)) for value in cell.centroid)
                rows.append({"dataset": dataset, "row_type": "node", "node_id": next_node, "t": frame, "z": z, "y": y, "x": x, "source_id": -1, "target_id": -1})
                next_node += 1
        edges = {(frame, source, frame + 1, target) for frame, source, target, _ in links}
        edges |= {(event.parent_frame, event.parent_id, event.division_frame, child) for event in divisions for child in (event.child1_id, event.child2_id)}
        for source_frame, source, target_frame, target in sorted(edges):
            if (source_frame, source) in node_ids and (target_frame, target) in node_ids:
                rows.append({"dataset": dataset, "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": node_ids[(source_frame, source)], "target_id": node_ids[(target_frame, target)]})
        return rows

    def write(self, rows: Iterable[dict[str, int | str]], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=HEADER)
            writer.writeheader()
            for identifier, row in enumerate(rows):
                writer.writerow({"id": identifier, **row})
