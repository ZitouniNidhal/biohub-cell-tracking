"""Build the CSV format required by the BioHub Kaggle competition."""

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import csv

from biohub_tracking.tracking.linker import Cell


HEADER = [
	"id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
	"source_id", "target_id",
]


class SubmissionBuilder:
	"""Serialize detected nodes and temporal edges into ``submission.csv``."""

	def build_rows(
		self,
		dataset: str,
		all_cells: Dict[int, Sequence[Cell]],
		links: Iterable[Tuple[int, int, int, float]],
		divisions: Iterable[object] = (),
	) -> List[dict]:
		node_ids = {}
		rows = []
		next_node_id = 1

		for frame in sorted(all_cells):
			for cell in sorted(all_cells[frame], key=lambda item: item.id):
				node_ids[(frame, cell.id)] = next_node_id
				z, y, x = (int(round(value)) for value in cell.centroid)
				rows.append({
					"dataset": dataset, "row_type": "node",
					"node_id": next_node_id, "t": frame,
					"z": z, "y": y, "x": x,
					"source_id": -1, "target_id": -1,
				})
				next_node_id += 1

		edge_keys = set()
		edges = []
		for frame, source_cell, target_cell, _confidence in links:
			source_key = (frame, source_cell)
			target_key = (frame + 1, target_cell)
			key = (source_key, target_key)
			if source_key in node_ids and target_key in node_ids and key not in edge_keys:
				edge_keys.add(key)
				edges.append(key)

		for division in divisions:
			parent = (division.parent_frame, division.parent_id)
			for child in (division.child1_id, division.child2_id):
				child_key = (division.division_frame, child)
				key = (parent, child_key)
				if parent in node_ids and child_key in node_ids and key not in edge_keys:
					edge_keys.add(key)
					edges.append(key)

		for (source_frame, source_cell), (target_frame, target_cell) in edges:
			rows.append({
				"dataset": dataset, "row_type": "edge", "node_id": -1,
				"t": -1, "z": -1, "y": -1, "x": -1,
				"source_id": node_ids[(source_frame, source_cell)],
				"target_id": node_ids[(target_frame, target_cell)],
			})
		return rows

	def write(self, rows: Iterable[dict], output_path: str | Path) -> None:
		"""Write rows with a consecutive throwaway ``id`` column."""
		output_path = Path(output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with output_path.open("w", newline="", encoding="utf-8") as handle:
			writer = csv.DictWriter(handle, fieldnames=HEADER)
			writer.writeheader()
			for row_id, row in enumerate(rows):
				writer.writerow({"id": row_id, **row})
