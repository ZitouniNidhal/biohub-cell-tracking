

import argparse
import logging
from pathlib import Path

from biohub_tracking.data.zarr_loader import iter_frames
from biohub_tracking.evaluation.submission_builder import SubmissionBuilder
from biohub_tracking.segmentation.segmenter import CellSegmenter
from biohub_tracking.tracking.division_classifier import DivisionClassifier
from biohub_tracking.tracking.linker import HungarianLinker

VOXEL_SIZE_UM = (1.625, 0.40625, 0.40625)


def process_sample(sample_path: Path, segmenter: CellSegmenter):
	all_cells = {}
	for frame_index, image in iter_frames(sample_path):
		_labels, cells = segmenter.segment_frame(image, frame_index)
		all_cells[frame_index] = cells
	linker = HungarianLinker(max_distance=7.0, use_volume_cost=False)
	links = []
	linked_ids = {}
	for frame in sorted(all_cells)[:-1]:
		frame_links = linker.link(all_cells[frame], all_cells[frame + 1])
		links.extend((frame, source, target, confidence)
					 for source, target, confidence in frame_links)
		linked_ids[frame] = [(source, target) for source, target, _ in frame_links]
	divisions = DivisionClassifier(max_distance_um=10.0).detect(
		all_cells, linked_ids
	)
	return all_cells, links, divisions


def generate_submission(test_dir: Path, output_path: Path) -> None:
	segmenter = CellSegmenter(
		method="blob", min_size=30, anisotropy=VOXEL_SIZE_UM[0] / VOXEL_SIZE_UM[1],
		voxel_size_um=VOXEL_SIZE_UM,
	)
	builder = SubmissionBuilder()
	rows = []
	samples = sorted(test_dir.glob("*.zarr"))
	if not samples:
		raise FileNotFoundError(f"No .zarr samples found in {test_dir}")
	for sample_path in samples:
		logging.info("Processing %s", sample_path.name)
		all_cells, links, divisions = process_sample(sample_path, segmenter)
		rows.extend(builder.build_rows(
			sample_path.stem, all_cells, links, divisions
		))
	builder.write(rows, output_path)
	logging.info("Wrote %d rows to %s", len(rows), output_path)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--test-dir", type=Path, required=True)
	parser.add_argument("--output", type=Path, default=Path("submission.csv"))
	args = parser.parse_args()
	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
	generate_submission(args.test_dir, args.output)


if __name__ == "__main__":
	main()
