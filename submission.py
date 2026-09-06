"""Run the BioHub segmentation, tracking, and CSV-export pipeline."""

import argparse
from pathlib import Path

from biohub_tracking.config import Config
from biohub_tracking.constants import DEFAULT_VOXEL_SIZE_UM
from biohub_tracking.data.zarr_loader import iter_frames
from biohub_tracking.evaluation.submission_builder import SubmissionBuilder
from biohub_tracking.segmentation.segmenter import CellSegmenter
from biohub_tracking.tracking.division_detector import DivisionDetector
from biohub_tracking.tracking.linker import HungarianLinker


def process_sample(path: Path, config: Config):
    voxel_size = tuple(config.get("tracking.voxel_size_um", DEFAULT_VOXEL_SIZE_UM))
    segmenter = CellSegmenter(
        method=config.get("segmentation.method", "blob"),
        diameter=config.get("segmentation.cellpose.diameter", 12.0),
        anisotropy=config.get("segmentation.cellpose.anisotropy", 4.0),
        flow_threshold=config.get("segmentation.cellpose.flow_threshold", 0.4),
        cellprob_threshold=config.get("segmentation.cellpose.cellprob_threshold", 0.0),
        min_size=config.get("segmentation.postprocess.min_volume", 30),
        max_volume=config.get("segmentation.postprocess.max_volume", 50_000),
        channels=tuple(config.get("segmentation.cellpose.channels", [0, 0])),
        voxel_size_um=voxel_size,
        model_type=config.get("segmentation.cellpose.model_type", "cyto3"),
        remove_border=config.get("segmentation.postprocess.remove_border", False),
    )
    cells = segmenter.segment_stack(iter_frames(path))
    linker = HungarianLinker(config.get("tracking.max_distance_um", 7.0))
    detector = DivisionDetector(
        config.get("division.min_size_ratio", 0.3), config.get("division.max_size_ratio", 0.8), config.get("division.max_distance_um", 10.0)
    )
    links, divisions = [], []
    for frame in sorted(cells)[:-1]:
        next_frame = frame + 1
        if next_frame not in cells:
            continue
        frame_links = linker.link(cells[frame], cells[next_frame])
        links.extend((frame, source, target, confidence) for source, target, confidence in frame_links)
        divisions.extend(detector.detect(cells[frame], cells[next_frame], {source for source, _, _ in frame_links}, {target for _, target, _ in frame_links}))
    return cells, links, divisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    samples = sorted(args.test_dir.glob("*.zarr"))
    if not samples:
        raise FileNotFoundError(f"No .zarr samples in {args.test_dir}")
    config, builder, output_rows = Config(args.config), SubmissionBuilder(), []
    for sample in samples:
        cells, links, divisions = process_sample(sample, config)
        output_rows.extend(builder.build_rows(sample.stem, cells, links, divisions))
    builder.write(output_rows, args.output)


if __name__ == "__main__":
    main()
