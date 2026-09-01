import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

from biohub_tracking.config import cfg
from biohub_tracking.constants import DEFAULT_VOXEL_SIZE_UM, DEFAULT_ANISOTROPY
from biohub_tracking.data.zarr_loader import iter_frames
from biohub_tracking.evaluation.submission_builder import SubmissionBuilder
from biohub_tracking.segmentation.segmenter import CellSegmenter
from biohub_tracking.tracking.division_detector import DivisionDetector
from biohub_tracking.tracking.linker import Cell, HungarianLinker

# Use logging for all pipeline messages
logger = logging.getLogger(__name__)


def _make_segmenter() -> CellSegmenter:
    """Build the segmenter with parameters from config.yaml."""
    return CellSegmenter(
        method=cfg.get("segmentation.method", "cellpose"),
        diameter=cfg.get("segmentation.cellpose.diameter", 12.0),
        do_3D=cfg.get("segmentation.cellpose.do_3D", True),
        anisotropy=cfg.get("segmentation.cellpose.anisotropy", DEFAULT_ANISOTROPY),
        flow_threshold=cfg.get("segmentation.cellpose.flow_threshold", 0.4),
        cellprob_threshold=cfg.get("segmentation.cellpose.cellprob_threshold", 0.0),
        min_size=cfg.get("segmentation.cellpose.min_size", 50),
        max_volume=cfg.get("segmentation.postprocess.max_volume", 50_000),
        channels=tuple(cfg.get("segmentation.cellpose.channels", [0, 0])),
        voxel_size_um=tuple(cfg.get("tracking.voxel_size_um", DEFAULT_VOXEL_SIZE_UM)),
        model_type=cfg.get("segmentation.cellpose.model_type", "cyto3"),
        remove_border=cfg.get("segmentation.postprocess.remove_border", False),
    )


def _make_linker():
    """Build the frame-to-frame or ILP tracker with parameters from config.yaml."""
    method = cfg.get("tracking.method", "hungarian")
    max_dist = cfg.get("tracking.max_distance_um", 7.0)
    
    if method == "ilp":
        return ILPTracker(
            max_distance=max_dist,
            max_frame_gap=cfg.get("tracking.max_frame_gap", 2),
            appearance_cost=cfg.get("tracking.appearance_cost", 20.0),
            disappearance_cost=cfg.get("tracking.disappearance_cost", 20.0),
            division_cost=cfg.get("tracking.ilp.division_cost", 15.0),
        )
    return HungarianLinker(
        max_distance=max_dist,
        use_volume_cost=cfg.get("tracking.use_volume_cost", True),
        volume_weight=cfg.get("tracking.volume_weight", 0.3),
    )


def process_sample(
    sample_path: Path,
    segmenter: CellSegmenter,
) -> Tuple[Dict[int, List[Cell]], List[Tuple], List]:
    """Segment, link (with gap bridging), and detect divisions for one sample.

    Returns:
        all_cells:  frame → list[Cell]
        links:      list of (frame, source_id, target_id, confidence)
        divisions:  list of DivisionEvent
    """
    # ── 1. Segmentation ──────────────────────────────────────────────────────
    all_cells: Dict[int, List[Cell]] = {}
    for frame_index, image in iter_frames(sample_path):
        _labels, cells = segmenter.segment_frame(image, frame_index)
        all_cells[frame_index] = cells
        logging.debug("  frame %d → %d cells", frame_index, len(cells))

    linker = _make_linker()
    links: List[Tuple] = []
    # frame → set of cell_ids that already have a forward link
    linked_sources: Dict[int, Set[int]] = {f: set() for f in all_cells}
    # frame → set of cell_ids that already have an incoming link
    linked_targets: Dict[int, Set[int]] = {f: set() for f in all_cells}

    sorted_frames = sorted(all_cells)

    # ── 2. Primary linking: consecutive frames ────────────────────────────────
    for frame in sorted_frames[:-1]:
        frame_links = linker.link(all_cells[frame], all_cells[frame + 1])
        for source, target, confidence in frame_links:
            links.append((frame, source, target, confidence))
            linked_sources[frame].add(source)
            linked_targets[frame + 1].add(target)

    # ── 3. Gap-2 bridging: try to reconnect unlinked cells across 1 gap ──────
    max_gap = cfg.get("tracking.max_frame_gap", 2)
    for gap in range(2, max_gap + 1):
        for frame in sorted_frames:
            t_next = frame + gap
            if t_next not in all_cells:
                continue

            # Cells at 'frame' without a forward link AND cells at t_next without an incoming link
            orphan_sources = [
                c for c in all_cells[frame]
                if c.id not in linked_sources[frame]
            ]
            orphan_targets = [
                c for c in all_cells[t_next]
                if c.id not in linked_targets[t_next]
            ]

            if not orphan_sources or not orphan_targets:
                continue

            bridge_links = linker.link(orphan_sources, orphan_targets)
            for source, target, confidence in bridge_links:
                # Record as a gap-N edge
                links.append((frame, source, target, confidence * (1.0 / gap)))
                linked_sources[frame].add(source)
                linked_targets[t_next].add(target)

    # ── 4. Division detection ─────────────────────────────────────────────────
    detector = DivisionDetector(
        min_size_ratio=cfg.get("division.min_size_ratio", 0.3),
        max_size_ratio=cfg.get("division.max_size_ratio", 0.8),
        max_distance_um=cfg.get("division.max_distance_um", 10.0),
        vol_balance_weight=cfg.get("division.vol_balance_weight", 0.5),
        dist_score_weight=cfg.get("division.dist_score_weight", 0.5),
    )

    # links is now a flat list of (frame, s, t, conf)
    divisions = detector.detect(all_cells, links)

    logging.info(
        "  %s: %d cells, %d links, %d divisions",
        sample_path.name,
        sum(len(v) for v in all_cells.values()),
        len(links),
        len(divisions),
    )
    return all_cells, links, divisions


def generate_submission(test_dir: Path, output_path: Path) -> None:
    """Run the full pipeline on every .zarr sample in *test_dir*."""
    segmenter = _make_segmenter()
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
    parser = argparse.ArgumentParser(
        description="Generate a BioHub Cell Tracking competition submission."
    )
    parser.add_argument("--test-dir", type=Path, required=True,
                        help="Directory containing *.zarr test samples.")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"),
                        help="Output CSV path (default: submission.csv).")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug-level logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    generate_submission(args.test_dir, args.output)


if __name__ == "__main__":
    main()
