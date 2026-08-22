import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

from biohub_tracking.data.zarr_loader import iter_frames
from biohub_tracking.evaluation.submission_builder import SubmissionBuilder
from biohub_tracking.segmentation.segmenter import CellSegmenter
from biohub_tracking.tracking.division_classifier import DivisionClassifier
from biohub_tracking.tracking.linker import Cell, HungarianLinker

# ─── Competition constants ────────────────────────────────────────────────────
# Physical voxel size in µm: [Z, Y, X]
VOXEL_SIZE_UM = (1.625, 0.40625, 0.40625)

# True anisotropy for Cellpose = Z_voxel / XY_voxel
ANISOTROPY = VOXEL_SIZE_UM[0] / VOXEL_SIZE_UM[1]          # ≈ 4.0

<<<<<<< HEAD
# Linking: use 10 µm as the linker cutoff (Kaggle *metric* threshold is 7 µm;
# giving the linker a bit more slack reduces FN edges at no cost to precision).
LINK_MAX_DIST_UM = 10.0

# Allow bridging across 1 missing frame (gap-2 tracking)
MAX_FRAME_GAP = 2


def _make_segmenter() -> CellSegmenter:
    """Build the segmenter with competition-tuned parameters."""
    return CellSegmenter(
        method="cellpose",              # tries Cellpose; falls back to blob
        diameter=12.0,
        do_3D=True,
        anisotropy=ANISOTROPY,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=50,
        max_volume=50_000,
        channels=(0, 0),               # grayscale
        voxel_size_um=VOXEL_SIZE_UM,
        model_type="cyto3",
    )


def _make_linker() -> HungarianLinker:
    """Build the frame-to-frame linker with competition-tuned parameters."""
    return HungarianLinker(
        max_distance=LINK_MAX_DIST_UM,
        use_volume_cost=True,           # penalise large volume jumps
        volume_weight=0.3,
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
    # frame → [(source_id, target_id)]  — used by division classifier
    linked_ids: Dict[int, List[Tuple[int, int]]] = {}

    sorted_frames = sorted(all_cells)

    # ── 2. Primary linking: consecutive frames ────────────────────────────────
    for frame in sorted_frames[:-1]:
        frame_links = linker.link(all_cells[frame], all_cells[frame + 1])
        for source, target, confidence in frame_links:
            links.append((frame, source, target, confidence))
            linked_sources[frame].add(source)
            linked_targets[frame + 1].add(target)
        linked_ids[frame] = [(s, t) for s, t, _ in frame_links]

    # ── 3. Gap-2 bridging: try to reconnect unlinked cells across 1 gap ──────
    for frame in sorted_frames[:-2]:
        t2 = frame + 2
        if t2 not in all_cells:
            continue
        # Cells at 'frame' without a forward link AND cells at t2 without an
        # incoming link — try to bridge them.
        orphan_sources = [
            c for c in all_cells[frame]
            if c.id not in linked_sources[frame]
        ]
        orphan_targets = [
            c for c in all_cells[t2]
            if c.id not in linked_targets[t2]
        ]
        if not orphan_sources or not orphan_targets:
            continue
        bridge_links = linker.link(orphan_sources, orphan_targets)
        for source, target, confidence in bridge_links:
            # Record as a gap-1 then gap-2 edge pair (frame→t1 virtual, t1→t2)
            # For the submission format we emit a direct frame→t2 edge which
            # the Kaggle evaluator treats as normal.  We mark them with a
            # slightly lower confidence.
            links.append((frame, source, target, confidence * 0.9))
            linked_ids.setdefault(frame, []).append((source, target))
            linked_sources[frame].add(source)
            linked_targets[t2].add(target)

    # ── 4. Division detection ─────────────────────────────────────────────────
    divisions = DivisionClassifier(max_distance_um=10.0).detect(
        all_cells, linked_ids
    )
    logging.info(
        "  %s: %d cells, %d links, %d divisions",
        sample_path.name,
        sum(len(v) for v in all_cells.values()),
        len(links),
        len(divisions),
    )
    return all_cells, links, divisions
=======
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
>>>>>>> a46ae14 (feat: implement submission generation and update sample processing logic)


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
