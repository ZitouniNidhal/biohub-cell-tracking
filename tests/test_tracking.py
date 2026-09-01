"""Unit tests for cell tracking (linking and division detection)."""

import numpy as np
import pytest

from biohub_tracking.tracking.linker import Cell, HungarianLinker
from biohub_tracking.tracking.division_detector import (
    DivisionDetector,
    DivisionEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cell(
    cell_id: int,
    frame: int,
    z: float,
    y: float,
    x: float,
    volume: float = 200.0,
) -> Cell:
    """Create a synthetic Cell at a given µm position."""
    pos = np.array([z, y, x])
    return Cell(
        id=cell_id,
        frame=frame,
        centroid=pos,
        centroid_um=pos,   # use µm == voxels for simplicity in tests
        volume=volume,
    )


# ---------------------------------------------------------------------------
# HungarianLinker tests
# ---------------------------------------------------------------------------

class TestHungarianLinker:

    def test_links_perfectly_matched_cells(self):
        """Three cells shift 2 µm between frames → 3 links expected."""
        cells_t = [
            _make_cell(1, 0, 0.0, 10.0, 10.0),
            _make_cell(2, 0, 0.0, 30.0, 10.0),
            _make_cell(3, 0, 0.0, 50.0, 10.0),
        ]
        cells_t1 = [
            _make_cell(1, 1, 0.0, 10.0, 12.0),   # shifted 2 µm in X
            _make_cell(2, 1, 0.0, 30.0, 12.0),
            _make_cell(3, 1, 0.0, 50.0, 12.0),
        ]
        linker = HungarianLinker(max_distance=10.0, use_volume_cost=False)
        links = linker.link(cells_t, cells_t1)

        assert len(links) == 3, f"Expected 3 links, got {len(links)}"
        # All links should map cell i → cell i (same IDs)
        for src, tgt, conf in links:
            assert src == tgt, f"Expected identity mapping, got {src}→{tgt}"
            assert 0.0 <= conf <= 1.0

    def test_no_links_when_cells_too_far(self):
        """Cells more than max_distance apart should not be linked."""
        cells_t = [_make_cell(1, 0, 0.0, 0.0, 0.0)]
        cells_t1 = [_make_cell(2, 1, 0.0, 0.0, 100.0)]  # 100 µm away
        linker = HungarianLinker(max_distance=10.0)
        links = linker.link(cells_t, cells_t1)
        assert len(links) == 0

    def test_empty_frames_return_empty(self):
        """Linking with empty frames should return an empty list."""
        linker = HungarianLinker(max_distance=10.0)
        assert linker.link([], []) == []
        cells = [_make_cell(1, 0, 0.0, 0.0, 0.0)]
        assert linker.link(cells, []) == []
        assert linker.link([], cells) == []

    def test_one_to_one_assignment(self):
        """Hungarian algorithm must produce 1-to-1 assignment (no duplicate targets)."""
        cells_t = [
            _make_cell(1, 0, 0.0, 0.0, 0.0),
            _make_cell(2, 0, 0.0, 5.0, 0.0),
        ]
        cells_t1 = [
            _make_cell(10, 1, 0.0, 0.1, 0.0),  # very close to cell 1
            _make_cell(11, 1, 0.0, 5.1, 0.0),  # very close to cell 2
        ]
        linker = HungarianLinker(max_distance=10.0)
        links = linker.link(cells_t, cells_t1)
        targets = [tgt for _, tgt, _ in links]
        assert len(targets) == len(set(targets)), "Duplicate target assignments detected"


# ---------------------------------------------------------------------------
# DivisionDetector tests
# ---------------------------------------------------------------------------

class TestDivisionDetector:

    def test_detects_single_division(self):
        """One mother + two daughters with correct volumes → 1 division event."""
        mother_vol = 400.0
        daughter_vol = 190.0          # daughter pair sum = 380 (close to 400 mother)

        mother = _make_cell(1, 0, 0.0, 25.0, 25.0, volume=mother_vol)
        d1 = _make_cell(10, 1, 0.0, 22.0, 25.0, volume=daughter_vol)
        d2 = _make_cell(11, 1, 0.0, 28.0, 25.0, volume=daughter_vol)

        all_cells = {0: [mother], 1: [d1, d2]}
        links = []          # mother has no forward link → candidate

        detector = DivisionDetector(
            max_distance_um=20.0,
            min_size_ratio=0.3,
            max_size_ratio=0.8,
        )
        divisions = detector.detect(all_cells, links)

        assert len(divisions) == 1, f"Expected 1 division, got {len(divisions)}"
        div: DivisionEvent = divisions[0]
        assert div.parent_id == mother.id
        assert div.parent_frame == 0
        assert div.division_frame == 1
        assert set([div.child1_id, div.child2_id]) == {d1.id, d2.id}

    def test_no_division_when_mother_already_linked(self):
        """If the mother already has a forward link it should NOT be a division."""
        mother_vol = 400.0
        daughter_vol = 190.0

        mother = _make_cell(1, 0, 0.0, 25.0, 25.0, volume=mother_vol)
        d1 = _make_cell(10, 1, 0.0, 22.0, 25.0, volume=daughter_vol)
        d2 = _make_cell(11, 1, 0.0, 28.0, 25.0, volume=daughter_vol)

        all_cells = {0: [mother], 1: [d1, d2]}
        # Mother IS linked → should not be treated as a division candidate
        links = [(0, mother.id, d1.id, 0.9)]

        detector = DivisionDetector(max_distance_um=20.0)
        divisions = detector.detect(all_cells, links)
        assert len(divisions) == 0

    def test_no_division_when_daughters_too_far(self):
        """Daughters beyond max_distance_um should not be matched."""
        mother = _make_cell(1, 0, 0.0, 25.0, 25.0, volume=400.0)
        d1 = _make_cell(10, 1, 0.0, 0.0, 0.0, volume=190.0)    # 25 µm away in Y
        d2 = _make_cell(11, 1, 0.0, 50.0, 50.0, volume=190.0)

        all_cells = {0: [mother], 1: [d1, d2]}
        links = []

        detector = DivisionDetector(max_distance_um=5.0)   # very tight
        divisions = detector.detect(all_cells, links)
        assert len(divisions) == 0

    def test_no_division_when_daughters_wrong_volume(self):
        """Daughters with volume far from mother volume sum should be rejected."""
        mother = _make_cell(1, 0, 0.0, 25.0, 25.0, volume=400.0)
        # Daughters sum = 40, error vs 400 is 90% -> low confidence
        d1 = _make_cell(10, 1, 0.0, 24.0, 25.0, volume=20.0)
        d2 = _make_cell(11, 1, 0.0, 26.0, 25.0, volume=20.0)

        all_cells = {0: [mother], 1: [d1, d2]}
        links = []

        detector = DivisionDetector(max_distance_um=20.0)
        divisions = detector.detect(all_cells, links)
        assert len(divisions) == 0


# ---------------------------------------------------------------------------
# ILPTracker tests
# ---------------------------------------------------------------------------

class TestILPTracker:

    def test_ilp_links_consecutive_frames(self):
        """Test ILP tracker on consecutive frames."""
        from biohub_tracking.tracking.ilp_tracker import ILPTracker

        cells_t0 = [_make_cell(1, 0, 0.0, 10.0, 10.0)]
        cells_t1 = [_make_cell(2, 1, 0.0, 10.0, 12.0)]
        all_cells = {0: cells_t0, 1: cells_t1}

        tracker = ILPTracker(max_distance=10.0)
        tracks, links = tracker.track(all_cells)

        assert len(links) == 1
        assert links[0][0] == 1 and links[0][1] == 2

    def test_ilp_gap_bridging(self):
        """Test ILP tracker bridging a missing frame detection (frame 0 -> frame 2)."""
        from biohub_tracking.tracking.ilp_tracker import ILPTracker

        c_t0 = _make_cell(1, 0, 0.0, 10.0, 10.0)
        c_t2 = _make_cell(2, 2, 0.0, 10.0, 13.0)  # frame 1 missing
        all_cells = {0: [c_t0], 1: [], 2: [c_t2]}

        tracker = ILPTracker(max_distance=10.0, max_frame_gap=2)
        tracks, links = tracker.track(all_cells)

        assert len(links) == 1
        assert links[0][0] == 1 and links[0][1] == 2

