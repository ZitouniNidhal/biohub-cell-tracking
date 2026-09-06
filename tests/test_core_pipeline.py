import numpy as np

from biohub_tracking.tracking.division_detector import DivisionDetector
from biohub_tracking.tracking.linker import Cell, HungarianLinker


def make_cell(identifier, frame, coordinate, volume=10.0):
    point = np.asarray(coordinate, dtype=float)
    return Cell(identifier, frame, point, point, volume)


def test_linker_matches_nearby_cells_only():
    linker = HungarianLinker(max_distance=5.0)
    source = make_cell(1, 0, (0, 0, 0))
    matched = make_cell(2, 1, (1, 0, 0))
    distant = make_cell(3, 1, (10, 0, 0))

    links = linker.link([source], [matched, distant])

    assert len(links) == 1
    assert links[0][:2] == (1, 2)


def test_division_detector_finds_two_balanced_daughters():
    parent = make_cell(1, 0, (0, 0, 0), volume=100)
    first = make_cell(2, 1, (1, 0, 0), volume=50)
    second = make_cell(3, 1, (-1, 0, 0), volume=50)

    events = DivisionDetector(max_distance_um=5.0).detect([parent], [first, second], set(), set())

    assert len(events) == 1
    assert {events[0].child1_id, events[0].child2_id} == {2, 3}
