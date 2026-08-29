"""Unit tests for 3D cell segmentation."""

import numpy as np
import pytest

from biohub_tracking.segmentation.segmenter import CellSegmenter
from biohub_tracking.segmentation.postprocess import postprocess_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blob_volume(
    shape=(10, 60, 60),
    centers=None,
    radius=4,
    intensity=1000,
) -> np.ndarray:
    """Create a synthetic 3D volume with bright spherical blobs."""
    vol = np.zeros(shape, dtype=np.uint16)
    if centers is None:
        centers = [
            (shape[0] // 2, shape[1] // 4, shape[2] // 4),
            (shape[0] // 2, shape[1] // 4, 3 * shape[2] // 4),
            (shape[0] // 2, 3 * shape[1] // 4, shape[2] // 2),
        ]
    for cz, cy, cx in centers:
        for z in range(max(0, cz - radius), min(shape[0], cz + radius + 1)):
            for y in range(max(0, cy - radius), min(shape[1], cy + radius + 1)):
                for x in range(max(0, cx - radius), min(shape[2], cx + radius + 1)):
                    if (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2:
                        vol[z, y, x] = intensity
    return vol


# ---------------------------------------------------------------------------
# Segmenter tests
# ---------------------------------------------------------------------------

class TestCellSegmenterBlob:
    """Tests using the blob fallback (no Cellpose required)."""

    def test_detects_cells_in_synthetic_volume(self):
        """Blob segmenter should detect ≥2 cells in a 3-blob volume."""
        vol = _make_blob_volume()
        seg = CellSegmenter(
            method="blob",
            diameter=8.0,
            min_size=10,
            anisotropy=1.0,
            voxel_size_um=(1.0, 1.0, 1.0),
        )
        labels, cells = seg.segment_frame(vol, frame_idx=0)
        assert len(cells) >= 2, (
            f"Expected ≥2 cells from a 3-blob volume, got {len(cells)}"
        )

    def test_returns_correct_frame_index(self):
        """Cells returned should carry the frame index passed to segment_frame."""
        vol = _make_blob_volume(centers=[(5, 30, 30)])
        seg = CellSegmenter(method="blob", min_size=5, anisotropy=1.0,
                            voxel_size_um=(1.0, 1.0, 1.0))
        _, cells = seg.segment_frame(vol, frame_idx=7)
        for cell in cells:
            assert cell.frame == 7

    def test_postprocessing_removes_debris(self):
        """Cells below min_size must be absent in the returned cell list."""
        vol = _make_blob_volume(centers=[(5, 30, 30)], radius=4)
        # Set min_size very large so even the real cell is filtered out
        seg = CellSegmenter(method="blob", min_size=100_000, anisotropy=1.0,
                            voxel_size_um=(1.0, 1.0, 1.0))
        _, cells = seg.segment_frame(vol, frame_idx=0)
        assert len(cells) == 0, (
            "All cells should be filtered when min_size is unrealistically large"
        )

    def test_labels_shape_matches_input(self):
        """Label array must have the same spatial shape as the input."""
        vol = _make_blob_volume()
        seg = CellSegmenter(method="blob", min_size=5, anisotropy=1.0,
                            voxel_size_um=(1.0, 1.0, 1.0))
        labels, _ = seg.segment_frame(vol, frame_idx=0)
        assert labels.shape == vol.shape


# ---------------------------------------------------------------------------
# Postprocess tests
# ---------------------------------------------------------------------------

class TestPostprocessLabels:

    def test_small_labels_removed(self):
        """Objects smaller than min_volume must be zeroed out."""
        labels = np.zeros((10, 20, 20), dtype=np.int32)
        # Large object (volume 8×8×8 = 512)
        labels[1:9, 1:9, 1:9] = 1
        # Tiny object (1 voxel)
        labels[0, 0, 0] = 2

        cleaned = postprocess_labels(labels, min_volume=10, max_volume=100_000)
        assert 1 in np.unique(cleaned), "Large object should survive"
        assert 2 not in np.unique(cleaned), "Tiny object should be removed"

    def test_large_labels_removed(self):
        """Objects larger than max_volume must be zeroed out."""
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[1:19, 1:19, 1:19] = 1   # volume = 18^3 = 5832

        cleaned = postprocess_labels(labels, min_volume=10, max_volume=100)
        assert 1 not in np.unique(cleaned), "Oversized object should be removed"

    def test_empty_input_stays_empty(self):
        """All-zero label array should remain all-zero."""
        labels = np.zeros((5, 10, 10), dtype=np.int32)
        cleaned = postprocess_labels(labels)
        assert cleaned.max() == 0
