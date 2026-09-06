"""
PyTorch Dataset implementations for cell tracking data.
"""

from typing import List, Tuple
from pathlib import Path


class CellTrackingDataset:
    """Dataset for loading cell images and segmentation masks."""

    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Tuple:
        return ()
