"""Evaluation metrics for cell tracking and segmentation."""

import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """Metrics for evaluating cell segmentation quality."""
    
    @staticmethod
    def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Compute Intersection over Union."""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return float(intersection / (union + 1e-8))
    
    @staticmethod
    def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Compute Dice coefficient."""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        return float(2.0 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-8))
    
    @staticmethod
    def compute_ap(
        pred_labels: np.ndarray,
        gt_labels: np.ndarray,
        iou_threshold: float = 0.5
    ) -> float:
        """Compute Average Precision for instance segmentation.
        
        Args:
            pred_labels: Predicted instance labels.
            gt_labels: Ground truth instance labels.
            iou_threshold: IoU threshold for true positive.
            
        Returns:
            Average precision.
        """
        pred_ids = np.unique(pred_labels)[1:]
        gt_ids = np.unique(gt_labels)[1:]
        
        if len(pred_ids) == 0 or len(gt_ids) == 0:
            return 0.0
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))
        
        for i, pid in enumerate(pred_ids):
            pred_mask = pred_labels == pid
            for j, gid in enumerate(gt_ids):
                gt_mask = gt_labels == gid
                iou_matrix[i, j] = SegmentationMetrics.compute_iou(pred_mask, gt_mask)
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        # Count true positives
        tp = sum(1 for i, j in zip(row_ind, col_ind) 
                if iou_matrix[i, j] >= iou_threshold)
        
        precision = tp / len(pred_ids) if len(pred_ids) > 0 else 0
        recall = tp / len(gt_ids) if len(gt_ids) > 0 else 0
        
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
    
    @staticmethod
    def compute_seg_score(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
        """Compute CTC SEG score."""
        return SegmentationMetrics.compute_ap(pred_labels, gt_labels, iou_threshold=0.5)


class TrackingMetrics:
    """Metrics for evaluating cell tracking quality."""
    
    @staticmethod
    def compute_tra_score(
        pred_tracks: Dict[int, List[Tuple[int, int]]],  # track_id -> [(frame, cell_id)]
        gt_tracks: Dict[int, List[Tuple[int, int]]]
    ) -> float:
        """Compute CTC TRA (tracking) score.
        
        Args:
            pred_tracks: Predicted tracks.
            gt_tracks: Ground truth tracks.
            
        Returns:
            TRA score in [0, 1].
        """
        # Simplified TRA computation
        # Full implementation would use CTC evaluation code
        
        # Match tracks based on temporal overlap
        track_ious = []
        
        for pred_id, pred_track in pred_tracks.items():
            pred_frames = {f: c for f, c in pred_track}
            
            best_iou = 0.0
            for gt_id, gt_track in gt_tracks.items():
                gt_frames = {f: c for f, c in gt_track}
                
                # Compute temporal overlap
                common_frames = set(pred_frames.keys()) & set(gt_frames.keys())
                if len(common_frames) == 0:
                    continue
                
                # Compute spatial overlap
                # This is simplified - would need actual masks
                overlap = len(common_frames)
                total = len(set(pred_frames.keys()) | set(gt_frames.keys()))
                iou = overlap / total
                
                if iou > best_iou:
                    best_iou = iou
            
            track_ious.append(best_iou)
        
        return np.mean(track_ious) if track_ious else 0.0
    
    @staticmethod
    def compute_det_score(
        pred_labels: np.ndarray,
        gt_labels: np.ndarray
    ) -> float:
        """Compute CTC DET (detection) score."""
        return SegmentationMetrics.compute_ap(pred_labels, gt_labels)
    
    @staticmethod
    def compute_lnk_score(
        pred_links: List[Tuple[int, int, int]],  # (frame, from, to)
        gt_links: List[Tuple[int, int, int]]
    ) -> float:
        """Compute CTC LNK (linking) score."""
        pred_set = set(pred_links)
        gt_set = set(gt_links)
        
        if len(gt_set) == 0:
            return 1.0 if len(pred_set) == 0 else 0.0
        
        tp = len(pred_set & gt_set)
        precision = tp / len(pred_set) if len(pred_set) > 0 else 0
        recall = tp / len(gt_set)
        
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
    
    @staticmethod
    def compute_division_f1(
        pred_divisions: List[Tuple[int, int, int]],  # (parent, child1, child2)
        gt_divisions: List[Tuple[int, int, int]]
    ) -> Tuple[float, float, float]:
        """Compute division detection F1 score.
        
        Returns:
            Tuple of (precision, recall, f1).
        """
        pred_set = set(tuple(sorted(d)) for d in pred_divisions)
        gt_set = set(tuple(sorted(d)) for d in gt_divisions)
        
        if len(gt_set) == 0:
            return (1.0, 1.0, 1.0) if len(pred_set) == 0 else (0.0, 0.0, 0.0)
        
        tp = len(pred_set & gt_set)
        precision = tp / len(pred_set) if len(pred_set) > 0 else 0
        recall = tp / len(gt_set)
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1