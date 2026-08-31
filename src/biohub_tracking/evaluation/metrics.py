"""Official Kaggle competition metrics: Edge Jaccard and Division Jaccard."""

from biohub_tracking.constants import DIVISION_SCORE_WEIGHT, MATCH_DIST_UM

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# Type aliases
NodeCoord = Tuple[float, float, float]  # (z, y, x) in µm
NodeId = str
Edge = Tuple[NodeId, NodeId]


# ---------------------------------------------------------------------------
# Kaggle official metric computation
# ---------------------------------------------------------------------------

class KaggleMetrics:
    """Compute the official BioHub Cell Tracking competition score.

    Score = Edge Jaccard + 0.1 * Division Jaccard

    Reference:
        https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation
    """

    MATCH_DIST_UM = MATCH_DIST_UM   # maximum matching distance (µm)

    def __init__(self, matching_distance_um: float = MATCH_DIST_UM):
        self.matching_distance_um = matching_distance_um

    # ------------------------------------------------------------------
    # Main scoring function
    # ------------------------------------------------------------------

    def score(
        self,
        pred_nodes: Dict[NodeId, NodeCoord],           # node_id -> (z, y, x) µm
        pred_edges: List[Edge],                         # list of (src_node_id, dst_node_id)
        gt_nodes: Dict[NodeId, NodeCoord],
        gt_edges: List[Edge],
    ) -> Dict[str, float]:
        """Compute the full Kaggle score.

        Args:
            pred_nodes: Predicted node positions in µm.
            pred_edges: Predicted temporal edges.
            gt_nodes:   Ground-truth node positions in µm.
            gt_edges:   Ground-truth temporal edges.

        Returns:
            Dictionary with keys: 'edge_jaccard', 'division_jaccard', 'score'.
        """
        # 1. Match predicted nodes to GT nodes (bipartite, optimal)
        pred_to_gt, gt_to_pred = self._match_nodes(pred_nodes, gt_nodes)

        # 2. Edge Jaccard
        ej = self._edge_jaccard(pred_edges, gt_edges, pred_to_gt)

        # 3. Division Jaccard
        dj = self._division_jaccard(pred_edges, gt_edges, pred_to_gt)

        total = ej + DIVISION_SCORE_WEIGHT * dj

        return {
            "edge_jaccard": ej,
            "division_jaccard": dj,
            "score": total,
            "n_pred_nodes": len(pred_nodes),
            "n_gt_nodes": len(gt_nodes),
            "n_matched_nodes": len(pred_to_gt),
        }

    # ------------------------------------------------------------------
    # Node matching
    # ------------------------------------------------------------------

    def _match_nodes(
        self,
        pred_nodes: Dict[NodeId, NodeCoord],
        gt_nodes: Dict[NodeId, NodeCoord],
    ) -> Tuple[Dict[NodeId, NodeId], Dict[NodeId, NodeId]]:
        """Optimal bipartite matching between predicted and GT nodes.

        Uses the Hungarian algorithm. Only pairs within
        ``self.matching_distance_um`` µm are considered.

        Returns:
            (pred_to_gt, gt_to_pred) bidirectional mapping.
        """
        if not pred_nodes or not gt_nodes:
            return {}, {}

        pred_ids = list(pred_nodes.keys())
        gt_ids = list(gt_nodes.keys())

        pred_pos = np.array([pred_nodes[i] for i in pred_ids])  # (n_pred, 3)
        gt_pos = np.array([gt_nodes[i] for i in gt_ids])        # (n_gt, 3)

        dist = cdist(pred_pos, gt_pos)  # (n_pred, n_gt)

        # Mask pairs beyond threshold
        infeasible = dist > self.matching_distance_um
        cost = dist.copy()
        cost[infeasible] = self.matching_distance_um * 100

        row_ind, col_ind = linear_sum_assignment(cost)

        pred_to_gt: Dict[NodeId, NodeId] = {}
        gt_to_pred: Dict[NodeId, NodeId] = {}

        for r, c in zip(row_ind, col_ind):
            if dist[r, c] <= self.matching_distance_um:
                pred_to_gt[pred_ids[r]] = gt_ids[c]
                gt_to_pred[gt_ids[c]] = pred_ids[r]

        return pred_to_gt, gt_to_pred

    # ------------------------------------------------------------------
    # Edge Jaccard
    # ------------------------------------------------------------------

    def _edge_jaccard(
        self,
        pred_edges: List[Edge],
        gt_edges: List[Edge],
        pred_to_gt: Dict[NodeId, NodeId],
    ) -> float:
        """Compute edge Jaccard score.

        An edge (u, v) in the prediction is a True Positive if:
          - both u and v are matched to GT nodes
          - the matched GT edge (gt_u, gt_v) exists in gt_edges

        Kaggle penalises over-prediction: FP = len(pred) - TP.
        FN = len(gt) - TP.

        Jaccard = TP / (TP + FP + FN)
        """
        gt_edge_set: Set[Edge] = set(gt_edges)

        tp = 0
        for u, v in pred_edges:
            gt_u = pred_to_gt.get(u)
            gt_v = pred_to_gt.get(v)
            if gt_u is not None and gt_v is not None:
                if (gt_u, gt_v) in gt_edge_set:
                    tp += 1

        fp = len(pred_edges) - tp
        fn = len(gt_edges) - tp
        denominator = tp + fp + fn

        return tp / denominator if denominator > 0 else 1.0

    # ------------------------------------------------------------------
    # Division Jaccard
    # ------------------------------------------------------------------

    def _division_jaccard(
        self,
        pred_edges: List[Edge],
        gt_edges: List[Edge],
        pred_to_gt: Dict[NodeId, NodeId],
    ) -> float:
        """Compute division Jaccard score.

        A predicted division (parent → child1, parent → child2) is a TP if:
          - All three nodes match GT nodes
          - The GT graph contains a component with the GT parent and both GT children

        Jaccard = TP / (TP + FP + FN)
        """
        # Build parent -> children maps
        pred_divs = self._find_divisions(pred_edges)
        gt_divs = self._find_divisions(gt_edges)

        # Map predicted division parents to GT parents
        gt_div_set: Set[Tuple[NodeId, ...]] = set()
        for gt_parent, gt_children in gt_divs.items():
            sorted_children = tuple(sorted(gt_children))
            gt_div_set.add((gt_parent,) + sorted_children)

        tp = 0
        for pred_parent, pred_children in pred_divs.items():
            gt_parent = pred_to_gt.get(pred_parent)
            if gt_parent is None:
                continue
            gt_children_pred = tuple(sorted(
                pred_to_gt.get(c, "") for c in pred_children
            ))
            if (gt_parent,) + gt_children_pred in gt_div_set:
                tp += 1

        fp = len(pred_divs) - tp
        fn = len(gt_divs) - tp
        denominator = tp + fp + fn

        return tp / denominator if denominator > 0 else 1.0

    @staticmethod
    def _find_divisions(edges: List[Edge]) -> Dict[NodeId, List[NodeId]]:
        """Identify nodes with 2 outgoing edges (divisions)."""
        from collections import defaultdict
        out_edges: Dict[NodeId, List[NodeId]] = defaultdict(list)
        for u, v in edges:
            out_edges[u].append(v)
        return {u: vs for u, vs in out_edges.items() if len(vs) == 2}


# ---------------------------------------------------------------------------
# Keep backward-compatible classes
# ---------------------------------------------------------------------------

class SegmentationMetrics:
    """Metrics for evaluating cell segmentation quality."""

    @staticmethod
    def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return float(intersection / (union + 1e-8))

    @staticmethod
    def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        return float(2.0 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-8))

    @staticmethod
    def compute_ap(
        pred_labels: np.ndarray,
        gt_labels: np.ndarray,
        iou_threshold: float = 0.5,
    ) -> float:
        pred_ids = np.unique(pred_labels)[1:]
        gt_ids = np.unique(gt_labels)[1:]
        if len(pred_ids) == 0 or len(gt_ids) == 0:
            return 0.0
        iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))
        for i, pid in enumerate(pred_ids):
            pred_mask = pred_labels == pid
            for j, gid in enumerate(gt_ids):
                gt_mask = gt_labels == gid
                iou_matrix[i, j] = SegmentationMetrics.compute_iou(pred_mask, gt_mask)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        tp = sum(1 for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] >= iou_threshold)
        precision = tp / len(pred_ids) if len(pred_ids) > 0 else 0
        recall = tp / len(gt_ids) if len(gt_ids) > 0 else 0
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0

    @staticmethod
    def compute_seg_score(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
        return SegmentationMetrics.compute_ap(pred_labels, gt_labels, iou_threshold=0.5)


class TrackingMetrics:
    """Metrics for evaluating cell tracking quality."""

    @staticmethod
    def compute_tra_score(
        pred_tracks: Dict[int, List[Tuple[int, int]]],
        gt_tracks: Dict[int, List[Tuple[int, int]]],
    ) -> float:
        track_ious = []
        for pred_id, pred_track in pred_tracks.items():
            pred_frames = {f: c for f, c in pred_track}
            best_iou = 0.0
            for gt_id, gt_track in gt_tracks.items():
                gt_frames = {f: c for f, c in gt_track}
                common_frames = set(pred_frames.keys()) & set(gt_frames.keys())
                if not common_frames:
                    continue
                iou = len(common_frames) / len(set(pred_frames) | set(gt_frames))
                if iou > best_iou:
                    best_iou = iou
            track_ious.append(best_iou)
        return float(np.mean(track_ious)) if track_ious else 0.0

    @staticmethod
    def compute_lnk_score(
        pred_links: List[Tuple[int, int, int]],
        gt_links: List[Tuple[int, int, int]],
    ) -> float:
        pred_set = set(pred_links)
        gt_set = set(gt_links)
        if not gt_set:
            return 1.0 if not pred_set else 0.0
        tp = len(pred_set & gt_set)
        precision = tp / len(pred_set) if pred_set else 0
        recall = tp / len(gt_set)
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0

    @staticmethod
    def compute_division_f1(
        pred_divisions: List[Tuple[int, int, int]],
        gt_divisions: List[Tuple[int, int, int]],
    ) -> Tuple[float, float, float]:
        pred_set = set(tuple(sorted(d)) for d in pred_divisions)
        gt_set = set(tuple(sorted(d)) for d in gt_divisions)
        if not gt_set:
            return (1.0, 1.0, 1.0) if not pred_set else (0.0, 0.0, 0.0)
        tp = len(pred_set & gt_set)
        precision = tp / len(pred_set) if pred_set else 0
        recall = tp / len(gt_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1