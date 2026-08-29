"""Unit tests for the Kaggle competition evaluation metrics."""

import pytest

from biohub_tracking.evaluation.metrics import KaggleMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nodes(*positions):
    """Build a node dict from (z, y, x) tuples. IDs are auto-assigned."""
    return {str(i): pos for i, pos in enumerate(positions)}


# ---------------------------------------------------------------------------
# KaggleMetrics — Edge Jaccard
# ---------------------------------------------------------------------------

class TestEdgeJaccard:

    def setup_method(self):
        self.m = KaggleMetrics(matching_distance_um=7.0)

    def test_perfect_prediction_score_is_one(self):
        """When pred == GT, edge Jaccard must be 1.0."""
        nodes = _nodes((0.0, 0.0, 0.0), (0.0, 0.0, 5.0))
        edges = [("0", "1")]
        result = self.m.score(nodes, edges, nodes, edges)
        assert result["edge_jaccard"] == pytest.approx(1.0)
        assert result["score"] >= 1.0

    def test_empty_prediction_score_is_zero(self):
        """Empty prediction with non-empty GT → edge Jaccard = 0."""
        gt_nodes = _nodes((0.0, 0.0, 0.0), (0.0, 0.0, 5.0))
        gt_edges = [("0", "1")]
        result = self.m.score({}, [], gt_nodes, gt_edges)
        assert result["edge_jaccard"] == pytest.approx(0.0)

    def test_all_wrong_edges_score_is_zero(self):
        """Edges between unmatched nodes → edge Jaccard = 0."""
        pred_nodes = _nodes((0.0, 0.0, 0.0), (0.0, 0.0, 5.0))
        gt_nodes = _nodes((100.0, 100.0, 100.0), (100.0, 100.0, 105.0))
        pred_edges = [("0", "1")]
        gt_edges = [("0", "1")]
        result = self.m.score(pred_nodes, pred_edges, gt_nodes, gt_edges)
        assert result["edge_jaccard"] == pytest.approx(0.0)

    def test_partial_prediction_jaccard(self):
        """Predict 1 of 2 GT edges → Jaccard = 1/(1+0+1) = 0.5."""
        nodes = _nodes((0.0, 0.0, 0.0), (0.0, 0.0, 5.0), (0.0, 0.0, 10.0))
        pred_edges = [("0", "1")]          # predict only first edge
        gt_edges = [("0", "1"), ("1", "2")]  # GT has two edges
        result = self.m.score(nodes, pred_edges, nodes, gt_edges)
        # TP=1, FP=0, FN=1 → Jaccard = 1/2 = 0.5
        assert result["edge_jaccard"] == pytest.approx(0.5)

    def test_over_prediction_penalised(self):
        """Predicting extra edges (FP) lowers the Jaccard score."""
        nodes = _nodes((0.0, 0.0, 0.0), (0.0, 0.0, 5.0), (0.0, 0.0, 10.0))
        pred_edges = [("0", "1"), ("1", "2")]  # predict two edges
        gt_edges = [("0", "1")]                # GT has only one
        result = self.m.score(nodes, pred_edges, nodes, gt_edges)
        # TP=1, FP=1, FN=0 → Jaccard = 1/2 = 0.5
        assert result["edge_jaccard"] == pytest.approx(0.5)

    def test_no_gt_no_pred_score_is_one(self):
        """No ground truth and no prediction → perfect score (vacuously true)."""
        nodes = _nodes((0.0, 0.0, 0.0))
        result = self.m.score(nodes, [], nodes, [])
        assert result["edge_jaccard"] == pytest.approx(1.0)

    def test_node_match_uses_distance_threshold(self):
        """Nodes beyond 7 µm should NOT be matched, so edges are FP."""
        pred_nodes = _nodes((0.0, 0.0, 0.0), (0.0, 0.0, 5.0))
        gt_nodes = _nodes((0.0, 0.0, 100.0), (0.0, 0.0, 105.0))  # 100 µm away
        pred_edges = [("0", "1")]
        gt_edges = [("0", "1")]
        result = self.m.score(pred_nodes, pred_edges, gt_nodes, gt_edges)
        assert result["edge_jaccard"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# KaggleMetrics — Division Jaccard
# ---------------------------------------------------------------------------

class TestDivisionJaccard:

    def setup_method(self):
        self.m = KaggleMetrics(matching_distance_um=7.0)

    def test_correct_division_detected(self):
        """A correctly predicted division (parent → 2 children) is a TP."""
        # parent node "0", children "1" and "2" — all at same position
        nodes = _nodes((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        edges = [("0", "1"), ("0", "2")]  # parent with 2 outgoing edges
        result = self.m.score(nodes, edges, nodes, edges)
        assert result["division_jaccard"] == pytest.approx(1.0)

    def test_no_predicted_divisions(self):
        """If pred has no divisions but GT does, division Jaccard = 0."""
        nodes = _nodes((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        gt_edges = [("0", "1"), ("0", "2")]
        # Predict only one child (not a division)
        pred_edges = [("0", "1")]
        result = self.m.score(nodes, pred_edges, nodes, gt_edges)
        assert result["division_jaccard"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# KaggleMetrics — Combined score
# ---------------------------------------------------------------------------

class TestCombinedScore:

    def test_score_formula(self):
        """Combined score = edge_jaccard + 0.1 * division_jaccard."""
        nodes = _nodes((0.0, 0.0, 0.0), (0.0, 0.0, 5.0))
        edges = [("0", "1")]
        m = KaggleMetrics()
        result = m.score(nodes, edges, nodes, edges)
        expected = result["edge_jaccard"] + 0.1 * result["division_jaccard"]
        assert result["score"] == pytest.approx(expected)

    def test_result_contains_all_keys(self):
        """Result dict must contain all documented metric keys."""
        nodes = _nodes((0.0, 0.0, 0.0))
        m = KaggleMetrics()
        result = m.score(nodes, [], nodes, [])
        for key in ("edge_jaccard", "division_jaccard", "score",
                    "n_pred_nodes", "n_gt_nodes", "n_matched_nodes"):
            assert key in result, f"Missing key: {key}"
