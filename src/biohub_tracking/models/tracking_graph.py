"""NetworkX-based tracking graph: nodes are detections, edges are temporal links."""

import logging
from typing import Dict, List, Tuple, Optional, Iterator

import numpy as np

from biohub_tracking.tracking.linker import Cell, Track
from biohub_tracking.tracking.division_classifier import DivisionEvent

logger = logging.getLogger(__name__)


try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    logger.warning("networkx not installed — TrackingGraph will be unavailable.")


class TrackingGraph:
    """Directed graph representing the full cell tracking solution.

    Nodes: (frame, cell_id) pairs, with attributes z, y, x (µm), volume.
    Edges: temporal links between nodes, with attribute confidence.
    Division edges are tagged with ``is_division=True``.

    The graph can be converted directly to the Kaggle submission format.
    """

    def __init__(self):
        if not _NX_AVAILABLE:
            raise ImportError("networkx is required. Install with: pip install networkx")
        self._g: "nx.DiGraph" = nx.DiGraph()

    # ------------------------------------------------------------------
    # Building the graph
    # ------------------------------------------------------------------

    def add_cells(self, all_cells: Dict[int, List[Cell]]) -> None:
        """Add all detected cells as nodes."""
        for frame, cells in all_cells.items():
            for cell in cells:
                node_id = self._node_id(frame, cell.id)
                pos = cell.centroid_um if cell.centroid_um is not None else cell.centroid
                self._g.add_node(
                    node_id,
                    frame=frame,
                    cell_id=cell.id,
                    z=float(pos[0]),
                    y=float(pos[1]),
                    x=float(pos[2]),
                    volume=float(cell.volume),
                )

    def add_links(
        self,
        links: List[Tuple[int, int, int, float]],  # (frame, cell_id_t, cell_id_t1, conf)
    ) -> None:
        """Add temporal edges from frame-to-frame links."""
        for frame, c_t, c_t1, conf in links:
            src = self._node_id(frame, c_t)
            dst = self._node_id(frame + 1, c_t1)
            if self._g.has_node(src) and self._g.has_node(dst):
                self._g.add_edge(src, dst, confidence=conf, is_division=False)

    def add_divisions(self, divisions: List[DivisionEvent]) -> None:
        """Tag division edges in the graph."""
        for div in divisions:
            parent = self._node_id(div.parent_frame, div.parent_id)
            child1 = self._node_id(div.division_frame, div.child1_id)
            child2 = self._node_id(div.division_frame, div.child2_id)

            for child in (child1, child2):
                if self._g.has_node(parent) and self._g.has_node(child):
                    if self._g.has_edge(parent, child):
                        self._g[parent][child]["is_division"] = True
                    else:
                        self._g.add_edge(
                            parent, child,
                            confidence=div.confidence,
                            is_division=True,
                        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def nodes(self) -> Iterator:
        return iter(self._g.nodes(data=True))

    def edges(self) -> Iterator:
        return iter(self._g.edges(data=True))

    def division_edges(self) -> List[Tuple]:
        return [
            (u, v, d) for u, v, d in self._g.edges(data=True)
            if d.get("is_division", False)
        ]

    def num_nodes(self) -> int:
        return self._g.number_of_nodes()

    def num_edges(self) -> int:
        return self._g.number_of_edges()

    def get_node_data(self, node_id: str) -> Optional[Dict]:
        return self._g.nodes.get(node_id)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_id(frame: int, cell_id: int) -> str:
        return f"f{frame:05d}_c{cell_id:06d}"

    @classmethod
    def from_cells_and_links(
        cls,
        all_cells: Dict[int, List[Cell]],
        links: List[Tuple[int, int, int, float]],
        divisions: Optional[List[DivisionEvent]] = None,
    ) -> "TrackingGraph":
        """Convenience constructor."""
        graph = cls()
        graph.add_cells(all_cells)
        graph.add_links(links)
        if divisions:
            graph.add_divisions(divisions)
        return graph

    def to_networkx(self) -> "nx.DiGraph":
        """Return underlying networkx DiGraph."""
        return self._g
