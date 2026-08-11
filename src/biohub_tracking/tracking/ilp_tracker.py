"""Integer Linear Programming tracker for global optimal tracking."""

import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np

from biohub_tracking.tracking.linker import Cell, Track

logger = logging.getLogger(__name__)


@dataclass
class Edge:
    """Edge in the tracking graph."""
    source: int  # cell index in frame t
    target: int  # cell index in frame t+1
    cost: float
    frame: int


class ILPTracker:
    """Global tracking using Integer Linear Programming.
    
    Formulates tracking as an ILP problem where:
    - Variables represent possible cell transitions
    - Constraints enforce biological consistency
    - Objective minimizes total assignment cost
    """
    
    def __init__(
        self,
        max_distance: float = 50.0,
        max_frame_gap: int = 3,
        appearance_cost: float = 100.0,
        disappearance_cost: float = 100.0,
        division_cost: float = 50.0
    ):
        """Initialize ILP tracker.
        
        Args:
            max_distance: Maximum distance for linking.
            max_frame_gap: Maximum frame gap.
            appearance_cost: Cost for track appearance.
            disappearance_cost: Cost for track disappearance.
            division_cost: Cost for cell division.
        """
        self.max_distance = max_distance
        self.max_frame_gap = max_frame_gap
        self.appearance_cost = appearance_cost
        self.disappearance_cost = disappearance_cost
        self.division_cost = division_cost
    
    def track(
        self,
        all_cells: Dict[int, List[Cell]]
    ) -> Tuple[List[Track], List[Tuple[int, int, int]]]:
        """Track cells across all frames using ILP.
        
        Args:
            all_cells: Dictionary mapping frame to list of cells.
            
        Returns:
            Tuple of (tracks, links).
        """
        try:
            import pulp
        except ImportError:
            logger.warning("PuLP not installed, falling back to Hungarian linker")
            from biohub_tracking.tracking.linker import HungarianLinker
            linker = HungarianLinker(max_distance=self.max_distance)
            
            tracks = []
            links = []
            for t in sorted(all_cells.keys())[:-1]:
                frame_links = linker.link(all_cells[t], all_cells[t + 1])
                for c1, c2, conf in frame_links:
                    links.append((c1, c2, t))
            return tracks, links
        
        # Build tracking graph
        edges = self._build_graph(all_cells)
        
        # Solve ILP
        solution = self._solve_ilp(all_cells, edges)
        
        # Extract tracks from solution
        tracks, links = self._extract_tracks(all_cells, solution)
        
        return tracks, links
    
    def _build_graph(
        self,
        all_cells: Dict[int, List[Cell]]
    ) -> List[Edge]:
        """Build tracking graph with all possible edges.
        
        Args:
            all_cells: Frame -> cells mapping.
            
        Returns:
            List of possible edges.
        """
        from scipy.spatial.distance import cdist
        
        edges = []
        frames = sorted(all_cells.keys())
        
        for i, t1 in enumerate(frames):
            cells1 = all_cells[t1]
            
            for t2 in frames[i + 1:i + 1 + self.max_frame_gap]:
                cells2 = all_cells[t2]
                
                if len(cells1) == 0 or len(cells2) == 0:
                    continue
                
                # Compute distances
                centroids1 = np.array([c.centroid for c in cells1])
                centroids2 = np.array([c.centroid for c in cells2])
                distances = cdist(centroids1, centroids2)
                
                # Add edges within distance threshold
                for i1, c1 in enumerate(cells1):
                    for i2, c2 in enumerate(cells2):
                        if distances[i1, i2] < self.max_distance:
                            # Cost based on distance and feature similarity
                            cost = self._compute_edge_cost(c1, c2, distances[i1, i2])
                            edges.append(Edge(i1, i2, cost, t1))
        
        return edges
    
    def _compute_edge_cost(self, c1: Cell, c2: Cell, distance: float) -> float:
        """Compute cost for an edge between two cells."""
        # Spatial cost
        spatial_cost = distance / self.max_distance
        
        # Volume similarity
        vol_ratio = min(c1.volume, c2.volume) / (max(c1.volume, c2.volume) + 1e-8)
        volume_cost = 1.0 - vol_ratio
        
        # Intensity similarity
        int1 = c1.features.get("intensity_mean", 0)
        int2 = c2.features.get("intensity_mean", 0)
        intensity_cost = abs(int1 - int2) / (max(int1, int2, 1) + 1e-8)
        
        # Combined cost
        cost = 0.5 * spatial_cost + 0.3 * volume_cost + 0.2 * intensity_cost
        
        return cost
    
    def _solve_ilp(
        self,
        all_cells: Dict[int, List[Cell]],
        edges: List[Edge]
    ) -> Dict:
        """Solve tracking as ILP problem.
        
        Args:
            all_cells: Frame -> cells.
            edges: Possible edges.
            
        Returns:
            Solution dictionary.
        """
        import pulp
        
        # Create problem
        prob = pulp.LpProblem("CellTracking", pulp.LpMinimize)
        
        # Create binary variables for edges
        edge_vars = {}
        for i, edge in enumerate(edges):
            var = pulp.LpVariable(f"edge_{i}", cat='Binary')
            edge_vars[i] = var
        
        # Objective: minimize total cost
        prob += pulp.lpSum([
            edge_vars[i] * edge.cost
            for i, edge in enumerate(edges)
        ])
        
        # Constraints: each cell can have at most one incoming and one outgoing edge
        # (Simplified - full implementation would include appearance/disappearance)
        
        frames = sorted(all_cells.keys())
        for t in frames[:-1]:
            cells_t = all_cells[t]
            for ci, cell in enumerate(cells_t):
                # Outgoing edges from this cell
                outgoing = [
                    edge_vars[i] for i, edge in enumerate(edges)
                    if edge.frame == t and edge.source == ci
                ]
                if outgoing:
                    prob += pulp.lpSum(outgoing) <= 1
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        solution = {
            'edges': [edges[i] for i, var in edge_vars.items() 
                     if pulp.value(var) > 0.5]
        }
        
        return solution
    
    def _extract_tracks(
        self,
        all_cells: Dict[int, List[Cell]],
        solution: Dict
    ) -> Tuple[List[Track], List[Tuple[int, int, int]]]:
        """Extract tracks from ILP solution."""
        # Simplified track extraction
        tracks = []
        links = []
        
        for edge in solution['edges']:
            # This is simplified - proper implementation would chain edges
            pass
        
        return tracks, links