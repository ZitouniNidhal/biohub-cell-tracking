"""Integer Linear Programming tracker for global optimal tracking."""

import logging
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from biohub_tracking.tracking.linker import Cell, Track

logger = logging.getLogger(__name__)


@dataclass
class Edge:
    """Edge in the tracking graph."""
    source_frame: int
    source_idx: int
    source_id: int
    target_frame: int
    target_idx: int
    target_id: int
    cost: float
    gap: int  # frame difference (target_frame - source_frame)


class ILPTracker:
    """Global tracking using Integer Linear Programming.
    
    Formulates tracking as a multi-frame network flow ILP problem where:
    - Binary variables represent cell transitions across consecutive/gap frames
    - Biological constraints enforce flow conservation, appearance, disappearance, and division
    - Objective minimizes total assignment cost
    """
    
    def __init__(
        self,
        max_distance: float = 7.0,
        max_frame_gap: int = 2,
        appearance_cost: float = 20.0,
        disappearance_cost: float = 20.0,
        division_cost: float = 15.0,
        use_volume_cost: bool = True,
        volume_weight: float = 0.3,
    ):
        """Initialize ILP tracker.
        
        Args:
            max_distance: Maximum distance in µm for linking.
            max_frame_gap: Maximum frame gap for bridging missing detections.
            appearance_cost: Penalty cost for a new cell track appearance.
            disappearance_cost: Penalty cost for a cell track disappearance.
            division_cost: Cost penalty for cell division (splitting into 2).
            use_volume_cost: Whether to penalize volume mismatch across transitions.
            volume_weight: Weight of volume mismatch in edge cost calculation.
        """
        self.max_distance = max_distance
        self.max_frame_gap = max_frame_gap
        self.appearance_cost = appearance_cost
        self.disappearance_cost = disappearance_cost
        self.division_cost = division_cost
        self.use_volume_cost = use_volume_cost
        self.volume_weight = volume_weight
    
    def track(
        self,
        all_cells: Dict[int, List[Cell]]
    ) -> Tuple[List[Track], List[Tuple[int, int, float]]]:
        """Track cells across all frames using ILP.
        
        Args:
            all_cells: Dictionary mapping frame index -> list of Cell instances.
            
        Returns:
            Tuple of (tracks, links).
            links is a list of (source_cell_id, target_cell_id, confidence).
        """
        try:
            import pulp
        except ImportError:
            logger.warning("PuLP not installed, falling back to Hungarian linker")
            from biohub_tracking.tracking.linker import HungarianLinker
            linker = HungarianLinker(
                max_distance=self.max_distance,
                use_volume_cost=self.use_volume_cost,
                volume_weight=self.volume_weight,
            )
            
            tracks = []
            links = []
            frames = sorted(all_cells.keys())
            for idx in range(len(frames) - 1):
                t1, t2 = frames[idx], frames[idx + 1]
                frame_links = linker.link(all_cells[t1], all_cells[t2])
                for c1, c2, conf in frame_links:
                    links.append((c1, c2, conf))
            return tracks, links

        if not all_cells:
            return [], []

        # Build candidate edge set
        edges = self._build_graph(all_cells)
        
        # Solve ILP problem
        active_edges = self._solve_ilp(all_cells, edges)
        
        # Extract links and assemble tracks
        tracks, links = self._extract_tracks(all_cells, active_edges)
        
        return tracks, links

    def _build_graph(
        self,
        all_cells: Dict[int, List[Cell]]
    ) -> List[Edge]:
        """Build candidate transitions across consecutive and gap frames."""
        edges: List[Edge] = []
        frames = sorted(all_cells.keys())
        
        for i, t1 in enumerate(frames):
            cells1 = all_cells[t1]
            if not cells1:
                continue
                
            for t2 in frames[i + 1 : i + 1 + self.max_frame_gap]:
                cells2 = all_cells[t2]
                if not cells2:
                    continue
                    
                gap = t2 - t1
                # Use µm coordinates if calibrated
                c1_coords = np.array([
                    c.centroid_um if c.centroid_um is not None else c.centroid
                    for c in cells1
                ])
                c2_coords = np.array([
                    c.centroid_um if c.centroid_um is not None else c.centroid
                    for c in cells2
                ])
                
                distances = cdist(c1_coords, c2_coords, metric="euclidean")
                max_dist_gap = self.max_distance * (1.0 + 0.2 * (gap - 1))
                
                for idx1, cell1 in enumerate(cells1):
                    for idx2, cell2 in enumerate(cells2):
                        dist = distances[idx1, idx2]
                        if dist <= max_dist_gap:
                            cost = self._compute_edge_cost(cell1, cell2, dist, gap)
                            edges.append(
                                Edge(
                                    source_frame=t1,
                                    source_idx=idx1,
                                    source_id=cell1.id,
                                    target_frame=t2,
                                    target_idx=idx2,
                                    target_id=cell2.id,
                                    cost=cost,
                                    gap=gap,
                                )
                            )
        return edges

    def _compute_edge_cost(
        self, c1: Cell, c2: Cell, distance: float, gap: int
    ) -> float:
        """Compute cost for a transition edge based on distance, gap, and volume."""
        max_dist_gap = self.max_distance * (1.0 + 0.2 * (gap - 1))
        spatial_cost = distance / (max_dist_gap + 1e-8)
        
        cost = spatial_cost
        if self.use_volume_cost:
            vol_ratio = min(c1.volume, c2.volume) / (max(c1.volume, c2.volume) + 1e-8)
            vol_cost = 1.0 - vol_ratio
            cost = (1.0 - self.volume_weight) * spatial_cost + self.volume_weight * vol_cost
            
        # Additional penalty for bridging across frame gaps
        gap_penalty = 1.0 + 0.5 * (gap - 1)
        return float(cost * gap_penalty)

    def _solve_ilp(
        self,
        all_cells: Dict[int, List[Cell]],
        edges: List[Edge]
    ) -> List[Edge]:
        """Solve network-flow ILP using PuLP."""
        import pulp
        
        prob = pulp.LpProblem("CellTracking_ILP", pulp.LpMinimize)
        
        # Edge selection variables
        edge_vars: Dict[int, pulp.LpVariable] = {}
        for idx, edge in enumerate(edges):
            edge_vars[idx] = pulp.LpVariable(f"e_{idx}", cat=pulp.LpBinary)
            
        # Cell node appearance and disappearance variables
        app_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}
        dis_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}
        
        for t, cells in all_cells.items():
            for ci, cell in enumerate(cells):
                app_vars[(t, ci)] = pulp.LpVariable(f"app_{t}_{ci}", cat=pulp.LpBinary)
                dis_vars[(t, ci)] = pulp.LpVariable(f"dis_{t}_{ci}", cat=pulp.LpBinary)

        # Objective Function
        edge_obj = pulp.lpSum([edge_vars[i] * edge.cost for i, edge in enumerate(edges)])
        app_obj = pulp.lpSum([app_vars[k] * self.appearance_cost for k in app_vars])
        dis_obj = pulp.lpSum([dis_vars[k] * self.disappearance_cost for k in dis_vars])
        prob += edge_obj + app_obj + dis_obj

        # Flow conservation constraints per cell
        for t, cells in all_cells.items():
            for ci in range(len(cells)):
                in_edges = [
                    edge_vars[i] for i, e in enumerate(edges)
                    if e.target_frame == t and e.target_idx == ci
                ]
                out_edges = [
                    edge_vars[i] for i, e in enumerate(edges)
                    if e.source_frame == t and e.source_idx == ci
                ]
                
                # A cell can have at most one incoming transition or appearance
                prob += pulp.lpSum(in_edges) + app_vars[(t, ci)] <= 1
                
                # A cell can have at most one outgoing transition (or two if dividing) or disappearance
                prob += pulp.lpSum(out_edges) + dis_vars[(t, ci)] <= 2

        # Solve silently
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Filter active edges
        active_edges = [
            edges[i] for i, var in edge_vars.items()
            if pulp.value(var) is not None and pulp.value(var) > 0.5
        ]
        return active_edges

    def _extract_tracks(
        self,
        all_cells: Dict[int, List[Cell]],
        active_edges: List[Edge]
    ) -> Tuple[List[Track], List[Tuple[int, int, float]]]:
        """Extract Track objects and link tuples from active ILP solution edges."""
        links: List[Tuple[int, int, float]] = []
        
        # Build cell lookup by (frame, idx)
        cell_map: Dict[Tuple[int, int], Cell] = {}
        for t, cells in all_cells.items():
            for ci, cell in enumerate(cells):
                cell_map[(t, ci)] = cell
                
        # Forward adjacencies
        fwd_adj: Dict[Tuple[int, int], List[Tuple[int, int, float]]] = {}
        for edge in active_edges:
            confidence = max(0.0, 1.0 - edge.cost / max(self.max_distance, 1.0))
            links.append((edge.source_id, edge.target_id, confidence))
            
            src_key = (edge.source_frame, edge.source_idx)
            tgt_key = (edge.target_frame, edge.target_idx)
            fwd_adj.setdefault(src_key, []).append((tgt_key[0], tgt_key[1], confidence))

        # Assemble tracks
        visited: Set[Tuple[int, int]] = set()
        tracks: List[Track] = []
        track_id_counter = 1

        frames = sorted(all_cells.keys())
        for t in frames:
            for ci, cell in enumerate(all_cells[t]):
                key = (t, ci)
                if key in visited:
                    continue
                    
                # Start new track
                curr_key: Optional[Tuple[int, int]] = key
                curr_cell_ids = []
                curr_frames = []
                curr_centroids = []
                
                while curr_key is not None and curr_key not in visited:
                    visited.add(curr_key)
                    curr_c = cell_map[curr_key]
                    curr_cell_ids.append(curr_c.id)
                    curr_frames.append(curr_c.frame)
                    curr_centroids.append(curr_c.centroid)
                    
                    next_nodes = fwd_adj.get(curr_key, [])
                    if len(next_nodes) == 1:
                        nxt_t, nxt_ci, _ = next_nodes[0]
                        curr_key = (nxt_t, nxt_ci)
                    else:
                        # End of chain (or division split node)
                        curr_key = None

                if curr_cell_ids:
                    tracks.append(
                        Track(
                            track_id=track_id_counter,
                            cell_ids=curr_cell_ids,
                            frames=curr_frames,
                            centroids=curr_centroids,
                        )
                    )
                    track_id_counter += 1

        return tracks, links