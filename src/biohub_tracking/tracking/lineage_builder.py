"""Build and manage cell lineages from tracking results."""

import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from biohub_tracking.tracking.linker import Track, Cell
from biohub_tracking.tracking.division_detector import DivisionEvent

logger = logging.getLogger(__name__)


@dataclass
class LineageTree:
    """Represents a complete cell lineage tree."""
    
    root_id: int
    tracks: Dict[int, Track] = field(default_factory=dict)
    divisions: List[DivisionEvent] = field(default_factory=list)
    
    def get_all_track_ids(self) -> List[int]:
        """Get all track IDs in this lineage."""
        return list(self.tracks.keys())
    
    def get_depth(self) -> int:
        """Get maximum depth of lineage tree."""
        def _depth(track_id: int, visited: Set[int]) -> int:
            if track_id in visited:
                return 0
            visited.add(track_id)
            
            track = self.tracks.get(track_id)
            if track is None or not track.children_ids:
                return 1
            
            return 1 + max(_depth(cid, visited) for cid in track.children_ids)
        
        return _depth(self.root_id, set())


class LineageBuilder:
    """Build cell lineages from frame-to-frame tracking results."""
    
    def __init__(self):
        """Initialize lineage builder."""
        self.tracks: Dict[int, Track] = {}
        self.divisions: List[DivisionEvent] = []
        self.next_track_id = 1
    
    def build_lineages(
        self,
        all_cells: Dict[int, List[Cell]],  # frame -> cells
        links: List[Tuple[int, int, int, float]],  # (frame, cell_id_t1, cell_id_t2, conf)
        divisions: Optional[List[DivisionEvent]] = None
    ) -> List[LineageTree]:
        """Build lineage trees from tracking results.
        
        Args:
            all_cells: Dictionary mapping frame index to list of cells.
            links: Frame-to-frame links with frame index.
            divisions: Optional detected division events.
            
        Returns:
            List of lineage trees.
        """
        self.tracks.clear()
        self.divisions.clear()
        self.next_track_id = 1

        # Build forward and backward link maps
        forward_links = defaultdict(list)  # (frame, cell_id) -> [(next_cell_id, conf)]
        backward_links = defaultdict(list)  # (frame, cell_id) -> [(prev_cell_id, conf)]
        
        for frame, c1, c2, conf in links:
            forward_links[(frame, c1)].append((c2, conf))
            backward_links[(frame + 1, c2)].append((c1, conf))
        
        # Track cells across frames
        cell_to_track = {}  # (frame, cell_id) -> track_id
        
        # Process frames in order
        frames = sorted(all_cells.keys())
        
        for frame in frames:
            cells = all_cells[frame]
            
            for cell in cells:
                key = (frame, cell.id)
                prev_links = backward_links.get(key, [])
                
                if len(prev_links) == 0:
                    # New track (birth or first appearance)
                    track_id = self.next_track_id
                    self.next_track_id += 1
                    
                    track = Track(
                        track_id=track_id,
                        cell_ids=[cell.id],
                        frames=[frame],
                        centroids=[cell.centroid]
                    )
                    self.tracks[track_id] = track
                    cell_to_track[key] = track_id
                
                elif len(prev_links) == 1:
                    # Continue existing track
                    prev_cell_id, _ = prev_links[0]
                    prev_key = (frame - 1, prev_cell_id)
                    
                    if prev_key in cell_to_track:
                        track_id = cell_to_track[prev_key]
                        self.tracks[track_id].cell_ids.append(cell.id)
                        self.tracks[track_id].frames.append(frame)
                        self.tracks[track_id].centroids.append(cell.centroid)
                        cell_to_track[key] = track_id
                
                else:
                    # Multiple parents - could be merge (handle as new track)
                    track_id = self.next_track_id
                    self.next_track_id += 1
                    
                    track = Track(
                        track_id=track_id,
                        cell_ids=[cell.id],
                        frames=[frame],
                        centroids=[cell.centroid],
                        parent_id=None  # Could link to one parent
                    )
                    self.tracks[track_id] = track
                    cell_to_track[key] = track_id
        
        # Handle divisions
        if divisions:
            self._integrate_divisions(divisions, cell_to_track)
        
        # Build lineage trees
        return self._build_trees()
    
    def _integrate_divisions(
        self,
        divisions: List[DivisionEvent],
        cell_to_track: Dict[Tuple[int, int], int]
    ) -> None:
        """Integrate division events into tracks."""
        for div in divisions:
            # Find parent track
            parent_key = (div.parent_frame, div.parent_id)
            if parent_key not in cell_to_track:
                continue
            
            parent_track_id = cell_to_track[parent_key]
            parent_track = self.tracks[parent_track_id]
            
            # Mark end of parent track at division
            # Parent track should end at division frame
            child_track_ids = []
            
            # Update child tracks to reference parent
            for child_id in (div.child1_id, div.child2_id):
                child_key = (div.division_frame, child_id)
                child_track_id = cell_to_track.get(child_key)
                if child_track_id is None or child_track_id == parent_track_id:
                    continue
                child_track_ids.append(child_track_id)
                self.tracks[child_track_id].parent_id = parent_track_id

            parent_track.children_ids = child_track_ids
            
            self.divisions.append(div)
    
    def _build_trees(self) -> List[LineageTree]:
        """Build lineage trees from tracks."""
        # Find root tracks (no parent)
        root_ids = [
            tid for tid, track in self.tracks.items()
            if track.parent_id is None
        ]
        
        trees = []
        for root_id in root_ids:
            # Collect all tracks in this lineage
            lineage_tracks = {}
            to_process = [root_id]
            visited = set()
            
            while to_process:
                tid = to_process.pop(0)
                if tid in visited:
                    continue
                visited.add(tid)
                
                if tid in self.tracks:
                    lineage_tracks[tid] = self.tracks[tid]
                    to_process.extend(self.tracks[tid].children_ids)
            
            tree = LineageTree(
                root_id=root_id,
                tracks=lineage_tracks,
                divisions=[
                    d for d in self.divisions
                    if (d.parent_frame, d.parent_id) in {
                        (frame, cell_id)
                        for track in lineage_tracks.values()
                        for frame, cell_id in zip(track.frames, track.cell_ids)
                    }
                ]
            )
            trees.append(tree)
        
        return trees
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        return self.tracks.get(track_id)
    
    def get_all_tracks(self) -> List[Track]:
        """Get all tracks."""
        return list(self.tracks.values())
    
    def export_to_ctc(self, output_dir: str) -> None:
        """Export lineages to CTC format.
        
        Args:
            output_dir: Output directory for CTC files.
        """
        from biohub_tracking.evaluation.ctc_formatter import CTCFormatter
        
        formatter = CTCFormatter()
        formatter.write_tracks(self.tracks, output_dir)