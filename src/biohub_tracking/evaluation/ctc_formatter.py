"""Small CTC track export retained for compatibility with lineage tools."""

from pathlib import Path
from typing import Mapping


class CTCFormatter:
	"""Write tracks as one tab-separated file per track."""

	def write_tracks(self, tracks: Mapping[int, object], output_dir: str) -> None:
		output_path = Path(output_dir)
		output_path.mkdir(parents=True, exist_ok=True)
		for track_id, track in tracks.items():
			with (output_path / f"track_{track_id}.txt").open(
				"w", encoding="utf-8"
			) as handle:
				for frame, centroid in zip(track.frames, track.centroids):
					z, y, x = (int(round(value)) for value in centroid)
					handle.write(f"{frame}\t{z}\t{y}\t{x}\n")
