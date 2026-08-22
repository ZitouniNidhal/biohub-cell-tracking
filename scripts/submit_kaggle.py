<<<<<<< HEAD
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
=======
"""Kaggle-friendly entry point for generating submission.csv."""

from pathlib import Path
>>>>>>> a46ae14 (feat: implement submission generation and update sample processing logic)

from submission import generate_submission


<<<<<<< HEAD
def find_test_dir(input_root: Path) -> Path:
	"""Find the competition test directory in Kaggle's mounted inputs."""
	candidates = sorted(
		path for path in input_root.glob("**/test")
		if any(path.glob("*.zarr"))
	)
	if candidates:
		return candidates[0]

	sample_dirs = sorted({path.parent for path in input_root.glob("**/*.zarr")})
	if sample_dirs:
		return sample_dirs[0]

	raise FileNotFoundError(
		f"No .zarr samples found below {input_root}. "
		"Attach the BioHub competition data to this Kaggle notebook."
	)


def main() -> None:
	input_root = Path("/kaggle/input")
	generate_submission(find_test_dir(input_root), Path("submission.csv"))
=======
def main() -> None:
	input_root = Path("/kaggle/input")
	test_dirs = sorted(input_root.glob("**/test"))
	if not test_dirs:
		raise FileNotFoundError("Could not find a Kaggle input/test directory")
	generate_submission(test_dirs[0], Path("submission.csv"))
>>>>>>> a46ae14 (feat: implement submission generation and update sample processing logic)


if __name__ == "__main__":
	main()
