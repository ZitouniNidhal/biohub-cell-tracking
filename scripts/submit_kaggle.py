"""Kaggle-friendly entry point for generating submission.csv."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from submission import generate_submission


def main() -> None:
	input_root = Path("/kaggle/input")
	test_dirs = sorted(input_root.glob("**/test"))
	if not test_dirs:
		raise FileNotFoundError("Could not find a Kaggle input/test directory")
	generate_submission(test_dirs[0], Path("submission.csv"))


if __name__ == "__main__":
	main()
