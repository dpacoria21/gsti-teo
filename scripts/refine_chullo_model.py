"""Run the predeclared validation-only experiments for the second Chullo pilot."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.refinement import run  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/training/chullo_local_v2/development_manifest.json")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/training/chullo_refinement_v2"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--suite", choices=("from_usa", "warm_start", "detail_scale"), default="from_usa")
    args = parser.parse_args()
    run(args.manifest, args.output, args.epochs, args.suite)
