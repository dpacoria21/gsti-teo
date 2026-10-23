"""Download verified SpaceNet reference imagery with geographic holdouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from src.training.acquisition import prepare_dataset

    parser = argparse.ArgumentParser(description="Prepare a spatially separated SpaceNet 2 Khartoum training subset.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/training/spacenet2_khartoum")
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--val", type=int, default=40)
    parser.add_argument("--test", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--gap-m", type=float, default=200)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare_dataset(
        args.output,
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed,
        gap_m=args.gap_m,
        workers=args.workers,
        callback=lambda event: print(json.dumps(event), flush=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
