"""Fine-tune a separate building detector; never replaces production weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.model import MODEL_PATH  # noqa: E402
from src.training.runner import TrainingConfig, run_training  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; must not already exist")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    arguments = vars(parser.parse_args())
    manifest, output_dir, checkpoint = (arguments.pop(key) for key in ("manifest", "output_dir", "checkpoint"))
    arguments["amp"] = not arguments.pop("no_amp")
    run_training(manifest, output_dir, TrainingConfig(**arguments), checkpoint)


if __name__ == "__main__":
    main()
