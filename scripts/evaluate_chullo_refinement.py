"""Freeze a final protocol, then explicitly evaluate the new geographic holdout."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.final_evaluation import evaluate_frozen, freeze_protocol  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze", help="Read reports/manifests and freeze rules without opening test imagery/labels"
    )
    freeze.add_argument("--run-dir", type=Path, required=True)
    freeze.add_argument("--development-manifest", type=Path, required=True)
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate the frozen protocol once; no tuning or promotion")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(freeze_protocol(args.run_dir, args.development_manifest, args.manifest, args.output))
    else:
        result = evaluate_frozen(args.protocol, args.device)
        print(f"Promotion gate passed: {result['comparison']['promotion_gate_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
