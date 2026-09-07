#!/usr/bin/env python3
"""Start the fixed trial once, or settle/report it without sending Telegram."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.prospective import experiment_summary, maintain_experiment, start_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true", help="Freeze current model and register 60 future slots")
    parser.add_argument("--settle", action="store_true", help="Evaluate stored snapshots on raw candle prices")
    args = parser.parse_args()
    if args.start:
        start_experiment()
    if args.settle:
        maintain_experiment()
    print(json.dumps(experiment_summary(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
