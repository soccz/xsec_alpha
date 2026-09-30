#!/usr/bin/env python3
"""Register prospective regime observation, or settle and summarize saved slots."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.regime_observer import refresh_observation, start_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if args.register:
        start_observation()
    summary = refresh_observation()
    print(json.dumps({k: summary.get(k) for k in (
        "status", "first_signal_at", "n_recorded", "n_matured", "n_missed", "automatic_switching")}, indent=2))


if __name__ == "__main__":
    main()
