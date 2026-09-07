#!/usr/bin/env python3
"""Register final-review rules or run the independent trial supervisor once."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.experiment_supervisor import SECONDARY, register_review, supervise


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--secondary-backup", type=Path, default=SECONDARY)
    args = parser.parse_args()
    if args.register:
        register_review()
    publisher = None
    if not args.no_publish:
        from scripts.fetch_and_rank import publish_dashboard
        publisher = publish_dashboard
    state = supervise(secondary=args.secondary_backup, publisher=publisher)
    print(json.dumps({key: state[key] for key in ("checked_at", "status", "milestones", "backup", "decision", "errors")}, indent=2))
    return 1 if state["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
