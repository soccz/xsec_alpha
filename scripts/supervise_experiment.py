#!/usr/bin/env python3
"""Register final-review rules or run the independent trial supervisor once."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.experiment_supervisor import register_review, seal_completed, supervise


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--seal-completed", action="store_true")
    parser.add_argument("--no-alert", action="store_true")
    parser.add_argument("--secondary-backup", type=Path,
                        help="Optional different-filesystem copy; default is verified local recovery only")
    args = parser.parse_args()
    if args.register:
        register_review()
    if args.seal_completed:
        seal_completed()
    publisher = None
    if not args.no_publish:
        from scripts.fetch_and_rank import publish_dashboard
        publisher = publish_dashboard
    state = supervise(secondary=args.secondary_backup, publisher=publisher)
    if not args.no_alert:
        from utils.ops_status import notify_operations
        notify_operations(secondary=args.secondary_backup)
    print(json.dumps({key: state[key] for key in ("checked_at", "status", "milestones", "backup", "decision", "errors")}, indent=2))
    return 1 if state["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
