#!/usr/bin/env python3
"""Register once or maintain the shadow rotation pilot without sending trade reports."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.rotation_pilot import backup_pilot, refresh_pilot, start_pilot


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if args.register:
        start_pilot()
    summary = refresh_pilot()
    backup = backup_pilot()
    print(json.dumps({"status": summary["status"], "protocol": summary.get("protocol"),
                      "decision": summary.get("decision"), "backup": backup}, indent=2))


if __name__ == "__main__":
    main()
