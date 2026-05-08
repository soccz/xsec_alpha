#!/usr/bin/env python3
"""Build encrypted dashboard payloads and write them to the public site repo.

Default target: /home/soccz/22tb/soccz.github.io/projects/xsec_alpha/dashboard/data
PIN default: 9963 (matches the Prelude dashboard).

Usage:
    python scripts/build_dashboard_export.py
    python scripts/build_dashboard_export.py --target /tmp/out --pin 1234 --plain  # skip encryption (debug)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.dashboard_export import (  # noqa: E402
    PIN_DEFAULT,
    build_accuracy_payload,
    build_history_payload,
    build_summary_payload,
    encrypt_payload,
)

DEFAULT_TARGET = Path(
    "/home/soccz/22tb/soccz.github.io/projects/xsec-alpha/dashboard/data"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(DEFAULT_TARGET),
                    help="output directory for {summary,history,accuracy}.json")
    ap.add_argument("--pin", default=PIN_DEFAULT)
    ap.add_argument("--history-days", type=int, default=60)
    ap.add_argument("--ic-days", type=int, default=60)
    ap.add_argument("--plain", action="store_true",
                    help="write plaintext (skip encryption) — debug only")
    args = ap.parse_args()

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    payloads = {
        "summary.json": build_summary_payload(),
        "history.json": build_history_payload(history_days=args.history_days,
                                              ic_days=args.ic_days),
        "accuracy.json": build_accuracy_payload(history_days=args.history_days),
    }

    for name, plain in payloads.items():
        body = json.dumps(plain, ensure_ascii=False, default=str).encode("utf-8")
        if args.plain:
            (target / name).write_bytes(body)
        else:
            env = encrypt_payload(body, args.pin)
            (target / name).write_text(json.dumps(env, ensure_ascii=False, indent=2))
        print(f"wrote {target/name}  ({len(body):,} bytes plaintext)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
