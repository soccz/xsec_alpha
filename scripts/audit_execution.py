#!/usr/bin/env python3
"""Register additional future-only execution diagnostics, or probe public APIs."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.execution_audit import fetch_public, observe_depth, observe_funding, register_execution
from utils.forecast_audit import backup_audit, refresh_audit
import pandas as pd


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        results = []
        for kind in ("depth", "funding"):
            payload, observed = fetch_public(kind, "BTCUSDT")
            data = payload["data"]
            policy = {"quote_window_seconds": 300, "max_age_seconds": 60, "future_tolerance_seconds": 5,
                      "funding_delay_hours": 9, "funding_deadline_hours": 36,
                      "funding_max_gap_hours": 8, "funding_boundary_seconds": 60}
            if kind == "depth":
                validated = observe_depth(observed.isoformat(), "entry", "BTCUSDT", observed, payload, observed, policy)
            else:
                validated = observe_funding(observed.isoformat(), "BTCUSDT", observed - pd.Timedelta(hours=20),
                                             observed - pd.Timedelta(hours=14), payload, observed, policy)
            results.append({"kind": kind, "observed_at": observed.isoformat(), "code": payload["code"],
                            "validation": validated["status"], "interval": "synthetic probe only; not a signal",
                            "rows": len(data) if isinstance(data, list) else {k: len(data[k]) for k in ("bids", "asks")},
                            "recorded_as_evidence": False})
        print(json.dumps(results, indent=2))
        return
    if args.register:
        register_execution()
    state = refresh_audit()
    if not state or not state.get("execution"):
        raise SystemExit("Execution diagnostics not registered")
    print(json.dumps({"execution": state["execution"], "backup": backup_audit()}, indent=2))


if __name__ == "__main__":
    main()
