#!/usr/bin/env python3
"""Register or refresh the saved-score audit; never sends messages or changes gates."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.forecast_audit import backup_audit, refresh_audit, start_audit


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--probe-quotes", action="store_true", help="Read public quotes without creating experiment evidence")
    args = parser.parse_args()
    if args.probe_quotes:
        from utils.venue_observer import fetch_quotes, observe_quotes
        payload, observed = fetch_quotes()
        intent = {"signal_at": observed.isoformat(), "entry_at": observed.isoformat(),
                  "symbols": {"KRW-BTC": "BTCUSDT", "KRW-ETH": "ETHUSDT", "KRW-XRP": "XRPUSDT"}}
        probe = observe_quotes(intent, "entry", payload, observed)
        print(json.dumps({"observed_at": observed.isoformat(), "api_rows": len(payload["data"]),
                          "probe_status": probe["status"], "fresh_quotes": len(probe["rows"]),
                          "errors": probe["errors"], "recorded_as_evidence": False}, indent=2))
        if probe["status"] != "observed":
            raise SystemExit(1)
        return
    if args.register:
        start_audit()
    state = refresh_audit()
    if state is None:
        raise SystemExit("Forecast audit not registered")
    print(json.dumps({"status": state["status"], "counts": state["counts"],
                      "first_signal_at": state["policy"]["first_signal_at"],
                      "backup": backup_audit()}, indent=2))


if __name__ == "__main__":
    main()
