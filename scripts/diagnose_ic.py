#!/usr/bin/env python3
"""Reconstruct the latest SHORT freeze without altering its gate or original evidence."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import prospective as trial
from utils.experiment_supervisor import _envelope, _write
from utils.ic_incident import build_incident, summary
from utils.model_release import model_release_guard
from utils.run_lock import run_lock, stable_data_read_lock


def main():
    os.umask(0o077)
    with run_lock("ic_incident", timeout_sec=5):
        with stable_data_read_lock(timeout_sec=480), model_release_guard():
            report = build_incident()
        checksum = trial._digest(report)
        folder = trial.ROOT / "output/ic_incident"
        _write(folder / "reports" / (checksum+".json"), _envelope(report), immutable=True)
        _write(folder / "summary.json", summary(report))
    print(json.dumps({"report_sha256":checksum,"controls":report["controls"],"events":[{
        k:r[k] for k in ("signal_at","recorded_ic","saved_ic","selected_proxy_net_pct","regime")
    } for r in report["events"]]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
