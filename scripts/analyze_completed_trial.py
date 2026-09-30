#!/usr/bin/env python3
"""Write separate diagnostics; never update the completed trial or live recommendations."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import prospective as trial
from utils.experiment_supervisor import _write
from utils.rotation_research import followup_plan
from utils.trial_diagnostics import build_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=trial.ROOT)
    args = parser.parse_args()
    report = build_diagnostics(args.root)
    report["followup_plan"] = followup_plan(report)
    folder = args.root / "output/trial_diagnostics"
    _write(folder / "report.json", report)
    # The dashboard gets aggregates only; per-coin attribution stays in the local report.
    summary = {key: value for key, value in report.items() if key not in ("windows", "coin_attribution")}
    _write(folder / "summary.json", summary)
    _write(folder / "followup_plan.json", report["followup_plan"])
    print(json.dumps({"report": str(folder / "report.json"), "n_windows": report["n_windows"],
                      "original_decision": report["original_decision"]["code"],
                      "paired_mean_pp": report["paired"]["mean_pct"],
                      "followup_status": report["followup_plan"]["status"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
