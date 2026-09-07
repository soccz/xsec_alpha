#!/usr/bin/env python3
"""Retry the newest unacknowledged operator report, without rerunning models."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.operator_report import ROOT, build_report, publish_report, retry_latest  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-error", action="store_true")
    args = parser.parse_args()
    if args.analysis_error:
        sent = publish_report(build_report(error=True, reason="운영 작업 실패 · 이전 후보 참고용"))
        refresh = True
    else:
        path = ROOT / "output" / "latest_operator_report.json"
        try:
            was_pending = json.loads(path.read_text()).get("telegram", {}).get("state") == "pending"
        except (OSError, ValueError):
            was_pending = False
        sent = retry_latest()
        refresh = was_pending and sent
    if refresh:
        from scripts.fetch_and_rank import publish_dashboard
        publish_dashboard()
    return 0 if sent else 1


if __name__ == "__main__":
    raise SystemExit(main())
