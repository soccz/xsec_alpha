"""Private receipt derived from the exact operator payload, not from export time."""

import hashlib
import json


def report_identity(report):
    picks = [report.get("signals"), report.get("ideas")]
    raw = json.dumps(
        picks, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()
    return {
        "run_id": report.get("run_id"),
        "data_asof": report.get("data_asof"),
        "telegram": report.get("telegram"),
        "picks_sha256": hashlib.sha256(raw).hexdigest(),
    }
