#!/usr/bin/env python3
"""OnFailure hook: tell the operator on Telegram that a systemd unit failed.

The September DB backup failed 20 days in a row and a 10-03 supervisor timeout
failed silently: user units had no OnFailure and the alpha OnFailure could
never fire. Units now point OnFailure at xsec-unit-failure[-user]@%n.service,
which runs this script. One notice per unit per 6 h; only the unit name leaves
the server (no paths, no raw errors), like the existing operations notice.

Usage: python scripts/notify_unit_failure.py UNIT_NAME
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "output" / "unit_failures" / "state.json"
QUIET = timedelta(hours=6)


def notify(unit: str, now: datetime | None = None, state_path: Path = STATE, sender=None) -> bool:
    now = now or datetime.now(timezone.utc)
    unit = re.sub(r"[^A-Za-z0-9@._:-]", "", unit)[:120] or "unknown"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    last = state.get(unit)
    if last and now - datetime.fromisoformat(last) < QUIET:
        return True
    message = (f"<b>xsec 서비스 실패</b>\n{unit}\n서버에서 상태 확인이 필요합니다.\n"
               "정기 코인 보고와 별도 상태입니다.")
    if sender is None:
        from utils.telegram import send_message as sender
    if not sender(message):
        return False
    state[unit] = now.isoformat()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, state_path)
    return True


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(0 if notify(sys.argv[1] if len(sys.argv) > 1 else "unknown") else 1)
