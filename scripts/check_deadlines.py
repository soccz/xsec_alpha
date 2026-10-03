#!/usr/bin/env python3
"""Daily governance deadline check — deadlines with defaults must live in code.

The README §14-bis deadman existed only as prose, so nothing happened on its
date (2026-09-01) and the verdict was recorded 33 days late. Every deadline that
carries a default now lives in docs/deadlines.json and is checked here daily:

- reminders at the configured days before the due time (each sent once);
- after the due time, if the done-condition is unmet, the default is recorded in
  output/deadlines/status.json and one Telegram notice is sent.

This script only records and announces. It never edits configs or stops
services; applying a default that changes live behaviour stays a recorded
human/agent action (DECISIONS.md, journal).

Usage: python scripts/check_deadlines.py [--dry-run] [--now 2026-10-31T15:00:00Z]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "deadlines.json"
STATE_DIR = ROOT / "output" / "deadlines"
KST = timezone(timedelta(hours=9))


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def condition_met(root: Path, cond: dict) -> bool:
    if "glob" in cond:
        return bool(glob.glob(str(root / cond["glob"])))
    if "json_field" in cond:
        spec = cond["json_field"]
        try:
            value = json.loads((root / spec["path"]).read_text())
            for key in spec["field"].split("."):
                value = value[key]
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return value == spec["equals"]
    raise ValueError(f"unknown deadline condition: {cond}")


def evaluate(registry: dict, root: Path, now: datetime, sent: dict) -> tuple[list, list]:
    """Return (events, rows). events: [(keys_to_mark, message)]; nothing is mutated."""
    events, rows = [], []
    for d in registry["deadlines"]:
        due = _utc(d["due_utc"])
        left = due - now
        done = condition_met(root, d["done_if"])
        status = "done" if done else ("overdue_default_applies" if left <= timedelta(0) else "pending")
        due_kst = due.astimezone(KST).strftime("%Y-%m-%d %H:%M KST")
        rows.append({"id": d["id"], "title": d["title"], "due_utc": d["due_utc"], "status": status,
                     "default": d["default"], "days_left": round(left.total_seconds() / 86400, 2)})
        if status == "overdue_default_applies":
            key = f"{d['id']}:overdue"
            if key not in sent:
                events.append(([key], f"⛔ xsec_alpha 기한 경과: {d['title']}\n마감 {due_kst} · 조건 미충족\n"
                                      f"기본값 확정: {d['default']}\nDECISIONS.md와 개발일지에 기록이 필요합니다."))
        elif status == "pending":
            due_thresholds = sorted(n for n in d.get("remind_days_before", []) if left <= timedelta(days=n))
            if due_thresholds:
                nearest = due_thresholds[0]
                keys = [f"{d['id']}:remind{n}" for n in due_thresholds]
                if f"{d['id']}:remind{nearest}" not in sent:
                    days = max(0, int(left.total_seconds() // 86400))
                    events.append((keys, f"⏰ xsec_alpha 기한 알림 (D-{days}): {d['title']}\n마감 {due_kst}\n"
                                         f"미충족 시 기본값: {d['default']}"))
    return events, rows


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def run(root: Path = ROOT, registry_path: Path = REGISTRY, state_dir: Path = STATE_DIR,
        now: datetime | None = None, sender=None, dry_run: bool = False) -> dict:
    now = now or datetime.now(timezone.utc)
    registry = json.loads(registry_path.read_text())
    state_path = state_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"sent": {}}
    events, rows = evaluate(registry, root, now, state["sent"])
    delivered = []
    for keys, message in events:
        if dry_run:
            print(message)
            continue
        if sender is None:
            from utils.telegram import send_message as sender  # lazy: needs the .env tokens
        if sender(message):
            for key in keys:
                state["sent"][key] = now.isoformat()
            delivered.append(keys[0])
    status = {"checked_at": now.isoformat(), "deadlines": rows, "notices_sent": delivered,
              "notices_pending": [keys[0] for keys, _ in events if keys[0] not in delivered]}
    if not dry_run:
        _write_json(state_path, state)
        _write_json(state_dir / "status.json", status)
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="print notices, send and write nothing")
    ap.add_argument("--now", help="override current time (ISO, UTC) for checks")
    args = ap.parse_args()
    status = run(now=_utc(args.now) if args.now else None, dry_run=args.dry_run)
    for row in status["deadlines"]:
        print(f"{row['id']}: {row['status']} (days_left {row['days_left']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
