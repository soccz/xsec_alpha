"""Read-only operational coverage beyond signal/data freshness."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent


def _load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def operational_checks(root=ROOT, now=None, secondary=None):
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    secondary = Path(secondary) if secondary is not None else root / "output/prospective_backups"
    rows = []
    for name, path in (("DISK", root), ("BACKUP_DISK", secondary)):
        try:
            free = shutil.disk_usage(path).free / 1024**3
            rows.append({"check": name, "status": "OK" if free >= 5 else "FAIL",
                         "detail": f"{free:.2f} GiB available; 5 GiB reserve required"})
        except OSError:
            rows.append({"check": name, "status": "FAIL", "detail": "storage unavailable"})
    state = _load(root / "output/experiment_supervision/status.json")
    pub = _load(root / "output/dashboard_publication/status.json")
    checks = [
        ("SUPERVISOR", state, "checked_at", 30, not state.get("errors")),
        ("BACKUP", state, "checked_at", 30, (state.get("backup") or {}).get("restore_verified") is True),
        ("PUBLICATION", pub, "checked_at", 90, pub.get("status") == "published"),
    ]
    if (root / "output/regime_observation/ledger.sqlite").exists():
        checks.append(("REGIME_BACKUP", state, "checked_at", 30,
                       (state.get("regime_backup") or {}).get("restore_verified") is True))
    for name, doc, key, max_minutes, good in checks:
        try:
            age = (now - datetime.fromisoformat(doc[key])).total_seconds() / 60
            status = "OK" if good and 0 <= age <= max_minutes else "FAIL"
            detail = f"last check {age:.1f} min ago; {'verified' if good else 'failed'}"
            if name == "BACKUP" and (state.get("backup") or {}).get("mode") == "same_disk":
                detail += "; same-disk recovery only, disk failure not protected"
        except (KeyError, TypeError, ValueError):
            status, detail = "FAIL", "missing or invalid status"
        rows.append({"check": name, "status": status, "detail": detail})
    return rows


def notify_operations(root=ROOT, now=None, secondary=None, sender=None):
    from utils.experiment_supervisor import _write
    from utils.run_lock import run_lock
    from utils.telegram import send_message

    root = Path(root)
    now = now or datetime.now(timezone.utc)
    with run_lock("operations_notice", lock_dir=str(root / "logs/locks")):
        checks = operational_checks(root, now, secondary)
        failed = sorted(row["check"] for row in checks if row["status"] != "OK")
        folder = root / "output/operations_notices"
        previous = _load(folder / "latest.json")
        if previous.get("failed") == failed and previous.get("acknowledged"):
            return True
        if not previous and not failed:
            return True
        # Only issue categories leave the server, never paths or raw exceptions.
        message = "<b>xsec 운영 점검</b>\n" + (
            "확인 필요: " + ", ".join(failed) if failed else "백업·배포 운영 점검이 정상으로 복구됐습니다."
        )
        message += '\n정기 코인 보고와 별도 상태입니다.\n<a href="https://soccz.github.io/projects/xsec-alpha/dashboard/">대시보드</a>'
        if secondary is None:
            message += "\n동일 디스크 복구용 사본을 사용합니다. 디스크 자체 고장에는 대비되지 않습니다."
        event = {"checked_at": now.isoformat(), "failed": failed, "acknowledged": False}
        _write(folder / "latest.json", event)
        event["acknowledged"] = bool((sender or send_message)(message))
        _write(folder / "latest.json", event)
        _write(folder / (now.strftime("%Y%m%dT%H%M%S%f") + ".json"), event)
        return event["acknowledged"]
