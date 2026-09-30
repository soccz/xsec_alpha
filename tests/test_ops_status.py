from datetime import datetime, timezone
import json
from types import SimpleNamespace

from utils import ops_status


def test_disk_and_stale_publication_cannot_hide_behind_fresh_signals(tmp_path, monkeypatch):
    monkeypatch.setattr(ops_status.shutil, "disk_usage", lambda path: SimpleNamespace(free=600 * 1024**2))
    rows = ops_status.operational_checks(tmp_path, secondary=tmp_path)
    assert all(row["status"] == "FAIL" for row in rows)


def test_all_operational_evidence_must_be_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(ops_status.shutil, "disk_usage", lambda path: SimpleNamespace(free=10 * 1024**3))
    for folder, doc in (
        ("experiment_supervision", {"checked_at": "2026-09-30T03:00:00+00:00", "errors": [],
                                    "backup": {"restore_verified": True}}),
        ("dashboard_publication", {"checked_at": "2026-09-30T03:00:00+00:00", "status": "published"}),
    ):
        path = tmp_path / "output" / folder / "status.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(doc))
    now = datetime(2026, 9, 30, 3, 10, tzinfo=timezone.utc)
    assert all(row["status"] == "OK" for row in ops_status.operational_checks(tmp_path, now, tmp_path))
    now = now.replace(hour=6)
    rows = ops_status.operational_checks(tmp_path, now, tmp_path)
    assert next(r for r in rows if r["check"] == "PUBLICATION")["status"] == "FAIL"


def test_notifications_deduplicate_acknowledged_incidents_and_recovery(tmp_path, monkeypatch):
    checks = [{"check": "BACKUP", "status": "FAIL"}]
    monkeypatch.setattr(ops_status, "operational_checks", lambda *args: checks)
    messages = []
    def send(text):
        messages.append(text)
        return len(messages) > 1
    assert not ops_status.notify_operations(tmp_path, sender=send)
    assert ops_status.notify_operations(tmp_path, sender=send)
    assert ops_status.notify_operations(tmp_path, sender=send)
    assert len(messages) == 2
    checks[0]["status"] = "OK"
    assert ops_status.notify_operations(tmp_path, sender=send)
    assert len(messages) == 3
    assert "복구" in messages[-1]
