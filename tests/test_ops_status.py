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
    assert "디스크 자체 고장" in messages[-1]


def test_default_storage_checks_stay_on_primary_volume(tmp_path, monkeypatch):
    visited = []
    def disk_usage(path):
        visited.append(path)
        assert path.is_relative_to(tmp_path)
        return SimpleNamespace(free=20 * 1024**3)
    monkeypatch.setattr(ops_status.shutil, "disk_usage", disk_usage)
    rows = ops_status.operational_checks(tmp_path)
    assert len(visited) == 2
    assert all(row["status"] == "OK" for row in rows[:2])


def test_registered_pilot_requires_fresh_processing_and_verified_backup(tmp_path):
    folder = tmp_path / "output/rotation_pilot"
    folder.mkdir(parents=True)
    (folder / "ledger.sqlite").touch()
    (folder / "summary.json").write_text(json.dumps({"status": "observing", "checked_at": "2026-09-30T03:00:00+00:00"}))
    parent = tmp_path / "output/experiment_supervision"
    parent.mkdir()
    (parent / "status.json").write_text(json.dumps({"checked_at": "2026-09-30T03:00:00+00:00",
                                                   "rotation_backup": {"restore_verified": True}}))
    now = datetime(2026, 9, 30, 3, 10, tzinfo=timezone.utc)
    rows = ops_status.operational_checks(tmp_path, now)
    assert all(r["status"] == "OK" for r in rows if r["check"].startswith("ROTATION"))
    (folder / "summary.json").write_text('{"status":"integrity_failure","checked_at":"2026-09-30T03:00:00+00:00"}')
    rows = ops_status.operational_checks(tmp_path, now)
    assert next(r for r in rows if r["check"] == "ROTATION_PILOT")["status"] == "FAIL"


def test_forecast_audit_must_be_fresh_and_backed_up_without_treating_ic_gap_as_failure(tmp_path):
    folder = tmp_path / "output/forecast_audit"
    folder.mkdir(parents=True)
    (folder / "ledger.sqlite").touch()
    (folder / "summary.json").write_text(json.dumps({"status": "monitoring", "checked_at": "2026-09-30T03:00:00+00:00",
                                                    "counts": {"differences": 2}}))
    supervisor = tmp_path / "output/experiment_supervision"
    supervisor.mkdir()
    (supervisor / "status.json").write_text(json.dumps({"checked_at": "2026-09-30T03:00:00+00:00",
                                                        "forecast_backup": {"restore_verified": True}}))
    now = datetime(2026, 9, 30, 3, 10, tzinfo=timezone.utc)
    assert all(r["status"] == "OK" for r in ops_status.operational_checks(tmp_path, now)
               if r["check"].startswith("FORECAST"))
    (folder / "summary.json").write_text('{"status":"attention","checked_at":"2026-09-30T03:00:00+00:00"}')
    rows = ops_status.operational_checks(tmp_path, now)
    assert next(r for r in rows if r["check"] == "FORECAST_AUDIT")["status"] == "FAIL"
