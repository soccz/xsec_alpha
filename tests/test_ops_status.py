from datetime import datetime, timedelta, timezone
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


def test_notifications_retry_failure_and_record_recovery_without_sending(tmp_path, monkeypatch):
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
    assert len(messages) == 2
    assert "디스크 자체 고장" in messages[-1]
    state = json.loads((tmp_path / "output/operations_notices/latest.json").read_text())
    assert state["failed"] == [] and state["telegram_state"] == "recorded_only"


def test_research_diagnostics_and_recovery_are_recorded_without_telegram(tmp_path, monkeypatch):
    checks = [{"check": name, "status": "FAIL"} for name in
              ("AUDIT_RUNTIME", "EXECUTION_DIAGNOSTICS", "VENUE_OBSERVATION")]
    monkeypatch.setattr(ops_status, "operational_checks", lambda *args: checks)
    messages = []
    for _ in range(2):
        assert ops_status.notify_operations(tmp_path, sender=lambda text: messages.append(text))
    folder = tmp_path / "output/operations_notices"
    state = json.loads((folder / "latest.json").read_text())
    assert len(state["failed"]) == 3 and state["notify_failed"] == []
    assert state["telegram_state"] == "recorded_only" and not state["acknowledged"]
    checks.clear()
    assert ops_status.notify_operations(tmp_path, sender=lambda text: messages.append(text))
    assert not messages
    assert len(list(folder.glob("2*.json"))) == 2


def test_new_critical_fault_alerts_even_when_research_warnings_change(tmp_path, monkeypatch):
    checks = [{"check": "AUDIT_RUNTIME", "status": "FAIL"}]
    monkeypatch.setattr(ops_status, "operational_checks", lambda *args: checks)
    messages = []
    def send(text):
        messages.append(text)
        return True
    now = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
    assert ops_status.notify_operations(tmp_path, now, sender=send)
    checks.append({"check": "BACKUP", "status": "FAIL"})
    assert ops_status.notify_operations(tmp_path, now, sender=send)
    checks[0]["check"] = "VENUE_OBSERVATION"
    assert ops_status.notify_operations(tmp_path, now + timedelta(minutes=10), sender=send)
    checks.append({"check": "PUBLICATION", "status": "FAIL"})
    assert ops_status.notify_operations(tmp_path, now + timedelta(minutes=20), sender=send)
    assert len(messages) == 2
    assert "BACKUP" in messages[0] and "PUBLICATION" in messages[1]
    assert all("AUDIT_RUNTIME" not in text and "VENUE_OBSERVATION" not in text for text in messages)


def test_flapping_cooldown_survives_recovery_and_does_not_repeat_open_incident(tmp_path, monkeypatch):
    checks = [{"check": "BACKUP", "status": "FAIL"}]
    monkeypatch.setattr(ops_status, "operational_checks", lambda *args: checks)
    messages = []
    def send(text):
        messages.append(text)
        return True
    now = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
    assert ops_status.notify_operations(tmp_path, now, sender=send)
    checks[0]["status"] = "OK"
    assert ops_status.notify_operations(tmp_path, now + timedelta(minutes=10), sender=send)
    checks[0]["status"] = "FAIL"
    assert ops_status.notify_operations(tmp_path, now + timedelta(minutes=20), sender=send)
    assert len(messages) == 1
    assert ops_status.notify_operations(tmp_path, now + timedelta(hours=6), sender=send)
    assert len(messages) == 2
    assert ops_status.notify_operations(tmp_path, now + timedelta(days=2), sender=send)
    assert len(messages) == 2


def test_recent_service_notice_prevents_duplicate_operational_alert(tmp_path, monkeypatch):
    checks = [{"check": "SUPERVISOR", "status": "FAIL"}]
    monkeypatch.setattr(ops_status, "operational_checks", lambda *args: checks)
    now = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
    path = tmp_path / "output/unit_failures/state.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"xsec-experiment-supervisor.service": now.isoformat()}))
    messages = []
    assert ops_status.notify_operations(tmp_path, now + timedelta(minutes=10), sender=lambda t: messages.append(t))
    assert not messages
    state = json.loads((tmp_path / "output/operations_notices/latest.json").read_text())
    assert state["notified_active"] == ["SUPERVISOR"]


def test_legacy_acknowledged_incident_does_not_resend_after_upgrade(tmp_path, monkeypatch):
    now = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
    folder = tmp_path / "output/operations_notices"
    folder.mkdir(parents=True)
    (folder / "latest.json").write_text(json.dumps({"checked_at": now.isoformat(),
                                                  "failed": ["BACKUP", "AUDIT_RUNTIME"], "acknowledged": True}))
    monkeypatch.setattr(ops_status, "operational_checks", lambda *args: [{"check": "BACKUP", "status": "FAIL"}])
    messages = []
    assert ops_status.notify_operations(tmp_path, now + timedelta(days=1), sender=lambda t: messages.append(t))
    assert not messages


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


def test_runtime_headroom_and_pages_receipt_have_independent_checks(tmp_path):
    from utils.experiment_supervisor import _write

    folder = tmp_path / "output/forecast_audit"
    folder.mkdir(parents=True)
    (folder / "ledger.sqlite").touch()
    stamp = "2026-09-30T03:40:00+00:00"
    doc = {"status": "monitoring", "checked_at": stamp, "timings": {"capture_seconds": 61, "total_seconds": 110}}
    _write(folder / "summary.json", doc)
    state = {"checked_at": stamp, "operating_acceptance": {"closed": {"closed_at": "2026-09-30T03:00:00+00:00"},
                                                          "publication": {"status": "waiting"}}}
    _write(tmp_path / "output/experiment_supervision/status.json", state)
    now = datetime(2026, 9, 30, 3, 45, tzinfo=timezone.utc)
    rows = {r["check"]: r["status"] for r in ops_status.operational_checks(tmp_path, now)}
    assert rows["AUDIT_RUNTIME"] == rows["OPERATING_PUBLICATION"] == "FAIL"
    doc["timings"] = {"capture_seconds": 2, "total_seconds": 10}
    state["operating_acceptance"]["publication"]["status"] = "verified"
    _write(folder / "summary.json", doc)
    _write(tmp_path / "output/experiment_supervision/status.json", state)
    rows = {r["check"]: r["status"] for r in ops_status.operational_checks(tmp_path, now)}
    assert rows["AUDIT_RUNTIME"] == rows["OPERATING_PUBLICATION"] == "OK"


def test_daily_db_backup_and_offhost_evidence_must_be_recent(tmp_path, monkeypatch):
    monkeypatch.setattr(ops_status.shutil, "disk_usage", lambda path: SimpleNamespace(free=10 * 1024**3))
    log = tmp_path / "logs/backup.log"
    log.parent.mkdir(parents=True)
    log.write_text("[backup_db] ok: /b/crypto_data_20261004T180447Z_1.db (kept 9)\n"
                   "[backup_db] insufficient secondary space: need 1 KiB\n")
    offhost = tmp_path / "output/evidence_offhost/status.json"
    offhost.parent.mkdir(parents=True)
    offhost.write_text(json.dumps({"checked_at": "2026-10-05T02:27:25+00:00", "ok": True}))
    rows = {r["check"]: r["status"] for r in ops_status.operational_checks(
        tmp_path, datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc), tmp_path)}
    assert rows["DB_BACKUP"] == "OK" and rows["EVIDENCE_OFFHOST"] == "OK"
    rows = {r["check"]: r["status"] for r in ops_status.operational_checks(
        tmp_path, datetime(2026, 10, 6, 0, 30, tzinfo=timezone.utc), tmp_path)}
    assert rows["DB_BACKUP"] == "FAIL" and rows["EVIDENCE_OFFHOST"] == "OK"  # 30.4 h vs 22 h
    offhost.write_text(json.dumps({"checked_at": "2026-10-06T12:00:00+00:00", "ok": False}))
    rows = {r["check"]: r["status"] for r in ops_status.operational_checks(
        tmp_path, datetime(2026, 10, 6, 12, 30, tzinfo=timezone.utc), tmp_path)}
    assert rows["EVIDENCE_OFFHOST"] == "FAIL"


def test_backup_log_without_any_success_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(ops_status.shutil, "disk_usage", lambda path: SimpleNamespace(free=10 * 1024**3))
    log = tmp_path / "logs/backup.log"
    log.parent.mkdir(parents=True)
    log.write_text("[backup_db] insufficient secondary space\n")
    rows = {r["check"]: r for r in ops_status.operational_checks(tmp_path, secondary=tmp_path)}
    assert rows["DB_BACKUP"]["status"] == "FAIL" and rows["DB_BACKUP"]["detail"] == "missing or invalid status"
