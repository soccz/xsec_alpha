import json
from datetime import datetime, timedelta, timezone

from scripts import check_deadlines as cd

NOW = datetime(2026, 10, 25, 0, 0, tzinfo=timezone.utc)


def _registry(tmp_path, due, done_if=None, remind=(7, 1)):
    path = tmp_path / "deadlines.json"
    path.write_text(json.dumps({"deadlines": [{
        "id": "d1", "title": "T", "due_utc": due.isoformat().replace("+00:00", "Z"),
        "done_if": done_if or {"glob": "docs/prereg/x*.json"}, "default": "D",
        "remind_days_before": list(remind)}]}))
    return path


class Sender:
    def __init__(self, ok=True):
        self.ok, self.messages = ok, []

    def __call__(self, text):
        self.messages.append(text)
        return self.ok


def _run(tmp_path, registry, now, sender, **kw):
    return cd.run(root=tmp_path, registry_path=registry, state_dir=tmp_path / "state", now=now, sender=sender, **kw)


def test_overdue_unmet_condition_records_default_and_notifies_once(tmp_path):
    reg = _registry(tmp_path, NOW - timedelta(hours=1))
    sender = Sender()
    first = _run(tmp_path, reg, NOW, sender)
    second = _run(tmp_path, reg, NOW + timedelta(days=1), sender)
    assert first["deadlines"][0]["status"] == "overdue_default_applies"
    assert first["notices_sent"] == ["d1:overdue"] and second["notices_sent"] == []
    assert len(sender.messages) == 1 and "기본값 확정: D" in sender.messages[0]


def test_done_condition_suppresses_notices(tmp_path):
    (tmp_path / "docs/prereg").mkdir(parents=True)
    (tmp_path / "docs/prereg/x_v1.json").write_text("{}")
    sender = Sender()
    status = _run(tmp_path, _registry(tmp_path, NOW - timedelta(days=1)), NOW, sender)
    assert status["deadlines"][0]["status"] == "done" and sender.messages == []


def test_only_nearest_reminder_is_sent_and_older_thresholds_are_consumed(tmp_path):
    reg = _registry(tmp_path, NOW + timedelta(hours=12))
    sender = Sender()
    _run(tmp_path, reg, NOW, sender)
    _run(tmp_path, reg, NOW + timedelta(hours=1), sender)
    assert len(sender.messages) == 1 and "D-0" in sender.messages[0]
    state = json.loads((tmp_path / "state/state.json").read_text())
    assert {"d1:remind1", "d1:remind7"} <= set(state["sent"])


def test_failed_delivery_is_retried_on_next_run(tmp_path):
    reg = _registry(tmp_path, NOW - timedelta(hours=1))
    failing = Sender(ok=False)
    first = _run(tmp_path, reg, NOW, failing)
    assert first["notices_pending"] == ["d1:overdue"]
    working = Sender()
    second = _run(tmp_path, reg, NOW + timedelta(hours=1), working)
    assert second["notices_sent"] == ["d1:overdue"] and len(working.messages) == 1


def test_json_field_condition(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out/s.json").write_text(json.dumps({"decision": {"terminal": True}}))
    cond = {"json_field": {"path": "out/s.json", "field": "decision.terminal", "equals": True}}
    assert cd.condition_met(tmp_path, cond)
    (tmp_path / "out/s.json").write_text(json.dumps({"decision": {"terminal": False}}))
    assert not cd.condition_met(tmp_path, cond)
    assert not cd.condition_met(tmp_path, {"json_field": {"path": "missing.json", "field": "a", "equals": 1}})


def test_dry_run_sends_and_writes_nothing(tmp_path, capsys):
    reg = _registry(tmp_path, NOW - timedelta(hours=1))
    sender = Sender()
    _run(tmp_path, reg, NOW, sender, dry_run=True)
    assert sender.messages == [] and not (tmp_path / "state").exists()
    assert "기한 경과" in capsys.readouterr().out


def test_repository_registry_is_valid_and_currently_pending():
    status = cd.run(now=NOW, dry_run=True)
    ids = {row["id"]: row["status"] for row in status["deadlines"]}
    assert set(ids) == {"model_zoo_prereg_v1", "rotation_pilot_terminal", "model_zoo_v1_deadman"}
    assert all(s in ("pending", "done") for s in ids.values())
