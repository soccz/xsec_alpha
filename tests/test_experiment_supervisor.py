import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from utils import experiment_supervisor as monitor
from utils import prospective as trial


class AuditRanker:
    _feature_names = ["reversal_4h"]

    def predict(self, frame):
        return -0.02 * frame["reversal_4h"].to_numpy()


@pytest.fixture
def registered(tmp_path):
    for name in trial.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen source\n")
    (tmp_path / "models/xsec_6h.pkl").write_bytes(pickle.dumps(AuditRanker()))
    trial.start_experiment(tmp_path, now="2026-09-07T04:00Z")
    plan = monitor.register_review(tmp_path, now="2026-09-07T04:05Z")
    return tmp_path, plan


def full_summary(interval, mean=0.1):
    return {"status": "ready_for_review", "runtime_matches": True, "frozen_model_matches": True,
            "n_matured": 60, "n_recorded": 60, "n_missed": 0, "n_invalid": 0,
            "paired_ci95_pct": interval, "paired_excess_pct": 0.2,
            "strategies": [{"name": "model_short5", "mean_net_pct": mean}]}


def test_review_is_bound_and_cannot_be_reset(registered):
    root, plan = registered
    assert plan["protocol_sha256"] == trial._digest(trial.experiment_summary(root)["protocol"])
    assert plan["automatic_promotion"] is False
    with pytest.raises(FileExistsError):
        monitor.register_review(root)


def test_no_post_outcome_preregistration(tmp_path):
    for name in trial.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source")
    (tmp_path / "models/xsec_6h.pkl").write_bytes(pickle.dumps(SimpleNamespace(_feature_names=[], predict=len)))
    trial.start_experiment(tmp_path, now="2026-09-07T04:00Z")
    with pytest.raises(ValueError, match="first slot"):
        monitor.register_review(tmp_path, now="2026-09-07T05:00Z")


@pytest.mark.parametrize("interval,code", [
    ([0.01, 0.2], "relative_edge"), ([-0.2, -0.01], "baseline_better"),
    ([-0.1, 0.2], "inconclusive"), ([0, 0.2], "inconclusive"), ([-0.2, 0], "inconclusive"),
])
def test_final_decision_preserves_uncertainty_and_no_auto_promotion(registered, interval, code):
    _, plan = registered
    outcome = monitor.decide(full_summary(interval), plan, now=plan["hard_review_at"])
    assert outcome["code"] == code and outcome["terminal"]
    assert outcome["automatic_promotion"] is False
    assert outcome["executable_profitability"] == "not_tested"


def test_relative_edge_is_not_profitability(registered):
    _, plan = registered
    result = monitor.decide(full_summary([0.1, 0.3], mean=-0.5), plan, now=plan["hard_review_at"])
    assert result["code"] == "relative_edge"
    assert result["paper_mean"] == "not_positive"


def test_partial_or_too_early_results_cannot_be_final(registered):
    _, plan = registered
    result = monitor.decide(full_summary([0.1, 0.3]), plan, now="2026-09-07T05:00Z")
    assert result["code"] == "collecting" and not result["terminal"]
    partial = {**full_summary([0.1, 0.3]), "n_matured": 59, "n_invalid": 1}
    result = monitor.decide(partial, plan, now=plan["hard_review_at"])
    assert result["code"] == "insufficient_evidence"


def test_integrity_failure_and_deadline_have_honest_terminal_outcomes(registered):
    root, plan = registered
    summary = trial.experiment_summary(root)
    summary["runtime_matches"] = False
    early = monitor.decide(summary, plan, now="2026-09-07T05:00Z")
    final = monitor.decide(summary, plan, now=plan["hard_review_at"])
    assert early["code"] == "integrity_blocked" and not early["terminal"]
    assert final["code"] == "integrity_failure" and final["terminal"]
    summary["runtime_matches"] = True
    assert monitor.decide(summary, plan, now=plan["hard_review_at"])["code"] == "insufficient_evidence"


def test_checkpoint_restore_is_idempotent_and_source_unchanged(registered, tmp_path):
    root, _ = registered
    source = root / "output/prospective/ledger.sqlite"
    before = source.read_bytes()
    offdisk = tmp_path / "secondary"
    one = monitor.backup_checkpoint(root, offdisk, require_separate=False)
    two = monitor.backup_checkpoint(root, offdisk, require_separate=False)
    assert one == two
    assert one["restore_verified"] and not one["separate_filesystem"]
    assert source.read_bytes() == before
    for destination in (one["primary"], one["secondary"]):
        assert monitor.verify_checkpoint(destination)["row_counts"] == [0, 0]
        assert (Path(destination) / "output/prospective/model.pkl").stat().st_mode & 0o777 == 0o600


def test_local_recovery_does_not_touch_offdisk_or_claim_disk_failure_protection(registered, monkeypatch):
    root, _ = registered
    visited = []
    def disk_usage(path):
        visited.append(Path(path))
        assert Path(path).is_relative_to(root)
        return SimpleNamespace(free=10 * 1024**3)
    monkeypatch.setattr(monitor.shutil, "disk_usage", disk_usage)
    one = monitor.backup_checkpoint(root)
    assert one == monitor.backup_checkpoint(root)
    assert one["mode"] == "same_disk" and one["secondary"] is None
    assert one["restore_verified"] and not one["separate_filesystem"]
    assert not one["disk_failure_protected"] and visited
    state = monitor.supervise(root, now="2026-09-07T04:10Z")
    assert not state["errors"]


def test_regime_ledger_local_snapshot_is_consistent_idempotent_and_verified(tmp_path):
    import sqlite3
    folder = tmp_path / "output/regime_observation"
    folder.mkdir(parents=True)
    source = folder / "ledger.sqlite"
    protocol = {"test": "regime"}
    with sqlite3.connect(source) as conn:
        conn.executescript("CREATE TABLE protocol(payload TEXT, sha256 TEXT);"
                           "CREATE TABLE observations(slot TEXT PRIMARY KEY, payload TEXT);"
                           "CREATE TABLE outcomes(slot TEXT PRIMARY KEY, payload TEXT);")
        conn.execute("INSERT INTO protocol VALUES (?, ?)", (trial._json(protocol), trial._digest(protocol)))
        conn.execute("INSERT INTO observations VALUES ('slot', '{}')")
    original = source.read_bytes()
    first = monitor.backup_regime_ledger(tmp_path)
    assert first == monitor.backup_regime_ledger(tmp_path)
    assert first["row_counts"] == [1, 0] and first["restore_verified"]
    assert not first["disk_failure_protected"]
    assert source.read_bytes() == original
    Path(first["primary"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="recovery copy changed"):
        monitor.backup_regime_ledger(tmp_path)


def test_same_filesystem_not_misrepresented_as_offdisk(registered, tmp_path):
    root, _ = registered
    with pytest.raises(ValueError, match="different filesystem"):
        monitor.backup_checkpoint(root, tmp_path / "secondary")


def test_corrupt_checkpoint_detected_without_touching_source(registered, tmp_path):
    root, _ = registered
    saved = monitor.backup_checkpoint(root, tmp_path / "secondary", require_separate=False)
    copied = Path(saved["secondary"]) / "output/prospective/model.pkl"
    copied.write_bytes(b"broken")
    with pytest.raises(ValueError, match="mismatch"):
        monitor.verify_checkpoint(saved["secondary"])
    with pytest.raises(ValueError, match="mismatch"):
        monitor.backup_checkpoint(root, tmp_path / "secondary", require_separate=False)
    assert trial.experiment_summary(root)["frozen_model_matches"]


def test_low_space_keeps_primary_and_previous_offdisk(registered, tmp_path, monkeypatch):
    root, _ = registered
    monkeypatch.setattr(monitor.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="reserve"):
        monitor.backup_checkpoint(root, tmp_path / "secondary", require_separate=False)
    assert len(list((root / "output/prospective_backups").glob("*/manifest.json"))) == 1
    assert not list((tmp_path / "secondary").glob("*/manifest.json"))


def test_supervisor_writes_status_and_deduplicates_dashboard_publish(registered, tmp_path):
    root, _ = registered
    calls = []
    def publish():
        calls.append(True)
        assert (root / "output/experiment_supervision/status.json").exists()
        return True
    first = monitor.supervise(root, tmp_path / "secondary", now="2026-09-07T04:10Z",
                               publisher=publish, require_separate=False)
    second = monitor.supervise(root, tmp_path / "secondary", now="2026-09-07T04:20Z",
                                publisher=publish, require_separate=False)
    assert len(calls) == 1 and not first["errors"] and not second["errors"]
    assert first["milestones"]["first_snapshot"] == "waiting"


def test_failed_publish_retries_next_pass(registered, tmp_path):
    root, _ = registered
    calls = []
    def publish():
        calls.append(True)
        return len(calls) > 1
    for now in ("2026-09-07T04:10Z", "2026-09-07T04:20Z"):
        state = monitor.supervise(root, tmp_path / "secondary", now=now, publisher=publish, require_separate=False)
        if len(calls) == 1:
            assert state["status"] == "attention"
            assert "publisher returned False" in state["errors"][-1]
    assert len(calls) == 2
    assert not state["errors"]


def test_rotation_failure_is_reported_without_losing_original_supervision(registered, monkeypatch):
    from utils import rotation_pilot
    root, _ = registered
    def fail(**kwargs):
        raise ValueError("pilot integrity test failure")
    monkeypatch.setattr(rotation_pilot, "refresh_pilot", fail)
    monkeypatch.setattr(rotation_pilot, "backup_pilot", lambda root: {"restore_verified": True})
    state = monitor.supervise(root, now="2026-09-07T04:10Z")
    assert state["status"] == "attention"
    assert state["errors"] == ["rotation_pilot: ValueError: pilot integrity test failure"]
    assert state["backup"]["restore_verified"]
    assert state["rotation_backup"]["restore_verified"]
    assert state["rotation_pilot"] is None
    assert state["decision"]["code"] == "collecting"


def test_rotation_heartbeat_does_not_publish_every_pass(registered, monkeypatch):
    from utils import rotation_pilot
    root, _ = registered
    calls = []
    monkeypatch.setattr(rotation_pilot, "refresh_pilot", lambda **kwargs: {
        "status": "observing", "checked_at": trial._utc(kwargs["now"]).isoformat(),
        "decision": {"counts": {"recorded": 0}},
    })
    for now in ("2026-09-07T04:10Z", "2026-09-07T04:20Z"):
        state = monitor.supervise(root, now=now, publisher=lambda: calls.append(True) or True)
        assert state["rotation_pilot"]["checked_at"] == trial._utc(now).isoformat()
        assert not state["errors"]
    assert len(calls) == 1


def test_forecast_failure_cannot_stop_existing_supervision(registered, monkeypatch):
    from utils import forecast_audit
    root, _ = registered
    def fail(**kwargs):
        raise ValueError("audit failure test")
    monkeypatch.setattr(forecast_audit, "refresh_audit", fail)
    monkeypatch.setattr(forecast_audit, "backup_audit", lambda root: {"restore_verified": True})
    state = monitor.supervise(root, now="2026-09-07T04:10Z")
    assert state["errors"] == ["forecast_audit: ValueError: audit failure test"]
    assert state["backup"]["restore_verified"] and state["forecast_backup"]["restore_verified"]
    assert state["decision"]["code"] == "collecting"


def test_forecast_heartbeat_alone_does_not_republish(registered, monkeypatch):
    from utils import forecast_audit
    root, _ = registered
    calls = []
    monkeypatch.setattr(forecast_audit, "refresh_audit", lambda **kwargs: {
        "status": "monitoring", "checked_at": trial._utc(kwargs["now"]).isoformat(),
        "counts": {"recorded": 0},
    })
    for now in ("2026-09-07T04:10Z", "2026-09-07T04:20Z"):
        state = monitor.supervise(root, now=now, publisher=lambda: calls.append(True) or True)
        assert not state["errors"]
    assert len(calls) == 1


def test_completed_archive_survives_live_code_changes_but_not_evidence_changes(registered, tmp_path):
    root, plan = registered
    monitor.supervise(root, tmp_path / "secondary", now=plan["hard_review_at"], require_separate=False)
    final = (root / "output/experiment_supervision/final_review.json").read_bytes()
    sealed = monitor.seal_completed(root)
    (root / "scripts/fetch_and_rank.py").write_text("new operational publisher\n")
    summary = monitor.review_summary(root)
    assert summary["runtime_matches"] and not summary["live_runtime_matches"]
    assert summary["integrity_basis"] == "sealed_completed_archive"
    assert monitor.seal_completed(root) == sealed
    state = monitor.supervise(root, tmp_path / "secondary", now=plan["hard_review_at"], require_separate=False)
    assert not state["errors"]
    assert (root / "output/experiment_supervision/final_review.json").read_bytes() == final
    assert monitor.verify_checkpoint(state["backup"]["secondary"])
    (root / "output/prospective/model.pkl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Completed evidence changed"):
        monitor.review_summary(root)


def test_cannot_seal_after_frozen_code_has_changed(registered, tmp_path):
    root, plan = registered
    monitor.supervise(root, tmp_path / "secondary", now=plan["hard_review_at"], require_separate=False)
    (root / "scripts/fetch_and_rank.py").write_text("already changed\n")
    with pytest.raises(ValueError, match="Cannot seal"):
        monitor.seal_completed(root)


def test_missed_first_run_and_terminal_report_are_automatic_and_immutable(registered, tmp_path):
    root, plan = registered
    offdisk = tmp_path / "secondary"
    first = monitor.supervise(root, offdisk, now="2026-09-07T05:50Z", require_separate=False)
    assert first["milestones"]["first_snapshot"] == "missing"
    assert not first["decision"]["terminal"]
    final = monitor.supervise(root, offdisk, now=plan["hard_review_at"], require_separate=False)
    assert final["decision"]["code"] == "insufficient_evidence"
    path = root / "output/experiment_supervision/final_review.json"
    original = path.read_bytes()
    again = monitor.supervise(root, offdisk, now=pd.Timestamp(plan["hard_review_at"]) + pd.Timedelta(days=1),
                               require_separate=False)
    assert path.read_bytes() == original and again["decision"] == final["decision"]
    archived = Path(again["backup"]["secondary"]) / "output/experiment_supervision/final_review.json"
    assert archived.read_bytes() == original


def test_backup_failure_does_not_hide_progress_or_overwrite_trial(registered, tmp_path, monkeypatch):
    root, _ = registered
    def fail(*args, **kwargs):
        raise OSError("injected disk failure")
    monkeypatch.setattr(monitor, "backup_checkpoint", fail)
    state = monitor.supervise(root, tmp_path / "secondary", now="2026-09-07T04:10Z", require_separate=False)
    assert state["status"] == "attention"
    assert "injected disk failure" in state["errors"][0]
    assert trial.experiment_summary(root)["n_recorded"] == 0


def test_public_summary_excludes_supervision_paths(tmp_path, monkeypatch):
    from utils import dashboard_export
    folder = tmp_path / "output/experiment_supervision"
    folder.mkdir(parents=True)
    (folder / "status.json").write_text(json.dumps({"backup": {"secondary": "private backup path"}}))
    monkeypatch.setattr(dashboard_export, "OUTPUT_DIR", tmp_path / "output")
    assert dashboard_export.build_summary_payload()["experiment_supervision"] is not None
    assert "experiment_supervision" not in dashboard_export.build_public_summary_payload()


def test_orphan_outcome_is_not_backed_up_as_valid_evidence(registered, tmp_path):
    root, _ = registered
    with trial._db(root) as conn:
        conn.execute("INSERT INTO outcomes VALUES (?, ?)", ("2026-09-07T05:00:00+00:00", '{"status":"matured"}'))
    with pytest.raises(ValueError, match="no recorded forecast"):
        monitor.backup_checkpoint(root, tmp_path / "secondary", require_separate=False)
    state = monitor.supervise(root, tmp_path / "secondary", now="2026-09-07T06:00Z", require_separate=False)
    assert state["decision"]["code"] == "integrity_blocked"
    assert state["status"] == "attention"


def test_publication_exception_keeps_durable_status_for_retry(registered, tmp_path):
    root, _ = registered
    def fail():
        raise OSError("publisher unavailable")
    state = monitor.supervise(root, tmp_path / "secondary", now="2026-09-07T04:10Z",
                               publisher=fail, require_separate=False)
    assert state["status"] == "attention"
    assert "publisher unavailable" in state["errors"][-1]
    assert not (root / "output/experiment_supervision/publication.json").exists()


def test_complete_simulated_trial_produces_audited_terminal_report(registered, tmp_path):
    root, plan = registered
    protocol = trial.experiment_summary(root)["protocol"]
    frame = pd.DataFrame({"reversal_4h": list(range(10))}, index=[f"KRW-T{i:02d}" for i in range(10)])
    for slot in trial._slots(protocol):
        trial.record_snapshot(slot, frame, frame, frame.index, root=root, now=slot + pd.Timedelta(minutes=5))
    entry = pd.Timestamp(protocol["first_signal_at"]) + pd.Timedelta(hours=1)
    prices = pd.DataFrame([
        {"timestamp": entry + pd.Timedelta(hours=6 * t), "market": market,
         "open": 100.0 * (1 - i / 100) ** t}
        for t in range(61) for i, market in enumerate(frame.index)
    ])
    trial.advance_experiment(root, now=plan["hard_review_at"], price_loader=lambda *args: prices)
    state = monitor.supervise(root, tmp_path / "secondary", now=plan["hard_review_at"], require_separate=False)
    assert not state["errors"]
    assert state["decision"]["code"] == "relative_edge"
    assert state["decision"]["paper_mean"] == "positive_only"
    assert state["milestones"]["matured"] == 60
    assert state["backup"]["row_counts"] == [60, 60]
    final = monitor._verified_document(root / "output/experiment_supervision/final_review.json")
    assert final["evidence"]["paired_excess_pct"] == pytest.approx(5.0)
    # SQLite integrity alone would not detect an accidentally edited cost/return.
    with trial._db(root) as conn:
        slot, raw = conn.execute("SELECT slot, payload FROM outcomes ORDER BY slot LIMIT 1").fetchone()
        outcome = json.loads(raw)
        outcome["strategies"]["model_short5"]["net_return"] += 1.0
        conn.execute("UPDATE outcomes SET payload=? WHERE slot=?", (json.dumps(outcome), slot))
    with pytest.raises(ValueError, match="basket return or cost"):
        monitor.backup_checkpoint(root, tmp_path / "secondary", require_separate=False)
