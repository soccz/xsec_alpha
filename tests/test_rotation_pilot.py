import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from utils import prospective as trial
from utils import regime_observer as observer
from utils import rotation_pilot as pilot


@pytest.fixture
def registered(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in pilot.SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    observer.start_observation(tmp_path, now="2026-09-30T03:00Z")
    protocol = pilot.start_pilot(tmp_path, now="2026-09-30T04:00Z")
    return tmp_path, protocol


def source_snapshot(root, slot):
    times = pd.date_range(end=slot, periods=24 * 45, freq="h")
    closes = pd.DataFrame({"KRW-BTC": 100 * np.exp(np.arange(len(times)) * .0001)}, index=times)
    scores = pd.Series(np.arange(10) * .01, index=[f"KRW-T{i:02d}" for i in range(10)])
    assert observer.record_observation(slot, closes, scores, -scores, "model-v1", root=root,
                                       now=slot + pd.Timedelta(minutes=5)) == "recorded"
    with observer._connect(root) as conn:
        return json.loads(conn.execute("SELECT payload FROM observations WHERE slot=?", (slot.isoformat(),)).fetchone()[0])


def source_outcome(root, snapshot):
    entry, exit_at = trial._utc(snapshot["entry_at"]), trial._utc(snapshot["exit_at"])
    prices = pd.DataFrame([{"timestamp": at, "market": row["market"], "open": 100. if at == entry else 110. - i}
                           for i, row in enumerate(snapshot["rows"]) for at in (entry, exit_at)])
    result = trial.evaluate_snapshot(snapshot, prices, snapshot["evaluation"], now=exit_at + pd.Timedelta(hours=1))
    with observer._connect(root) as conn, conn:
        trial._append(conn, "outcomes", snapshot["signal_at"], result)
    return result


def test_registration_future_only_and_append_only(registered):
    root, protocol = registered
    assert protocol["first_signal_at"] == "2026-09-30T05:00:00+00:00"
    assert protocol["n_slots"] == 336 and not protocol["automatic_live_switching"]
    with pytest.raises(FileExistsError):
        pilot.start_pilot(root)
    with pilot._connection(pilot._folder(root) / "ledger.sqlite", writable=True) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE protocol SET payload='{}'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM protocol")
    assert not (root / "output/latest.csv").exists()


def test_capture_is_before_entry_idempotent_and_source_unchanged(registered):
    root, protocol = registered
    first = trial._utc(protocol["first_signal_at"])
    source_snapshot(root, first)
    before = (root / "output/regime_observation/ledger.sqlite").read_bytes()
    state = pilot.refresh_pilot(root, now=first + pd.Timedelta(minutes=10))
    assert state["decision"]["counts"]["recorded"] == 1
    assert state["latest"]["weights"]["reversal"] == 0
    ledger = pilot._folder(root) / "ledger.sqlite"
    original = ledger.read_bytes()
    pilot.refresh_pilot(root, now=first + pd.Timedelta(minutes=20))
    assert ledger.read_bytes() == original
    assert (root / "output/regime_observation/ledger.sqlite").read_bytes() == before


def test_late_source_never_backfills_pilot_and_remains_missed(registered):
    root, protocol = registered
    slot = trial._utc(protocol["first_signal_at"])
    source_snapshot(root, slot)
    state = pilot.refresh_pilot(root, now=slot + pd.Timedelta(minutes=46))
    assert state["decision"]["counts"]["missed"] == 1
    assert state["decision"]["counts"]["recorded"] == 0
    again = pilot.refresh_pilot(root, now=slot + pd.Timedelta(hours=2))
    assert again["decision"]["counts"]["missed"] == 1


def test_exact_saved_outcome_and_cost_are_used_without_forecast_rerun(registered):
    root, protocol = registered
    slot = trial._utc(protocol["first_signal_at"])
    snapshot = source_snapshot(root, slot)
    pilot.refresh_pilot(root, now=slot + pd.Timedelta(minutes=10))
    result = source_outcome(root, snapshot)
    before = (root / "output/regime_observation/ledger.sqlite").read_bytes()
    state = pilot.refresh_pilot(root, now=slot + pd.Timedelta(hours=8))
    assert state["decision"]["counts"]["matured"] == 1
    assert state["decision"]["statistics"]["paired_pp"] == 0
    assert state["decision"]["statistics"]["model_net_pct"] == pytest.approx(100 * result["strategies"]["model_short5"]["net_return"])
    assert (root / "output/regime_observation/ledger.sqlite").read_bytes() == before


def test_missing_outcome_waits_until_grace_then_is_not_zero(registered):
    root, protocol = registered
    slot = trial._utc(protocol["first_signal_at"])
    source_snapshot(root, slot)
    pilot.refresh_pilot(root, now=slot + pd.Timedelta(minutes=10))
    state = pilot.refresh_pilot(root, now=slot + pd.Timedelta(hours=54))
    assert state["decision"]["counts"]["pending"] == 1
    state = pilot.refresh_pilot(root, now=slot + pd.Timedelta(hours=55))
    assert state["decision"]["counts"]["invalid"] == 1
    assert state["decision"]["statistics"]["paired_pp"] is None


@pytest.mark.parametrize("changed", ["runtime", "snapshot", "outcome"])
def test_integrity_failure_is_durable_and_does_not_rewrite_old_decision(registered, changed):
    root, protocol = registered
    slot = trial._utc(protocol["first_signal_at"])
    snapshot = source_snapshot(root, slot)
    pilot.refresh_pilot(root, now=slot + pd.Timedelta(minutes=10))
    source_outcome(root, snapshot)
    pilot.refresh_pilot(root, now=slot + pd.Timedelta(hours=8))
    original = (pilot._folder(root) / "ledger.sqlite").read_bytes()
    if changed == "runtime":
        (root / "utils/rotation_research.py").write_text("changed")
    else:
        table = "observations" if changed == "snapshot" else "outcomes"
        with observer._connect(root) as conn, conn:
            conn.execute(f"UPDATE {table} SET payload='{{}}' WHERE slot=?", (slot.isoformat(),))
    with pytest.raises(ValueError, match="changed"):
        pilot.refresh_pilot(root, now=slot + pd.Timedelta(hours=9))
    assert (pilot._folder(root) / "ledger.sqlite").read_bytes() == original
    failure = (pilot._folder(root) / "integrity_failure.json").read_bytes()
    with pytest.raises(ValueError, match="no automatic restart"):
        pilot.refresh_pilot(root, now=slot + pd.Timedelta(hours=10))
    assert (pilot._folder(root) / "integrity_failure.json").read_bytes() == failure


def test_weekly_capture_uses_two_distinct_reviews_and_real_weight_change_cost(registered):
    root, protocol = registered
    # Training is prospective source evidence; no old fixed-trial rows are imported.
    for slot in pd.date_range(protocol["first_signal_at"], periods=40, freq="6h"):
        source_outcome(root, source_snapshot(root, slot))
    pilot.refresh_pilot(root, now="2026-10-12T00:10Z")
    pilot.refresh_pilot(root, now="2026-10-19T00:10Z")
    slot = trial._utc("2026-10-19T05:00Z")
    snapshot = source_snapshot(root, slot)
    state = pilot.refresh_pilot(root, now=slot + pd.Timedelta(minutes=10))
    assert state["n_reviews"] == 2
    assert state["latest"]["weights"]["reversal"] == .25
    result = source_outcome(root, snapshot)
    state = pilot.refresh_pilot(root, now=slot + pd.Timedelta(hours=8))
    expected = .25 * 100 * (result["strategies"]["reversal_short5"]["net_return"]
                            - result["strategies"]["model_short5"]["net_return"]) - .025
    assert state["decision"]["statistics"]["paired_pp"] == pytest.approx(expected)
    assert state["decision"]["counts"]["exercised"] == 1


def test_all_missed_finishes_and_final_report_never_changes(registered):
    root, protocol = registered
    end = protocol["hard_review_at"]
    state = pilot.refresh_pilot(root, now=end)
    assert state["status"] == "insufficient_evidence"
    assert state["decision"]["counts"]["missed"] == 336
    final = (pilot._folder(root) / "final_review.json").read_bytes()
    state2 = pilot.refresh_pilot(root, now=trial._utc(end) + pd.Timedelta(days=1))
    assert state2["decision"] == state["decision"]
    assert (pilot._folder(root) / "final_review.json").read_bytes() == final


def test_complete_336_window_fixture_is_audited_and_ends_without_promotion(registered):
    root, protocol = registered
    folder = pilot._folder(root)
    with pilot._connection(folder / "ledger.sqlite", writable=True) as conn, conn:
        for slot in trial._slots(protocol):
            source = source_snapshot(root, slot)
            result = source_outcome(root, source)
            row = {"status": "recorded", "signal_at": slot.isoformat(),
                   "recorded_at": (slot + pd.Timedelta(minutes=10)).isoformat(), "source_snapshot": source,
                   "review_key": None, "weights": {"model": 1., "reversal": 0., "reason": "missing_review_fallback"},
                   "previous_reversal_weight": 0.}
            pilot._append(conn, "observations", slot.isoformat(), row)
            pilot._append(conn, "outcomes", slot.isoformat(), pilot._settled(row, result, protocol, slot + pd.Timedelta(hours=9)))
    state = pilot.refresh_pilot(root, now=protocol["hard_review_at"])
    assert state["status"] == "policy_not_exercised"
    assert state["decision"]["counts"]["matured"] == 336
    assert not state["decision"]["confirmatory_evidence"]
    saved = pilot.backup_pilot(root)
    assert saved["restore_verified"] and saved["row_counts"]["terminal"] == 1


def test_local_backup_restores_frozen_code_protocol_and_ledger(registered):
    root, _ = registered
    first = pilot.backup_pilot(root)
    second = pilot.backup_pilot(root)
    assert first == second and not first["disk_failure_protected"]
    assert pilot.verify_backup(first["primary"])["row_counts"] == dict.fromkeys(pilot.TABLES, 0)
    (Path(first["primary"]) / "sources/utils/rotation_pilot.py").write_text("damaged")
    with pytest.raises(ValueError, match="backup changed"):
        pilot.backup_pilot(root)


def test_public_payload_excludes_rotation_pilot(tmp_path, monkeypatch):
    from utils import dashboard_export

    folder = tmp_path / "output/rotation_pilot"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text('{"status":"observing"}')
    monkeypatch.setattr(dashboard_export, "OUTPUT_DIR", tmp_path / "output")
    assert dashboard_export.build_summary_payload()["rotation_pilot"]["status"] == "observing"
    assert "rotation_pilot" not in dashboard_export.build_public_summary_payload()


@pytest.mark.parametrize("mean", [-1., 1.])
def test_exercised_pilot_ends_even_on_negative_result_without_promotion(registered, mean):
    _, protocol = registered
    tables = {"observations": {}, "outcomes": {}}
    for slot in trial._slots(protocol):
        key = slot.isoformat()
        tables["observations"][key] = {"status": "recorded", "weights": {"reversal": .25}}
        tables["outcomes"][key] = {"status": "matured", "model_net_pct": 0., "mixed_net_pct": mean, "paired_pp": mean}
    decision = pilot._decision(protocol, tables, trial._utc(protocol["hard_review_at"]))
    assert decision["code"] == "pilot_complete_not_confirmatory"
    assert decision["statistics"]["paired_pp"] == mean
    assert not decision["automatic_promotion"] and not decision["confirmatory_evidence"]


def test_future_saved_state_is_not_silently_accepted(registered):
    root, protocol = registered
    slot = trial._utc(protocol["first_signal_at"])
    source_snapshot(root, slot)
    pilot.refresh_pilot(root, now=slot + pd.Timedelta(minutes=10))
    with pytest.raises(ValueError, match="future"):
        pilot.refresh_pilot(root, now=slot + pd.Timedelta(minutes=9))


def test_pilot_backup_enforces_primary_reserve(registered, monkeypatch):
    from types import SimpleNamespace

    root, _ = registered
    monkeypatch.setattr(pilot.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024))
    with pytest.raises(OSError, match="5 GiB reserve"):
        pilot.backup_pilot(root)
