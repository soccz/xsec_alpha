from contextlib import closing
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from utils import forecast_audit as audit
from utils import prospective as trial
from utils.experiment_supervisor import _verified_document
from utils.score_evidence import record_score_evidence

SIGNAL = "2026-09-30T11:00:00+00:00"


@pytest.fixture
def registered(tmp_path):
    for name in ("scripts/fetch_and_rank.py", "scripts/track_ic.py", "utils/score_evidence.py",
                 "data/features.py", "data/database.py", "models/xgb_ranker.py", "config.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test source\n")
    audit.start_audit(tmp_path, now="2026-09-30T06:00Z")
    return tmp_path


def capture(root, signal=SIGNAL, recorded="2026-09-30T11:05Z", kind="signal", invert=False):
    names = [f"KRW-T{i:02d}" for i in range(12)]
    inputs = pd.DataFrame({"a": np.arange(12) / 100, "b": [np.nan] + [.1] * 11}, index=names)
    scores = inputs.a * (-1 if invert else 1)
    actual = pd.Series(np.arange(12) / 100, index=names) if kind == "measurement" else None
    path = record_score_evidence(kind, signal, inputs, scores, "a" * 64,
                                 actual=actual, root=root, now=recorded)
    return _verified_document(path), path


def prices(signal=SIGNAL):
    return pd.DataFrame([{"timestamp": trial._utc(signal) + pd.Timedelta(hours=h),
                          "market": f"KRW-T{i:02d}", "open": 100. if h == 1 else 100. + i}
                         for h in (1, 7) for i in range(12)])


def read(root):
    with closing(audit._connect(root / "output/forecast_audit/ledger.sqlite", "ro")) as conn:
        return audit._read(conn)


def history(root, value=-1.):
    row = {"timestamp": SIGNAL, "ic": value, "n_coins": 12, "side": "short", "horizon_h": 6,
           "contract_version": audit.LEGACY_CONTRACT}
    (root / "output/ic_history.json").write_text(json.dumps([row]))
    return row


def report(root, witness, acknowledged=True, linked=True):
    body = {"run_id": "20260930T110800", "generated_at": "2026-09-30T11:08:00+00:00",
            "data_asof": SIGNAL, "status": "watch", "signals": [], "ideas": [{"market": "KRW-T00"}],
            "score_evidence_sha256": trial._digest(witness) if linked else "b" * 64,
            "telegram": {"state": "sent" if acknowledged else "pending"}}
    if acknowledged:
        body["telegram"]["acknowledged_at"] = "2026-09-30T11:09:00+00:00"
    folder = root / "output/operator_reports"
    folder.mkdir(exist_ok=True)
    (folder / (body["run_id"] + ".json")).write_text(json.dumps(body))
    return body


def test_registration_is_future_only_cannot_reset_and_preserves_other_trials(registered):
    policy, tables = read(registered)
    assert policy["first_signal_at"] == SIGNAL and not policy["changes_gate"]
    assert all(not values for values in tables.values())
    with pytest.raises(FileExistsError):
        audit.start_audit(registered)
    assert not (registered / "output/prospective").exists()
    assert not (registered / "output/rotation_pilot").exists()
    state = audit.refresh_audit(registered, now="2026-09-30T10:00Z")
    assert state["first_cycle"] is None and state["counts"]["scheduled"] == 0


def test_exact_saved_score_ic_no_model_replay_and_first_cycle_link(registered):
    witness, _ = capture(registered)
    report(registered, witness)
    before = audit.refresh_audit(registered, now="2026-09-30T11:10Z")
    assert before["first_cycle"]["delivery"] == "verified"
    assert before["counts"]["evaluated"] == 0
    state = audit.refresh_audit(registered, now="2026-09-30T19:10Z", price_loader=lambda *args: prices())
    assert state["first_cycle"]["saved_ic"] == pytest.approx(1.)
    assert state["first_cycle"]["n_coins"] == 12
    original = read(registered)[1]["outcomes"][SIGNAL]
    audit.refresh_audit(registered, now="2026-09-30T19:20Z", price_loader=lambda *args: pytest.fail("must not reload settled prices"))
    assert read(registered)[1]["outcomes"][SIGNAL] == original
    assert not (registered / "output/ic_history.json").exists()
    assert not (registered / "output/gate_state.json").exists()


def test_never_evaluate_before_exit_candle_is_complete(registered):
    witness, _ = capture(registered)
    for now in ("2026-09-30T11:10Z", "2026-09-30T18:59Z"):
        result, pending = audit.evaluate(witness, prices(), now)
        assert result is None and pending == "waiting_maturity"
    assert audit.evaluate(witness, prices(), "2026-09-30T19:00Z")[0]["status"] == "evaluated"


@pytest.mark.parametrize("issue", ["missing", "duplicate", "zero", "negative", "nan", "infinite"])
def test_bad_raw_prices_never_shrink_fill_or_reweight_pool(registered, issue):
    witness, _ = capture(registered)
    raw = prices()
    if issue == "missing":
        raw = raw.iloc[1:]
    elif issue == "duplicate":
        raw = pd.concat([raw, raw.iloc[:1]])
    else:
        raw.loc[0, "open"] = {"zero": 0, "negative": -1, "nan": np.nan, "infinite": np.inf}[issue]
    result, reason = audit.evaluate(witness, raw, "2026-09-30T19:10Z")
    assert result is None and "prices" in reason
    result, _ = audit.evaluate(witness, raw, "2026-10-02T18:00Z")
    assert result["status"] == "invalid" and result["n_coins"] == 12 and result["saved_ic"] is None


def test_late_evaluation_cannot_backfill_a_success_even_with_complete_prices(registered):
    witness, _ = capture(registered)
    result, _ = audit.evaluate(witness, prices(), "2026-10-02T18:01Z")
    assert result["reason"] == "evaluation_deadline" and result["saved_ic"] is None


def test_constant_targets_are_invalid_not_zero_ic(registered):
    witness, _ = capture(registered)
    raw = prices()
    raw["open"] = 100.
    outcome, _ = audit.evaluate(witness, raw, "2026-09-30T19:01Z")
    assert outcome["reason"] == "constant_ranks" and outcome["saved_ic"] is None
    state = audit.refresh_audit(registered, now="2026-09-30T19:01Z", price_loader=lambda *args: raw)
    assert state["counts"]["invalid"] == 1
    read(registered)


def test_late_signals_are_counted_missing_and_never_repaired_by_backfill(registered):
    capture(registered, recorded="2026-09-30T11:46Z")
    state = audit.refresh_audit(registered, now="2026-09-30T11:50Z")
    assert state["counts"]["missed"] == 1 and state["counts"]["recorded"] == 0
    capture(registered, recorded="2026-09-30T11:05Z")
    state = audit.refresh_audit(registered, now="2026-09-30T11:55Z")
    assert state["counts"]["missed"] == 1 and state["counts"]["recorded"] == 0


def test_first_saved_witness_is_not_replaced_by_better_rerun(registered):
    first, _ = capture(registered)
    audit.refresh_audit(registered, now="2026-09-30T11:10Z")
    capture(registered, recorded="2026-09-30T11:12Z", invert=True)
    audit.refresh_audit(registered, now="2026-09-30T11:20Z")
    assert read(registered)[1]["signals"][SIGNAL]["witness"] == first


def test_corrupt_and_wrong_kind_witness_cannot_become_original_evidence(registered):
    _, path = capture(registered)
    path.write_text('{"document":{},"sha256":"bad"}')
    with pytest.raises(ValueError, match="checksum"):
        audit.refresh_audit(registered, now="2026-09-30T11:10Z")
    assert json.loads((registered / "output/forecast_audit/summary.json").read_text())["status"] == "error"
    assert not read(registered)[1]["signals"]


def test_measurement_and_history_compare_are_append_only_and_keep_gates_unchanged(registered):
    witness, _ = capture(registered)
    measurement, _ = capture(registered, kind="measurement", recorded="2026-09-30T19:01Z", invert=True)
    legacy = history(registered)
    old = (registered / "output/ic_history.json").read_bytes()
    state = audit.refresh_audit(registered, now="2026-09-30T19:10Z", price_loader=lambda *args: prices())
    row = state["first_cycle"]
    assert row["comparison_warning"] and row["difference"] == pytest.approx(2.)
    assert row["measurement_matches_history"] and row["same_model"] and row["same_pool"]
    records = read(registered)[1]["comparisons"]
    result = next(iter(records.values()))
    assert result["measurement_raw_ic"] == pytest.approx(-1.)
    assert result["max_target_difference"] < 1e-12
    assert result["legacy_history"] == legacy and result["measurement"] == measurement
    audit.refresh_audit(registered, now="2026-09-30T19:20Z")
    assert read(registered)[1]["comparisons"] == records
    assert (registered / "output/ic_history.json").read_bytes() == old


def test_model_and_membership_changes_are_not_presented_as_identical_replay(registered):
    witness, _ = capture(registered)
    measurement, _ = capture(registered, kind="measurement", recorded="2026-09-30T19:01Z")
    measurement["model_sha256"] = "b" * 64
    measurement["rows"] = measurement["rows"][:-1]
    outcome, _ = audit.evaluate(witness, prices(), "2026-09-30T19:10Z")
    result = audit.compare(witness, outcome, history(registered, 1.), measurement, "2026-09-30T19:10Z")
    assert result["same_model"] is False and result["same_pool"] is False
    assert result["measurement_raw_ic"] is None and not result["measurement_matches_history"]


@pytest.mark.parametrize("acknowledged,linked", [(False, True), (True, False), (False, False)])
def test_report_delivery_requires_ack_and_exact_original_witness_link(registered, acknowledged, linked):
    witness, _ = capture(registered)
    report(registered, witness, acknowledged, linked)
    state = audit.refresh_audit(registered, now="2026-09-30T11:50Z")
    assert state["first_cycle"]["delivery"] == "unverified"
    assert state["counts"]["delivery_verified"] == 0
    report(registered, witness)
    state = audit.refresh_audit(registered, now="2026-09-30T11:55Z")
    assert state["first_cycle"]["delivery"] == "verified"
    assert len(read(registered)[1]["deliveries"]) == 2


def test_ledger_prevents_mutation_and_backup_restores_self_contained_evidence(registered):
    witness, _ = capture(registered)
    report(registered, witness)
    audit.refresh_audit(registered, now="2026-09-30T19:10Z", price_loader=lambda *args: prices())
    before = read(registered)
    with closing(audit._connect(registered / "output/forecast_audit/ledger.sqlite")) as conn:
        writer = json.loads(conn.execute("SELECT payload FROM outcomes LIMIT 1").fetchone()[0])["_writer"]
        assert set(writer["packages"]) == {"numpy", "pandas", "scipy"}
        assert len(writer["sources"]["forecast_audit.py"]) == 64
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM outcomes")
    one = audit.backup_audit(registered)
    assert audit.backup_audit(registered) == one
    assert one["restore_verified"] and not one["disk_failure_protected"]
    assert one["row_counts"]["outcomes"] == 1
    restored = registered / "output/forecast_audit_backups" / (one["checkpoint"] + ".sqlite")
    with closing(audit._connect(restored, "ro")) as conn:
        assert audit._read(conn) == before
    restored.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="recovery copy changed"):
        audit.backup_audit(registered)


def test_new_export_is_private_only(tmp_path, monkeypatch):
    from utils import dashboard_export
    folder = tmp_path / "forecast_audit"
    folder.mkdir()
    (folder / "summary.json").write_text('{"status":"monitoring"}')
    monkeypatch.setattr(dashboard_export, "OUTPUT_DIR", tmp_path)
    assert dashboard_export.build_summary_payload()["forecast_audit"]["status"] == "monitoring"
    assert "forecast_audit" not in dashboard_export.build_public_summary_payload()


def test_pending_prices_can_recover_before_deadline_and_are_visible(registered):
    capture(registered)
    first = audit.refresh_audit(registered, now="2026-09-30T19:10Z", price_loader=lambda *args: prices().iloc[1:])
    assert first["first_cycle"]["reason"] == "missing_prices"
    assert first["counts"]["evaluated"] == 0
    later = audit.refresh_audit(registered, now="2026-09-30T19:20Z", price_loader=lambda *args: prices())
    assert later["counts"]["evaluated"] == 1


def test_expired_evaluation_does_not_need_a_working_price_database(registered):
    capture(registered)
    state = audit.refresh_audit(registered, now="2026-10-02T18:01Z",
                                price_loader=lambda *args: pytest.fail("expired window must not read prices"))
    assert state["first_cycle"]["reason"] == "evaluation_deadline"
    assert state["counts"]["invalid"] == 1


def test_no_timely_signal_and_no_delivery_are_operational_issues(registered):
    state = audit.refresh_audit(registered, now="2026-09-30T11:46Z")
    assert state["status"] == "attention"
    assert state["operational_issues"] == ["latest_signal_missing", "latest_delivery_unverified"]


def test_a_concurrently_written_future_witness_is_deferred_not_accepted(registered):
    capture(registered, recorded="2026-09-30T11:10:01Z")
    state = audit.refresh_audit(registered, now="2026-09-30T11:10Z")
    assert state["counts"]["recorded"] == 0
    assert audit.refresh_audit(registered, now="2026-09-30T11:20Z")["counts"]["recorded"] == 1


def test_real_operator_report_path_preserves_hash_and_acknowledgment(registered, monkeypatch):
    from utils import operator_report, telegram
    witness, path = capture(registered)
    monkeypatch.setattr(operator_report, "_now", lambda: "2026-09-30T11:08:00+00:00")
    sent = []
    monkeypatch.setattr(telegram, "send_message", lambda message: sent.append(message) or True)
    document = operator_report.build_report(candidates=pd.DataFrame({"market": ["KRW-T00"]}),
                                            asof=SIGNAL, root=registered)
    document["run_id"] = "20260930T110800_integration"
    document["score_evidence_sha256"] = path.stem
    assert operator_report.publish_report(document, root=registered)
    state = audit.refresh_audit(registered, now="2026-09-30T11:10Z")
    assert len(sent) == 1 and state["first_cycle"]["delivery"] == "verified"
    assert next(iter(read(registered)[1]["deliveries"].values()))["report"]["score_evidence_sha256"] == trial._digest(witness)
