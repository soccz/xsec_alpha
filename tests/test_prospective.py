import json
import pickle

import numpy as np
import pandas as pd
import pytest

from utils import prospective as trial


class SimpleRanker:
    _feature_names = ["reversal_4h", "reversal_1h"]

    def predict(self, frame):
        return frame["reversal_1h"].to_numpy() * 0.02


@pytest.fixture
def experiment(tmp_path):
    for name in trial.SOURCE_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("test source\n")
    (tmp_path / "models/xsec_6h.pkl").write_bytes(pickle.dumps(SimpleRanker()))
    protocol = trial.start_experiment(tmp_path, now="2026-09-07T04:30Z")
    return tmp_path, protocol


def inputs():
    return pd.DataFrame({"reversal_4h": np.arange(10, dtype=float),
                         "reversal_1h": -np.arange(10, dtype=float)},
                        index=[f"KRW-T{i:02d}" for i in range(10)])


def record(root, signal="2026-09-07T05:00Z", now="2026-09-07T05:05Z", frame=None):
    frame = inputs() if frame is None else frame
    return trial.record_snapshot(signal, frame, frame, frame.index, root=root, now=now,
                                 context={"short_gate": "FREEZE"})


def snapshot(root):
    with trial._db(root) as conn:
        return json.loads(conn.execute("SELECT payload FROM observations ORDER BY slot LIMIT 1").fetchone()[0])


def prices(row):
    return pd.DataFrame([
        {"timestamp": ts, "market": item["market"], "open": float(value)}
        for i, item in enumerate(row["rows"])
        for ts, value in [(row["entry_at"], 100), (row["exit_at"], 100 - i)]
    ])


def test_start_is_future_only_and_does_not_modify_production(experiment):
    root, protocol = experiment
    assert protocol["first_signal_at"] == "2026-09-07T05:00:00+00:00"
    assert len(trial._slots(protocol)) == 60
    assert (root / "models/xsec_6h.pkl").read_bytes() == (root / "output/prospective/model.pkl").read_bytes()
    with pytest.raises(FileExistsError):
        trial.start_experiment(root)
    assert trial.experiment_summary(root)["n_recorded"] == 0
    assert not (root / "output/recommendation_ledger.csv").exists()


@pytest.mark.parametrize("signal,now,expected", [
    ("2026-09-06T23:00Z", "2026-09-07T05:05Z", "outside_protocol"),
    ("2026-09-07T06:00Z", "2026-09-07T06:05Z", "outside_protocol"),
    ("2026-09-07T05:00Z", "2026-09-07T04:59Z", "outside_capture_window"),
    ("2026-09-07T05:00Z", "2026-09-07T05:46Z", "outside_capture_window"),
])
def test_no_backfill_or_lookahead(experiment, signal, now, expected):
    root, _ = experiment
    assert record(root, signal, now) == expected
    assert trial.experiment_summary(root)["n_recorded"] == 0


def test_immutable_prediction_and_same_universe_baseline(experiment):
    root, _ = experiment
    assert record(root) == "recorded"
    original = snapshot(root)
    assert record(root, frame=inputs() * -1) == "already_recorded"
    assert snapshot(root) == original
    assert original["entry_at"] == "2026-09-07T06:00:00+00:00"
    assert original["exit_at"] == "2026-09-07T12:00:00+00:00"
    assert sum(r["model_selected"] for r in original["rows"]) == 5
    assert sum(r["baseline_selected"] for r in original["rows"]) == 5
    assert original["context"]["short_gate"] == "FREEZE"
    assert not (root / "output/latest.csv").exists()


def test_frozen_model_survives_production_replacement(experiment):
    root, _ = experiment
    (root / "models/xsec_6h.pkl").write_bytes(b"replacement production model")
    assert record(root) == "recorded"
    row = snapshot(root)["rows"][-1]
    assert row["score"] == pytest.approx(-0.18)


@pytest.mark.parametrize("path", ["data/features.py", "output/prospective/model.pkl"])
def test_changed_contract_or_frozen_model_fails_closed(experiment, path):
    root, _ = experiment
    (root / path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        record(root)
    summary = trial.experiment_summary(root)
    assert summary["n_recorded"] == 0
    assert not (summary["runtime_matches"] and summary["frozen_model_matches"])


def test_missing_input_mask_and_sensitivity_sign(experiment):
    root, _ = experiment
    frame = inputs()
    raw = frame.copy()
    raw.iloc[-1, 1] = np.nan
    trial.record_snapshot("2026-09-07T05:00Z", frame, raw, frame.index,
                          root=root, now="2026-09-07T05:05Z")
    row = snapshot(root)["rows"][-1]
    assert row["missing_inputs"] == ["reversal_1h"]
    assert row["sensitivity"][0] == {"feature": "reversal_1h", "score_delta": pytest.approx(-0.09)}


def test_maturity_uses_exact_raw_future_opens_and_signed_costs(experiment):
    root, protocol = experiment
    record(root)
    row = snapshot(root)
    data = prices(row)
    assert trial.evaluate_snapshot(row, data, protocol, now="2026-09-07T12:59Z") is None
    outcome = trial.evaluate_snapshot(row, data, protocol, now="2026-09-07T13:00Z")
    assert outcome["strategies"]["model_short5"]["gross_return"] == pytest.approx(0.07)
    assert outcome["strategies"]["reversal_short5"]["gross_return"] == pytest.approx(0.02)
    assert outcome["strategies"]["model_short5"]["net_return"] == pytest.approx(0.067)
    assert outcome["model_ic"] == pytest.approx(1.0)
    assert outcome["baseline_ic"] == pytest.approx(-1.0)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "zero", "nan", "inf", "wrong_time"])
def test_bad_price_never_shrinks_basket_or_becomes_zero(experiment, corruption):
    root, protocol = experiment
    record(root)
    row = snapshot(root)
    data = prices(row)
    if corruption == "missing":
        data = data.iloc[:-1]
    elif corruption == "duplicate":
        data = pd.concat([data, data.iloc[[-1]]])
    elif corruption == "wrong_time":
        data.loc[data.index[-1], "timestamp"] = "2026-09-07T13:00Z"
    else:
        data.loc[data.index[-1], "open"] = {"zero": 0, "nan": np.nan, "inf": np.inf}[corruption]
    assert trial.evaluate_snapshot(row, data, protocol, now="2026-09-07T13:00Z") is None
    outcome = trial.evaluate_snapshot(row, data, protocol, now="2026-09-09T12:00Z")
    assert outcome["status"] == "invalid_prices"
    assert "strategies" not in outcome


def test_missing_slots_and_outcomes_are_not_rewritten(experiment):
    root, _ = experiment
    record(root)
    original = snapshot(root)
    trial.advance_experiment(root, now="2026-09-07T13:00Z", price_loader=lambda *a: prices(original))
    summary = trial.experiment_summary(root)
    assert summary["n_matured"] == 1 and summary["n_missed"] == 1
    assert summary["paired_excess_pct"] == pytest.approx(5)
    assert summary["paired_ci95_pct"] is None
    assert summary["context_breakdown"][0]["short_gate"] == "FREEZE"
    matured = next(row for row in summary["recent_windows"] if row["status"] == "matured")
    assert matured["model_net_pct"] == pytest.approx(6.7)
    assert len(matured["evidence"]) == 5
    assert matured["evidence"][-1]["net_pct"] == pytest.approx(8.7)
    assert matured["evidence"][-1]["sensitivity"] == original["rows"][-1]["sensitivity"]
    trial.advance_experiment(root, now="2026-09-07T13:00Z", price_loader=lambda *a: pytest.fail("reran outcome"))
    assert trial.experiment_summary(root) == summary


def test_sixty_missed_slots_do_not_become_a_success(experiment):
    root, _ = experiment
    trial.advance_experiment(root, now="2026-09-30T00:00Z")
    summary = trial.experiment_summary(root)
    assert summary["n_missed"] == 60
    assert summary["status"] == "incomplete_evidence"
    assert summary["paired_ci95_pct"] is None
    assert summary["paired_excess_pct"] is None


def test_sixty_windows_not_three_hundred_independent_coins(experiment):
    root, protocol = experiment
    for slot in trial._slots(protocol):
        assert record(root, signal=slot, now=slot + pd.Timedelta(minutes=5)) == "recorded"
    with trial._db(root) as conn:
        all_rows = [json.loads(raw) for (raw,) in conn.execute("SELECT payload FROM observations ORDER BY slot")]
    all_prices = pd.concat([prices(row) for row in all_rows]).drop_duplicates(["timestamp", "market"])
    trial.advance_experiment(root, now="2026-09-30T00:00Z", price_loader=lambda *a: all_prices)
    summary = trial.experiment_summary(root)
    assert summary["n_matured"] == 60
    assert summary["status"] == "ready_for_review"
    assert all(s["n_windows"] == 60 for s in summary["strategies"])
    assert summary["paired_ci95_pct"] is not None
    assert all(np.isfinite(v) for v in summary["paired_ci95_pct"])


def test_provenance_or_post_entry_snapshot_is_rejected(experiment):
    root, protocol = experiment
    record(root)
    row = snapshot(root)
    row["recorded_at"] = row["entry_at"]
    with pytest.raises(ValueError, match="before entry"):
        trial.evaluate_snapshot(row, prices(row), protocol, now="2026-09-30T00:00Z")


def test_public_export_does_not_include_experiment(tmp_path, monkeypatch):
    from utils import dashboard_export

    monkeypatch.setattr(dashboard_export, "OUTPUT_DIR", tmp_path / "output")
    private = dashboard_export.build_summary_payload()
    public = dashboard_export.build_public_summary_payload()
    assert private["prospective_experiment"] == {"status": "not_started"}
    assert "prospective_experiment" not in public


def test_maintenance_failure_is_nonfatal(tmp_path, monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("injected experiment failure")
    monkeypatch.setattr(trial, "advance_experiment", broken)
    trial.maintain_experiment(root=tmp_path)


def test_protocol_corruption_is_not_silently_accepted(experiment):
    root, _ = experiment
    with trial._db(root) as conn:
        conn.execute("UPDATE protocol SET payload='{}'")
    with pytest.raises(ValueError, match="checksum"):
        record(root)


def test_raw_price_reader_does_not_fill_or_drop_delisted_names(tmp_path, monkeypatch):
    import sqlite3

    db_path = tmp_path / "prices.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE crypto_data(timestamp TEXT, market TEXT, open REAL)")
        conn.executemany("INSERT INTO crypto_data VALUES (?, ?, ?)", [
            ("2026-09-07T06:00:00", "KRW-DELISTED", 42.0),
            ("2026-09-07 12:00:00", "KRW-OTHER", 7.0),
            ("2026-09-06T06:00:00", "KRW-DELISTED", 100.0),
        ])
    monkeypatch.setattr(trial.config.General, "DB_PATH", str(db_path))
    data = trial._raw_prices("2026-09-07T06:00Z", "2026-09-07T12:00Z")
    assert len(data) == 2
    assert set(data["market"]) == {"KRW-DELISTED", "KRW-OTHER"}
    assert data["open"].sum() == 49
