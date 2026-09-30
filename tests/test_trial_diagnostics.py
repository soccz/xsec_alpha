import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from utils import prospective as trial
from utils import trial_diagnostics as audit


def rows():
    protocol = {"first_signal_at": "2026-09-07T05:00:00+00:00", "n_slots": 60,
                "basket_size": 5, "model_sha256": "frozen"}
    observations, outcomes = [], []
    for i, slot in enumerate(trial._slots(protocol)):
        key = slot.isoformat()
        raw = np.linspace(-.04, .04, 10) * (1 if i % 2 else -1) + i / 100000
        members = [{"market": f"T{j}", "model_selected": j < 5, "baseline_selected": j >= 5,
                    "missing_inputs": ["x"] if j == 0 else []} for j in range(10)]
        observations.append((key, json.dumps({"status": "recorded", "rows": members,
                                             "cutoff_gap_sigma": .1, "context": {"short_gate": "WARN"}})))
        strategies = {}
        for name, values in zip(audit.STRATEGIES, (raw[:5], raw[5:], raw)):
            strategies[name] = {"net_return": -float(np.mean(values)) - .003, "cost_return": .003}
        outcomes.append((key, json.dumps({"status": "matured", "strategies": strategies,
                                          "prices": [{"market": f"T{j}", "raw_return": v} for j, v in enumerate(raw)]})))
    return protocol, observations, outcomes


def test_contributions_reconcile_and_use_windows_not_coins():
    protocol, obs, out = rows()
    result = audit.analyze_rows(protocol, obs, out, {})
    assert result["n_windows"] == result["power"]["pilot_n"] == 60
    assert result["selection"]["selected_coin_rows"] == 300
    assert result["selection"]["selected_missing_input_rows"] == 60
    coins = result["coin_attribution"]
    assert sum(c["model_mean_contribution_pp"] for c in coins) == pytest.approx(result["strategies"]["model_short5"]["mean_pct"])
    assert sum(c["paired_mean_contribution_pp"] for c in coins) == pytest.approx(result["paired"]["mean_pct"])
    assert result["groups"]["regime"][0]["label"] == "unknown"
    json.dumps(result, allow_nan=False)


def test_missing_or_invalid_window_is_not_silently_dropped():
    p, obs, out = rows()
    with pytest.raises(ValueError, match="complete trial"):
        audit.analyze_rows(p, obs, out[:-1], {})
    out[0] = (out[0][0], '{"status":"invalid_prices"}')
    with pytest.raises(ValueError, match="complete trial"):
        audit.analyze_rows(p, obs, out, {})


def test_common_cost_changes_absolute_not_paired_return():
    result = audit.analyze_rows(*rows(), {})
    first, last = result["cost_sensitivity"][0], result["cost_sensitivity"][-1]
    assert first["paired_mean_pp"] == last["paired_mean_pp"]
    assert first["model_mean_net_pct"] - last["model_mean_net_pct"] == pytest.approx(.3)


def test_bootstrap_reproduces_registered_four_window_interval():
    data = np.random.default_rng(7).normal(size=60)
    assert np.quantile(audit.block_means(data, 4, 2000), [.025, .975]) == pytest.approx(trial._block_interval(data))


def test_power_scales_with_variance_and_effect_not_observed_mean():
    values = np.tile([-2., -1., 1., 2.], 15)
    one = audit.power_scenarios(values)
    shifted = audit.power_scenarios(values + 100)
    assert one == shifted
    base = {r["effect_pp"]: r for r in one["scenarios"] if r["variance_multiplier"] == 1}
    assert base[.25]["n_windows"] >= 3.9 * base[.5]["n_windows"]
    assert one["planning_variance_pp2"] >= one["variance_estimates_pp2"]["iid"]
    assert one["adaptive_policy_power_established"] is False


@pytest.mark.parametrize("values", [[1.], [1., np.nan], [1.] * 60])
def test_invalid_or_zero_variance_pilot_rejected(values):
    with pytest.raises(ValueError):
        audit.power_scenarios(values)


def test_historical_labels_ignore_future_candles_and_never_write_database(tmp_path):
    db = tmp_path / "prices.sqlite"
    index = pd.date_range("2026-07-01", periods=24 * 70, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": index.strftime("%Y-%m-%dT%H:%M:%S"), "market": "KRW-BTC",
                          "close": 100 * np.exp(np.sin(np.arange(len(index)) / 30) * .01)})
    with sqlite3.connect(db) as conn:
        frame.to_sql("crypto_data", conn, index=False)
    signals = ["2026-08-20T05:00:00+00:00", "2026-08-21T05:00:00+00:00"]
    before = db.read_bytes()
    first = audit.reconstruct_regimes(signals, db)
    assert db.read_bytes() == before
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE crypto_data SET close=999999 WHERE timestamp>=?", ("2026-08-21T05:00:00",))
    assert audit.reconstruct_regimes(signals, db) == first
    assert first[1]["candle_revision_history_available"] is False


def test_unsealed_or_corrupt_trial_cannot_generate_diagnostics(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit.build_diagnostics(tmp_path)


def test_diagnostics_stay_out_of_public_export(tmp_path, monkeypatch):
    from utils import dashboard_export

    folder = tmp_path / "output/trial_diagnostics"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text('{"status":"post_hoc_diagnostic"}')
    monkeypatch.setattr(dashboard_export, "OUTPUT_DIR", tmp_path / "output")
    assert dashboard_export.build_summary_payload()["trial_diagnostics"]["status"] == "post_hoc_diagnostic"
    assert "trial_diagnostics" not in dashboard_export.build_public_summary_payload()
