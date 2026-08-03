"""Contract tests for temporal-holdout, next-open sigma calibration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import rebuild_calibration


def _feature_frame(timestamps, markets=("KRW-A", "KRW-B", "KRW-C")):
    index = pd.MultiIndex.from_product(
        [timestamps, markets],
        names=["timestamp", "market"],
    )
    values = np.tile([1.0, 2.0, 3.0], len(timestamps))
    return pd.DataFrame({"factor": values}, index=index)


class _FactorModel:
    def predict(self, features):
        assert not features.isna().any().any()
        return features["factor"].to_numpy(dtype=float)


def test_next_open_return_has_one_bar_lag_and_full_horizon():
    timeline = pd.date_range("2026-01-01T05:00:00Z", periods=8, freq="h")
    opens = pd.DataFrame({"KRW-A": 100.0}, index=timeline)
    opens.loc[timeline[0], "KRW-A"] = 10_000.0  # signal-bar open: must be ignored
    opens.loc[timeline[1], "KRW-A"] = 100.0     # entry at next open
    opens.loc[timeline[7], "KRW-A"] = 110.0     # six hours after entry

    returns = rebuild_calibration._next_open_returns(opens, horizon_h=6)

    assert returns.loc[timeline[0], "KRW-A"] == pytest.approx(0.10)


def test_holdout_scan_uses_exact_anchors_live_fill_and_all_zero_filter():
    timeline = pd.date_range("2026-01-01T05:00:00Z", periods=20, freq="h")
    exact_5 = timeline[0]
    non_anchor = timeline[1]
    exact_11 = timeline[6]
    non_exact_11 = exact_11 + pd.Timedelta(minutes=30)
    non_exact_11_ns = exact_11 + pd.Timedelta(nanoseconds=1)
    features = _feature_frame(
        [exact_5, non_anchor, exact_11, non_exact_11, non_exact_11_ns]
    )
    features.loc[(exact_5, "KRW-A"), "factor"] = np.nan
    features.loc[(exact_11, "KRW-A"), "factor"] = np.nan
    opens = pd.DataFrame(100.0, index=timeline, columns=["KRW-A", "KRW-B", "KRW-C"])
    opens.loc[timeline[7], "KRW-B"] = 110.0
    opens.loc[timeline[13], "KRW-C"] = 90.0

    samples = rebuild_calibration._scan_holdout(
        _FactorModel(),
        features,
        opens,
        horizon_h=6,
        anchors_utc=(5, 11, 17, 23),
    )

    assert set(samples["timestamp"]) == {exact_5, exact_11}
    assert set(samples["market"]) == {"KRW-B", "KRW-C"}
    assert non_anchor not in set(samples["timestamp"])
    assert non_exact_11 not in set(samples["timestamp"])
    assert non_exact_11_ns not in set(samples["timestamp"])


def test_long_scan_requires_both_strict_finite_regime_gates():
    timeline = pd.date_range("2026-01-01T11:00:00Z", periods=50, freq="h")
    anchors = [timeline[i] for i in (0, 12, 24, 36)]
    features = _feature_frame(anchors)
    opens = pd.DataFrame(100.0, index=timeline, columns=["KRW-A", "KRW-B", "KRW-C"])
    btc_context = pd.DataFrame(
        {
            "btc_ret_7d": [0.0, 0.01, np.nan, 0.01],
            "btc_ret_30d": [0.0, -0.10, 0.0, -0.09],
        },
        index=anchors,
    )

    samples = rebuild_calibration._scan_holdout(
        _FactorModel(),
        features,
        opens,
        horizon_h=12,
        anchors_utc=(11, 23),
        btc_context=btc_context,
        btc_7d_gate=0.0,
        btc_30d_floor=-0.10,
    )

    assert set(samples["timestamp"]) == {anchors[-1]}


def test_bucket_output_preserves_existing_json_schema():
    samples = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="h"),
            "market": [f"KRW-{i}" for i in range(5)],
            "sigma": [0.25, 0.75, 1.25, 1.75, 2.25],
            "score": [1.0, -1.0, 1.0, -1.0, 1.0],
            "ret": [0.01, -0.02, -0.03, 0.04, 0.05],
        }
    )

    buckets = rebuild_calibration._calibrate_buckets(samples)

    expected_keys = {
        "sigma_low",
        "sigma_high",
        "n",
        "hit_rate",
        "mean_signed_return_pct",
        "mean_abs_return_pct",
        "std_return_pct",
    }
    assert len(buckets) == 5
    assert all(set(bucket) == expected_keys for bucket in buckets)
    assert buckets[-1]["sigma_high"] is None


def test_document_uses_only_temporal_holdouts_and_production_models():
    dataset_calls = []
    model_paths = []

    def dataset_builder(**kwargs):
        dataset_calls.append(kwargs)
        horizon = kwargs["horizon"]
        signal = pd.Timestamp(
            "2026-01-01T05:00:00Z" if horizon == 6 else "2026-01-01T11:00:00Z"
        )
        timeline = pd.date_range(signal, periods=horizon + 2, freq="h")
        opens = pd.DataFrame(
            100.0,
            index=timeline,
            columns=["KRW-A", "KRW-B", "KRW-C"],
        )
        return {
            "X_train": "must not be read",
            "X_holdout": _feature_frame([signal]),
            "opens": opens,
            "closes": opens,
        }

    def model_loader(path):
        model_paths.append(Path(path).name)
        return _FactorModel()

    def regime_builder(closes):
        return pd.DataFrame(
            {"btc_ret_7d": 0.01, "btc_ret_30d": -0.09},
            index=closes.index,
        )

    document = rebuild_calibration.build_calibration_document(
        dataset_builder=dataset_builder,
        model_loader=model_loader,
        regime_builder=regime_builder,
        generated_at="2026-01-02T00:00:00+00:00",
    )

    assert [call["holdout_ratio"] for call in dataset_calls] == [0.2, 0.2]
    assert [call["horizon"] for call in dataset_calls] == [6, 12]
    assert all(call["side"] == "unified" for call in dataset_calls)
    assert all(call["target"] == "absolute" for call in dataset_calls)
    assert model_paths == ["xsec_6h.pkl", "xsec_12h.pkl"]
    assert document["short_6h"]
    assert document["long_12h"]
    assert document["generated_at"] == "2026-01-02T00:00:00+00:00"
    assert document["_provenance"]["oos"] is True
    assert document["_provenance"]["lag_bars"] == 1
    assert document["_provenance"]["anchors_utc"] == {
        "short_6h": [5, 11, 17, 23],
        "long_12h": [11, 23],
    }


def test_retrain_pipeline_calls_audited_calibration_script(monkeypatch):
    from scripts import retrain_pipeline

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr(retrain_pipeline.subprocess, "run", fake_run)

    assert retrain_pipeline.rebuild_calibration() is True
    assert captured["command"] == [
        retrain_pipeline.sys.executable,
        str(retrain_pipeline.ROOT / "scripts" / "rebuild_calibration.py"),
    ]
    assert captured["kwargs"]["cwd"] == str(retrain_pipeline.ROOT)
    assert captured["kwargs"]["check"] is True
