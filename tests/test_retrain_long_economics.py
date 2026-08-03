"""Regression tests for production-aligned 12h retrain promotion."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import retrain_pipeline


def _scores_by_timestamp(rows):
    index = pd.MultiIndex.from_tuples(
        [(ts, market) for ts, market, _ in rows],
        names=["timestamp", "market"],
    )
    return pd.Series([score for _, _, score in rows], index=index, name="score")


def test_long_economics_uses_exact_anchor_and_next_bar_open_return():
    timeline = pd.date_range("2026-01-01T11:00:00Z", periods=15, freq="h")
    opens = pd.DataFrame(100.0, index=timeline, columns=["KRW-A", "KRW-B"])
    opens.loc[timeline[0], "KRW-A"] = 10_000.0
    opens.loc[timeline[1], "KRW-A"] = 100.0
    opens.loc[timeline[13], "KRW-A"] = 110.0
    opens.loc[timeline[1], "KRW-B"] = 100.0
    opens.loc[timeline[13], "KRW-B"] = 90.0

    scores = _scores_by_timestamp(
        [
            (timeline[0], "KRW-A", 2.0),
            (timeline[0], "KRW-B", 1.0),
            (timeline[1], "KRW-A", 0.0),
            (timeline[1], "KRW-B", 10.0),
        ]
    )
    btc_context = pd.DataFrame(
        {"btc_ret_7d": [0.1, 0.1], "btc_ret_30d": [0.0, 0.0]},
        index=[timeline[0], timeline[1]],
    )

    stats = retrain_pipeline._measure_long_holdout_economics(
        scores,
        opens,
        btc_context,
        long_n=1,
        round_trip_cost=0.002,
    )

    assert stats["n_long_periods"] == 1
    assert stats["long_gross"] == pytest.approx(0.10)
    assert stats["long_net"] == pytest.approx(0.098)
    assert stats["long_hit_rate"] == pytest.approx(1.0)


def test_long_economics_applies_both_regime_gates_and_turnover_cost():
    timeline = pd.date_range("2026-01-01T11:00:00Z", periods=62, freq="h")
    markets = ["KRW-A", "KRW-B", "KRW-C"]
    opens = pd.DataFrame(100.0, index=timeline, columns=markets)
    endpoints = [1, 13, 25, 37, 49, 61]
    for market in markets:
        for step, position in enumerate(endpoints):
            opens.loc[timeline[position], market] = 100.0 * (1.1 ** step)

    anchors = [timeline[position] for position in (0, 12, 24, 36, 48)]
    rankings = [
        {"KRW-A": 3.0, "KRW-B": 2.0, "KRW-C": 1.0},
        {"KRW-A": 3.0, "KRW-B": 1.0, "KRW-C": 2.0},
        {"KRW-A": 3.0, "KRW-B": 1.0, "KRW-C": 2.0},
        {"KRW-A": 3.0, "KRW-B": 1.0, "KRW-C": 2.0},
        {"KRW-A": 3.0, "KRW-B": 1.0, "KRW-C": 2.0},
    ]
    scores = _scores_by_timestamp(
        [
            (ts, market, score)
            for ts, ranking in zip(anchors, rankings)
            for market, score in ranking.items()
        ]
    )
    btc_context = pd.DataFrame(
        {
            # Equality must fail both strict production gates.
            "btc_ret_7d": [0.1, 0.1, 0.0, 0.1, 0.1],
            "btc_ret_30d": [0.0, 0.0, 0.0, -0.10, 0.0],
        },
        index=anchors,
    )

    stats = retrain_pipeline._measure_long_holdout_economics(
        scores,
        opens,
        btc_context,
        long_n=2,
        btc_7d_gate=0.0,
        btc_30d_floor=-0.10,
        round_trip_cost=0.02,
    )

    # Active periods are anchors 0, 1, and 4. Turnover is 100%, 50%, 100%:
    # net returns are 8%, 9%, and 8% after the 2% round-trip cost.
    assert stats["n_long_periods"] == 3
    assert stats["long_gross"] == pytest.approx(0.10)
    assert stats["long_net"] == pytest.approx((0.08 + 0.09 + 0.08) / 3)
    assert stats["long_hit_rate"] == pytest.approx(1.0)


def test_long_economics_filters_to_live_tradable_universe():
    timeline = pd.date_range("2026-01-01T11:00:00Z", periods=14, freq="h")
    opens = pd.DataFrame(100.0, index=timeline, columns=["KRW-A", "KRW-B"])
    opens.loc[timeline[13], "KRW-A"] = 120.0
    opens.loc[timeline[13], "KRW-B"] = 90.0
    scores = _scores_by_timestamp([
        (timeline[0], "KRW-A", 2.0),
        (timeline[0], "KRW-B", 1.0),
    ])
    btc_context = pd.DataFrame(
        {"btc_ret_7d": [0.1], "btc_ret_30d": [0.0]},
        index=[timeline[0]],
    )

    stats = retrain_pipeline._measure_long_holdout_economics(
        scores,
        opens,
        btc_context,
        long_n=1,
        round_trip_cost=0.0,
        tradable_markets={"KRW-B"},
    )

    assert stats["long_gross"] == pytest.approx(-0.10)


def test_aligned_ic_uses_only_exact_next_open_anchor():
    timeline = pd.date_range("2026-01-01T05:00:00Z", periods=8, freq="h")
    markets = [f"KRW-{i:02d}" for i in range(20)]
    opens = pd.DataFrame(100.0, index=timeline, columns=markets)
    opens.loc[timeline[7]] = np.arange(100.0, 120.0)
    scores = _scores_by_timestamp(
        [(timeline[0], market, float(index)) for index, market in enumerate(markets)]
        + [(timeline[1], market, float(19 - index)) for index, market in enumerate(markets)]
    )

    values = retrain_pipeline._measure_aligned_ic(
        scores,
        opens,
        horizon=6,
        anchor_hours_utc=(5, 11, 17, 23),
    )

    assert values == pytest.approx([1.0])


@pytest.mark.parametrize(
    (
        "horizon",
        "new_ic",
        "old_ic",
        "long_net",
        "n_long_periods",
        "force",
        "expected_promoted",
        "expected_long_pass",
    ),
    [
        (6, 0.05, 0.06, None, 0, False, True, True),
        (12, 0.05, 0.06, 0.001, 10, False, True, True),
        (12, 0.05, 0.06, 0.001, 9, False, False, False),
        (12, 0.05, 0.06, 0.0, 10, False, False, False),
        (12, 0.0, 0.10, -0.01, 0, True, True, False),
    ],
)
def test_promotion_gate_is_economic_only_for_12h_and_force_still_bypasses(
    horizon,
    new_ic,
    old_ic,
    long_net,
    n_long_periods,
    force,
    expected_promoted,
    expected_long_pass,
):
    decision = retrain_pipeline._evaluate_promotion_gate(
        horizon=horizon,
        new_ic=new_ic,
        old_ic=old_ic,
        min_delta=-0.015,
        force=force,
        long_net=long_net,
        n_long_periods=n_long_periods,
    )

    assert decision["promoted"] is expected_promoted
    assert decision["pass_long_economics"] is expected_long_pass


def test_measure_holdout_stats_neutral_fills_features_before_prediction(monkeypatch):
    from data import dataset
    from models import xgb_ranker

    timestamp = pd.Timestamp("2026-01-01T05:00:00Z")
    markets = [f"KRW-{i:02d}" for i in range(20)]
    index = pd.MultiIndex.from_product(
        [[timestamp], markets],
        names=["timestamp", "market"],
    )
    features = pd.DataFrame(
        {"factor": [float("nan"), *range(1, 20)]},
        index=index,
    )
    target = pd.Series(np.arange(20, dtype=float), index=index)
    timeline = pd.date_range(timestamp, periods=8, freq="h")
    opens = pd.DataFrame(100.0, index=timeline, columns=markets)
    opens.loc[timeline[7]] = np.arange(100.0, 120.0)
    monkeypatch.setattr(
        dataset,
        "build_dataset",
        lambda **kwargs: {
            "X_holdout": features,
            "y_holdout": target,
            "opens": opens,
        },
    )

    class FakeModel:
        def predict(self, frame):
            assert not frame.isna().any().any()
            return frame["factor"].to_numpy()

    class FakeRanker:
        @staticmethod
        def load(path):
            return FakeModel()

    monkeypatch.setattr(xgb_ranker, "XSecRanker", FakeRanker)

    stats = retrain_pipeline.measure_holdout_stats(Path("unused.pkl"), horizon=6)

    assert stats["ic"] == pytest.approx(1.0)
    assert stats["n_periods"] == 1
    assert "long_net" not in stats
