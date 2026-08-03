import numpy as np
import pandas as pd
import pytest

from scripts.wf_holdout_harness import (
    SIDE_CONTRACTS,
    basket_turnover,
    compute_next_open_forward_returns,
    evaluate_top_n_slot,
    long_regime_allows,
    select_exact_anchor_timestamps,
    turnover_adjusted_net_return,
)


def test_side_contracts_pin_production_models_and_exact_utc_anchors():
    assert SIDE_CONTRACTS["short"] == {
        "horizon_h": 6,
        "anchor_hours_utc": (5, 11, 17, 23),
        "model_file": "xsec_6h.pkl",
    }
    assert SIDE_CONTRACTS["long"] == {
        "horizon_h": 12,
        "anchor_hours_utc": (11, 23),
        "model_file": "xsec_12h.pkl",
    }


def test_exact_anchor_selection_rejects_subhour_and_enforces_non_overlap():
    timestamps = list(pd.date_range("2026-08-01", periods=36, freq="h", tz="UTC"))
    timestamps.extend(
        [
            pd.Timestamp("2026-08-01 05:30", tz="UTC"),
            pd.Timestamp("2026-08-01 05:00", tz="UTC"),
            pd.Timestamp("2026-08-01 08:00", tz="UTC"),
        ]
    )

    short = select_exact_anchor_timestamps(timestamps, (5, 8, 11, 17, 23), horizon_h=6)

    assert short == [
        pd.Timestamp("2026-08-01 05:00", tz="UTC"),
        pd.Timestamp("2026-08-01 11:00", tz="UTC"),
        pd.Timestamp("2026-08-01 17:00", tz="UTC"),
        pd.Timestamp("2026-08-01 23:00", tz="UTC"),
        pd.Timestamp("2026-08-02 05:00", tz="UTC"),
        pd.Timestamp("2026-08-02 11:00", tz="UTC"),
    ]


def test_long_anchor_windows_touch_only_at_the_boundary():
    timestamps = pd.date_range("2026-08-01", periods=48, freq="h", tz="UTC")

    selected = select_exact_anchor_timestamps(timestamps, (11, 23), horizon_h=12)

    assert [ts.hour for ts in selected] == [11, 23, 11, 23]
    assert all(
        current - previous == pd.Timedelta(hours=12)
        for previous, current in zip(selected, selected[1:])
    )


def test_next_open_forward_return_uses_lag_one_entry_and_horizon_later_open():
    index = pd.date_range("2026-08-01", periods=9, freq="h", tz="UTC")
    opens = pd.DataFrame(
        {"KRW-BTC": [100.0, 110.0, 121.0, 133.1, 146.41, 161.051, 177.1561, 194.87171, 214.358881]},
        index=index,
    )

    returns = compute_next_open_forward_returns(opens, horizon_h=2)

    # Signal t0 enters open[t1]=110 and exits open[t3]=133.1.
    assert returns.loc[index[0], "KRW-BTC"] == pytest.approx(133.1 / 110.0 - 1.0)
    assert np.isnan(returns.loc[index[-3], "KRW-BTC"])


@pytest.mark.parametrize(
    ("btc_7d", "btc_30d", "expected"),
    [
        (0.01, -0.09, True),
        (0.00, -0.09, False),
        (0.01, -0.10, False),
        (np.nan, 0.10, False),
        (0.10, np.nan, False),
    ],
)
def test_long_regime_requires_both_strict_finite_gates(btc_7d, btc_30d, expected):
    row = pd.Series({"btc_ret_7d": btc_7d, "btc_ret_30d": btc_30d})
    assert long_regime_allows(row, btc_7d_gate=0.0, btc_30d_floor=-0.10) is expected


def test_turnover_and_round_trip_cost_scale_with_new_basket_fraction():
    previous = {"A", "B", "C", "D", "E"}
    unchanged = set(previous)
    one_replacement = {"A", "B", "C", "D", "F"}

    assert basket_turnover(unchanged, previous) == 0.0
    assert basket_turnover(one_replacement, previous) == pytest.approx(0.2)
    assert basket_turnover(one_replacement, set()) == 1.0
    assert turnover_adjusted_net_return(0.01, 1.0, 6.0, 4.0) == pytest.approx(0.008)
    assert turnover_adjusted_net_return(0.01, 0.2, 6.0, 4.0) == pytest.approx(0.0096)


def test_top5_slot_records_gross_net_turnover_and_hits():
    markets = list("ABCDEFG")
    scores = pd.Series([7, 6, 5, 4, 3, 2, 1], index=markets, dtype=float)
    forward_returns = pd.Series(
        [0.010, 0.020, -0.005, 0.015, 0.010, 0.50, 0.50],
        index=markets,
        dtype=float,
    )

    slot = evaluate_top_n_slot(
        scores,
        forward_returns,
        previous={"A", "B", "C", "D", "F"},
        top_n=5,
        fee_bps=6.0,
        slippage_bps=4.0,
    )

    expected_gross = (0.010 + 0.020 - 0.005 + 0.015 + 0.010) / 5
    assert slot["picks"] == {"A", "B", "C", "D", "E"}
    assert slot["gross_return"] == pytest.approx(expected_gross)
    assert slot["turnover"] == pytest.approx(0.2)
    assert slot["net_return"] == pytest.approx(expected_gross - 0.2 * 0.002)
    assert slot["gross_hit"] == 1.0
    assert slot["net_hit"] == 1.0


def test_top5_selection_never_uses_future_return_availability():
    scores = pd.Series([6, 5, 4, 3, 2, 1], index=list("ABCDEF"), dtype=float)
    forward_returns = pd.Series(
        [np.nan, 0.01, 0.01, 0.01, 0.01, 0.50], index=list("ABCDEF"), dtype=float
    )

    slot = evaluate_top_n_slot(
        scores,
        forward_returns,
        previous=set(),
        top_n=5,
        fee_bps=6.0,
        slippage_bps=4.0,
    )

    assert slot["picks"] == {"A", "B", "C", "D", "E"}
    assert slot["gross_return"] is None
    assert "F" not in slot["picks"]
