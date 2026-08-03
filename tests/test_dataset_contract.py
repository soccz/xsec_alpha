"""Unit tests for training/live feature parity and temporal holdout isolation."""

import pandas as pd
import pytest

from data.dataset import (
    _prepare_training_frame,
    _split_timestamps_with_horizon_purge,
)


def test_temporal_split_purges_labels_reaching_holdout_boundary():
    timestamps = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")

    train, holdout, split_ts = _split_timestamps_with_horizon_purge(
        timestamps,
        holdout_ratio=0.2,
        horizon=12,
    )

    assert split_ts == timestamps[80]
    assert train[-1] == timestamps[67]
    assert timestamps[68:80].tolist() == [
        ts for ts in timestamps if ts not in train and ts not in holdout
    ]
    assert all(ts + pd.Timedelta(hours=12) < split_ts for ts in train)
    assert holdout == timestamps[80:].tolist()


@pytest.mark.parametrize(
    ("periods", "holdout_ratio", "expected_holdout"),
    [
        (100, 0.2, 20),
        (11, 0.2, 2),
        (4, 0.01, 1),
    ],
)
def test_temporal_split_preserves_requested_holdout_size(
    periods,
    holdout_ratio,
    expected_holdout,
):
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h")

    _, holdout, split_ts = _split_timestamps_with_horizon_purge(
        timestamps,
        holdout_ratio=holdout_ratio,
        horizon=1,
    )

    assert len(holdout) == expected_holdout
    assert holdout == timestamps[-expected_holdout:].tolist()
    assert split_ts == timestamps[-expected_holdout]


def test_prepare_training_frame_matches_live_fill_and_all_zero_filter():
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-01T00:00:00Z"), "KRW-A"),
            (pd.Timestamp("2026-01-01T00:00:00Z"), "KRW-B"),
            (pd.Timestamp("2026-01-01T00:00:00Z"), "KRW-C"),
            (pd.Timestamp("2026-01-01T00:00:00Z"), "KRW-D"),
        ],
        names=["timestamp", "market"],
    )
    factors = pd.DataFrame(
        {
            "factor_a": [float("nan"), 2.0, float("nan"), float("nan")],
            "factor_b": [1.0, float("nan"), float("nan"), float("nan")],
        },
        index=index,
    )
    target = pd.Series(
        [0.1, -0.2, float("nan"), 0.3],
        index=index,
        name="fwd_return",
    )

    combined = _prepare_training_frame(
        factors,
        target,
        factor_cols=["factor_a", "factor_b"],
        target_col="fwd_return",
    )

    assert combined.index.tolist() == index[:2].tolist()
    assert combined[["factor_a", "factor_b"]].to_numpy().tolist() == [
        [0.0, 1.0],
        [2.0, 0.0],
    ]
    assert combined["fwd_return"].tolist() == [0.1, -0.2]
    assert not combined.isna().any().any()
