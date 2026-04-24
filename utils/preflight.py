"""Pre-flight gates before emitting actionable predictions.

Batch IV consensus (live-ops + risk): check data freshness, feature NaN rate,
universe stability, and known-bad coin states (listing window, suppression)
BEFORE marking a prediction actionable. If any gate trips, set the row's
`suppression` column to a human-readable reason and set `actionable=False`
on affected coins.

Gates (all soft — they suppress rather than crash):
  1. Data freshness: latest factor timestamp within ≤90 min of now
  2. Feature NaN rate: each column ≤ 20% NaN (universe-level)
  3. Universe churn: top-100 membership change ≤ 30% day-over-day
  4. Per-coin: insufficient history (< 24 hours of closes)
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd


def check_data_freshness(latest_ts, max_age_minutes: int = 90) -> Optional[str]:
    """Returns None if fresh, else a suppression reason string."""
    if latest_ts is None:
        return "data_missing"
    now = datetime.now(timezone.utc)
    ts = pd.Timestamp(latest_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    age_min = (now - ts.to_pydatetime()).total_seconds() / 60
    if age_min > max_age_minutes:
        return f"data_stale_{int(age_min)}min"
    return None


def check_feature_nan(latest_factors: "pd.DataFrame", max_nan_frac: float = 0.20) -> dict[str, float]:
    """Returns per-column NaN fraction. Caller decides which columns trip the gate."""
    if latest_factors is None or len(latest_factors) == 0:
        return {}
    frac = latest_factors.isna().mean().to_dict()
    return {k: float(v) for k, v in frac.items() if v > max_nan_frac}


def check_universe_churn(current_universe: set, previous_universe: set,
                          max_churn_frac: float = 0.30,
                          min_comparable: int = 50) -> Optional[str]:
    """Compare membership sets, return warning if >max_churn_frac turnover.

    Requires previous_universe to have at least `min_comparable` entries —
    otherwise the comparison is meaningless (e.g. prev was top-5 basket only).
    """
    if not current_universe or not previous_universe:
        return None
    if len(previous_universe) < min_comparable:
        return None
    added = current_universe - previous_universe
    removed = previous_universe - current_universe
    churn = max(len(added), len(removed)) / max(len(current_universe), 1)
    if churn > max_churn_frac:
        return f"universe_churn_{int(churn*100)}%"
    return None


def check_coin_history(coin_closes: "pd.Series", min_hours: int = 24) -> Optional[str]:
    """Returns suppression reason if coin has insufficient history."""
    if coin_closes is None or coin_closes.dropna().shape[0] < min_hours:
        return f"insufficient_history"
    return None


def run_preflight(
    latest_ts,
    latest_factors,
    current_universe: set,
    previous_universe: set | None = None,
    max_age_minutes: int = 90,
    max_nan_frac: float = 0.20,
    max_churn_frac: float = 0.30,
) -> dict:
    """Run all batch-level gates. Returns a dict with:
      - batch_suppression: None or reason (applies to all coins)
      - bad_features: list of columns with high NaN
      - warnings: list of non-blocking notes
    """
    result = {
        "batch_suppression": None,
        "bad_features": [],
        "warnings": [],
    }

    # 1. Data freshness — batch-blocking
    fr = check_data_freshness(latest_ts, max_age_minutes=max_age_minutes)
    if fr:
        result["batch_suppression"] = fr
        return result

    # 2. Feature NaN
    bad = check_feature_nan(latest_factors, max_nan_frac=max_nan_frac)
    if bad:
        result["bad_features"] = list(bad.keys())
        result["warnings"].append(
            "high_nan_features=" + ",".join(f"{k}:{v:.0%}" for k, v in bad.items())
        )

    # 3. Universe churn
    if previous_universe is not None:
        ch = check_universe_churn(current_universe, previous_universe, max_churn_frac=max_churn_frac)
        if ch:
            result["warnings"].append(ch)

    return result
