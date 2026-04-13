from __future__ import annotations

import numpy as np
import pandas as pd


def compute_score_diagnostics(
    sorted_scores: pd.Series,
    long_n: int,
    short_n: int,
) -> dict[str, float]:
    """Per-timestamp score separation metrics for rank stability."""
    n = len(sorted_scores)
    score_std = float(sorted_scores.std()) if n > 1 else float("nan")
    top_mean = float(sorted_scores.head(long_n).mean()) if long_n > 0 else float("nan")
    bottom_mean = float(sorted_scores.tail(short_n).mean()) if short_n > 0 else float("nan")
    top_bottom_gap = top_mean - bottom_mean

    long_cut_gap = float("nan")
    if 0 < long_n < n:
        long_cut_gap = float(sorted_scores.iloc[long_n - 1] - sorted_scores.iloc[long_n])

    short_cut_gap = float("nan")
    short_start = n - short_n
    if 0 < short_n < n and short_start - 1 >= 0:
        short_cut_gap = float(sorted_scores.iloc[short_start - 1] - sorted_scores.iloc[short_start])

    return {
        "score_std": score_std,
        "top_bottom_gap": float(top_bottom_gap),
        "top_bottom_gap_z": float(top_bottom_gap / score_std) if score_std and not np.isnan(score_std) else float("nan"),
        "long_cut_gap": long_cut_gap,
        "long_cut_gap_z": float(long_cut_gap / score_std) if score_std and not np.isnan(long_cut_gap) else float("nan"),
        "short_cut_gap": short_cut_gap,
        "short_cut_gap_z": float(short_cut_gap / score_std) if score_std and not np.isnan(short_cut_gap) else float("nan"),
    }


def build_btc_context(
    closes: pd.DataFrame,
    return_lookback_hours: int = 24 * 7,
    vol_lookback_hours: int = 24 * 7,
    trend_lookback_hours: int = 24 * 30,
) -> pd.DataFrame:
    """BTC regime panel keyed by timestamp."""
    btc_col = "KRW-BTC"
    if btc_col not in closes.columns:
        raise ValueError("KRW-BTC not in closes — cannot compute BTC regime context")

    btc = closes[btc_col].copy()
    btc_ret_7d = btc.pct_change(return_lookback_hours)
    btc_ret_30d = btc.pct_change(trend_lookback_hours)
    btc_vol_7d = btc.pct_change().rolling(vol_lookback_hours).std()

    vol_median = btc_vol_7d.dropna().median()
    direction = np.where(btc_ret_7d >= 0, "bull", "bear")
    vol_bucket = np.where(btc_vol_7d >= vol_median, "highvol", "lowvol")
    regime = pd.Series(direction + "_" + vol_bucket, index=btc.index, name="regime")
    regime[btc_ret_7d.isna() | btc_vol_7d.isna()] = "unknown"

    return pd.DataFrame(
        {
            "btc_ret_7d": btc_ret_7d,
            "btc_ret_30d": btc_ret_30d,
            "btc_vol_7d": btc_vol_7d,
            "regime": regime,
        }
    )


def select_positions_with_buffer(
    score_sorted: pd.Series,
    long_n: int,
    short_n: int,
    prev_longs: set[str] | None = None,
    prev_shorts: set[str] | None = None,
    rebal_buffer: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """Mirror live rebalancing-buffer selection in research evaluators."""
    score_desc = score_sorted.sort_values(ascending=False)
    score_asc = score_sorted.sort_values(ascending=True)

    prev_longs = set(prev_longs or [])
    prev_shorts = set(prev_shorts or [])
    empty = pd.Series(dtype=float)

    if rebal_buffer <= 0 or not (prev_longs or prev_shorts):
        longs = score_desc.head(long_n) if long_n > 0 else empty
        shorts = score_asc.head(short_n) if short_n > 0 else empty
        return longs, shorts

    final_longs: list[str] = []
    if long_n > 0:
        extended_longs = list(score_desc.head(long_n + rebal_buffer).index)
        keep_longs = [m for m in extended_longs if m in prev_longs]
        new_long_candidates = [m for m in score_desc.head(long_n).index if m not in keep_longs]
        final_longs = (keep_longs + new_long_candidates)[:long_n]

    final_shorts: list[str] = []
    if short_n > 0:
        extended_shorts = list(score_asc.head(short_n + rebal_buffer).index)
        keep_shorts = [m for m in extended_shorts if m in prev_shorts and m not in final_longs]
        new_short_candidates = [
            m for m in score_asc.head(short_n).index
            if m not in keep_shorts and m not in final_longs
        ]
        final_shorts = (keep_shorts + new_short_candidates)[:short_n]

    longs = score_desc.loc[final_longs] if final_longs else empty
    shorts = score_asc.loc[final_shorts] if final_shorts else empty
    return longs, shorts
