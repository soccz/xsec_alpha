"""Build (X, y) matrix for XGBoost training with temporal holdout split."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import config
from utils.logger import logger
from data.features import (
    load_and_pivot,
    load_binance_pivot,
    compute_factors,
    compute_forward_returns,
    compute_residual_returns,
    crosssection_zscore,
    CALENDAR_COLS,
    build_top_liquidity_universe_index,
    filter_long_frame_by_universe,
    filter_long_series_by_universe,
)

# Optional holdout ratio from config (may not exist)
_HOLDOUT_RATIO_DEFAULT = 0.2
try:
    _HOLDOUT_RATIO_CONFIG = config.Model.HOLDOUT_RATIO
except AttributeError:
    _HOLDOUT_RATIO_CONFIG = _HOLDOUT_RATIO_DEFAULT

OUTLIER_THRESHOLD = 0.5  # |residual| > 0.5 in target horizon = data error


def build_dataset(
    days: int = 90,
    holdout_ratio: float = _HOLDOUT_RATIO_CONFIG,
    horizon: int | None = None,
) -> dict:
    """
    Load data, compute factors and residual returns, split temporally.

    Returns dict with keys:
        X_train:           pd.DataFrame, index=(timestamp, market), columns=factor_names
        y_train:           pd.Series,    index=(timestamp, market), values=residual_return
        X_holdout:         pd.DataFrame  (same structure)
        y_holdout:         pd.Series     (same structure)
        train_timestamps:  list of timestamps
        holdout_timestamps: list of timestamps
        factor_cols:       list of factor column names
        split_ts:          the timestamp where train ends / holdout begins
    """
    # ------------------------------------------------------------------
    # 1. Load OHLCV and compute features
    # ------------------------------------------------------------------
    closes, opens, highs, lows, volumes = load_and_pivot(days=days)

    horizon     = horizon if horizon is not None else config.Data.PREDICT_HORIZON
    beta_window = config.Data.BETA_ROLLING_WINDOW   # 168

    binance_closes = load_binance_pivot(closes.columns.tolist(), days=days)
    factor_df  = compute_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
    residuals  = compute_residual_returns(closes, horizon=horizon, beta_window=beta_window)
    fwd_returns = compute_forward_returns(closes, horizon=horizon)

    # ------------------------------------------------------------------
    # 2. Normalise factors (cross-sectional z-score per timestamp)
    # ------------------------------------------------------------------
    factor_cols  = factor_df.columns.tolist()
    zscore_cols  = [c for c in factor_cols if c not in CALENDAR_COLS]
    factor_df    = crosssection_zscore(factor_df, cols=zscore_cols)

    # ------------------------------------------------------------------
    # 3. Stack residuals to long format (timestamp, market)
    # ------------------------------------------------------------------
    residuals_long = residuals.stack(future_stack=True)
    residuals_long.index.names = ["timestamp", "market"]
    residuals_long.name = "residual_return"

    top_n = getattr(config.Data, "LIQUIDITY_TOP_N", 0)
    selected_index, coverage = build_top_liquidity_universe_index(closes, volumes, top_n=top_n)
    factor_df = filter_long_frame_by_universe(factor_df, selected_index)
    residuals_long = filter_long_series_by_universe(residuals_long, selected_index)
    logger.info(
        "Active universe: top %s by %sh traded value (avg selected=%.1f, min=%s, max=%s)",
        top_n,
        getattr(config.Data, "LIQUIDITY_LOOKBACK_HOURS", 24),
        coverage["avg_selected"],
        coverage["min_selected"],
        coverage["max_selected"],
    )

    # ------------------------------------------------------------------
    # 4. Join X and y; drop rows where either is NaN
    # ------------------------------------------------------------------
    import pandas as pd

    combined = factor_df.join(residuals_long, how="inner").dropna()

    # ------------------------------------------------------------------
    # 5. Drop extreme outliers: |y| > 0.5 (data error)
    # ------------------------------------------------------------------
    pre_len = len(combined)
    combined = combined[combined["residual_return"].abs() <= OUTLIER_THRESHOLD]
    n_dropped = pre_len - len(combined)
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} rows with |residual_return| > {OUTLIER_THRESHOLD}")

    # ------------------------------------------------------------------
    # 6. Temporal split — NEVER shuffle, NEVER mix
    # ------------------------------------------------------------------
    all_timestamps = combined.index.get_level_values("timestamp").unique().sort_values()
    n_ts           = len(all_timestamps)
    n_holdout      = max(1, int(n_ts * holdout_ratio))
    n_train        = n_ts - n_holdout

    train_timestamps   = all_timestamps[:n_train].tolist()
    holdout_timestamps = all_timestamps[n_train:].tolist()
    split_ts           = holdout_timestamps[0]

    train_mask   = combined.index.get_level_values("timestamp").isin(train_timestamps)
    holdout_mask = ~train_mask

    train_data   = combined[train_mask]
    holdout_data = combined[holdout_mask]

    X_train   = train_data[factor_cols]
    y_train   = train_data["residual_return"]
    X_holdout = holdout_data[factor_cols]
    y_holdout = holdout_data["residual_return"]

    # ------------------------------------------------------------------
    # 7. Log sizes
    # ------------------------------------------------------------------
    logger.info(
        f"Train: {len(X_train)} rows, "
        f"Holdout: {len(X_holdout)} rows, "
        f"Split at: {split_ts}"
    )

    return {
        "X_train":           X_train,
        "y_train":           y_train,
        "X_holdout":         X_holdout,
        "y_holdout":         y_holdout,
        "fwd_returns":       fwd_returns,
        "closes":            closes,
        "opens":             opens,
        "train_timestamps":  train_timestamps,
        "holdout_timestamps": holdout_timestamps,
        "factor_cols":       factor_cols,
        "split_ts":          split_ts,
        "horizon":           horizon,
    }
