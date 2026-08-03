"""Build (X, y) matrix for XGBoost training with temporal holdout split."""
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import config
from utils.logger import logger
from data.features import (
    load_and_pivot,
    load_binance_pivot,
    compute_factors,
    compute_long_factors,
    compute_unified_factors,
    compute_forward_returns,
    compute_residual_returns,
    compute_btc_regime,
    crosssection_zscore,
    CALENDAR_COLS,
    LONG_CALENDAR_COLS,
    UNIFIED_CALENDAR_COLS,
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


def _prepare_training_frame(
    factor_df: pd.DataFrame,
    target_long: pd.Series,
    factor_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Apply live's neutral NaN fill and all-zero row filter to training data."""
    combined = factor_df.join(target_long, how="inner")
    combined = combined.dropna(subset=[target_col]).copy()
    combined.loc[:, factor_cols] = combined.loc[:, factor_cols].fillna(0.0)
    combined = combined[combined.loc[:, factor_cols].any(axis=1)]
    return combined


def _split_timestamps_with_horizon_purge(
    all_timestamps: pd.DatetimeIndex,
    holdout_ratio: float,
    horizon: int,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp], pd.Timestamp]:
    """Split chronologically and purge train labels that reach the holdout."""
    timestamps = pd.DatetimeIndex(all_timestamps).unique().sort_values()
    if len(timestamps) == 0:
        raise ValueError("Cannot split an empty timestamp index")

    n_holdout = max(1, int(len(timestamps) * holdout_ratio))
    n_train = len(timestamps) - n_holdout
    holdout_timestamps = timestamps[n_train:]
    split_ts = holdout_timestamps[0]

    candidate_train = timestamps[:n_train]
    label_horizon = pd.Timedelta(hours=horizon)
    train_timestamps = candidate_train[candidate_train + label_horizon < split_ts]

    return train_timestamps.tolist(), holdout_timestamps.tolist(), split_ts


def build_dataset(
    days: int = 90,
    holdout_ratio: float = _HOLDOUT_RATIO_CONFIG,
    horizon: int | None = None,
    side: str = "short",
    target: str = "residual",
    regime_filter: str = "default",
    include_macro_globals: bool = True,
) -> dict:
    """
    Load data, compute factors and target returns, split temporally.

    Parameters
    ----------
    target : str
        "residual" (default)  → beta-adjusted return (coin_fwd − β · BTC_fwd)
        "absolute"            → raw coin forward return (what the user trades on)

    "absolute" is preferred for production predictions because it directly
    matches what the user sees as "% move in the next H hours", without the
    residual→absolute translation gap that compressed pred_std to ~0.003.
    """
    # ------------------------------------------------------------------
    # 1. Load OHLCV and compute features
    # ------------------------------------------------------------------
    closes, opens, highs, lows, volumes = load_and_pivot(days=days)

    if horizon is not None:
        pass  # explicit override
    elif side == "long" and hasattr(config, "LongModel"):
        horizon = getattr(config.LongModel, "PREDICT_HORIZON", config.Data.PREDICT_HORIZON)
    else:
        horizon = config.Data.PREDICT_HORIZON
    beta_window = config.Data.BETA_ROLLING_WINDOW   # 168

    binance_closes = load_binance_pivot(closes.columns.tolist(), days=days)
    if side == "unified":
        factor_df = compute_unified_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
        logger.info("Using UNIFIED factor library (F1: same features for 6h and 12h)")
    elif side == "long":
        factor_df = compute_long_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes,
                                          include_macro_globals=include_macro_globals)
        logger.info(f"Using LONG-specialist factors (macro_globals={include_macro_globals})")
    else:
        factor_df = compute_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
    residuals  = compute_residual_returns(closes, horizon=horizon, beta_window=beta_window)
    fwd_returns = compute_forward_returns(closes, horizon=horizon)

    # Pick target series based on mode
    if target == "absolute":
        target_wide = fwd_returns
        target_col = "fwd_return"
        logger.info(f"Target = absolute forward return ({horizon}h coin_fwd)")
    else:
        target_wide = residuals
        target_col = "residual_return"
        logger.info(f"Target = residual return ({horizon}h coin_fwd − β · BTC_fwd)")

    # ------------------------------------------------------------------
    # 2. Normalise factors (cross-sectional z-score per timestamp)
    # ------------------------------------------------------------------
    factor_cols  = factor_df.columns.tolist()
    if side == "unified":
        cal_cols = UNIFIED_CALENDAR_COLS
    elif side == "long":
        cal_cols = LONG_CALENDAR_COLS
    else:
        cal_cols = CALENDAR_COLS
    zscore_cols  = [c for c in factor_cols if c not in cal_cols]
    factor_df    = crosssection_zscore(factor_df, cols=zscore_cols)

    # ------------------------------------------------------------------
    # 3. Stack residuals to long format (timestamp, market)
    # ------------------------------------------------------------------
    target_long = target_wide.stack(future_stack=True)
    target_long.index.names = ["timestamp", "market"]
    target_long.name = target_col

    top_n = getattr(config.Data, "LIQUIDITY_TOP_N", 0)
    selected_index, coverage = build_top_liquidity_universe_index(closes, volumes, top_n=top_n)
    factor_df = filter_long_frame_by_universe(factor_df, selected_index)
    target_long = filter_long_series_by_universe(target_long, selected_index)
    logger.info(
        "Active universe: top %s by %sh traded value (avg selected=%.1f, min=%s, max=%s)",
        top_n,
        getattr(config.Data, "LIQUIDITY_LOOKBACK_HOURS", 24),
        coverage["avg_selected"],
        coverage["min_selected"],
        coverage["max_selected"],
    )

    # ------------------------------------------------------------------
    # 4. Join X and y; retain feature-NaN rows using live's neutral fill policy
    # ------------------------------------------------------------------
    combined = _prepare_training_frame(
        factor_df,
        target_long,
        factor_cols,
        target_col,
    )

    # ------------------------------------------------------------------
    # 4b. Regime filter for long side: keep only bull timestamps
    #
    # regime_filter:
    #   "default" → bull-only for long side (legacy, creates baseline drift)
    #   "all"     → no regime filter, keep all timestamps (unifies baseline
    #                with short model → resolves 90% direction disagreement)
    # ------------------------------------------------------------------
    if side == "unified":
        # Unified side: all-regime training, no filter
        pass
    elif side == "long" and regime_filter == "all":
        logger.info("Regime filter DISABLED (regime_filter='all') — training on all regimes")
    elif side == "long":
        btc_regime = compute_btc_regime(closes)
        bull_timestamps = btc_regime.index[btc_regime["regime_bull"] == 1.0]
        pre_regime = len(combined)
        combined = combined[combined.index.get_level_values("timestamp").isin(bull_timestamps)]
        n_filtered = pre_regime - len(combined)
        bull_pct = len(combined) / max(pre_regime, 1) * 100
        logger.info(
            f"Regime filter (long): kept {len(combined)} rows ({bull_pct:.1f}%), "
            f"dropped {n_filtered} bear/sideways rows"
        )
        if combined.empty:
            raise ValueError(
                "Regime filter removed all rows — no bull timestamps in data. "
                "Try increasing --days or check BTC regime thresholds."
            )

    # ------------------------------------------------------------------
    # 5. Drop extreme outliers: |y| > 0.5 (data error)
    # ------------------------------------------------------------------
    pre_len = len(combined)
    combined = combined[combined[target_col].abs() <= OUTLIER_THRESHOLD]
    n_dropped = pre_len - len(combined)
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} rows with |{target_col}| > {OUTLIER_THRESHOLD}")

    # ------------------------------------------------------------------
    # 6. Temporal split — NEVER shuffle, NEVER mix
    # ------------------------------------------------------------------
    all_timestamps = combined.index.get_level_values("timestamp").unique().sort_values()
    n_ts = len(all_timestamps)
    train_timestamps, holdout_timestamps, split_ts = (
        _split_timestamps_with_horizon_purge(
            all_timestamps,
            holdout_ratio=holdout_ratio,
            horizon=horizon,
        )
    )

    row_timestamps = combined.index.get_level_values("timestamp")
    train_mask = row_timestamps.isin(train_timestamps)
    holdout_mask = row_timestamps.isin(holdout_timestamps)

    train_data   = combined[train_mask]
    holdout_data = combined[holdout_mask]

    X_train   = train_data[factor_cols]
    y_train   = train_data[target_col]
    X_holdout = holdout_data[factor_cols]
    y_holdout = holdout_data[target_col]

    # ------------------------------------------------------------------
    # 7. Log sizes
    # ------------------------------------------------------------------
    logger.info(
        f"Train: {len(X_train)} rows, "
        f"Holdout: {len(X_holdout)} rows, "
        f"Split at: {split_ts}, "
        f"Purged: {n_ts - len(train_timestamps) - len(holdout_timestamps)} timestamps"
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
        "target":            target,
        "target_col":        target_col,
    }
