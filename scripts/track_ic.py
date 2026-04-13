#!/usr/bin/env python3
"""
Live IC tracker — appends per-period Spearman IC to output/ic_history.json.

Called after each fetch_and_rank run (12h after predictions, when outcomes are known).

Usage:
    cd /mnt/20t/main/gan_t/xsec_alpha
    python scripts/track_ic.py [--days 30]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import config
from utils.logger import logger
from data.features import (
    load_and_pivot,
    load_binance_pivot,
    compute_factors,
    crosssection_zscore,
    compute_residual_returns,
    build_top_liquidity_universe_index,
    filter_long_frame_by_universe,
)
from models.xgb_ranker import XSecRanker

IC_HISTORY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "ic_history.json")


def _load_history() -> list:
    if os.path.exists(IC_HISTORY_PATH):
        with open(IC_HISTORY_PATH, "r") as f:
            return json.load(f)
    return []


def _save_history(history: list) -> None:
    Path(IC_HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(IC_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def _check_alerts(history: list) -> None:
    """Check consecutive low-IC readings and print warnings/alerts."""
    if len(history) < 2:
        return

    # Check last 3 for ALERT (position freeze)
    if len(history) >= 3:
        last3 = [h["ic"] for h in history[-3:]]
        if all(ic < 0.03 for ic in last3):
            print(f"ALERT: position freeze recommended — IC < 0.03 for 3 consecutive readings: {last3}")
            return

    # Check last 2 for WARNING
    last2 = [h["ic"] for h in history[-2:]]
    if all(ic < 0.05 for ic in last2):
        print(f"WARNING: IC < 0.05 for 2 consecutive readings: {last2}")


def main():
    parser = argparse.ArgumentParser(description="Track live IC over time")
    parser.add_argument("--days", type=int, default=30, help="Days of history to load (default 30)")
    args = parser.parse_args()

    # 1. Load model
    model_path = config.Model.MODEL_PATH
    abs_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
    if not os.path.exists(abs_model_path):
        logger.error(f"Model not found: {abs_model_path}")
        sys.exit(1)

    model = XSecRanker.load(abs_model_path)
    logger.info(f"Loaded model from {abs_model_path}")

    # 2. Load recent data
    logger.info(f"Loading {args.days} days of data...")
    closes, opens, highs, lows, volumes = load_and_pivot(days=args.days)

    warmup = config.Data.MIN_ROWS_PER_COIN
    closes  = closes.iloc[warmup:]
    opens   = opens.iloc[warmup:]
    highs   = highs.iloc[warmup:]
    lows    = lows.iloc[warmup:]
    volumes = volumes.iloc[warmup:]
    logger.info(f"After warmup drop: {closes.shape}")

    # 3. Compute factors
    binance_closes = load_binance_pivot(closes.columns.tolist(), days=args.days)
    if not binance_closes.empty:
        binance_closes = binance_closes.iloc[warmup:]

    factor_df = compute_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
    factor_df = crosssection_zscore(factor_df)
    top_n = getattr(config.Data, "LIQUIDITY_TOP_N", 0)
    selected_index, coverage = build_top_liquidity_universe_index(closes, volumes, top_n=top_n)
    factor_df = filter_long_frame_by_universe(factor_df, selected_index)
    logger.info(
        "Active universe: top %s by %sh traded value (avg selected=%.1f, min=%s, max=%s)",
        top_n,
        getattr(config.Data, "LIQUIDITY_LOOKBACK_HOURS", 24),
        coverage["avg_selected"],
        coverage["min_selected"],
        coverage["max_selected"],
    )

    # 4. Compute residual returns
    horizon = config.Data.PREDICT_HORIZON
    residuals_wide = compute_residual_returns(closes, horizon=horizon, beta_window=config.Data.BETA_ROLLING_WINDOW)

    # 5. Find the most recent complete period (has both factors and realized returns)
    # residuals_wide has NaN for the last `horizon` rows, so the last valid timestamp is
    # the most recent row where residuals are not all-NaN
    valid_mask = residuals_wide.notna().any(axis=1)
    if not valid_mask.any():
        logger.error("No valid residual return data — cannot compute IC")
        sys.exit(1)

    last_valid_ts = residuals_wide.index[valid_mask][-1]
    logger.info(f"Most recent complete period: {last_valid_ts}")

    # 6. Get model predictions and actual residuals for that timestamp
    # Factor row for that timestamp
    ts_factors = factor_df.loc[last_valid_ts]  # DataFrame: index=market, columns=factor_cols
    ts_residuals = residuals_wide.loc[last_valid_ts]  # Series: index=market

    # Drop coins with NaN in either factors or residuals
    valid_coins = ts_factors.dropna().index.intersection(ts_residuals.dropna().index)

    if len(valid_coins) < 5:
        logger.error(f"Only {len(valid_coins)} valid coins at {last_valid_ts} — need at least 5")
        sys.exit(1)

    X = ts_factors.loc[valid_coins]
    actual = ts_residuals.loc[valid_coins]

    predicted = model.predict(X)

    # 7. Compute Spearman IC
    ic, pvalue = spearmanr(predicted, actual.values)

    if np.isnan(ic):
        logger.error("IC is NaN — likely zero variance in predictions or actuals")
        sys.exit(1)

    n_coins = len(valid_coins)
    ts_str = last_valid_ts.isoformat() if hasattr(last_valid_ts, 'isoformat') else str(last_valid_ts)

    logger.info(f"IC = {ic:.4f} | n_coins = {n_coins} | timestamp = {ts_str} | p = {pvalue:.4f}")

    # 8. Append to history
    history = _load_history()

    # Avoid duplicating the same timestamp
    existing_ts = {h["timestamp"] for h in history}
    if ts_str in existing_ts:
        logger.info(f"Timestamp {ts_str} already in history — skipping append")
    else:
        entry = {
            "timestamp": ts_str,
            "ic": round(float(ic), 6),
            "n_coins": int(n_coins),
        }
        history.append(entry)
        _save_history(history)
        logger.info(f"Appended to {IC_HISTORY_PATH} (total {len(history)} entries)")

    # 9. Check alerts
    _check_alerts(history)

    # Print summary
    print(f"IC={ic:+.4f}  n={n_coins}  ts={ts_str}")


if __name__ == "__main__":
    main()
