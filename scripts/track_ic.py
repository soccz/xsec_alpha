#!/usr/bin/env python3
"""
Live IC tracker — appends per-period Spearman IC to output/ic_history*.json.

Called before each fetch_and_rank run. Default mode tracks both:
  - short: 6h horizon
  - long: 12h horizon, bull-regime only

Usage:
    cd /mnt/20t/main/gan_t/xsec_alpha
    python scripts/track_ic.py [--days 30] [--side short|long|both]
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
    compute_long_factors,
    crosssection_zscore,
    compute_btc_regime,
    compute_residual_returns,
    CALENDAR_COLS,
    LONG_CALENDAR_COLS,
    build_top_liquidity_universe_index,
    filter_long_frame_by_universe,
)
from models.xgb_ranker import XSecRanker

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
SHORT_IC_HISTORY_PATH = os.path.join(OUTPUT_DIR, "ic_history.json")
LONG_IC_HISTORY_PATH = os.path.join(OUTPUT_DIR, "ic_history_long.json")


def _history_path_for_side(side: str) -> str:
    return LONG_IC_HISTORY_PATH if side == "long" else SHORT_IC_HISTORY_PATH


def _load_history(path: str) -> list:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def _save_history(path: str, history: list) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def _check_alerts(history: list, side: str) -> None:
    """Check consecutive low-IC readings and print warnings/alerts."""
    if len(history) < 2:
        return

    # Check last 3 for ALERT (position freeze)
    if len(history) >= 3:
        last3 = [h["ic"] for h in history[-3:]]
        if all(ic < 0.03 for ic in last3):
            print(f"ALERT[{side}]: position freeze recommended — IC < 0.03 for 3 consecutive readings: {last3}")
            return

    # Check last 2 for WARNING
    last2 = [h["ic"] for h in history[-2:]]
    if all(ic < 0.05 for ic in last2):
        print(f"WARNING[{side}]: IC < 0.05 for 2 consecutive readings: {last2}")


def _track_side(side: str, days: int) -> None:
    model_path = (
        getattr(config.LongModel, "MODEL_PATH", "models/xsec_long.pkl")
        if side == "long"
        else config.Model.MODEL_PATH
    )
    abs_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
    if not os.path.exists(abs_model_path):
        logger.error("[%s] Model not found: %s", side, abs_model_path)
        return

    model = XSecRanker.load(abs_model_path)
    logger.info("[%s] Loaded model from %s", side, abs_model_path)

    logger.info("[%s] Loading %s days of data...", side, days)
    closes, opens, highs, lows, volumes = load_and_pivot(days=days)

    warmup = config.Data.MIN_ROWS_PER_COIN
    closes = closes.iloc[warmup:]
    opens = opens.iloc[warmup:]
    highs = highs.iloc[warmup:]
    lows = lows.iloc[warmup:]
    volumes = volumes.iloc[warmup:]
    logger.info("[%s] After warmup drop: %s", side, closes.shape)

    binance_closes = load_binance_pivot(closes.columns.tolist(), days=days)
    if not binance_closes.empty:
        binance_closes = binance_closes.iloc[warmup:]

    if side == "long":
        factor_df = compute_long_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
        cal_cols = LONG_CALENDAR_COLS
        logger.info("[%s] Using LONG-specialist factors", side)
    else:
        factor_df = compute_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
        cal_cols = CALENDAR_COLS

    zscore_cols = [c for c in factor_df.columns if c not in cal_cols]
    factor_df = crosssection_zscore(factor_df, cols=zscore_cols)

    top_n = getattr(config.Data, "LIQUIDITY_TOP_N", 0)
    selected_index, coverage = build_top_liquidity_universe_index(closes, volumes, top_n=top_n)
    factor_df = filter_long_frame_by_universe(factor_df, selected_index)
    logger.info(
        "[%s] Active universe: top %s by %sh traded value (avg selected=%.1f, min=%s, max=%s)",
        side,
        top_n,
        getattr(config.Data, "LIQUIDITY_LOOKBACK_HOURS", 24),
        coverage["avg_selected"],
        coverage["min_selected"],
        coverage["max_selected"],
    )

    horizon = (
        getattr(config.LongModel, "PREDICT_HORIZON", config.Data.PREDICT_HORIZON)
        if side == "long"
        else config.Data.PREDICT_HORIZON
    )
    residuals_wide = compute_residual_returns(closes, horizon=horizon, beta_window=config.Data.BETA_ROLLING_WINDOW)

    valid_mask = residuals_wide.notna().any(axis=1)
    if side == "long":
        btc_regime = compute_btc_regime(closes)
        bull_ts = btc_regime.index[btc_regime["regime_bull"] == 1.0]
        factor_df = factor_df[factor_df.index.get_level_values("timestamp").isin(bull_ts)]
        valid_mask = valid_mask & residuals_wide.index.isin(bull_ts)
        logger.info("[%s] Regime filter: %s bull timestamps", side, int(valid_mask.sum()))

    if not valid_mask.any():
        logger.error("[%s] No valid residual return data — cannot compute IC", side)
        return

    last_valid_ts = residuals_wide.index[valid_mask][-1]
    logger.info("[%s] Most recent complete period: %s", side, last_valid_ts)

    try:
        ts_factors = factor_df.loc[last_valid_ts]
    except KeyError:
        logger.error("[%s] No factor snapshot at %s", side, last_valid_ts)
        return
    ts_residuals = residuals_wide.loc[last_valid_ts]

    valid_coins = ts_factors.dropna().index.intersection(ts_residuals.dropna().index)
    if len(valid_coins) < 5:
        logger.error("[%s] Only %s valid coins at %s — need at least 5", side, len(valid_coins), last_valid_ts)
        return

    X = ts_factors.loc[valid_coins]
    actual = ts_residuals.loc[valid_coins]
    predicted = model.predict(X)
    ic, pvalue = spearmanr(predicted, actual.values)
    if np.isnan(ic):
        logger.error("[%s] IC is NaN — likely zero variance in predictions or actuals", side)
        return

    n_coins = len(valid_coins)
    ts_str = last_valid_ts.isoformat() if hasattr(last_valid_ts, "isoformat") else str(last_valid_ts)
    logger.info("[%s] IC = %.4f | n_coins = %s | timestamp = %s | p = %.4f", side, ic, n_coins, ts_str, pvalue)

    history_path = _history_path_for_side(side)
    history = _load_history(history_path)
    existing_ts = {h["timestamp"] for h in history}
    if ts_str in existing_ts:
        logger.info("[%s] Timestamp %s already in history — skipping append", side, ts_str)
    else:
        history.append({
            "timestamp": ts_str,
            "ic": round(float(ic), 6),
            "n_coins": int(n_coins),
            "side": side,
            "horizon_h": int(horizon),
        })
        _save_history(history_path, history)
        logger.info("[%s] Appended to %s (total %s entries)", side, history_path, len(history))

    _check_alerts(history, side)
    print(f"{side.upper()} IC={ic:+.4f}  n={n_coins}  h={horizon}  ts={ts_str}")


def main():
    parser = argparse.ArgumentParser(description="Track live IC over time")
    parser.add_argument("--days", type=int, default=30, help="Days of history to load (default 30)")
    parser.add_argument("--side", type=str, default="both", choices=["short", "long", "both"], help="Track short, long, or both (default)")
    args = parser.parse_args()
    sides = ["short", "long"] if args.side == "both" else [args.side]
    for side in sides:
        _track_side(side, args.days)


if __name__ == "__main__":
    main()
