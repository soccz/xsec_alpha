#!/usr/bin/env python3
"""
Live IC tracker — appends per-period Spearman IC to output/ic_history*.json.

Called before each fetch_and_rank run. Default mode tracks both:
  - short: 6h horizon
  - long: 12h horizon

Both sides use the production absolute-return contract: exact live anchors,
next-bar-open execution, and the same neutral feature imputation as inference.

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
    compute_unified_factors,
    UNIFIED_CALENDAR_COLS,
    crosssection_zscore,
    compute_btc_regime,
    compute_forward_returns,
    build_top_liquidity_universe_index,
    filter_long_frame_by_universe,
)
from models.xgb_ranker import XSecRanker
from utils.run_lock import stable_data_read_lock

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
SHORT_IC_HISTORY_PATH = os.path.join(OUTPUT_DIR, "ic_history.json")
LONG_IC_HISTORY_PATH = os.path.join(OUTPUT_DIR, "ic_history_long.json")
IC_CONTRACT_VERSION = "absolute_open_lag1_anchors_v1"
_DATA_ACCESS_LOCK_WAIT_SEC = 480.0


def _anchor_hours_for_side(side: str) -> tuple[int, ...]:
    return (11, 23) if side == "long" else (5, 11, 17, 23)


def _latest_complete_anchor(
    returns: pd.DataFrame,
    side: str,
    min_coins: int = 5,
) -> pd.Timestamp | None:
    """Return the latest exact live anchor with a fully observable return."""
    valid_counts = returns.notna().sum(axis=1)
    anchors = set(_anchor_hours_for_side(side))
    candidates = [
        ts
        for ts, count in valid_counts.items()
        if int(count) >= int(min_coins) and pd.Timestamp(ts).hour in anchors
    ]
    return pd.Timestamp(candidates[-1]) if candidates else None


def _long_regime_eligible(regime_row) -> bool:
    if regime_row is None:
        return False
    try:
        btc_7d = float(regime_row.get("btc_ret_7d"))
        btc_30d = float(regime_row.get("btc_ret_30d"))
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(
        np.isfinite(btc_7d)
        and np.isfinite(btc_30d)
        and btc_7d > float(getattr(config.LongModel, "BTC_7D_RETURN_GATE", 0.0))
        and btc_30d > float(getattr(config.LongModel, "BTC_30D_RETURN_FLOOR", -0.10))
    )


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
    aligned = [
        row for row in history
        if row.get("contract_version") == IC_CONTRACT_VERSION
    ]
    if aligned:
        history = aligned
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
    regime_closes = closes

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

    # F1 unified (2026-04-25): both sides use same 10-feature factor library
    factor_df = compute_unified_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
    cal_cols = UNIFIED_CALENDAR_COLS
    logger.info("[%s] Using UNIFIED factor library (F1)", side)

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
    execution_lag_bars = 1
    absolute_returns = compute_forward_returns(
        opens,
        horizon=horizon,
        execution_lag=execution_lag_bars,
    )
    last_valid_ts = _latest_complete_anchor(absolute_returns, side=side)
    if last_valid_ts is None:
        logger.error("[%s] No complete production-anchor return data — cannot compute IC", side)
        return
    logger.info("[%s] Most recent complete period: %s", side, last_valid_ts)

    try:
        ts_factors = factor_df.loc[last_valid_ts]
    except KeyError:
        logger.error("[%s] No factor snapshot at %s", side, last_valid_ts)
        return
    ts_actual = absolute_returns.loc[last_valid_ts]

    ts_factors = ts_factors.fillna(0.0)
    ts_factors = ts_factors[ts_factors.any(axis=1)]
    valid_coins = ts_factors.index.intersection(ts_actual.dropna().index)
    if len(valid_coins) < 5:
        logger.error("[%s] Only %s valid coins at %s — need at least 5", side, len(valid_coins), last_valid_ts)
        return

    X = ts_factors.loc[valid_coins]
    actual = ts_actual.loc[valid_coins]
    predicted = model.predict(X)
    ic, pvalue = spearmanr(predicted, actual.values)
    if np.isnan(ic):
        logger.error("[%s] IC is NaN — likely zero variance in predictions or actuals", side)
        return

    n_coins = len(valid_coins)
    ts_str = last_valid_ts.isoformat() if hasattr(last_valid_ts, "isoformat") else str(last_valid_ts)
    logger.info("[%s] IC = %.4f | n_coins = %s | timestamp = %s | p = %.4f", side, ic, n_coins, ts_str, pvalue)

    regime_eligible = None
    btc_ret_7d = None
    btc_ret_30d = None
    if side == "long":
        btc_regime = compute_btc_regime(regime_closes)
        regime_row = btc_regime.loc[last_valid_ts] if last_valid_ts in btc_regime.index else None
        regime_eligible = _long_regime_eligible(regime_row)
        if regime_row is not None:
            raw_7d = pd.to_numeric(regime_row.get("btc_ret_7d"), errors="coerce")
            raw_30d = pd.to_numeric(regime_row.get("btc_ret_30d"), errors="coerce")
            btc_ret_7d = float(raw_7d) if pd.notna(raw_7d) else None
            btc_ret_30d = float(raw_30d) if pd.notna(raw_30d) else None

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
            "target": "absolute",
            "price_source": "open",
            "execution_lag_bars": execution_lag_bars,
            "anchor_hours_utc": list(_anchor_hours_for_side(side)),
            "regime_eligible": regime_eligible,
            "btc_ret_7d": btc_ret_7d,
            "btc_ret_30d": btc_ret_30d,
            "contract_version": IC_CONTRACT_VERSION,
        })
        _save_history(history_path, history)
        logger.info("[%s] Appended to %s (total %s entries)", side, history_path, len(history))

    _check_alerts(history, side)
    print(f"{side.upper()} IC={ic:+.4f}  n={n_coins}  h={horizon}  ts={ts_str}")


def main():
    with stable_data_read_lock(timeout_sec=_DATA_ACCESS_LOCK_WAIT_SEC):
        _main_locked()


def _main_locked():
    parser = argparse.ArgumentParser(description="Track live IC over time")
    parser.add_argument("--days", type=int, default=45, help="Days of history to load (default 45; preserves 30d regime context after warmup)")
    parser.add_argument("--side", type=str, default="both", choices=["short", "long", "both"], help="Track short, long, or both (default)")
    args = parser.parse_args()
    sides = ["short", "long"] if args.side == "both" else [args.side]
    for side in sides:
        _track_side(side, args.days)


if __name__ == "__main__":
    main()
