#!/usr/bin/env python3
"""
Main entry point for xsec_alpha (systemd / cron).

Steps:
  1. Collect fresh OHLCV data for all KRW markets
  2. Compute cross-sectional factors
  3. Rank coins → Long top 20 / Short bottom 20
  4. Save recommendations CSV

Usage:
    cd /mnt/20t/main/gan_t/xsec_alpha
    python scripts/fetch_and_rank.py [--collect] [--dry-run]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

from config import config
from utils.bitget import filter_markets_to_bitget, load_bitget_usdt_perp_map, market_to_bitget_symbol
from utils.eval_metrics import select_positions_with_buffer
from utils.logger import logger
from utils.run_lock import run_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="Run data collection before ranking")
    parser.add_argument("--dry-run", action="store_true", help="Print recommendations without saving")
    parser.add_argument("--no-model", action="store_true", help="Force equal-weight fallback (ignore XGBoost model)")
    parser.add_argument("--no-telegram", action="store_true", help="Suppress Telegram notification")
    args = parser.parse_args()

    with run_lock("fetch_and_rank"):
        _run(args)


def _compute_score(latest_factors: "pd.DataFrame", args) -> "pd.Series":
    """Return composite score Series (index=market).

    Priority:
      1. XGBoost model (XSecRanker) if model file exists and --no-model not set
      2. Equal-weight mean of all factors (fallback)
    """
    if args.no_model:
        logger.info("--no-model flag set: using equal-weight fallback")
        return latest_factors.mean(axis=1)

    try:
        model_path = getattr(getattr(config, "Model", None), "MODEL_PATH", "models/xsec_xgb.pkl")
        abs_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)

        if not os.path.exists(abs_model_path):
            logger.warning(f"Model not found at {abs_model_path}: using equal-weight fallback")
            return latest_factors.mean(axis=1)

        from models.xgb_ranker import XSecRanker
        model = XSecRanker.load(abs_model_path)
        scores_array = model.predict(latest_factors)
        score = pd.Series(scores_array, index=latest_factors.index)
        logger.info(f"XGBoost model loaded from {abs_model_path}: {len(score)} coins scored")
        return score

    except Exception as e:
        logger.warning(f"Model load/predict failed ({e}): using equal-weight fallback")
        return latest_factors.mean(axis=1)


def _run(args):
    logger.info("=== fetch_and_rank START ===")

    # 1. Collect data if requested
    if args.collect:
        logger.info("Collecting fresh data...")
        from data.collector import run_all
        run_all(days=120)

    # 2. Load and compute
    from data.features import (
        load_and_pivot,
        load_binance_pivot,
        compute_factors,
        crosssection_zscore,
        build_top_liquidity_universe_index,
        filter_long_frame_by_universe,
    )

    closes, opens, highs, lows, volumes = load_and_pivot(days=30)
    warmup = config.Data.MIN_ROWS_PER_COIN
    closes  = closes.iloc[warmup:]
    opens   = opens.iloc[warmup:]
    highs   = highs.iloc[warmup:]
    lows    = lows.iloc[warmup:]
    volumes = volumes.iloc[warmup:]

    binance_closes = load_binance_pivot(closes.columns.tolist(), days=30)
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

    # 3. Use latest timestamp
    latest_ts = factor_df.index.get_level_values("timestamp").max()
    latest_factors = factor_df.xs(latest_ts, level="timestamp")
    # Binance factors may be NaN if binance data lags — fill with 0 (neutral signal)
    latest_factors = latest_factors.fillna(0)
    # Drop coins with all zeros (truly no data)
    latest_factors = latest_factors[latest_factors.any(axis=1)]

    # Composite score: XGBoost model if available, else equal-weight fallback
    score = _compute_score(latest_factors, args)
    score_sorted = score.sort_values(ascending=False)

    execution_mode = getattr(config.Portfolio, "LIVE_EXECUTION_MODE", "short_only")
    watch_long_n = getattr(config.Portfolio, "LIVE_WATCH_LONG_N", config.Portfolio.LONG_N)
    exec_short_n = getattr(config.Portfolio, "LIVE_EXEC_SHORT_N", config.Portfolio.SHORT_N)
    require_bitget = getattr(config.Portfolio, "LIVE_REQUIRE_BITGET_TRADABLE", False)

    contract_map = {}
    tradable_score_sorted = score_sorted
    if require_bitget:
        contract_map = load_bitget_usdt_perp_map()
        tradable_markets = filter_markets_to_bitget(score_sorted.index.tolist(), contract_map)
        tradable_score_sorted = score_sorted.loc[tradable_markets]
        logger.info(
            "Tradable Bitget universe: %s/%s scored coins",
            len(tradable_score_sorted),
            len(score_sorted),
        )

    if execution_mode == "short_only":
        long_n = watch_long_n
        short_n = exec_short_n
        long_side_label = "WATCH_LONG"
    else:
        long_n = config.Portfolio.LONG_N
        short_n = config.Portfolio.SHORT_N
        long_side_label = "LONG"

        # Calendar filter: suppress LONG on Fri/Sat/Sun KST
        now_kst = datetime.now(timezone.utc) + timedelta(hours=9)
        dow_kst = now_kst.weekday()  # 0=Mon … 6=Sun
        is_bullish_day = dow_kst < 4  # Mon/Tue/Wed/Thu OK, Fri/Sat/Sun suppress longs
        if not is_bullish_day:
            logger.warning(f"Calendar filter: {now_kst.strftime('%A')} KST — suppressing LONG signals (Fri/weekend)")
            long_n = 0

    # --- Rebalancing buffer: reduce turnover by keeping existing positions ---
    rebal_buffer = config.Portfolio.REBAL_BUFFER
    prev_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "latest.csv")
    prev_longs = set()
    prev_shorts = set()
    try:
        if os.path.exists(prev_path):
            prev_df = pd.read_csv(os.path.realpath(prev_path))
            side_col = prev_df["side"].astype(str).str.lower()
            prev_longs = set(prev_df[side_col.str.contains("long", na=False)]["market"])
            prev_shorts = set(prev_df[side_col.str.contains("short", na=False)]["market"])
    except Exception:
        pass

    selection_scores = tradable_score_sorted if require_bitget else score_sorted

    if prev_longs or prev_shorts:
        longs, shorts = select_positions_with_buffer(
            selection_scores,
            long_n=long_n,
            short_n=short_n,
            prev_longs=prev_longs,
            prev_shorts=prev_shorts,
            rebal_buffer=rebal_buffer,
        )
        turnover_l = len(set(longs.index) - prev_longs) / max(len(longs), 1)
        turnover_s = len(set(shorts.index) - prev_shorts) / max(len(shorts), 1)
        logger.info(f"Rebal buffer: turnover L={turnover_l:.0%} S={turnover_s:.0%} (buffer={rebal_buffer})")
    else:
        longs = selection_scores.head(long_n) if long_n > 0 else pd.Series(dtype=float)
        shorts = selection_scores.tail(short_n)

    # Guard: need minimum coins
    if len(longs) < 5 and long_n > 0:
        logger.warning(f"Too few long candidates: {len(longs)} → watch-only for longs")
    if len(shorts) < 5:
        logger.warning(f"Too few coins for ranking: shorts={len(shorts)} → watch-only")

    logger.info(f"As of {latest_ts}")
    if execution_mode == "short_only":
        logger.info(f"WATCH LONG ({len(longs)}): {longs.index.tolist()}")
        logger.info(f"SHORT EXEC ({len(shorts)}): {shorts.index.tolist()}")
    else:
        logger.info(f"LONG  ({len(longs)}): {longs.index.tolist()}")
        logger.info(f"SHORT ({len(shorts)}): {shorts.index.tolist()}")

    if not args.dry_run:
        _save_recommendations(
            latest_ts,
            longs,
            shorts,
            long_side_label=long_side_label,
            short_side_label="SHORT",
            contract_map=contract_map,
        )

    # Telegram notification with live prices + previous performance
    if not args.dry_run and not args.no_telegram:
        try:
            from utils.telegram import send_message, format_report, format_performance
            from data.collector import get_current_price
            import time as _time

            # --- Previous recommendations performance ---
            perf_msg = ""
            prev_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "latest.csv")
            try:
                if os.path.exists(prev_path):
                    prev_df = pd.read_csv(os.path.realpath(prev_path))
                    if "entry_price" in prev_df.columns:
                        prev_active = prev_df[prev_df["side"].isin(["LONG", "SHORT"])].copy()
                        if len(prev_active) > 0:
                            now_prices = {}
                            for mkt in prev_active["market"]:
                                p = get_current_price(mkt)
                                if p:
                                    now_prices[mkt] = p
                                _time.sleep(0.1)
                            perf_msg = format_performance(prev_active, now_prices)
                            logger.info(f"Previous performance: {len(now_prices)} coins checked")
            except Exception as e:
                logger.warning(f"Performance calc failed: {e}")

            # --- Current recommendations ---
            all_coins = list(longs.index) + list(shorts.index)
            prices = {}
            for mkt in all_coins:
                p = get_current_price(mkt)
                if p:
                    prices[mkt] = p
                _time.sleep(0.1)
            logger.info(f"Fetched {len(prices)}/{len(all_coins)} live prices")

            horizon = config.Data.PREDICT_HORIZON
            capital = getattr(config.Portfolio, "TOTAL_CAPITAL_KRW", 10_000_000)
            stop_loss = getattr(config.Portfolio, "STOP_LOSS_PCT", 3.0)
            msg = format_report(
                longs,
                shorts,
                latest_ts,
                prices=prices,
                horizon_h=horizon,
                all_scores=selection_scores,
                min_sigma=0.0,
                capital=capital,
                stop_loss_pct=stop_loss,
                long_header="WATCH LONG" if execution_mode == "short_only" else "LONG",
                long_watch_only=(execution_mode == "short_only"),
            )

            # Send performance first, then new recommendations
            if perf_msg:
                send_message(perf_msg)
            sent = send_message(msg)
            if sent:
                logger.info("Telegram report sent")
        except Exception as e:
            logger.warning(f"Telegram notification failed: {e}")

    logger.info("=== fetch_and_rank END ===")


def _save_recommendations(ts, longs, shorts, long_side_label="LONG", short_side_label="SHORT", contract_map=None):
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    ts_str = ts.strftime("%Y%m%dT%H%M") if hasattr(ts, "strftime") else str(ts)[:16].replace(" ", "T")
    csv_path = os.path.join(out_dir, f"recommendations_{ts_str}.csv")

    # Fetch entry prices for LONG/SHORT positions
    from data.collector import get_current_price
    import time as _time

    rows = []
    for market, score in longs.items():
        side = long_side_label
        entry_price = None
        entry_price = get_current_price(market)
        _time.sleep(0.1)
        rows.append({
            "market": market,
            "score": round(float(score), 4),
            "side": side,
            "entry_price": entry_price,
            "bitget_symbol": market_to_bitget_symbol(market, contract_map or {}),
            "actionable": side == "SHORT",
        })

    for market, score in shorts.items():
        entry_price = get_current_price(market)
        _time.sleep(0.1)
        rows.append({
            "market": market,
            "score": round(float(score), 4),
            "side": short_side_label,
            "entry_price": entry_price,
            "bitget_symbol": market_to_bitget_symbol(market, contract_map or {}),
            "actionable": True,
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    # Latest symlink
    latest_path = os.path.join(out_dir, "latest.csv")
    try:
        if os.path.lexists(latest_path):
            os.remove(latest_path)
        os.symlink(os.path.basename(csv_path), latest_path)
    except Exception as e:
        logger.warning(f"Could not update latest symlink: {e}")


if __name__ == "__main__":
    main()
