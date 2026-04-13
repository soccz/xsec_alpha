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


def _load_previous_recommendations(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            return pd.read_csv(os.path.realpath(path))
    except Exception as e:
        logger.warning(f"Could not load previous recommendations ({e})")
    return pd.DataFrame()


def _is_rebalance_slot(ts, horizon_h: int, anchor_hour_utc: int) -> bool:
    if horizon_h <= 0:
        return True
    ts_hour = pd.Timestamp(ts).tz_convert("UTC").hour if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).hour
    return ((ts_hour - anchor_hour_utc) % horizon_h) == 0


def _next_rebalance_ts(ts, horizon_h: int, anchor_hour_utc: int | None = None):
    base = pd.Timestamp(ts)
    if horizon_h <= 0:
        return base
    if anchor_hour_utc is None:
        return base + pd.Timedelta(hours=horizon_h)
    probe = base
    for _ in range(1, horizon_h + 2):
        probe = probe + pd.Timedelta(hours=1)
        if _is_rebalance_slot(probe, horizon_h, anchor_hour_utc):
            return probe
    return base + pd.Timedelta(hours=horizon_h)


def _update_realized_ledger(closes: "pd.DataFrame", latest_ts) -> None:
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    ledger_path = os.path.join(out_dir, "recommendation_ledger.csv")
    os.makedirs(out_dir, exist_ok=True)
    ledger_cols = [
        "market",
        "side",
        "actionable",
        "entry_time",
        "exit_time_target",
        "exit_time_actual",
        "horizon_h",
        "entry_price",
        "exit_price",
        "realized_return",
        "refresh_reason",
        "source_file",
    ]

    key_cols = ["market", "side", "entry_time", "horizon_h"]
    if os.path.exists(ledger_path):
        try:
            ledger_df = pd.read_csv(ledger_path)
        except Exception as e:
            logger.warning(f"Could not read ledger ({e}) — rebuilding from scratch")
            ledger_df = pd.DataFrame()
    else:
        ledger_df = pd.DataFrame()

    existing_keys = set()
    if not ledger_df.empty and set(key_cols).issubset(ledger_df.columns):
        for row in ledger_df[key_cols].itertuples(index=False):
            existing_keys.add((str(row.market), str(row.side), str(row.entry_time), int(row.horizon_h)))

    latest_ts = pd.Timestamp(latest_ts)
    matured_rows = []

    for fname in sorted(os.listdir(out_dir)):
        if not (fname.startswith("recommendations_") and fname.endswith(".csv")):
            continue
        fpath = os.path.join(out_dir, fname)
        try:
            rec_df = pd.read_csv(fpath)
        except Exception:
            continue
        required = {"market", "side", "entry_price", "entry_time", "horizon_h"}
        if not required.issubset(rec_df.columns):
            continue

        for row in rec_df.to_dict(orient="records"):
            market = str(row.get("market", ""))
            side = str(row.get("side", ""))
            entry_price = pd.to_numeric(row.get("entry_price"), errors="coerce")
            horizon_h = pd.to_numeric(row.get("horizon_h"), errors="coerce")
            entry_time = pd.to_datetime(row.get("entry_time"), utc=True, errors="coerce")
            if not market or pd.isna(entry_price) or entry_price <= 0 or pd.isna(horizon_h) or pd.isna(entry_time):
                continue

            horizon_h = int(horizon_h)
            key = (market, side, entry_time.isoformat(), horizon_h)
            if key in existing_keys:
                continue

            target_exit_ts = entry_time + pd.Timedelta(hours=horizon_h)
            if target_exit_ts > latest_ts:
                continue
            if market not in closes.columns:
                continue

            eligible = closes.index[closes.index <= target_exit_ts]
            if len(eligible) == 0:
                continue
            exit_ts = eligible[-1]
            exit_price = closes.at[exit_ts, market]
            if pd.isna(exit_price):
                continue

            side_lower = side.lower()
            if "short" in side_lower:
                realized_return = (entry_price - exit_price) / entry_price
            else:
                realized_return = (exit_price - entry_price) / entry_price

            matured_rows.append({
                "market": market,
                "side": side,
                "actionable": bool(row.get("actionable", False)),
                "entry_time": entry_time.isoformat(),
                "exit_time_target": target_exit_ts.isoformat(),
                "exit_time_actual": pd.Timestamp(exit_ts).isoformat(),
                "horizon_h": horizon_h,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "realized_return": float(realized_return),
                "refresh_reason": row.get("refresh_reason", ""),
                "source_file": fname,
            })
            existing_keys.add(key)

    if not matured_rows:
        if not os.path.exists(ledger_path):
            pd.DataFrame(columns=ledger_cols).to_csv(ledger_path, index=False)
            logger.info("Ledger initialized with no matured positions yet -> %s", ledger_path)
            return
        logger.info("Ledger unchanged: no newly matured positions")
        return

    append_df = pd.DataFrame(matured_rows)
    ledger_df = pd.concat([ledger_df, append_df], ignore_index=True) if not ledger_df.empty else append_df
    ledger_df = ledger_df.sort_values(["entry_time", "market", "side"]).reset_index(drop=True)
    ledger_df.to_csv(ledger_path, index=False)

    long_mask = ~append_df["side"].astype(str).str.lower().str.contains("short", na=False)
    short_mask = append_df["side"].astype(str).str.lower().str.contains("short", na=False)
    logger.info(
        "Ledger updated: +%s matured positions (long/watch=%s, short=%s) -> %s",
        len(append_df),
        int(long_mask.sum()),
        int(short_mask.sum()),
        ledger_path,
    )


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
        compute_long_factors,
        compute_btc_regime,
        crosssection_zscore,
        build_top_liquidity_universe_index,
        filter_long_frame_by_universe,
        LONG_CALENDAR_COLS,
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
    _update_realized_ledger(closes, latest_ts)
    short_horizon_h = config.Data.PREDICT_HORIZON
    long_horizon_h = getattr(config.LongModel, "PREDICT_HORIZON", short_horizon_h)
    long_anchor_hour = getattr(config.LongModel, "REBALANCE_ANCHOR_HOUR_UTC", 11)
    long_rebalance_due = _is_rebalance_slot(latest_ts, long_horizon_h, long_anchor_hour)
    next_short_rebalance_ts = _next_rebalance_ts(latest_ts, short_horizon_h)
    next_long_rebalance_ts = _next_rebalance_ts(latest_ts, long_horizon_h, long_anchor_hour)
    latest_factors = factor_df.xs(latest_ts, level="timestamp")
    # Binance factors may be NaN if binance data lags — fill with 0 (neutral signal)
    latest_factors = latest_factors.fillna(0)
    # Drop coins with all zeros (truly no data)
    latest_factors = latest_factors[latest_factors.any(axis=1)]

    # Composite score (short model): XGBoost model if available, else equal-weight fallback
    score = _compute_score(latest_factors, args)
    score_sorted = score.sort_values(ascending=False)

    # --- Long model scoring (independent, regime-gated, execution-mode-gated) ---
    long_model_score_sorted = None
    long_model_active = False
    long_execution_mode = getattr(config.LongModel, "EXECUTION_MODE", "disabled")
    try:
        long_model_path = getattr(config.LongModel, "MODEL_PATH", "models/xsec_long.pkl")
        abs_long_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), long_model_path)

        if long_execution_mode == "disabled":
            logger.info("Long model DISABLED (LongModel.EXECUTION_MODE='disabled')")
        elif os.path.exists(abs_long_model_path) and not args.no_model:
            # Check BTC regime gate
            btc_regime = compute_btc_regime(closes)
            latest_regime = btc_regime.loc[latest_ts] if latest_ts in btc_regime.index else None

            if latest_regime is not None and latest_regime["regime_bull"] == 1.0:
                # Compute long factors for latest timestamp
                long_factor_df = compute_long_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
                long_cal_cols = LONG_CALENDAR_COLS
                long_zscore_cols = [c for c in long_factor_df.columns if c not in long_cal_cols]
                long_factor_df = crosssection_zscore(long_factor_df, cols=long_zscore_cols)
                long_factor_df = filter_long_frame_by_universe(long_factor_df, selected_index)
                latest_long_factors = long_factor_df.xs(latest_ts, level="timestamp")
                latest_long_factors = latest_long_factors.fillna(0)
                latest_long_factors = latest_long_factors[latest_long_factors.any(axis=1)]

                from models.xgb_ranker import XSecRanker
                long_model = XSecRanker.load(abs_long_model_path)
                long_scores = long_model.predict(latest_long_factors)
                long_model_score_sorted = pd.Series(long_scores, index=latest_long_factors.index).sort_values(ascending=False)
                long_model_active = True
                logger.info(
                    f"Long model ACTIVE (regime_bull=1, btc_7d={latest_regime['btc_ret_7d']:+.2%}, "
                    f"btc_30d={latest_regime['btc_ret_30d']:+.2%}): {len(long_model_score_sorted)} coins scored"
                )
            else:
                regime_info = ""
                if latest_regime is not None:
                    regime_info = f" (btc_7d={latest_regime['btc_ret_7d']:+.2%}, btc_30d={latest_regime['btc_ret_30d']:+.2%})"
                logger.info(f"Long model INACTIVE: BTC regime not bullish{regime_info}")
    except Exception as e:
        logger.warning(f"Long model scoring failed: {e}")

    execution_mode = getattr(config.Portfolio, "LIVE_EXECUTION_MODE", "short_only")
    watch_long_n = getattr(config.Portfolio, "LIVE_WATCH_LONG_N", config.Portfolio.LONG_N)
    exec_short_n = getattr(config.Portfolio, "LIVE_EXEC_SHORT_N", config.Portfolio.SHORT_N)
    require_bitget = getattr(config.Portfolio, "LIVE_REQUIRE_BITGET_TRADABLE", False)
    prev_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "latest.csv")
    prev_df = _load_previous_recommendations(prev_path)

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
    prev_longs = set()
    prev_shorts = set()
    prev_long_rows = pd.DataFrame()
    if not prev_df.empty and "side" in prev_df.columns and "market" in prev_df.columns:
        side_col = prev_df["side"].astype(str).str.lower()
        prev_long_rows = prev_df[side_col.str.contains("long", na=False)].copy()
        prev_longs = set(prev_long_rows["market"])
        prev_shorts = set(prev_df[side_col.str.contains("short", na=False)]["market"])

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

    # --- Override long picks with long-specialist model if active ---
    carry_longs = False
    long_refresh_reason = "watch_seed"
    if long_model_active and long_model_score_sorted is not None:
        long_model_n = getattr(config.LongModel, "LONG_N", 5)
        if require_bitget:
            tradable_long = filter_markets_to_bitget(long_model_score_sorted.index.tolist(), contract_map)
            long_model_tradable = long_model_score_sorted.loc[tradable_long]
        else:
            long_model_tradable = long_model_score_sorted

        if long_rebalance_due or prev_long_rows.empty:
            longs = long_model_tradable.head(long_model_n)
            carry_longs = False
            long_refresh_reason = "refresh_12h"
            logger.info(
                "Long model rebalance due: refreshing %s picks (h=%sh, anchor=%s UTC)",
                len(longs), long_horizon_h, long_anchor_hour,
            )
        else:
            carry_longs = True
            long_refresh_reason = "carry_12h"
            prev_long_rows = prev_long_rows.drop_duplicates(subset=["market"], keep="last")
            carry_markets = [m for m in prev_long_rows["market"].tolist() if m in long_model_tradable.index]
            if carry_markets:
                longs = prev_long_rows.set_index("market").loc[carry_markets, "score"]
                logger.info(
                    "Long model HOLD: carrying %s previous picks until next %sh rebalance",
                    len(longs), long_horizon_h,
                )
            else:
                longs = long_model_tradable.head(long_model_n)
                carry_longs = False
                long_refresh_reason = "refresh_12h"
                logger.info("Long model carry unavailable: seeding fresh picks")
    elif long_execution_mode == "disabled":
        long_refresh_reason = "disabled"
    elif not long_model_active:
        long_refresh_reason = "regime_off"

    # Only upgrade to actionable LONG if execution mode permits
    if long_model_active and long_execution_mode == "long_only":
        long_side_label = "LONG"
        if carry_longs:
            logger.info("Long model EXECUTE HOLD: carrying existing LONG basket")
        else:
            logger.info(f"Long model EXECUTE: {len(longs)} positions (EXECUTION_MODE=long_only)")
    elif long_model_active:
        long_side_label = "WATCH_LONG"
        mode_note = "HOLD" if carry_longs else "WATCH"
        logger.info(f"Long model {mode_note}: {len(longs)} picks (EXECUTION_MODE={long_execution_mode}, not yet approved)")

    # Guard: need minimum coins
    if len(longs) < 5 and long_n > 0:
        logger.warning(f"Too few long candidates: {len(longs)} → watch-only for longs")
    if len(shorts) < 5:
        logger.warning(f"Too few coins for ranking: shorts={len(shorts)} → watch-only")

    logger.info(f"As of {latest_ts}")
    logger.info(f"{long_side_label} ({len(longs)}): {longs.index.tolist()}")
    logger.info(f"SHORT {'EXEC' if execution_mode == 'short_only' else ''} ({len(shorts)}): {shorts.index.tolist()}")

    if not args.dry_run:
        _save_recommendations(
            latest_ts,
            longs,
            shorts,
            long_side_label=long_side_label,
            short_side_label="SHORT",
            contract_map=contract_map,
            long_model_active=long_model_active,
            long_horizon_h=long_horizon_h if long_model_active else short_horizon_h,
            short_horizon_h=short_horizon_h,
            carry_longs=carry_longs,
            prev_long_rows=prev_long_rows,
            long_refresh_reason=long_refresh_reason,
            next_long_rebalance_ts=next_long_rebalance_ts,
            next_short_rebalance_ts=next_short_rebalance_ts,
        )

    # Telegram notification with live prices + previous performance
    if not args.dry_run and not args.no_telegram:
        try:
            from utils.telegram import send_message, format_report, format_performance
            from data.collector import get_current_price
            import time as _time

            # --- Previous recommendations performance ---
            perf_msg = ""
            try:
                if not prev_df.empty and "entry_price" in prev_df.columns:
                    prev_active = prev_df[prev_df["side"].isin(["LONG", "WATCH_LONG", "SHORT"])].copy()
                    if len(prev_active) > 0:
                        if "entry_time" in prev_active.columns and "horizon_h" in prev_active.columns:
                            entry_ts = pd.to_datetime(prev_active["entry_time"], utc=True, errors="coerce")
                            horizon_vals = pd.to_numeric(prev_active["horizon_h"], errors="coerce")
                            age_hours = (pd.Timestamp(latest_ts) - entry_ts).dt.total_seconds() / 3600.0
                            prev_active = prev_active[age_hours >= horizon_vals]
                        if len(prev_active) > 0:
                            now_prices = {}
                            for mkt in prev_active["market"]:
                                p = get_current_price(mkt)
                                if p:
                                    now_prices[mkt] = p
                                _time.sleep(0.1)
                            perf_msg = format_performance(prev_active, now_prices)
                            logger.info(f"Previous performance: {len(now_prices)} matured coins checked")
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

            capital = getattr(config.Portfolio, "TOTAL_CAPITAL_KRW", 10_000_000)
            stop_loss = getattr(config.Portfolio, "STOP_LOSS_PCT", 3.0)
            msg = format_report(
                longs,
                shorts,
                latest_ts,
                prices=prices,
                horizon_h=short_horizon_h,
                all_scores=selection_scores,
                min_sigma=0.0,
                capital=capital,
                stop_loss_pct=stop_loss,
                long_header="LONG" if (long_model_active and long_execution_mode == "long_only") else "WATCH LONG",
                long_watch_only=not (long_model_active and long_execution_mode == "long_only"),
                long_horizon_h=long_horizon_h if long_model_active else short_horizon_h,
                short_horizon_h=short_horizon_h,
                next_long_rebalance_at=next_long_rebalance_ts,
                next_short_rebalance_at=next_short_rebalance_ts,
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


def _save_recommendations(
    ts,
    longs,
    shorts,
    long_side_label="LONG",
    short_side_label="SHORT",
    contract_map=None,
    long_model_active=False,
    long_horizon_h=6,
    short_horizon_h=6,
    carry_longs=False,
    prev_long_rows=None,
    long_refresh_reason="watch_seed",
    next_long_rebalance_ts=None,
    next_short_rebalance_ts=None,
):
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    ts_str = ts.strftime("%Y%m%dT%H%M") if hasattr(ts, "strftime") else str(ts)[:16].replace(" ", "T")
    csv_path = os.path.join(out_dir, f"recommendations_{ts_str}.csv")

    # Fetch entry prices for LONG/SHORT positions
    from data.collector import get_current_price
    import time as _time

    rows = []
    prev_long_meta = {}
    if prev_long_rows is not None and not prev_long_rows.empty and "market" in prev_long_rows.columns:
        prev_long_meta = prev_long_rows.drop_duplicates(subset=["market"], keep="last").set_index("market").to_dict("index")

    for market, score in longs.items():
        side = long_side_label
        prev_meta = prev_long_meta.get(market, {})
        if carry_longs and prev_meta:
            entry_price = prev_meta.get("entry_price")
            entry_time = prev_meta.get("entry_time", pd.Timestamp(ts).isoformat())
            bitget_symbol = prev_meta.get("bitget_symbol") or market_to_bitget_symbol(market, contract_map or {})
        else:
            entry_price = get_current_price(market)
            entry_time = pd.Timestamp(ts).isoformat()
            bitget_symbol = market_to_bitget_symbol(market, contract_map or {})
            _time.sleep(0.1)
        rows.append({
            "market": market,
            "score": round(float(score), 4),
            "side": side,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "horizon_h": long_horizon_h,
            "refresh_reason": long_refresh_reason,
            "next_rebalance_at": pd.Timestamp(next_long_rebalance_ts).isoformat() if next_long_rebalance_ts is not None else None,
            "bitget_symbol": bitget_symbol,
            "actionable": (side == "SHORT") or (side == "LONG" and long_model_active),
        })

    for market, score in shorts.items():
        entry_price = get_current_price(market)
        _time.sleep(0.1)
        rows.append({
            "market": market,
            "score": round(float(score), 4),
            "side": short_side_label,
            "entry_price": entry_price,
            "entry_time": pd.Timestamp(ts).isoformat(),
            "horizon_h": short_horizon_h,
            "refresh_reason": "refresh_6h",
            "next_rebalance_at": pd.Timestamp(next_short_rebalance_ts).isoformat() if next_short_rebalance_ts is not None else None,
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
