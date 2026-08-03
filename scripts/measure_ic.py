#!/usr/bin/env python3
"""
IC measurement script — Phase 1 gate check.

Usage:
    cd /mnt/20t/main/gan_t/xsec_alpha
    python scripts/measure_ic.py [--days 120] [--no-zscore]

Pass gate:  IC > 0.05 AND t-stat > 1.5 on at least one factor.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import math
import pandas as pd

from config import config
from utils.logger import logger
from utils.run_lock import stable_data_read_lock
from data.features import (
    load_and_pivot,
    load_binance_pivot,
    compute_factors,
    compute_long_factors,
    compute_btc_regime,
    compute_residual_returns,
    crosssection_zscore,
    compute_ic_series,
    summarize_ic,
    CALENDAR_COLS,
    LONG_CALENDAR_COLS,
    build_top_liquidity_universe_index,
    filter_long_frame_by_universe,
    filter_long_series_by_universe,
)


_DATA_ACCESS_LOCK_WAIT_SEC = 480.0


def main():
    with stable_data_read_lock(timeout_sec=_DATA_ACCESS_LOCK_WAIT_SEC):
        _main_locked()


def _main_locked():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",     type=int, default=120, help="Days of history to load")
    parser.add_argument("--no-zscore", action="store_true", help="Skip cross-sectional z-score normalization")
    parser.add_argument("--rolling-window", type=int, default=168,
                        help="Rolling IC window: last N timestamps (default 168 = 7 days of hourly data)")
    parser.add_argument(
        "--top-liquidity",
        type=int,
        default=0,
        help="Restrict IC calculation to top N coins by 24h traded value at each timestamp (0 = no filter)",
    )
    parser.add_argument(
        "--side",
        type=str,
        default="short",
        choices=["short", "long"],
        help="Factor set to measure: short (default) or long (momentum/breakout)",
    )
    args = parser.parse_args()

    logger.info(f"=== measure_ic.py | days={args.days} zscore={not args.no_zscore} side={args.side} ===")

    # 1. Load data
    closes, opens, highs, lows, volumes = load_and_pivot(days=args.days)
    logger.info(f"Shape: {closes.shape} ({closes.shape[0]} timestamps × {closes.shape[1]} coins)")

    # 2. Drop warmup rows (per coin — but since pivoted, drop first N rows globally)
    warmup = config.Data.MIN_ROWS_PER_COIN
    closes  = closes.iloc[warmup:]
    opens   = opens.iloc[warmup:]
    highs   = highs.iloc[warmup:]
    lows    = lows.iloc[warmup:]
    volumes = volumes.iloc[warmup:]
    logger.info(f"After warmup drop ({warmup} rows): {closes.shape}")

    # 3. Load Binance data and compute factors
    logger.info("Loading Binance data...")
    binance_closes = load_binance_pivot(closes.columns.tolist(), days=args.days)
    if not binance_closes.empty:
        binance_closes = binance_closes.iloc[warmup:]
    if args.side == "long":
        logger.info("Computing LONG-specialist factors...")
        factor_df = compute_long_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
    else:
        logger.info("Computing factors...")
        factor_df = compute_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)

    if not args.no_zscore:
        logger.info("Applying cross-sectional z-score normalization...")
        cal_cols = LONG_CALENDAR_COLS if args.side == "long" else CALENDAR_COLS
        zscore_cols = [c for c in factor_df.columns if c not in cal_cols]
        factor_df = crosssection_zscore(factor_df, cols=zscore_cols)

    # 4. Compute residual returns
    logger.info("Computing residual returns (this may take a moment)...")
    horizon = (
        getattr(config.LongModel, "PREDICT_HORIZON", config.Data.PREDICT_HORIZON)
        if args.side == "long"
        else config.Data.PREDICT_HORIZON
    )
    beta_window = config.Data.BETA_ROLLING_WINDOW
    residuals_wide = compute_residual_returns(closes, horizon=horizon, beta_window=beta_window)
    residuals_long = residuals_wide.stack(future_stack=True)
    residuals_long.index.names = ["timestamp", "market"]

    # Regime filter for long side: IC only on bull timestamps
    if args.side == "long":
        btc_regime = compute_btc_regime(closes)
        bull_ts = btc_regime.index[btc_regime["regime_bull"] == 1.0]
        pre_len = len(factor_df)
        factor_df = factor_df[factor_df.index.get_level_values("timestamp").isin(bull_ts)]
        residuals_long = residuals_long[residuals_long.index.get_level_values("timestamp").isin(bull_ts)]
        logger.info(
            "Regime filter (long): kept %d/%d rows (%.1f%% bull)",
            len(factor_df), pre_len, len(factor_df) / max(pre_len, 1) * 100,
        )

    universe_note = "full universe"
    if args.top_liquidity > 0:
        selected_index, coverage = build_top_liquidity_universe_index(
            closes=closes,
            volumes=volumes,
            top_n=args.top_liquidity,
        )
        factor_df = filter_long_frame_by_universe(factor_df, selected_index)
        residuals_long = filter_long_series_by_universe(residuals_long, selected_index)
        universe_note = (
            f"top {args.top_liquidity} by 24h traded value "
            f"(avg selected={coverage['avg_selected']:.1f}, min={coverage['min_selected']}, max={coverage['max_selected']})"
        )
        logger.info("Liquidity universe filter applied: %s", universe_note)

    # 5. Measure IC per factor
    factor_cols = factor_df.columns.tolist()
    logger.info(
        "Measuring IC for %s factors over %sh horizon on %s...",
        len(factor_cols),
        horizon,
        universe_note,
    )

    results = []
    rolling_results = []
    split_results = []
    for col in factor_cols:
        ic_series = compute_ic_series(factor_df, residuals_long, col)
        summary = summarize_ic(ic_series)
        results.append(summary)
        status = _gate_label(summary["mean_ic"], summary["t_stat"])
        logger.info(
            f"  {col:22s}  IC={summary['mean_ic']:+.4f}  t={summary['t_stat']:+.2f}"
            f"  std={summary['ic_std']:.4f}  n={summary['n_periods']}  [{status}]"
        )

        # Rolling IC (last N timestamps)
        rolling_slice = ic_series.iloc[-args.rolling_window:]
        rolling_summary = summarize_ic(rolling_slice)
        rolling_results.append((summary, rolling_summary))
        split_results.append((summary, *_split_ic_summary(ic_series)))

    # 6. Gate check
    print("\n" + "=" * 70)
    print(f"IC Summary (horizon={horizon}h, days={args.days}, universe={universe_note})")
    print("=" * 70)
    print(f"{'Factor':<24} {'mean_IC':>8} {'t_stat':>7} {'IC_std':>7} {'n':>5}  Gate")
    print("-" * 70)
    gate_passed = False
    for r in results:
        status = _gate_label(r["mean_ic"], r["t_stat"])
        if status == "PASS":
            gate_passed = True
        print(f"{r['factor']:<24} {r['mean_ic']:>+8.4f} {r['t_stat']:>+7.2f} {r['ic_std']:>7.4f} {r['n_periods']:>5}  {status}")
    print("=" * 70)

    # 7. Rolling IC section
    print(f"\nRolling IC (last {args.rolling_window} periods):")
    print(f"{'Factor':<24} {'rolling_IC':>10} {'rolling_t':>9}  Tag")
    print("-" * 55)
    for full_summary, rolling_summary in rolling_results:
        ric  = rolling_summary["mean_ic"]
        rt   = rolling_summary["t_stat"]
        full_ic = full_summary["mean_ic"]
        tag = _rolling_tag(ric, rt, full_ic)
        print(f"{rolling_summary['factor']:<24} {ric:>+10.4f} {rt:>+9.2f}  {tag}")
    print("-" * 55)

    print("\nStability Check (first half vs second half):")
    print(f"{'Factor':<24} {'1H_IC':>8} {'2H_IC':>8} {'1H_t':>7} {'2H_t':>7}  Tag")
    print("-" * 70)
    for full_summary, first_summary, second_summary in split_results:
        tag = _stability_tag(full_summary["mean_ic"], first_summary["mean_ic"], second_summary["mean_ic"])
        print(
            f"{full_summary['factor']:<24} "
            f"{first_summary['mean_ic']:>+8.4f} {second_summary['mean_ic']:>+8.4f} "
            f"{first_summary['t_stat']:>+7.2f} {second_summary['t_stat']:>+7.2f}  {tag}"
        )
    print("-" * 70)

    if gate_passed:
        print("✓ Phase 1 GATE PASSED: IC > 0.05 on at least one factor")
        sys.exit(0)
    else:
        print("✗ Phase 1 GATE FAILED: No factor reached IC > 0.05 with t > 1.5")
        print("  → Check data pipeline before proceeding to Phase 2")
        sys.exit(1)


def _rolling_tag(rolling_ic: float, rolling_t: float, full_ic: float) -> str:
    """Tag rolling IC vs full-sample IC."""
    if _sign(rolling_ic) != 0 and _sign(full_ic) != 0 and _sign(rolling_ic) != _sign(full_ic):
        return "[FLIP]"
    if abs(rolling_ic) > 0.05 and abs(rolling_t) > 1.5:
        return "[FRESH]"
    # STALE: rolling IC dropped below half of full-sample IC (same sign direction)
    if full_ic != 0 and abs(rolling_ic) < abs(full_ic) / 2:
        return "[STALE]"
    return ""


def _gate_label(mean_ic: float, t_stat: float) -> str:
    if abs(mean_ic) > 0.05 and abs(t_stat) > 1.5:
        return "PASS"
    if abs(mean_ic) > 0.02:
        return "WEAK"
    return "FAIL"

def _split_ic_summary(ic_series: pd.Series):
    clean = ic_series.dropna()
    if clean.empty:
        empty = {"factor": ic_series.name, "mean_ic": float("nan"), "t_stat": float("nan")}
        return empty, empty

    mid = max(1, math.ceil(len(clean) / 2))
    first = summarize_ic(clean.iloc[:mid])
    second = summarize_ic(clean.iloc[mid:]) if mid < len(clean) else {
        "factor": ic_series.name,
        "mean_ic": float("nan"),
        "t_stat": float("nan"),
    }
    return first, second


def _sign(x: float) -> int:
    if pd.isna(x) or x == 0:
        return 0
    return 1 if x > 0 else -1


def _stability_tag(full_ic: float, first_ic: float, second_ic: float) -> str:
    s_full = _sign(full_ic)
    s_first = _sign(first_ic)
    s_second = _sign(second_ic)
    if s_first != 0 and s_second != 0 and s_first != s_second:
        return "[FLIP]"
    if s_full != 0 and ((s_first != 0 and s_first != s_full) or (s_second != 0 and s_second != s_full)):
        return "[DRIFT]"
    if not pd.isna(second_ic) and abs(second_ic) < abs(full_ic) / 2:
        return "[DECAY]"
    return "[STABLE]"


if __name__ == "__main__":
    main()
