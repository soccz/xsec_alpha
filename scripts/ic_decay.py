#!/usr/bin/env python3
"""
IC decay analysis — measures IC at multiple horizons to understand signal decay.

Usage:
    cd /mnt/20t/main/gan_t/xsec_alpha
    python scripts/ic_decay.py [--days 90] [--horizons "1,2,4,6,12,24,48"]

Shows IC at each horizon for each factor. Helps determine optimal rebalancing frequency.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import pandas as pd

from config import config
from utils.logger import logger
from data.features import (
    load_and_pivot,
    load_binance_pivot,
    compute_factors,
    compute_residual_returns,
    crosssection_zscore,
    compute_ic_series,
    summarize_ic,
    build_top_liquidity_universe_index,
    filter_long_frame_by_universe,
    filter_long_series_by_universe,
)


def main():
    parser = argparse.ArgumentParser(description="IC decay analysis across multiple horizons")
    parser.add_argument("--days",     type=int,  default=90,
                        help="Days of history to load (default: 90)")
    parser.add_argument("--horizons", type=str,  default="1,2,4,6,12,24,48",
                        help="Comma-separated list of horizons in hours (default: '1,2,4,6,12,24,48')")
    parser.add_argument(
        "--top-liquidity",
        type=int,
        default=getattr(config.Data, "LIQUIDITY_TOP_N", 0),
        help="Restrict analysis to top N coins by 24h traded value per timestamp",
    )
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    beta_window = config.Data.BETA_ROLLING_WINDOW

    logger.info(f"=== ic_decay.py | days={args.days} horizons={horizons} ===")

    # 1. Load and prepare data
    closes, opens, highs, lows, volumes = load_and_pivot(days=args.days)
    logger.info(f"Loaded: {closes.shape[0]} timestamps × {closes.shape[1]} coins")

    # 2. Drop warmup rows
    warmup = config.Data.MIN_ROWS_PER_COIN
    closes  = closes.iloc[warmup:]
    opens   = opens.iloc[warmup:]
    highs   = highs.iloc[warmup:]
    lows    = lows.iloc[warmup:]
    volumes = volumes.iloc[warmup:]
    logger.info(f"After warmup drop ({warmup} rows): {closes.shape}")

    # 3. Compute factors once (reused across all horizons)
    logger.info("Loading Binance data...")
    binance_closes = load_binance_pivot(closes.columns.tolist(), days=args.days)
    if not binance_closes.empty:
        binance_closes = binance_closes.iloc[warmup:]

    logger.info("Computing factors...")
    factor_df = compute_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
    logger.info("Applying cross-sectional z-score normalization...")
    factor_df = crosssection_zscore(factor_df)
    selected_index, coverage = build_top_liquidity_universe_index(
        closes,
        volumes,
        top_n=args.top_liquidity,
    )
    factor_df = filter_long_frame_by_universe(factor_df, selected_index)
    logger.info(
        "Active universe: top %s by %sh traded value (avg selected=%.1f, min=%s, max=%s)",
        args.top_liquidity,
        getattr(config.Data, "LIQUIDITY_LOOKBACK_HOURS", 24),
        coverage["avg_selected"],
        coverage["min_selected"],
        coverage["max_selected"],
    )

    factor_cols = factor_df.columns.tolist()
    logger.info(f"Factors: {factor_cols}")

    # 4. For each horizon: compute residuals and measure IC per factor
    all_results = []  # list of dicts: {horizon, factor, mean_ic, t_stat, n_periods}

    for h in horizons:
        logger.info(f"Computing residual returns at horizon={h}h ...")
        residuals_wide = compute_residual_returns(closes, horizon=h, beta_window=beta_window)
        residuals_long = residuals_wide.stack(future_stack=True)
        residuals_long.index.names = ["timestamp", "market"]
        residuals_long = filter_long_series_by_universe(residuals_long, selected_index)

        for col in factor_cols:
            ic_series = compute_ic_series(factor_df, residuals_long, col)
            summary = summarize_ic(ic_series)
            all_results.append({
                "horizon":   h,
                "factor":    col,
                "mean_ic":   summary["mean_ic"],
                "t_stat":    summary["t_stat"],
                "n_periods": summary["n_periods"],
            })
            logger.info(
                f"  h={h:3d}h  {col:22s}  IC={summary['mean_ic']:+.4f}  t={summary['t_stat']:+.2f}"
                f"  n={summary['n_periods']}"
            )

    # 5. Print result table
    col_w = max(len(c) for c in factor_cols)
    header_w = 14 + col_w + 10 + 8 + 10
    sep = "-" * max(header_w, 64)

    print("\n" + "=" * max(header_w, 64))
    print(f"IC Decay Table (days={args.days}, beta_window={beta_window}h)")
    print("=" * max(header_w, 64))
    print(f"{'horizon':>7}  {'factor':<{col_w}}  {'mean_IC':>8}  {'t_stat':>7}  {'n_periods':>9}")
    print(sep)

    for r in all_results:
        sig = "*" if abs(r["mean_ic"]) > 0.05 and abs(r["t_stat"]) > 1.5 else " "
        print(
            f"{r['horizon']:>6}h  {r['factor']:<{col_w}}  "
            f"{r['mean_ic']:>+8.4f}  {r['t_stat']:>+7.2f}  {r['n_periods']:>9}  {sig}"
        )

    print("=" * max(header_w, 64))
    print("  * = IC > 0.05 AND t-stat > 1.5")

    # 6. Best horizon per factor
    print("\nBest horizon per factor (by |mean_IC|):")
    print(sep)
    results_df = pd.DataFrame(all_results)
    for col in factor_cols:
        sub = results_df[results_df["factor"] == col].copy()
        sub["abs_ic"] = sub["mean_ic"].abs()
        valid = sub.dropna(subset=["abs_ic"])
        if valid.empty:
            print(f"  {col:<{col_w}}  best h= N/A  IC=nan  t=nan")
            continue
        best = valid.loc[valid["abs_ic"].idxmax()]
        sig = "*" if abs(best["mean_ic"]) > 0.05 and abs(best["t_stat"]) > 1.5 else " "
        print(
            f"  {col:<{col_w}}  best h={int(best['horizon']):3d}h  "
            f"IC={best['mean_ic']:+.4f}  t={best['t_stat']:+.2f}  {sig}"
        )
    print(sep)


if __name__ == "__main__":
    main()
