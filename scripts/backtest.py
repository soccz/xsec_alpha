#!/usr/bin/env python3
"""
Backtest: simulate Long/Short portfolio returns on holdout period.

Answers: "If we traded this model for the last N days, what would P&L look like?"

Usage:
    cd /mnt/20t/main/gan_t/xsec_alpha
    python scripts/backtest.py [--days 90] [--long-n 20] [--short-n 20] [--fee-bps 5]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.stats import spearmanr

from config import config
from utils.logger import logger
from data.features import (
    load_and_pivot,
    load_binance_pivot,
    compute_factors,
    compute_residual_returns,
    crosssection_zscore,
    CALENDAR_COLS,
    build_top_liquidity_universe_index,
    filter_long_frame_by_universe,
    filter_long_series_by_universe,
)
from models.xgb_ranker import XSecRanker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--long-n", type=int, default=20)
    parser.add_argument("--short-n", type=int, default=20)
    parser.add_argument("--fee-bps", type=float, default=5, help="One-way fee in bps (5 = 0.05%%)")
    parser.add_argument("--model-path", default="models/xsec_xgb.pkl")
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--rebalance-hours", type=int, default=config.Data.PREDICT_HORIZON)
    args = parser.parse_args()

    fee = args.fee_bps / 10000  # 5 bps = 0.0005

    logger.info(f"=== BACKTEST | days={args.days} long={args.long_n} short={args.short_n} fee={args.fee_bps}bps ===")

    # 1. Load data
    closes, opens, highs, lows, volumes = load_and_pivot(days=args.days)
    warmup = config.Data.MIN_ROWS_PER_COIN
    closes  = closes.iloc[warmup:]
    opens   = opens.iloc[warmup:]
    highs   = highs.iloc[warmup:]
    lows    = lows.iloc[warmup:]
    volumes = volumes.iloc[warmup:]

    binance_closes = load_binance_pivot(closes.columns.tolist(), days=args.days)
    if not binance_closes.empty:
        binance_closes = binance_closes.iloc[warmup:]

    # 2. Compute factors
    factor_df = compute_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
    zscore_cols = [c for c in factor_df.columns if c not in CALENDAR_COLS]
    factor_df = crosssection_zscore(factor_df, cols=zscore_cols)
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

    # 3. Load model
    abs_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.model_path)
    model = XSecRanker.load(abs_model_path)
    logger.info(f"Model: {type(model).__name__}")

    # 4. Temporal split — only backtest on holdout period
    all_timestamps = closes.index.sort_values()
    n_holdout = max(1, int(len(all_timestamps) * args.holdout_ratio))
    holdout_start = all_timestamps[-n_holdout]
    logger.info(f"Holdout: {holdout_start} → {all_timestamps[-1]} ({n_holdout} hours)")

    # 5. Forward returns (actual, not residual — this is what you earn)
    horizon = args.rebalance_hours
    fwd_returns = closes.shift(-horizon) / closes - 1
    residuals_long = compute_residual_returns(
        closes,
        horizon=horizon,
        beta_window=config.Data.BETA_ROLLING_WINDOW,
    ).stack(future_stack=True)
    residuals_long.index.names = ["timestamp", "market"]
    residuals_long = filter_long_series_by_universe(residuals_long, selected_index)

    # 6. Simulate: rebalance every `horizon` hours
    rebalance_ts = all_timestamps[all_timestamps >= holdout_start][::horizon]
    logger.info(f"Rebalance points: {len(rebalance_ts)}")

    pnl_rows = []
    prev_longs = set()
    prev_shorts = set()

    for ts in rebalance_ts:
        # Get factor snapshot at this timestamp
        try:
            snapshot = factor_df.xs(ts, level="timestamp").dropna()
        except KeyError:
            continue

        if len(snapshot) < args.long_n + args.short_n:
            continue

        # Predict
        scores = model.predict(snapshot)
        score_series = pd.Series(scores, index=snapshot.index).sort_values(ascending=False)

        # Select long/short
        long_coins  = set(score_series.head(args.long_n).index)
        short_coins = set(score_series.tail(args.short_n).index)

        # Forward return for this period
        if ts not in fwd_returns.index:
            continue
        period_returns = fwd_returns.loc[ts]

        # Portfolio returns (equal weight)
        long_ret  = period_returns[list(long_coins)].mean() if long_coins else 0.0
        short_ret = period_returns[list(short_coins)].mean() if short_coins else 0.0

        # Long/Short P&L: long - short (market neutral)
        ls_gross = long_ret - short_ret
        if pd.isna(ls_gross):
            continue

        # Turnover cost
        long_turnover  = len(long_coins - prev_longs) / max(len(long_coins), 1)
        short_turnover = len(short_coins - prev_shorts) / max(len(short_coins), 1)
        avg_turnover = (long_turnover + short_turnover) / 2
        # Each side: buy + sell = 2 * fee per turnover unit
        cost = avg_turnover * 2 * fee
        ls_net = ls_gross - cost

        actual_res_ts = residuals_long.xs(ts, level="timestamp").dropna() if ts in residuals_long.index.get_level_values("timestamp") else pd.Series(dtype=float)
        common = score_series.index.intersection(actual_res_ts.index)
        if len(common) > 5:
            ic_val, _ = spearmanr(score_series[common], actual_res_ts[common])
        else:
            ic_val = float("nan")

        pnl_rows.append({
            "timestamp": ts,
            "long_ret": float(long_ret),
            "short_ret": float(short_ret),
            "ls_gross": float(ls_gross),
            "turnover": float(avg_turnover),
            "cost": float(cost),
            "ls_net": float(ls_net),
            "ic": float(ic_val),
            "n_long": len(long_coins),
            "n_short": len(short_coins),
        })

        prev_longs = long_coins
        prev_shorts = short_coins

    if not pnl_rows:
        print("No backtest periods available.")
        sys.exit(1)

    df = pd.DataFrame(pnl_rows).set_index("timestamp")

    # 7. Compute stats
    cum_gross = (1 + df["ls_gross"]).cumprod()
    cum_net   = (1 + df["ls_net"]).cumprod()
    n_periods = len(df)
    periods_per_year = 365 * 24 / horizon

    ann_factor = np.sqrt(periods_per_year)
    sharpe_gross = df["ls_gross"].mean() / df["ls_gross"].std() * ann_factor if df["ls_gross"].std() > 0 else 0
    sharpe_net   = df["ls_net"].mean()   / df["ls_net"].std()   * ann_factor if df["ls_net"].std() > 0 else 0

    total_gross = cum_gross.iloc[-1] - 1
    total_net   = cum_net.iloc[-1] - 1
    max_dd_net  = (cum_net / cum_net.cummax() - 1).min()

    win_rate = (df["ls_net"] > 0).mean()
    avg_turnover = df["turnover"].mean()
    total_cost = df["cost"].sum()

    long_hit = (df["long_ret"] > 0).mean()
    short_hit = (df["short_ret"] < 0).mean()

    # 8. Print results
    print("\n" + "=" * 70)
    print(f"BACKTEST RESULTS | {df.index[0]} → {df.index[-1]}")
    print(f"  Rebalance: {horizon}h | Long: {args.long_n} | Short: {args.short_n} | Fee: {args.fee_bps}bps")
    print("=" * 70)
    print(f"  Periods:           {n_periods}")
    print(f"  Total return (gross): {total_gross:+.2%}")
    print(f"  Total return (net):   {total_net:+.2%}")
    print(f"  Sharpe (gross):       {sharpe_gross:+.2f}")
    print(f"  Sharpe (net):         {sharpe_net:+.2f}")
    print(f"  Max drawdown:         {max_dd_net:+.2%}")
    print(f"  Win rate (net):       {win_rate:.1%}")
    print(f"  Avg turnover:         {avg_turnover:.1%}")
    print(f"  Total cost:           {total_cost:.4f} ({total_cost:.2%})")
    print("-" * 70)
    print(f"  Long hit rate:        {long_hit:.1%} (fraction of periods where longs > 0)")
    print(f"  Short hit rate:       {short_hit:.1%} (fraction of periods where shorts < 0)")
    print(f"  Avg long ret/period:  {df['long_ret'].mean():+.4f}")
    print(f"  Avg short ret/period: {df['short_ret'].mean():+.4f}")
    print(f"  Avg L-S spread:       {df['ls_gross'].mean():+.4f}")
    print("=" * 70)

    # Gate check
    if sharpe_net > 0.5:
        print(f"  BACKTEST PASSED: Sharpe(net)={sharpe_net:.2f} > 0.5")
    elif sharpe_net > 0:
        print(f"  BACKTEST MARGINAL: Sharpe(net)={sharpe_net:.2f} (positive but < 0.5)")
    else:
        print(f"  BACKTEST FAILED: Sharpe(net)={sharpe_net:.2f} ≤ 0")

    # Save detailed results
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "backtest_result.csv")
    df.to_csv(csv_path)
    print(f"\n  Detailed results: {csv_path}")

    sys.exit(0 if sharpe_net > 0.5 else 1)


if __name__ == "__main__":
    main()
