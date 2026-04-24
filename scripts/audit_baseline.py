#!/usr/bin/env python3
"""
Baseline audit script — Phase A signal quality check.

Answers two questions:
  1. Is the IC real? (fold stability, half-split sign consistency, monthly breakdown)
  2. Is the universe realistic? (volume filter, stale/dead coin removal)

Usage:
    cd /mnt/20t/main/gan_t/xsec_alpha
    python scripts/audit_baseline.py [--days 180] [--min-volume-krw 500000000]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
import pandas as pd
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
)
from models.xgb_ranker import EnsembleRanker, RidgeRanker


# ------------------------------------------------------------------ #
# Walk-forward engine (same as backtest_wf.py but captures coin lists)
# ------------------------------------------------------------------ #

def run_walk_forward(
    factor_df, residuals_wide, fwd_returns, volumes_wide,
    feature_cols, horizon, train_hours, test_hours,
    long_n, short_n, fee, model_type="ensemble",
):
    """Run walk-forward and return detailed per-period results with coin lists."""
    from models.xgb_ranker import EnsembleRanker, RidgeRanker, LGBMRanker, XSecRanker

    residuals_long = residuals_wide.stack(future_stack=True)
    residuals_long.index.names = ["timestamp", "market"]

    combined = factor_df.join(residuals_long.rename("residual"), how="inner").dropna()
    combined = combined[combined["residual"].abs() <= 0.5]

    all_ts = combined.index.get_level_values("timestamp").unique().sort_values()

    pnl_rows = []
    ic_rows = []
    coin_lists = []
    n_retrains = 0
    prev_longs = set()
    prev_shorts = set()

    cursor = train_hours
    while cursor + test_hours <= len(all_ts):
        train_ts = all_ts[max(0, cursor - train_hours):cursor]
        test_ts = all_ts[cursor:min(cursor + test_hours, len(all_ts))]

        train_mask = combined.index.get_level_values("timestamp").isin(train_ts)
        train_data = combined[train_mask]
        X_train = train_data[feature_cols]
        y_train = train_data["residual"]

        if len(X_train) < 100:
            cursor += test_hours
            continue

        if model_type == "ensemble":
            model = EnsembleRanker()
        elif model_type == "ridge":
            model = RidgeRanker(alpha=1.0)
        else:
            model = EnsembleRanker()
        model.fit(X_train, y_train)
        n_retrains += 1

        rebalance_points = test_ts[::horizon]
        for ts in rebalance_points:
            try:
                snapshot = factor_df.xs(ts, level="timestamp").dropna()
            except KeyError:
                continue
            if len(snapshot) < long_n + short_n:
                continue

            scores = model.predict(snapshot[feature_cols])
            score_series = pd.Series(scores, index=snapshot.index).sort_values(ascending=False)

            long_coins = list(score_series.head(long_n).index)
            short_coins = list(score_series.tail(short_n).index)

            if ts not in fwd_returns.index:
                continue
            period_ret = fwd_returns.loc[ts]

            long_ret = period_ret[long_coins].mean() if long_coins else 0.0
            short_ret = period_ret[short_coins].mean() if short_coins else 0.0
            ls_gross = long_ret - short_ret
            if pd.isna(ls_gross):
                continue

            lt = len(set(long_coins) - prev_longs) / max(len(long_coins), 1)
            st = len(set(short_coins) - prev_shorts) / max(len(short_coins), 1)
            turnover = (lt + st) / 2
            cost = turnover * 2 * fee
            ls_net = ls_gross - cost

            # IC
            ic_val = np.nan
            if ts in residuals_wide.index:
                actual_res = residuals_wide.loc[ts]
                common = snapshot.index.intersection(actual_res.dropna().index)
                if len(common) > 5:
                    ic_val, _ = spearmanr(
                        pd.Series(scores, index=snapshot.index)[common],
                        actual_res[common],
                    )

            # Volume of selected coins
            long_vol = short_vol = np.nan
            if ts in volumes_wide.index:
                vol_row = volumes_wide.loc[ts]
                lv = vol_row.reindex(long_coins).dropna()
                sv = vol_row.reindex(short_coins).dropna()
                long_vol = float(lv.median()) if len(lv) > 0 else np.nan
                short_vol = float(sv.median()) if len(sv) > 0 else np.nan

            pnl_rows.append({
                "timestamp": ts,
                "long_ret": float(long_ret),
                "short_ret": float(short_ret),
                "ls_gross": float(ls_gross),
                "ls_net": float(ls_net),
                "turnover": float(turnover),
                "cost": float(cost),
                "retrain_id": n_retrains,
                "ic": float(ic_val) if not np.isnan(ic_val) else None,
                "long_median_vol": long_vol,
                "short_median_vol": short_vol,
            })

            coin_lists.append({
                "timestamp": str(ts),
                "long_coins": long_coins,
                "short_coins": short_coins,
            })

            prev_longs = set(long_coins)
            prev_shorts = set(short_coins)

        cursor += test_hours

    return pd.DataFrame(pnl_rows), coin_lists, n_retrains


# ------------------------------------------------------------------ #
# Audit reports
# ------------------------------------------------------------------ #

def compute_sharpe(returns: pd.Series, periods_per_year: float) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def audit_ic_stability(df: pd.DataFrame) -> dict:
    """Fold-level IC, half-split consistency, monthly IC."""
    ic_series = df["ic"].dropna()
    if len(ic_series) == 0:
        return {"error": "no IC data"}

    # Overall
    overall_ic = float(ic_series.mean())
    overall_std = float(ic_series.std())
    n = len(ic_series)
    t_stat = overall_ic / (overall_std / np.sqrt(n)) if overall_std > 0 and n > 1 else 0

    # Per-fold IC
    fold_ics = {}
    for rid, grp in df.groupby("retrain_id"):
        fold_ic = grp["ic"].dropna()
        fold_ics[int(rid)] = {
            "mean_ic": round(float(fold_ic.mean()), 4) if len(fold_ic) > 0 else None,
            "n": int(len(fold_ic)),
            "positive_pct": round(float((fold_ic > 0).mean()), 2) if len(fold_ic) > 0 else None,
        }

    # Half-split
    mid = len(ic_series) // 2
    first_half_ic = float(ic_series.iloc[:mid].mean())
    second_half_ic = float(ic_series.iloc[mid:].mean())
    sign_consistent = (first_half_ic > 0) == (second_half_ic > 0)

    # Monthly
    df_ic = df[["ic"]].copy()
    df_ic.index = pd.to_datetime(df["timestamp"])
    monthly = df_ic["ic"].resample("ME").agg(["mean", "count"])
    monthly_ics = {
        str(idx.date()): {"mean_ic": round(float(row["mean"]), 4), "n": int(row["count"])}
        for idx, row in monthly.iterrows()
        if row["count"] > 0
    }

    return {
        "overall_ic": round(overall_ic, 4),
        "ic_std": round(overall_std, 4),
        "t_stat": round(t_stat, 2),
        "n_periods": n,
        "first_half_ic": round(first_half_ic, 4),
        "second_half_ic": round(second_half_ic, 4),
        "half_split_sign_consistent": sign_consistent,
        "fold_ics": fold_ics,
        "monthly_ics": monthly_ics,
    }


def audit_universe(coin_lists: list, volumes_wide: pd.DataFrame, min_vol: float) -> dict:
    """Check which coins appear in recommendations and their volume."""
    from collections import Counter

    long_counter = Counter()
    short_counter = Counter()
    for entry in coin_lists:
        for c in entry["long_coins"]:
            long_counter[c] += 1
        for c in entry["short_coins"]:
            short_counter[c] += 1

    # Median daily volume per coin (KRW)
    daily_vol = volumes_wide.resample("D").sum().median()

    # Coins below min_vol threshold
    below_threshold_long = []
    below_threshold_short = []
    for coin, count in short_counter.most_common(50):
        vol = daily_vol.get(coin, 0)
        if vol < min_vol:
            below_threshold_short.append({
                "coin": coin, "appearances": count,
                "median_daily_vol_krw": round(float(vol), 0),
            })
    for coin, count in long_counter.most_common(50):
        vol = daily_vol.get(coin, 0)
        if vol < min_vol:
            below_threshold_long.append({
                "coin": coin, "appearances": count,
                "median_daily_vol_krw": round(float(vol), 0),
            })

    return {
        "total_unique_long": len(long_counter),
        "total_unique_short": len(short_counter),
        "top10_short_repeaters": [
            {"coin": c, "count": n} for c, n in short_counter.most_common(10)
        ],
        "top10_long_repeaters": [
            {"coin": c, "count": n} for c, n in long_counter.most_common(10)
        ],
        "low_volume_short_coins": below_threshold_short[:20],
        "low_volume_long_coins": below_threshold_long[:20],
        "min_volume_threshold_krw": min_vol,
    }


def audit_legs(df: pd.DataFrame, horizon: int) -> dict:
    """Long-only, short-only, and combined metrics."""
    ppyr = 365 * 24 / horizon

    return {
        "long_only": {
            "mean_6h": round(float(df["long_ret"].mean()), 5),
            "hit_rate": round(float((df["long_ret"] > 0).mean()), 3),
            "sharpe_6h": round(compute_sharpe(df["long_ret"], ppyr), 2),
        },
        "short_alpha": {
            "mean_6h": round(float((-df["short_ret"]).mean()), 5),
            "hit_rate": round(float((df["short_ret"] < 0).mean()), 3),
            "sharpe_6h": round(compute_sharpe(-df["short_ret"], ppyr), 2),
        },
        "ls_combined": {
            "mean_6h": round(float(df["ls_net"].mean()), 5),
            "hit_rate": round(float((df["ls_net"] > 0).mean()), 3),
            "sharpe_6h": round(compute_sharpe(df["ls_net"], ppyr), 2),
        },
        "daily_sharpe": _daily_sharpe(df),
    }


def _daily_sharpe(df: pd.DataFrame) -> dict:
    """Aggregate to daily returns, then compute Sharpe."""
    df2 = df.copy()
    df2.index = pd.to_datetime(df2["timestamp"])
    daily = df2[["ls_net", "long_ret", "short_ret"]].resample("D").apply(
        lambda x: (1 + x).prod() - 1
    )
    ppyr = 365
    return {
        "ls_net": round(compute_sharpe(daily["ls_net"], ppyr), 2),
        "long_only": round(compute_sharpe(daily["long_ret"], ppyr), 2),
        "short_alpha": round(compute_sharpe(-daily["short_ret"], ppyr), 2),
    }


# ------------------------------------------------------------------ #
# Filtered re-run
# ------------------------------------------------------------------ #

def filter_universe(closes, opens, highs, lows, volumes, min_daily_vol_krw: float):
    """Remove coins whose median daily KRW volume is below threshold."""
    daily_vol = volumes.resample("D").sum()
    median_vol = daily_vol.median()
    keep = median_vol[median_vol >= min_daily_vol_krw].index.tolist()

    # Always keep BTC
    if "KRW-BTC" not in keep:
        keep.append("KRW-BTC")

    logger.info(
        f"Volume filter: {len(keep)}/{len(closes.columns)} coins pass "
        f"(min_daily_vol={min_daily_vol_krw/1e8:.1f}억 KRW)"
    )
    return (
        closes[keep], opens[keep], highs[keep], lows[keep], volumes[keep],
    )


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=14)
    parser.add_argument("--fee-bps", type=float, default=5)
    parser.add_argument("--long-n", type=int, default=20)
    parser.add_argument("--short-n", type=int, default=20)
    parser.add_argument("--min-volume-krw", type=float, default=5e8,
                        help="Min median daily volume in KRW (default: 5억)")
    args = parser.parse_args()

    horizon = config.Data.PREDICT_HORIZON
    fee = args.fee_bps / 10000
    train_hours = args.train_days * 24
    test_hours = args.test_days * 24

    print(f"=== BASELINE AUDIT | days={args.days} horizon={horizon}h ===\n")

    # Load data
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

    # ============================================================ #
    # PART 1: Unfiltered baseline (same as current backtest_wf)
    # ============================================================ #
    print("─" * 60)
    print("PART 1: UNFILTERED BASELINE")
    print("─" * 60)

    factor_df = compute_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
    zscore_cols = [c for c in factor_df.columns if c not in CALENDAR_COLS]
    factor_df = crosssection_zscore(factor_df, cols=zscore_cols)
    feature_cols = factor_df.columns.tolist()

    residuals_wide = compute_residual_returns(closes, horizon=horizon, beta_window=config.Data.BETA_ROLLING_WINDOW)
    fwd_returns = closes.shift(-horizon) / closes - 1

    df_unf, coins_unf, n_ret_unf = run_walk_forward(
        factor_df, residuals_wide, fwd_returns, volumes,
        feature_cols, horizon, train_hours, test_hours,
        args.long_n, args.short_n, fee,
    )

    ic_audit_unf = audit_ic_stability(df_unf)
    universe_audit = audit_universe(coins_unf, volumes, args.min_volume_krw)
    legs_unf = audit_legs(df_unf, horizon)

    print(f"\n  Periods: {len(df_unf)}, Retrains: {n_ret_unf}")
    print(f"\n  [IC Stability]")
    print(f"    Overall IC: {ic_audit_unf['overall_ic']:+.4f} (t={ic_audit_unf['t_stat']:.1f})")
    print(f"    Half-split: 1st={ic_audit_unf['first_half_ic']:+.4f}, 2nd={ic_audit_unf['second_half_ic']:+.4f}, consistent={ic_audit_unf['half_split_sign_consistent']}")
    print(f"    Fold ICs:")
    for fid, fdata in ic_audit_unf["fold_ics"].items():
        print(f"      Fold {fid}: IC={fdata['mean_ic']:+.4f}, n={fdata['n']}, pos%={fdata['positive_pct']:.0%}")
    print(f"    Monthly ICs:")
    for month, mdata in ic_audit_unf["monthly_ics"].items():
        print(f"      {month}: IC={mdata['mean_ic']:+.4f} (n={mdata['n']})")

    print(f"\n  [Leg Decomposition]")
    for label, d in legs_unf.items():
        if label == "daily_sharpe":
            print(f"    Daily Sharpe: LS={d['ls_net']:.1f}, Long={d['long_only']:.1f}, Short={d['short_alpha']:.1f}")
        else:
            print(f"    {label}: mean={d['mean_6h']:+.5f}, hit={d['hit_rate']:.1%}, sharpe_6h={d['sharpe_6h']:.1f}")

    print(f"\n  [Universe]")
    print(f"    Unique coins: long={universe_audit['total_unique_long']}, short={universe_audit['total_unique_short']}")
    print(f"    Top-10 short repeaters:")
    for item in universe_audit["top10_short_repeaters"]:
        print(f"      {item['coin']}: {item['count']} times")
    n_low_vol_s = len(universe_audit["low_volume_short_coins"])
    n_low_vol_l = len(universe_audit["low_volume_long_coins"])
    print(f"    Low-vol coins (<{args.min_volume_krw/1e8:.0f}억): short={n_low_vol_s}, long={n_low_vol_l}")
    if universe_audit["low_volume_short_coins"]:
        print(f"    Low-vol short examples:")
        for item in universe_audit["low_volume_short_coins"][:5]:
            print(f"      {item['coin']}: {item['appearances']}x, vol={item['median_daily_vol_krw']/1e8:.2f}억")

    # ============================================================ #
    # PART 2: Volume-filtered universe
    # ============================================================ #
    print("\n" + "─" * 60)
    print(f"PART 2: FILTERED (min daily vol >= {args.min_volume_krw/1e8:.0f}억 KRW)")
    print("─" * 60)

    c_f, o_f, h_f, l_f, v_f = filter_universe(
        closes, opens, highs, lows, volumes, args.min_volume_krw
    )

    bn_f = binance_closes
    if not bn_f.empty:
        common = [c for c in c_f.columns if c in bn_f.columns]
        bn_f = bn_f[common] if common else pd.DataFrame()

    factor_f = compute_factors(c_f, o_f, h_f, l_f, v_f, binance_closes=bn_f)
    zscore_cols_f = [c for c in factor_f.columns if c not in CALENDAR_COLS]
    factor_f = crosssection_zscore(factor_f, cols=zscore_cols_f)
    feature_cols_f = factor_f.columns.tolist()

    residuals_f = compute_residual_returns(c_f, horizon=horizon, beta_window=config.Data.BETA_ROLLING_WINDOW)
    fwd_f = c_f.shift(-horizon) / c_f - 1

    df_fil, coins_fil, n_ret_fil = run_walk_forward(
        factor_f, residuals_f, fwd_f, v_f,
        feature_cols_f, horizon, train_hours, test_hours,
        args.long_n, args.short_n, fee,
    )

    ic_audit_fil = audit_ic_stability(df_fil)
    legs_fil = audit_legs(df_fil, horizon)

    print(f"\n  Periods: {len(df_fil)}, Retrains: {n_ret_fil}")
    print(f"\n  [IC Stability]")
    print(f"    Overall IC: {ic_audit_fil['overall_ic']:+.4f} (t={ic_audit_fil['t_stat']:.1f})")
    print(f"    Half-split: 1st={ic_audit_fil['first_half_ic']:+.4f}, 2nd={ic_audit_fil['second_half_ic']:+.4f}, consistent={ic_audit_fil['half_split_sign_consistent']}")
    print(f"    Fold ICs:")
    for fid, fdata in ic_audit_fil["fold_ics"].items():
        print(f"      Fold {fid}: IC={fdata['mean_ic']:+.4f}, n={fdata['n']}, pos%={fdata['positive_pct']:.0%}")

    print(f"\n  [Leg Decomposition]")
    for label, d in legs_fil.items():
        if label == "daily_sharpe":
            print(f"    Daily Sharpe: LS={d['ls_net']:.1f}, Long={d['long_only']:.1f}, Short={d['short_alpha']:.1f}")
        else:
            print(f"    {label}: mean={d['mean_6h']:+.5f}, hit={d['hit_rate']:.1%}, sharpe_6h={d['sharpe_6h']:.1f}")

    # ============================================================ #
    # PART 3: Comparison
    # ============================================================ #
    print("\n" + "=" * 60)
    print("COMPARISON: UNFILTERED vs FILTERED")
    print("=" * 60)

    def _cmp(label, val_unf, val_fil):
        delta = val_fil - val_unf
        print(f"  {label:30s}  {val_unf:+.4f}  →  {val_fil:+.4f}  (Δ={delta:+.4f})")

    _cmp("IC (overall)", ic_audit_unf["overall_ic"], ic_audit_fil["overall_ic"])
    _cmp("IC (1st half)", ic_audit_unf["first_half_ic"], ic_audit_fil["first_half_ic"])
    _cmp("IC (2nd half)", ic_audit_unf["second_half_ic"], ic_audit_fil["second_half_ic"])

    daily_unf = legs_unf["daily_sharpe"]
    daily_fil = legs_fil["daily_sharpe"]
    _cmp("Daily Sharpe (LS net)", daily_unf["ls_net"], daily_fil["ls_net"])
    _cmp("Daily Sharpe (Long)", daily_unf["long_only"], daily_fil["long_only"])
    _cmp("Daily Sharpe (Short)", daily_unf["short_alpha"], daily_fil["short_alpha"])

    ls_unf_mean = legs_unf["ls_combined"]["mean_6h"]
    ls_fil_mean = legs_fil["ls_combined"]["mean_6h"]
    _cmp("LS spread (6h mean)", ls_unf_mean, ls_fil_mean)

    print("=" * 60)

    if ic_audit_fil["overall_ic"] > 0.05 and ic_audit_fil["t_stat"] > 2.0:
        print("  VERDICT: Filtered IC still significant — signal likely real")
    elif ic_audit_fil["overall_ic"] > 0.02:
        print("  VERDICT: Filtered IC marginal — proceed with caution")
    else:
        print("  VERDICT: Filtered IC collapsed — unfiltered baseline was inflated")

    # Save full audit
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    audit_path = os.path.join(out_dir, "audit_baseline.json")
    audit_data = {
        "unfiltered": {
            "ic": ic_audit_unf,
            "legs": legs_unf,
            "universe": universe_audit,
            "n_periods": len(df_unf),
        },
        "filtered": {
            "ic": ic_audit_fil,
            "legs": legs_fil,
            "n_periods": len(df_fil),
            "min_volume_krw": args.min_volume_krw,
        },
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, default=str)
    print(f"\n  Full audit saved: {audit_path}")


if __name__ == "__main__":
    main()
