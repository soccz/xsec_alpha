#!/usr/bin/env python3
"""
Holdout evaluation script — Phase 2 gate check.

Usage:
    cd /mnt/20t/main/gan_t/xsec_alpha
    python scripts/evaluate_holdout.py [--days 90] [--model-path models/xsec_6h.pkl]

Phase 2 Gate: holdout IC > 0.10 AND t-stat > 2.0
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import pickle
from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import config
from data.features import compute_forward_returns, compute_residual_returns
from utils.eval_metrics import build_btc_context, compute_score_diagnostics, select_positions_with_buffer
from utils.logger import logger
from data.dataset import build_dataset


# ---------------------------------------------------------------------------
# Gate thresholds
# ---------------------------------------------------------------------------
IC_GATE  = 0.10
TST_GATE = 2.0


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 gate: evaluate XGBoost ranker on holdout split."
    )
    parser.add_argument("--days",          type=int,   default=90,                    help="Days of history to load (default: 90)")
    parser.add_argument("--model-path",    type=str,   default=config.Model.MODEL_PATH, help="Path to trained model pickle")
    parser.add_argument("--holdout-ratio", type=float, default=0.2,                   help="Fraction of timestamps to hold out (default: 0.2)")
    parser.add_argument("--horizon",       type=int,   default=None,                  help="Override prediction horizon in hours")
    parser.add_argument("--rebalance-hours", type=int, default=None,                  help="Evaluation rebalance interval in hours (default: same as horizon)")
    parser.add_argument("--execution-lag-bars", type=int, default=0,                  help="Delay execution by N bars: score_t -> enter at t+N")
    parser.add_argument("--long-n",        type=int,   default=config.Portfolio.LONG_N, help="Number of long positions")
    parser.add_argument("--short-n",       type=int,   default=config.Portfolio.SHORT_N, help="Number of short positions")
    parser.add_argument("--rebal-buffer",  type=int,   default=config.Portfolio.REBAL_BUFFER, help="Keep prior positions if they remain within N+buffer")
    parser.add_argument("--fee-bps",       type=float, default=5.0,                   help="One-way fee in bps")
    parser.add_argument("--slippage-bps",  type=float, default=0.0,                   help="One-way slippage assumption in bps")
    parser.add_argument("--short-extra-cost-bps", type=float, default=10.0,          help="Additional per-period cost charged to the short leg (research baseline: 10bps)")
    parser.add_argument("--side",              type=str,   default="unified", choices=["unified", "short", "long"], help="Factor library/model side")
    parser.add_argument("--target",            type=str,   default="absolute", choices=["absolute", "residual"], help="Holdout target to rank against")
    args = parser.parse_args()

    logger.info(
        "=== evaluate_holdout.py | days=%s holdout=%s horizon=%s model=%s ===",
        args.days,
        args.holdout_ratio,
        args.horizon,
        args.model_path,
    )

    # ------------------------------------------------------------------
    # 1. Build dataset (same split as training)
    # ------------------------------------------------------------------
    logger.info("Building dataset (side=%s, target=%s)...", args.side, args.target)
    ds = build_dataset(
        days=args.days,
        holdout_ratio=args.holdout_ratio,
        horizon=args.horizon,
        side=args.side,
        target=args.target,
    )

    X_holdout          = ds["X_holdout"]
    closes             = ds["closes"]
    opens              = ds["opens"]
    holdout_timestamps = ds["holdout_timestamps"]
    split_ts           = ds["split_ts"]
    horizon            = ds["horizon"]
    rebalance_hours    = args.rebalance_hours or horizon
    execution_lag_bars = max(0, args.execution_lag_bars)
    eval_timestamps    = _select_rebalance_timestamps(holdout_timestamps, rebalance_hours)
    eval_prices        = opens if execution_lag_bars > 0 else closes
    eval_fwd_returns   = compute_forward_returns(
        eval_prices,
        horizon=horizon,
        execution_lag=execution_lag_bars,
    )
    eval_residuals_wide = compute_residual_returns(
        eval_prices,
        horizon=horizon,
        beta_window=config.Data.BETA_ROLLING_WINDOW,
        execution_lag=execution_lag_bars,
    )
    eval_target_wide = eval_fwd_returns if args.target == "absolute" else eval_residuals_wide
    btc_context = build_btc_context(closes)

    logger.info(
        "Holdout: %s rows across %s timestamps (eval every %sh, lag=%s bars, %s eval points, split at %s)",
        len(X_holdout),
        len(holdout_timestamps),
        rebalance_hours,
        execution_lag_bars,
        len(eval_timestamps),
        split_ts,
    )

    if len(X_holdout) == 0:
        logger.error("Holdout set is empty — check --days and --holdout-ratio")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    model_path = args.model_path
    # Only auto-redirect to long model when user didn't explicitly set --model-path
    user_set_model_path = any(
        a in sys.argv for a in ("--model-path",)
    ) or any(a.startswith("--model-path=") for a in sys.argv)
    if not user_set_model_path and args.side == "long":
        model_path = config.LongModel.MODEL_PATH
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)

    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.error("Train a model first (e.g. python scripts/train_xgb.py) then re-run.")
        sys.exit(1)

    logger.info(f"Loading model from {model_path}...")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Predict on holdout
    # ------------------------------------------------------------------
    logger.info("Predicting on holdout set...")
    try:
        scores = model.predict(X_holdout)
    except Exception as e:
        logger.error(f"model.predict() failed: {e}")
        sys.exit(1)

    scores_series = pd.Series(scores, index=X_holdout.index, name="score")

    # ------------------------------------------------------------------
    # 4. Compute IC per timestamp
    # ------------------------------------------------------------------
    logger.info("Computing per-timestamp Spearman IC...")
    ic_values      = {}
    longshort_hits = []  # 1 if long > short, 0 otherwise
    gross_spreads  = []
    net_spreads    = []
    turnovers      = []
    period_rows    = []
    prev_longs     = set()
    prev_shorts    = set()
    fee_total      = (args.fee_bps + args.slippage_bps) / 10000
    short_extra_cost = args.short_extra_cost_bps / 10000

    for ts in eval_timestamps:
        try:
            sc_ts   = scores_series.xs(ts, level="timestamp")
        except KeyError:
            continue

        if ts not in eval_target_wide.index:
            continue
        actual_target_ts = eval_target_wide.loc[ts].dropna()

        combined = pd.concat([sc_ts, actual_target_ts], axis=1).dropna()
        combined.columns = ["score", "target"]

        if len(combined) < max(5, args.long_n + args.short_n):
            continue

        ic, _ = spearmanr(combined["score"], combined["target"])
        ic_values[ts] = ic

        # Long/Short hit rate
        sorted_scores = combined["score"].sort_values(ascending=False)
        score_diag = compute_score_diagnostics(sorted_scores, args.long_n, args.short_n)
        longs, shorts = select_positions_with_buffer(
            sorted_scores,
            long_n=args.long_n,
            short_n=args.short_n,
            prev_longs=prev_longs,
            prev_shorts=prev_shorts,
            rebal_buffer=args.rebal_buffer,
        )
        long_idx = list(longs.index)
        short_idx = list(shorts.index)

        long_ret = combined.loc[long_idx, "target"].mean() if long_idx else 0.0
        short_ret = combined.loc[short_idx, "target"].mean() if short_idx else 0.0

        if long_idx and short_idx:
            longshort_hits.append(1 if long_ret > short_ret else 0)
        elif short_idx:
            longshort_hits.append(1 if short_ret < 0 else 0)
        elif long_idx:
            longshort_hits.append(1 if long_ret > 0 else 0)

        if ts not in eval_fwd_returns.index:
            continue
        period_returns = eval_fwd_returns.loc[ts]
        gross_long = period_returns.reindex(long_idx).dropna().mean() if len(long_idx) > 0 else 0.0
        gross_short = period_returns.reindex(short_idx).dropna().mean() if len(short_idx) > 0 else 0.0
        gross_spread = gross_long - gross_short
        if pd.isna(gross_spread):
            continue

        current_longs = set(long_idx)
        current_shorts = set(short_idx)
        long_turnover = len(current_longs - prev_longs) / max(len(current_longs), 1)
        short_turnover = len(current_shorts - prev_shorts) / max(len(current_shorts), 1)
        turnover = (long_turnover + short_turnover) / 2
        cost = turnover * 2 * fee_total
        short_cost = short_extra_cost if short_idx is not None and len(short_idx) > 0 else 0.0
        net_spread = gross_spread - cost - short_cost

        gross_spreads.append(float(gross_spread))
        net_spreads.append(float(net_spread))
        turnovers.append(float(turnover))
        ctx = btc_context.loc[ts] if ts in btc_context.index else pd.Series(dtype=float)
        period_rows.append(
            {
                "timestamp": ts,
                "ic": float(ic),
                "gross_spread": float(gross_spread),
                "net_spread": float(net_spread),
                "turnover": float(turnover),
                "long_ret": float(gross_long),
                "short_ret": float(gross_short),
                "short_extra_cost": float(short_cost),
                "regime": ctx.get("regime", "unknown"),
                "btc_ret_7d": ctx.get("btc_ret_7d", float("nan")),
                "btc_ret_30d": ctx.get("btc_ret_30d", float("nan")),
                "btc_vol_7d": ctx.get("btc_vol_7d", float("nan")),
                **score_diag,
            }
        )
        prev_longs = current_longs
        prev_shorts = current_shorts

    ic_series = pd.Series(ic_values, name="holdout_IC")
    period_df = pd.DataFrame(period_rows).set_index("timestamp") if period_rows else pd.DataFrame()

    if len(ic_series) == 0:
        logger.error("IC series is empty — no valid timestamps in holdout.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Summarize
    # ------------------------------------------------------------------
    n         = len(ic_series.dropna())
    mean_ic   = float(ic_series.mean())
    ic_std    = float(ic_series.std())
    t_stat    = mean_ic / (ic_std / sqrt(n)) if n > 1 and ic_std > 0 else 0.0
    ic_ir     = mean_ic / ic_std if ic_std > 0 else 0.0
    ls_hitrate = float(np.mean(longshort_hits)) if longshort_hits else float("nan")
    mean_gross_spread = float(np.mean(gross_spreads)) if gross_spreads else float("nan")
    mean_net_spread = float(np.mean(net_spreads)) if net_spreads else float("nan")
    avg_turnover = float(np.mean(turnovers)) if turnovers else float("nan")
    mean_short_extra_cost = float(period_df["short_extra_cost"].mean()) if not period_df.empty else float("nan")
    mean_score_std = float(period_df["score_std"].mean()) if not period_df.empty else float("nan")
    median_top_bottom_gap = float(period_df["top_bottom_gap"].median()) if not period_df.empty else float("nan")
    median_top_bottom_gap_z = float(period_df["top_bottom_gap_z"].median()) if not period_df.empty else float("nan")
    median_long_cut_gap = float(period_df["long_cut_gap"].median()) if not period_df.empty else float("nan")
    median_short_cut_gap = float(period_df["short_cut_gap"].median()) if not period_df.empty else float("nan")
    btc_30d_min = float(period_df["btc_ret_30d"].min()) if not period_df.empty else float("nan")
    btc_30d_max = float(period_df["btc_ret_30d"].max()) if not period_df.empty else float("nan")
    strong_trend_share = float((period_df["btc_ret_30d"].abs() >= 0.30).mean()) if not period_df.empty else float("nan")

    # Prediction sanity
    pred_mean = float(scores_series.mean())
    pred_std  = float(scores_series.std())

    # ------------------------------------------------------------------
    # 6. Print table
    # ------------------------------------------------------------------
    gate_ic  = mean_ic > IC_GATE
    gate_t   = t_stat  > TST_GATE
    gate_std = pred_std >= 0.0005
    gate_mean = abs(pred_mean) <= 0.10
    gate_pass = gate_ic and gate_t and gate_std and gate_mean

    print()
    print("=" * 65)
    print(
        f"Phase 2 Holdout Evaluation  (split: {split_ts}, horizon: {horizon}h, "
        f"rebalance: {rebalance_hours}h, exec lag: {execution_lag_bars}, target: {args.target})"
    )
    print("=" * 65)
    print(f"  {'Metric':<30} {'Value':>10}  {'Gate':>6}")
    print("-" * 65)
    print(f"  {'Holdout timestamps':<30} {n:>10}")
    print(f"  {'Mean IC':<30} {mean_ic:>+10.4f}  {'PASS' if gate_ic else 'FAIL':>6}  (>{IC_GATE})")
    print(f"  {'IC Std':<30} {ic_std:>10.4f}")
    print(f"  {'t-stat':<30} {t_stat:>+10.3f}  {'PASS' if gate_t else 'FAIL':>6}  (>{TST_GATE})")
    print(f"  {'IC IR':<30} {ic_ir:>+10.3f}")
    print(f"  {'Long/Short hit rate':<30} {ls_hitrate:>10.1%}  {'(informational)':>6}")
    print(f"  {'Long / Short count':<30} {f'{args.long_n}/{args.short_n}':>10}  {'(info)':>6}")
    print(f"  {'Rebal buffer':<30} {args.rebal_buffer:>10}  {'(info)':>6}")
    print(f"  {'Gross spread / period':<30} {mean_gross_spread:>+10.4f}  {'(info)':>6}")
    print(f"  {'Net spread / period':<30} {mean_net_spread:>+10.4f}  {'(info)':>6}")
    print(f"  {'Avg turnover':<30} {avg_turnover:>10.1%}  {'(info)':>6}")
    print(f"  {'Short extra cost / period':<30} {mean_short_extra_cost:>+10.4f}  {'(info)':>6}")
    print(f"  {'Mean xs score std':<30} {mean_score_std:>10.4f}  {'(info)':>6}")
    print(f"  {'Median top-bottom gap':<30} {median_top_bottom_gap:>+10.4f}  {'(info)':>6}")
    print(f"  {'Median top-bottom gap z':<30} {median_top_bottom_gap_z:>+10.3f}  {'(info)':>6}")
    print(f"  {'Median long cut gap':<30} {median_long_cut_gap:>+10.5f}  {'(info)':>6}")
    print(f"  {'Median short cut gap':<30} {median_short_cut_gap:>+10.5f}  {'(info)':>6}")
    print(f"  {'BTC 30d ret range':<30} {btc_30d_min:>+4.1%} ~ {btc_30d_max:+4.1%}")
    print(f"  {'|BTC 30d ret| >= 30%':<30} {strong_trend_share:>10.1%}  {'(info)':>6}")
    print(f"  {'Pred mean':<30} {pred_mean:>+10.4f}")
    print(f"  {'Pred std':<30} {pred_std:>10.4f}")
    print("=" * 65)

    if gate_pass:
        print(f"  PHASE 2 GATE PASSED: IC={mean_ic:+.4f} > {IC_GATE}, t={t_stat:+.2f} > {TST_GATE}")
        print(f"  -> Ready for Phase 3 (live deployment)")
    else:
        reasons = []
        if not gate_ic:
            reasons.append(f"IC {mean_ic:+.4f} <= {IC_GATE}")
        if not gate_t:
            reasons.append(f"t-stat {t_stat:+.2f} <= {TST_GATE}")
        if not gate_std:
            reasons.append(f"pred std {pred_std:.4f} < 0.0005")
        if not gate_mean:
            reasons.append(f"|pred mean| {abs(pred_mean):.4f} > 0.10")
        print(f"  PHASE 2 GATE FAILED: {', '.join(reasons)}")
        print(f"  -> Revisit feature engineering or increase training data")
    print("=" * 65)
    print()

    if not period_df.empty:
        print("Regime Slice (BTC 7d sign x 7d vol median):")
        print(f"{'Regime':<18} {'n':>5} {'IC':>8} {'gross':>9} {'net':>9} {'hit':>8}")
        print("-" * 65)
        regime_summary = (
            period_df.groupby("regime")
            .agg(
                n=("ic", "size"),
                mean_ic=("ic", "mean"),
                gross=("gross_spread", "mean"),
                net=("net_spread", "mean"),
                hit=("gross_spread", lambda s: float((s > 0).mean())),
            )
            .sort_values("n", ascending=False)
        )
        for regime, row in regime_summary.iterrows():
            print(
                f"{regime:<18} {int(row['n']):>5} {row['mean_ic']:>+8.4f} "
                f"{row['gross']:>+9.4f} {row['net']:>+9.4f} {row['hit']:>8.1%}"
            )
        print("-" * 65)

    # Prediction std sanity follows the current ranking-model contract:
    # cross-section std around 0.001~0.01 can be valid when IC is alive.
    if pred_std < 0.0005:
        logger.error(f"Prediction std={pred_std:.4f} < 0.0005 — zero-signal risk; stop and investigate")
    elif pred_std < 0.001:
        logger.warning(f"Prediction std={pred_std:.4f} < 0.001 — below normal ranking-model range")
    elif pred_std > 0.50:
        logger.warning(f"Prediction std={pred_std:.4f} > 0.50 — above normal ranking-model range")

    sys.exit(0 if gate_pass else 1)


def _select_rebalance_timestamps(
    timestamps: Iterable[pd.Timestamp],
    rebalance_hours: int,
) -> list[pd.Timestamp]:
    """Pick timestamps spaced by at least rebalance_hours for evaluation."""
    ordered = sorted(pd.to_datetime(list(timestamps)))
    if rebalance_hours <= 1 or not ordered:
        return ordered

    selected = [ordered[0]]
    min_gap = pd.Timedelta(hours=rebalance_hours)
    for ts in ordered[1:]:
        if ts - selected[-1] >= min_gap:
            selected.append(ts)
    return selected


if __name__ == "__main__":
    main()
