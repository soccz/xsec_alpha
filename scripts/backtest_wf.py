#!/usr/bin/env python3
"""
Walk-forward backtest: retrain model every `retrain_days`, test on next window.

No data snooping: each test period only uses model trained on prior data.
Factor design bias still exists (factors were chosen by looking at full data),
but at least model coefficients are truly out-of-sample.

Usage:
    python scripts/backtest_wf.py [--days 180] [--train-days 60] [--test-days 14] [--fee-bps 5]
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
from utils.eval_metrics import build_btc_context, compute_score_diagnostics, select_positions_with_buffer
from utils.logger import logger
from data.features import (
    load_and_pivot,
    load_binance_pivot,
    compute_factors,
    compute_forward_returns,
    compute_residual_returns,
    crosssection_zscore,
    CALENDAR_COLS,
    build_top_liquidity_universe_index,
    filter_long_frame_by_universe,
    filter_long_series_by_universe,
)
from models.xgb_ranker import EnsembleRanker, LGBMRanker, RidgeRanker, XSecRanker


def build_model(model_type: str):
    """Match the production training model family used by scripts/train.py."""
    if model_type == "ensemble":
        return EnsembleRanker()
    if model_type == "lgbm":
        return LGBMRanker(min_child_samples=50)
    if model_type == "xgb":
        return XSecRanker(
            n_estimators=config.Model.XGB_N_ESTIMATORS,
            max_depth=config.Model.XGB_MAX_DEPTH,
            learning_rate=config.Model.XGB_LEARNING_RATE,
            subsample=config.Model.XGB_SUBSAMPLE,
            colsample_bytree=config.Model.XGB_COLSAMPLE_BYTREE,
            min_child_weight=config.Model.XGB_MIN_CHILD_WEIGHT,
            random_state=config.Model.XGB_RANDOM_STATE,
        )
    model = RidgeRanker(alpha=1.0)
    return model


def _append_position_rows(
    position_rows: list[dict[str, object]],
    *,
    timestamp,
    retrain_id: int,
    regime: str,
    side: str,
    positions: pd.Series,
    realized_returns: pd.Series,
    shared_cost: float,
    shared_short_cost: float = 0.0,
):
    """Store per-position realized return and attributed contribution."""
    if positions.empty:
        return

    realized = realized_returns.reindex(positions.index)
    valid_count = int(realized.dropna().shape[0])
    if valid_count <= 0:
        return

    gross_sign = 1.0 if side == "long" else -1.0
    cost_share = shared_cost / valid_count
    short_cost_share = shared_short_cost / valid_count if side == "short" else 0.0

    for rank, (market, score) in enumerate(positions.items(), start=1):
        raw_ret = realized.get(market, np.nan)
        if pd.isna(raw_ret):
            continue
        gross_contrib = gross_sign * float(raw_ret) / valid_count
        net_contrib = gross_contrib - cost_share - short_cost_share
        position_rows.append(
            {
                "timestamp": timestamp,
                "retrain_id": retrain_id,
                "regime": regime,
                "side": side,
                "rank": rank,
                "market": market,
                "score": float(score),
                "raw_return": float(raw_ret),
                "gross_contribution": float(gross_contrib),
                "net_contribution": float(net_contrib),
                "cost_share": float(cost_share),
                "short_extra_cost_share": float(short_cost_share),
            }
        )


def _build_position_summary(position_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated-name dependence for audit."""
    if position_df.empty:
        return pd.DataFrame()

    grouped = (
        position_df.groupby(["side", "market"], as_index=False)
        .agg(
            periods_selected=("timestamp", "size"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            mean_rank=("rank", "mean"),
            mean_score=("score", "mean"),
            mean_raw_return=("raw_return", "mean"),
            worst_raw_return=("raw_return", "max"),
            best_raw_return=("raw_return", "min"),
            gross_contribution_sum=("gross_contribution", "sum"),
            net_contribution_sum=("net_contribution", "sum"),
            positive_periods=("gross_contribution", lambda s: int((s > 0).sum())),
        )
    )

    side_totals = (
        grouped.groupby("side")[["gross_contribution_sum", "net_contribution_sum"]]
        .sum()
        .rename(
            columns={
                "gross_contribution_sum": "side_gross_total",
                "net_contribution_sum": "side_net_total",
            }
        )
        .reset_index()
    )
    grouped = grouped.merge(side_totals, on="side", how="left")
    grouped["gross_share_of_side"] = grouped["gross_contribution_sum"] / grouped["side_gross_total"].replace(0, np.nan)
    grouped["net_share_of_side"] = grouped["net_contribution_sum"] / grouped["side_net_total"].replace(0, np.nan)
    grouped["positive_rate"] = grouped["positive_periods"] / grouped["periods_selected"].replace(0, np.nan)

    return grouped.sort_values(["side", "net_contribution_sum"], ascending=[True, False])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180, help="Total data days to load")
    parser.add_argument("--train-days", type=int, default=60, help="Rolling train window (days)")
    parser.add_argument("--test-days", type=int, default=14, help="Test window before retrain (days)")
    parser.add_argument("--fee-bps", type=float, default=5)
    parser.add_argument("--long-n", type=int, default=20)
    parser.add_argument("--short-n", type=int, default=20)
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["ridge", "lgbm", "xgb", "ensemble"],
        default="ensemble",
        help="Model family used for each retrain window (default: ensemble)",
    )
    parser.add_argument("--rebalance-hours", type=int, default=config.Data.PREDICT_HORIZON)
    parser.add_argument("--execution-lag-bars", type=int, default=0, help="Delay execution by N bars: score_t -> enter at t+N")
    parser.add_argument("--rebal-buffer", type=int, default=config.Portfolio.REBAL_BUFFER, help="Keep prior positions if they remain within N+buffer")
    parser.add_argument("--short-extra-cost-bps", type=float, default=10.0, help="Additional per-period cost charged to the short leg (research baseline: 10bps)")
    args = parser.parse_args()

    fee = args.fee_bps / 10000
    horizon = args.rebalance_hours
    execution_lag_bars = max(0, args.execution_lag_bars)
    short_extra_cost = args.short_extra_cost_bps / 10000
    train_hours = args.train_days * 24
    test_hours  = args.test_days * 24

    logger.info(
        "=== WALK-FORWARD BACKTEST | model=%s days=%s train=%sd test=%sd horizon=%sh lag=%s ===",
        args.model_type,
        args.days,
        args.train_days,
        args.test_days,
        horizon,
        execution_lag_bars,
    )

    # 1. Load all data
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

    # 2. Compute factors (once — factor formulas are fixed, only model weights change)
    factor_df = compute_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
    zscore_cols = [c for c in factor_df.columns if c not in CALENDAR_COLS]
    factor_df = crosssection_zscore(factor_df, cols=zscore_cols)

    # 3. Compute residual returns (for training target)
    beta_window = config.Data.BETA_ROLLING_WINDOW
    residuals_wide = compute_residual_returns(closes, horizon=horizon, beta_window=beta_window)
    residuals_long = residuals_wide.stack(future_stack=True)
    residuals_long.index.names = ["timestamp", "market"]
    top_n = getattr(config.Data, "LIQUIDITY_TOP_N", 0)
    selected_index, coverage = build_top_liquidity_universe_index(closes, volumes, top_n=top_n)
    factor_df = filter_long_frame_by_universe(factor_df, selected_index)
    residuals_long = filter_long_series_by_universe(residuals_long, selected_index)
    logger.info(
        "Active universe: top %s by %sh traded value (avg selected=%.1f, min=%s, max=%s)",
        top_n,
        getattr(config.Data, "LIQUIDITY_LOOKBACK_HOURS", 24),
        coverage["avg_selected"],
        coverage["min_selected"],
        coverage["max_selected"],
    )

    # 4. Execution-aligned returns (for P&L and lagged IC checks)
    eval_prices = opens if execution_lag_bars > 0 else closes
    eval_fwd_returns = compute_forward_returns(
        eval_prices,
        horizon=horizon,
        execution_lag=execution_lag_bars,
    )
    eval_residuals_wide = compute_residual_returns(
        eval_prices,
        horizon=horizon,
        beta_window=beta_window,
        execution_lag=execution_lag_bars,
    )
    btc_context = build_btc_context(closes)

    # 5. Build combined dataset
    combined = factor_df.join(residuals_long.rename("residual"), how="inner").dropna()
    combined = combined[combined["residual"].abs() <= 0.5]
    feature_cols = [c for c in factor_df.columns.tolist()]

    all_ts = combined.index.get_level_values("timestamp").unique().sort_values()
    logger.info(f"Total timestamps: {len(all_ts)} ({len(all_ts)/24:.0f} days)")

    # 6. Walk-forward loop
    min_start = train_hours  # need at least train_hours of history
    pnl_rows = []
    ic_rows = []
    position_rows = []
    n_retrains = 0
    prev_longs = set()
    prev_shorts = set()

    cursor = min_start  # index into all_ts
    while cursor + test_hours <= len(all_ts):
        # Train window
        train_end_idx = cursor
        train_start_idx = max(0, cursor - train_hours)
        train_ts = all_ts[train_start_idx:train_end_idx]

        # Test window
        test_start_idx = cursor
        test_end_idx = min(cursor + test_hours, len(all_ts))
        test_ts = all_ts[test_start_idx:test_end_idx]

        # Train data
        train_mask = combined.index.get_level_values("timestamp").isin(train_ts)
        train_data = combined[train_mask]
        X_train = train_data[feature_cols]
        y_train = train_data["residual"]

        if len(X_train) < 100:
            cursor += test_hours
            continue

        # Train model
        model = build_model(args.model_type)
        model.fit(X_train, y_train)
        n_retrains += 1

        # Test: simulate trading on each rebalance point in test window
        rebalance_points = test_ts[::horizon]
        for ts in rebalance_points:
            try:
                snapshot = factor_df.xs(ts, level="timestamp").dropna()
            except KeyError:
                continue
            if len(snapshot) < max(5, args.long_n + args.short_n):
                continue

            scores = model.predict(snapshot[feature_cols])
            score_series = pd.Series(scores, index=snapshot.index).sort_values(ascending=False)
            score_diag = compute_score_diagnostics(score_series, args.long_n, args.short_n)
            longs, shorts = select_positions_with_buffer(
                score_series,
                long_n=args.long_n,
                short_n=args.short_n,
                prev_longs=prev_longs,
                prev_shorts=prev_shorts,
                rebal_buffer=args.rebal_buffer,
            )
            long_coins = set(longs.index)
            short_coins = set(shorts.index)

            if ts not in eval_fwd_returns.index:
                continue
            period_ret = eval_fwd_returns.loc[ts]
            long_realized = period_ret.reindex(list(long_coins)).dropna() if long_coins else pd.Series(dtype=float)
            short_realized = period_ret.reindex(list(short_coins)).dropna() if short_coins else pd.Series(dtype=float)

            long_ret = long_realized.mean() if not long_realized.empty else 0.0
            short_ret = short_realized.mean() if not short_realized.empty else 0.0
            ls_gross = long_ret - short_ret

            if pd.isna(ls_gross):
                continue

            # Turnover
            lt = len(long_coins - prev_longs) / max(len(long_coins), 1)
            st = len(short_coins - prev_shorts) / max(len(short_coins), 1)
            turnover = (lt + st) / 2
            cost = turnover * 2 * fee
            short_cost = short_extra_cost if short_coins else 0.0
            ls_net = ls_gross - cost - short_cost

            # IC for this period
            ic_val = float("nan")
            if ts in eval_residuals_wide.index:
                actual_res = eval_residuals_wide.loc[ts]
                common = snapshot.index.intersection(actual_res.dropna().index)
                if len(common) > 5:
                    ic_val, _ = spearmanr(
                        pd.Series(scores, index=snapshot.index)[common],
                        actual_res[common]
                    )
                    ic_rows.append({"timestamp": ts, "ic": float(ic_val)})

            ctx = btc_context.loc[ts] if ts in btc_context.index else pd.Series(dtype=float)
            regime = ctx.get("regime", "unknown")

            _append_position_rows(
                position_rows,
                timestamp=ts,
                retrain_id=n_retrains,
                regime=regime,
                side="long",
                positions=longs,
                realized_returns=long_realized,
                shared_cost=cost,
            )
            _append_position_rows(
                position_rows,
                timestamp=ts,
                retrain_id=n_retrains,
                regime=regime,
                side="short",
                positions=shorts,
                realized_returns=short_realized,
                shared_cost=cost,
                shared_short_cost=short_cost,
            )

            pnl_rows.append({
                "timestamp": ts,
                "long_ret": float(long_ret),
                "short_ret": float(short_ret),
                "ls_gross": float(ls_gross),
                "ls_net": float(ls_net),
                "turnover": float(turnover),
                "cost": float(cost),
                "short_extra_cost": float(short_cost),
                "ic": float(ic_val),
                "regime": regime,
                "btc_ret_7d": ctx.get("btc_ret_7d", float("nan")),
                "btc_ret_30d": ctx.get("btc_ret_30d", float("nan")),
                "btc_vol_7d": ctx.get("btc_vol_7d", float("nan")),
                **score_diag,
                "retrain_id": n_retrains,
                "long_names": json.dumps(sorted(long_coins)),
                "short_names": json.dumps(sorted(short_coins)),
            })

            prev_longs = long_coins
            prev_shorts = short_coins

        cursor += test_hours

    if not pnl_rows:
        print("No walk-forward periods available. Need more data.")
        sys.exit(1)

    df = pd.DataFrame(pnl_rows).set_index("timestamp")
    ic_df = pd.DataFrame(ic_rows).set_index("timestamp") if ic_rows else pd.DataFrame()
    position_df = pd.DataFrame(position_rows)
    position_summary_df = _build_position_summary(position_df)

    # 7. Stats
    n = len(df)
    periods_per_year = 365 * 24 / horizon
    ann = np.sqrt(periods_per_year)

    sharpe_gross = df["ls_gross"].mean() / df["ls_gross"].std() * ann if df["ls_gross"].std() > 0 else 0
    sharpe_net   = df["ls_net"].mean()   / df["ls_net"].std()   * ann if df["ls_net"].std() > 0 else 0

    cum_net = (1 + df["ls_net"]).cumprod()
    total_net = cum_net.iloc[-1] - 1
    max_dd = (cum_net / cum_net.cummax() - 1).min()
    win_rate = (df["ls_net"] > 0).mean()

    mean_ic = df["ic"].mean() if not df["ic"].dropna().empty else float("nan")
    ic_std  = df["ic"].std() if not df["ic"].dropna().empty else float("nan")
    mean_score_std = float(df["score_std"].mean()) if "score_std" in df.columns else float("nan")
    median_top_bottom_gap = float(df["top_bottom_gap"].median()) if "top_bottom_gap" in df.columns else float("nan")
    median_top_bottom_gap_z = float(df["top_bottom_gap_z"].median()) if "top_bottom_gap_z" in df.columns else float("nan")
    median_long_cut_gap = float(df["long_cut_gap"].median()) if "long_cut_gap" in df.columns else float("nan")
    median_short_cut_gap = float(df["short_cut_gap"].median()) if "short_cut_gap" in df.columns else float("nan")
    btc_30d_min = float(df["btc_ret_30d"].min()) if "btc_ret_30d" in df.columns else float("nan")
    btc_30d_max = float(df["btc_ret_30d"].max()) if "btc_ret_30d" in df.columns else float("nan")
    strong_trend_share = float((df["btc_ret_30d"].abs() >= 0.30).mean()) if "btc_ret_30d" in df.columns else float("nan")

    if sharpe_net > 0.5:
        status = "passed"
    elif sharpe_net > 0:
        status = "marginal"
    else:
        status = "failed"

    # 8. Print
    print("\n" + "=" * 70)
    print(f"WALK-FORWARD BACKTEST | {df.index[0]} → {df.index[-1]}")
    print(
        f"  Model: {args.model_type} | Train: {args.train_days}d rolling | "
        f"Test: {args.test_days}d | Retrain count: {n_retrains}"
    )
    print(
        f"  Rebalance: {horizon}h | Exec lag: {execution_lag_bars} | "
        f"Long: {args.long_n} | Short: {args.short_n} | Buffer: {args.rebal_buffer} | Fee: {args.fee_bps}bps"
    )
    print("=" * 70)
    print(f"  Periods:             {n}")
    print(f"  Total return (net):  {total_net:+.2%}")
    print(f"  Sharpe (gross):      {sharpe_gross:+.2f}")
    print(f"  Sharpe (net):        {sharpe_net:+.2f}")
    print(f"  Max drawdown:        {max_dd:+.2%}")
    print(f"  Win rate:            {win_rate:.1%}")
    print(f"  Avg turnover:        {df['turnover'].mean():.1%}")
    print(f"  Total cost:          {df['cost'].sum():.2%}")
    print(f"  Short extra cost:    {df['short_extra_cost'].sum():.2%}")
    print(f"  Mean OOS IC:         {mean_ic:+.4f}")
    print(f"  IC std:              {ic_std:.4f}")
    print("-" * 70)
    print(f"  Avg L-S spread/period: {df['ls_gross'].mean():+.4f}")
    print(f"  Avg long/period:       {df['long_ret'].mean():+.4f}")
    print(f"  Avg short/period:      {df['short_ret'].mean():+.4f}")
    print(f"  Mean xs score std:     {mean_score_std:.4f}")
    print(f"  Median top-bottom gap: {median_top_bottom_gap:+.4f}")
    print(f"  Median top-bottom z:   {median_top_bottom_gap_z:+.3f}")
    print(f"  Median long cut gap:   {median_long_cut_gap:+.5f}")
    print(f"  Median short cut gap:  {median_short_cut_gap:+.5f}")
    print(f"  BTC 30d ret range:     {btc_30d_min:+.1%} ~ {btc_30d_max:+.1%}")
    print(f"  |BTC 30d ret|>=30%:    {strong_trend_share:.1%}")

    # Per-retrain window stats
    print("-" * 70)
    print("  Per-retrain window:")
    for rid, grp in df.groupby("retrain_id"):
        g_net = grp["ls_net"]
        s = g_net.mean() / g_net.std() * ann if g_net.std() > 0 else 0
        print(f"    Window {rid}: {len(grp)} periods, ret={g_net.sum():+.2%}, Sharpe={s:+.1f}")
    print("-" * 70)
    print("  Regime Slice:")
    regime_summary = (
        df.groupby("regime")
        .agg(
            n=("ls_net", "size"),
            mean_ic=("ic", "mean"),
            gross=("ls_gross", "mean"),
            net=("ls_net", "mean"),
            hit=("ls_net", lambda s: float((s > 0).mean())),
        )
        .sort_values("n", ascending=False)
    )
    for regime, row in regime_summary.iterrows():
        print(
            f"    {regime:<14} n={int(row['n']):>4} "
            f"IC={row['mean_ic']:+.4f} gross={row['gross']:+.4f} "
            f"net={row['net']:+.4f} hit={row['hit']:.1%}"
        )

    if not position_summary_df.empty:
        print("-" * 70)
        print("  Top Name Dependence:")
        for side in ["long", "short"]:
            side_df = position_summary_df[position_summary_df["side"] == side].head(5)
            if side_df.empty:
                continue
            print(f"    {side}:")
            for _, row in side_df.iterrows():
                print(
                    f"      {row['market']:<12} periods={int(row['periods_selected']):>4} "
                    f"net_sum={row['net_contribution_sum']:+.4f} "
                    f"net_share={row['net_share_of_side']:.1%} "
                    f"worst_raw={row['worst_raw_return']:+.2%}"
                )

    print("=" * 70)
    if status == "passed":
        print(f"  WALK-FORWARD PASSED: Sharpe(net)={sharpe_net:.2f}")
    elif status == "marginal":
        print(f"  WALK-FORWARD MARGINAL: Sharpe(net)={sharpe_net:.2f}")
    else:
        print(f"  WALK-FORWARD FAILED: Sharpe(net)={sharpe_net:.2f}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "backtest_wf.csv")
    df.to_csv(out_path)
    positions_path = os.path.join(out_dir, "backtest_wf_positions.csv")
    position_df.to_csv(positions_path, index=False)
    position_summary_path = os.path.join(out_dir, "backtest_wf_position_summary.csv")
    position_summary_df.to_csv(position_summary_path, index=False)
    summary_path = os.path.join(out_dir, "backtest_wf_summary.json")

    top_name_dependence = {}
    if not position_summary_df.empty:
        for side in ["long", "short"]:
            side_df = position_summary_df[position_summary_df["side"] == side]
            if side_df.empty:
                continue
            top_name_dependence[side] = {
                "distinct_names": int(side_df["market"].nunique()),
                "top1_net_share": float(side_df.iloc[0]["net_share_of_side"]),
                "top3_net_share": float(side_df.head(3)["net_share_of_side"].sum()),
                "top5_net_share": float(side_df.head(5)["net_share_of_side"].sum()),
            }

    summary = {
        "status": status,
        "model_type": args.model_type,
        "days": args.days,
        "train_days": args.train_days,
        "test_days": args.test_days,
        "rebalance_hours": horizon,
        "execution_lag_bars": execution_lag_bars,
        "rebal_buffer": args.rebal_buffer,
        "long_n": args.long_n,
        "short_n": args.short_n,
        "fee_bps": args.fee_bps,
        "short_extra_cost_bps": args.short_extra_cost_bps,
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "periods": int(n),
        "retrain_count": int(n_retrains),
        "feature_count": int(len(feature_cols)),
        "features": feature_cols,
        "avg_ls_gross": float(df["ls_gross"].mean()),
        "avg_long_ret": float(df["long_ret"].mean()),
        "avg_short_ret": float(df["short_ret"].mean()),
        "avg_turnover": float(df["turnover"].mean()),
        "total_cost": float(df["cost"].sum()),
        "total_short_extra_cost": float(df["short_extra_cost"].sum()),
        "total_net": float(total_net),
        "sharpe_gross": float(sharpe_gross),
        "sharpe_net": float(sharpe_net),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "mean_ic": None if np.isnan(mean_ic) else float(mean_ic),
        "ic_std": None if np.isnan(ic_std) else float(ic_std),
        "mean_score_std": None if np.isnan(mean_score_std) else float(mean_score_std),
        "median_top_bottom_gap": None if np.isnan(median_top_bottom_gap) else float(median_top_bottom_gap),
        "median_top_bottom_gap_z": None if np.isnan(median_top_bottom_gap_z) else float(median_top_bottom_gap_z),
        "median_long_cut_gap": None if np.isnan(median_long_cut_gap) else float(median_long_cut_gap),
        "median_short_cut_gap": None if np.isnan(median_short_cut_gap) else float(median_short_cut_gap),
        "btc_30d_ret_min": None if np.isnan(btc_30d_min) else float(btc_30d_min),
        "btc_30d_ret_max": None if np.isnan(btc_30d_max) else float(btc_30d_max),
        "strong_trend_share": None if np.isnan(strong_trend_share) else float(strong_trend_share),
        "position_artifacts": {
            "period_path": positions_path,
            "summary_path": position_summary_path,
        },
        "top_name_dependence": top_name_dependence,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results: {out_path}")
    print(f"  Positions: {positions_path}")
    print(f"  Position summary: {position_summary_path}")
    print(f"  Summary: {summary_path}")

    sys.exit(0 if status == "passed" else 1)


if __name__ == "__main__":
    main()
