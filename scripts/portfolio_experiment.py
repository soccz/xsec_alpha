#!/usr/bin/env python3
"""Portfolio-level experiment harness for xsec_alpha.

This script compares portfolio construction ideas on the same timestamp grid,
using the current production models and the F1 unified factor library. It is a
research harness, not a live recommendation path.

The unit of observation is one rebalance window, not one coin pick. This keeps
the evaluation closer to a real portfolio process: five names opened at the
same timestamp share the same market regime and should not be counted as five
independent experiments.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import config
from data.features import (
    UNIFIED_CALENDAR_COLS,
    build_top_liquidity_universe_index,
    compute_all_betas,
    compute_forward_returns,
    compute_unified_factors,
    crosssection_zscore,
    filter_long_frame_by_universe,
    load_and_pivot,
    load_binance_pivot,
)
from models.xgb_ranker import XSecRanker
from utils.eval_metrics import build_btc_context, select_positions_with_buffer
from utils.logger import logger
from utils.magnitude import _find_bucket, _load_calibration

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare portfolio construction rules.")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--holdout-ratio", type=float, default=0.35)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--short-n", type=int, default=5)
    p.add_argument("--long-n", type=int, default=5)
    p.add_argument("--rebalance-hours", type=int, default=None)
    p.add_argument("--fee-bps", type=float, default=6.0, help="One-way fee in bps")
    p.add_argument("--slippage-bps", type=float, default=4.0, help="One-way slippage in bps")
    p.add_argument("--short-extra-cost-bps", type=float, default=10.0,
                   help="Per-window funding/borrow cost for any active short basket")
    p.add_argument("--quality-entry-sigma", type=float, default=1.5)
    p.add_argument("--quality-stay-sigma", type=float, default=1.1)
    p.add_argument("--quality-min-net-bps", type=float, default=20.0,
                   help="Minimum expected net edge for new quality entries")
    p.add_argument("--no-bitget-filter", action="store_true")
    p.add_argument("--out-prefix", default="portfolio_experiment")
    return p.parse_args()


def _load_tradable_markets(markets: list[str]) -> set[str]:
    cache_path = OUTPUT_DIR / "bitget_usdt_contracts.json"
    bases: set[str] = set()
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text()).get("payload", {})
            for row in payload.get("data", []):
                if row.get("symbolStatus") == "normal" and row.get("baseCoin"):
                    bases.add(row["baseCoin"])
        except Exception as exc:
            logger.warning("Could not read Bitget cache: %s", exc)

    if not bases:
        from utils.bitget import load_bitget_usdt_perp_map
        bases = set(load_bitget_usdt_perp_map().keys())

    return {m for m in markets if m.replace("KRW-", "") in bases}


def _expected_profit_pct(side: str, sigma: float) -> float:
    buckets = _load_calibration().get(side, [])
    bucket = _find_bucket(buckets, float(sigma))
    if not bucket:
        return 0.0
    return float(bucket.get("mean_signed_return_pct") or 0.0)


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def _ann_sharpe(returns: pd.Series, horizon_h: int) -> float | None:
    clean = returns.dropna()
    if len(clean) < 2:
        return None
    std = clean.std(ddof=1)
    if std <= 0 or pd.isna(std):
        return None
    return float(clean.mean() / std * np.sqrt(365 * 24 / horizon_h))


def _tstat(returns: pd.Series) -> float | None:
    clean = returns.dropna()
    if len(clean) < 2:
        return None
    std = clean.std(ddof=1)
    if std <= 0 or pd.isna(std):
        return None
    return float(clean.mean() / (std / np.sqrt(len(clean))))


def _strategy_summary(df: pd.DataFrame, positions: pd.DataFrame, horizon_h: int) -> dict:
    if df.empty:
        return {"periods": 0}
    net = df["net_return"].dropna()
    gross = df["gross_return"].dropna()
    btc = df["btc_return"].reindex(net.index).dropna()
    common = net.index.intersection(btc.index)
    beta_to_btc = None
    if len(common) > 3 and btc.loc[common].var(ddof=1) > 0:
        beta_to_btc = float(net.loc[common].cov(btc.loc[common]) / btc.loc[common].var(ddof=1))

    side_positions = positions[positions["strategy"].eq(df["strategy"].iloc[0])] if not positions.empty else pd.DataFrame()
    top_short_share = None
    distinct_shorts = 0
    if not side_positions.empty:
        shorts = side_positions[side_positions["side"].eq("short")]
        distinct_shorts = int(shorts["market"].nunique())
        contrib = shorts.groupby("market")["gross_contribution"].sum().sort_values(ascending=False)
        total = contrib.sum()
        if total != 0 and len(contrib):
            top_short_share = float(contrib.iloc[0] / total)

    return {
        "periods": int(len(net)),
        "mean_gross": float(gross.mean()) if len(gross) else None,
        "mean_net": float(net.mean()) if len(net) else None,
        "median_net": float(net.median()) if len(net) else None,
        "tstat_window": _tstat(net),
        "sharpe_ann": _ann_sharpe(net, horizon_h),
        "win_rate": float((net > 0).mean()) if len(net) else None,
        "total_compound": float((1.0 + net).prod() - 1.0) if len(net) else None,
        "max_drawdown": _max_drawdown(net),
        "p05": float(net.quantile(0.05)) if len(net) else None,
        "worst": float(net.min()) if len(net) else None,
        "avg_cost": float((df["gross_return"] - df["net_return"]).mean()),
        "avg_turnover": float(df["turnover"].mean()),
        "avg_n_short": float(df["n_short"].mean()),
        "avg_n_long": float(df["n_long"].mean()),
        "beta_to_btc": beta_to_btc,
        "distinct_shorts": distinct_shorts,
        "top_short_gross_share": top_short_share,
    }


def _compute_cost(
    longs: set[str],
    shorts: set[str],
    prev_longs: set[str],
    prev_shorts: set[str],
    one_way_cost: float,
    short_extra_cost: float,
    hedge_beta: float | None = None,
    prev_hedge_beta: float | None = None,
) -> tuple[float, float]:
    long_turnover = len(longs - prev_longs) / max(len(longs), 1) if longs else 0.0
    short_turnover = len(shorts - prev_shorts) / max(len(shorts), 1) if shorts else 0.0
    long_cost = long_turnover * 2.0 * one_way_cost if longs else 0.0
    short_cost = short_turnover * 2.0 * one_way_cost if shorts else 0.0
    if shorts:
        short_cost += short_extra_cost

    hedge_cost = 0.0
    if hedge_beta is not None:
        prev = prev_hedge_beta or 0.0
        hedge_cost = abs(float(hedge_beta) - float(prev)) * 2.0 * one_way_cost

    turnover = (long_turnover + short_turnover) / max((1 if longs else 0) + (1 if shorts else 0), 1)
    return long_cost + short_cost + hedge_cost, turnover


def _record_positions(
    rows: list[dict],
    ts,
    strategy: str,
    side: str,
    positions: pd.Series,
    realized: pd.Series,
) -> None:
    if positions.empty:
        return
    side_sign = 1.0 if side == "long" else -1.0
    valid = realized.reindex(positions.index).dropna()
    n = len(valid)
    if n == 0:
        return
    for rank, (market, score) in enumerate(positions.items(), start=1):
        raw = valid.get(market, np.nan)
        if pd.isna(raw):
            continue
        rows.append({
            "timestamp": ts,
            "strategy": strategy,
            "side": side,
            "rank": rank,
            "market": market,
            "score": float(score),
            "raw_return": float(raw),
            "gross_contribution": float(side_sign * raw / n),
        })


def _quality_shorts(
    candidates: pd.DataFrame,
    prev_shorts: set[str],
    short_n: int,
    entry_sigma: float,
    stay_sigma: float,
    min_net: float,
) -> pd.Series:
    rows = []
    ordered = candidates.sort_values("score", ascending=True)
    for market, row in ordered.iterrows():
        if row["direction"] >= 0:
            continue
        is_prev = market in prev_shorts
        sigma_req = stay_sigma if is_prev else entry_sigma
        net_req = 0.0 if is_prev else min_net
        strong_override = row["sigma"] >= 2.0
        if row["sigma"] < sigma_req:
            continue
        if row["expected_net"] < net_req:
            continue
        if not bool(row["consensus"]) and not strong_override:
            continue
        rows.append((market, row["score"]))
        if len(rows) >= short_n:
            break
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series(dict(rows), dtype=float)


def main() -> int:
    args = parse_args()
    rebalance_h = args.rebalance_hours or args.horizon
    one_way_cost = (args.fee_bps + args.slippage_bps) / 10000.0
    short_extra_cost = args.short_extra_cost_bps / 10000.0
    min_net = args.quality_min_net_bps / 10000.0

    logger.info("=== portfolio_experiment START ===")
    closes, opens, highs, lows, volumes = load_and_pivot(days=args.days)
    warmup = config.Data.MIN_ROWS_PER_COIN
    closes = closes.iloc[warmup:]
    opens = opens.iloc[warmup:]
    highs = highs.iloc[warmup:]
    lows = lows.iloc[warmup:]
    volumes = volumes.iloc[warmup:]

    binance_closes = load_binance_pivot(closes.columns.tolist(), days=args.days)
    if not binance_closes.empty:
        binance_closes = binance_closes.iloc[warmup:]

    factors = compute_unified_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
    zcols = [c for c in factors.columns if c not in UNIFIED_CALENDAR_COLS]
    factors = crosssection_zscore(factors, cols=zcols)
    selected_index, coverage = build_top_liquidity_universe_index(
        closes,
        volumes,
        top_n=getattr(config.Data, "LIQUIDITY_TOP_N", 100),
    )
    factors = filter_long_frame_by_universe(factors, selected_index)
    logger.info(
        "Active universe: avg selected=%.1f, min=%s, max=%s",
        coverage["avg_selected"],
        coverage["min_selected"],
        coverage["max_selected"],
    )

    tradable = set(closes.columns)
    if not args.no_bitget_filter:
        tradable = _load_tradable_markets(closes.columns.tolist())
        logger.info("Bitget tradable filter: %s/%s markets", len(tradable), len(closes.columns))

    short_model = XSecRanker.load(str(ROOT / config.Model.MODEL_PATH))
    long_model = XSecRanker.load(str(ROOT / config.LongModel.MODEL_PATH))
    fwd_returns = compute_forward_returns(closes, horizon=args.horizon)
    btc_context = build_btc_context(closes)
    returns_1h = closes.pct_change(1, fill_method=None)
    betas = compute_all_betas(returns_1h, window=config.Data.BETA_ROLLING_WINDOW)

    all_ts = sorted(factors.index.get_level_values("timestamp").unique())
    holdout_start = int(len(all_ts) * (1.0 - args.holdout_ratio))
    eval_ts = all_ts[holdout_start::rebalance_h]
    logger.info("Evaluation windows: %s (holdout_ratio=%.2f)", len(eval_ts), args.holdout_ratio)

    strategies = [
        "raw_short5_no_buffer",
        "raw_short5_buffer",
        "quality_short5",
        "quality_short5_btc_hedged",
        "long_short5",
    ]
    prev_longs: dict[str, set[str]] = {s: set() for s in strategies}
    prev_shorts: dict[str, set[str]] = {s: set() for s in strategies}
    prev_hedge_beta: dict[str, float] = defaultdict(float)

    period_rows: list[dict] = []
    position_rows: list[dict] = []

    for ts in eval_ts:
        if ts not in fwd_returns.index:
            continue
        try:
            snap = factors.xs(ts, level="timestamp").fillna(0)
        except KeyError:
            continue
        snap = snap[snap.any(axis=1)]
        if len(snap) < max(args.short_n, args.long_n) + 5:
            continue

        if tradable:
            snap = snap.loc[[m for m in snap.index if m in tradable]]
        if len(snap) < max(args.short_n, args.long_n) + 5:
            continue

        score = pd.Series(short_model.predict(snap), index=snap.index, name="score")
        score12 = pd.Series(long_model.predict(snap), index=snap.index, name="score12")
        score_std = score.std()
        if score_std <= 0 or pd.isna(score_std):
            continue
        candidates = pd.DataFrame({"score": score, "score12": score12})
        candidates["direction"] = np.sign(candidates["score"]).astype(int)
        candidates["sigma"] = (candidates["score"].abs() / score_std).astype(float)
        candidates["consensus"] = np.sign(candidates["score"]).eq(np.sign(candidates["score12"]))
        candidates["expected_profit"] = [
            _expected_profit_pct("short_6h", sigma) / 100.0
            for sigma in candidates["sigma"]
        ]
        candidates["expected_net"] = candidates["expected_profit"] - short_extra_cost

        period_ret = fwd_returns.loc[ts].reindex(candidates.index)
        if period_ret.dropna().empty:
            continue
        btc_ret = float(fwd_returns.loc[ts, "KRW-BTC"]) if "KRW-BTC" in fwd_returns.columns else 0.0
        regime = btc_context.loc[ts, "regime"] if ts in btc_context.index else "unknown"
        beta_row = betas.loc[ts].reindex(candidates.index) if ts in betas.index else pd.Series(dtype=float)

        raw_short_no_buffer = score.sort_values(ascending=True).head(args.short_n)
        _, raw_short_buffer = select_positions_with_buffer(
            score.sort_values(ascending=False),
            long_n=0,
            short_n=args.short_n,
            prev_longs=set(),
            prev_shorts=prev_shorts["raw_short5_buffer"],
            rebal_buffer=config.Portfolio.REBAL_BUFFER,
        )
        quality_short = _quality_shorts(
            candidates,
            prev_shorts=prev_shorts["quality_short5"],
            short_n=args.short_n,
            entry_sigma=args.quality_entry_sigma,
            stay_sigma=args.quality_stay_sigma,
            min_net=min_net,
        )
        quality_short_hedged = _quality_shorts(
            candidates,
            prev_shorts=prev_shorts["quality_short5_btc_hedged"],
            short_n=args.short_n,
            entry_sigma=args.quality_entry_sigma,
            stay_sigma=args.quality_stay_sigma,
            min_net=min_net,
        )
        long_leg = score.sort_values(ascending=False).head(args.long_n)
        short_leg = score.sort_values(ascending=True).head(args.short_n)

        books = {
            "raw_short5_no_buffer": (pd.Series(dtype=float), raw_short_no_buffer, None),
            "raw_short5_buffer": (pd.Series(dtype=float), raw_short_buffer, None),
            "quality_short5": (pd.Series(dtype=float), quality_short, None),
            "quality_short5_btc_hedged": (pd.Series(dtype=float), quality_short_hedged, "btc"),
            "long_short5": (long_leg, short_leg, None),
        }

        for strategy, (longs, shorts, hedge) in books.items():
            long_markets = set(longs.index)
            short_markets = set(shorts.index)
            long_ret = period_ret.reindex(longs.index).dropna().mean() if len(longs) else 0.0
            short_ret = period_ret.reindex(shorts.index).dropna().mean() if len(shorts) else 0.0
            gross = float(long_ret - short_ret)
            hedge_beta = None
            if hedge == "btc" and len(shorts):
                hedge_beta = float(beta_row.reindex(shorts.index).dropna().mean())
                if not np.isnan(hedge_beta):
                    gross += hedge_beta * btc_ret

            cost, turnover = _compute_cost(
                long_markets,
                short_markets,
                prev_longs[strategy],
                prev_shorts[strategy],
                one_way_cost,
                short_extra_cost,
                hedge_beta=hedge_beta,
                prev_hedge_beta=prev_hedge_beta[strategy],
            )
            net = gross - cost

            period_rows.append({
                "timestamp": ts,
                "strategy": strategy,
                "gross_return": gross,
                "net_return": net,
                "cost": cost,
                "turnover": turnover,
                "n_long": len(longs),
                "n_short": len(shorts),
                "btc_return": btc_ret,
                "hedge_beta": hedge_beta,
                "regime": regime,
                "long_names": json.dumps(list(longs.index)),
                "short_names": json.dumps(list(shorts.index)),
            })
            _record_positions(position_rows, ts, strategy, "long", longs, period_ret)
            _record_positions(position_rows, ts, strategy, "short", shorts, period_ret)
            prev_longs[strategy] = long_markets
            prev_shorts[strategy] = short_markets
            if hedge_beta is not None and not np.isnan(hedge_beta):
                prev_hedge_beta[strategy] = hedge_beta

    period_df = pd.DataFrame(period_rows)
    position_df = pd.DataFrame(position_rows)
    if period_df.empty:
        print("No experiment rows generated.")
        return 1

    period_df["timestamp"] = pd.to_datetime(period_df["timestamp"], utc=True)
    period_df = period_df.sort_values(["strategy", "timestamp"]).reset_index(drop=True)
    position_df = position_df.sort_values(["strategy", "timestamp", "side", "rank"]).reset_index(drop=True)

    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config": {
            "days": args.days,
            "holdout_ratio": args.holdout_ratio,
            "horizon": args.horizon,
            "rebalance_hours": rebalance_h,
            "short_n": args.short_n,
            "long_n": args.long_n,
            "fee_bps": args.fee_bps,
            "slippage_bps": args.slippage_bps,
            "short_extra_cost_bps": args.short_extra_cost_bps,
            "quality_entry_sigma": args.quality_entry_sigma,
            "quality_stay_sigma": args.quality_stay_sigma,
            "quality_min_net_bps": args.quality_min_net_bps,
            "bitget_filter": not args.no_bitget_filter,
        },
        "strategies": {},
        "regime": {},
    }

    indexed = period_df.set_index("timestamp")
    for strategy, grp in indexed.groupby("strategy"):
        summary["strategies"][strategy] = _strategy_summary(grp, position_df, args.horizon)
        regime_summary = {}
        for regime, rgrp in grp.groupby("regime"):
            regime_summary[str(regime)] = {
                "periods": int(len(rgrp)),
                "mean_net": float(rgrp["net_return"].mean()),
                "win_rate": float((rgrp["net_return"] > 0).mean()),
            }
        summary["regime"][strategy] = regime_summary

    OUTPUT_DIR.mkdir(exist_ok=True)
    period_path = OUTPUT_DIR / f"{args.out_prefix}.csv"
    pos_path = OUTPUT_DIR / f"{args.out_prefix}_positions.csv"
    summary_path = OUTPUT_DIR / f"{args.out_prefix}_summary.json"
    period_df.to_csv(period_path, index=False)
    position_df.to_csv(pos_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print("\nPortfolio experiment summary (net per rebalance window):")
    print(f"{'strategy':<30} {'n':>4} {'mean':>9} {'t':>7} {'sharpe':>8} {'win':>7} {'mdd':>9} {'beta':>8}")
    print("-" * 92)
    for strategy, stats in sorted(summary["strategies"].items()):
        print(
            f"{strategy:<30} {stats['periods']:>4} "
            f"{(stats['mean_net'] or 0):>+9.4f} "
            f"{(stats['tstat_window'] if stats['tstat_window'] is not None else float('nan')):>+7.2f} "
            f"{(stats['sharpe_ann'] if stats['sharpe_ann'] is not None else float('nan')):>+8.2f} "
            f"{(stats['win_rate'] or 0):>7.1%} "
            f"{(stats['max_drawdown'] or 0):>+9.1%} "
            f"{(stats['beta_to_btc'] if stats['beta_to_btc'] is not None else float('nan')):>+8.2f}"
        )
    print("-" * 92)
    print(f"Wrote: {period_path}")
    print(f"Wrote: {pos_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
