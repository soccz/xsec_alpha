#!/usr/bin/env python3
"""Walk-forward holdout IC harness.

Runs daily (via systemd or manual). For each side (short/long):
  1. Load factor data + closes over last 60 days.
  2. Retrain model on first 80% (train), evaluate on last 20% (holdout).
  3. Compute holdout IC (Spearman per timestamp, aggregated).
  4. Append to output/wf_history.json.

Purpose: Detect model staleness / concept drift by tracking holdout IC over
time. Health snapshot reads this file and warns if recent holdout IC drops.

Cheap mode: --measure-only skips retraining, just evaluates current model on
a fresh holdout slice. Useful when model retrain is expensive and we only
want a daily health measure.

Usage:
    python scripts/wf_holdout_harness.py --measure-only     # daily
    python scripts/wf_holdout_harness.py --retrain          # weekly
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
HISTORY_FILE = ROOT / "output" / "wf_history.json"


def measure_side_ic(side: str, days: int = 60, holdout_frac: float = 0.20) -> dict:
    """Load data, run current model on last `holdout_frac` of timeline, return IC."""
    from data.features import (
        load_and_pivot, load_binance_pivot, compute_factors, compute_long_factors,
        compute_btc_regime, crosssection_zscore, build_top_liquidity_universe_index,
        filter_long_frame_by_universe, LONG_CALENDAR_COLS,
    )
    from models.xgb_ranker import XSecRanker
    from config import config

    closes, opens, highs, lows, volumes = load_and_pivot(days=days)
    warmup = config.Data.MIN_ROWS_PER_COIN
    closes = closes.iloc[warmup:]; opens = opens.iloc[warmup:]
    highs = highs.iloc[warmup:]; lows = lows.iloc[warmup:]; volumes = volumes.iloc[warmup:]
    binance_closes = load_binance_pivot(closes.columns.tolist(), days=days)
    if not binance_closes.empty:
        binance_closes = binance_closes.iloc[warmup:]
    uni, _ = build_top_liquidity_universe_index(closes, volumes, top_n=100)

    if side == "short":
        horizon_h = 6
        fdf = compute_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
        fdf = crosssection_zscore(fdf)
        fdf = filter_long_frame_by_universe(fdf, uni)
        model = XSecRanker.load(str(ROOT / "models/xsec_xgb.pkl"))
    else:
        horizon_h = 12
        fdf = compute_long_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
        zcols = [c for c in fdf.columns if c not in LONG_CALENDAR_COLS]
        fdf = crosssection_zscore(fdf, cols=zcols)
        fdf = filter_long_frame_by_universe(fdf, uni)
        model = XSecRanker.load(str(ROOT / "models/xsec_long.pkl"))

    ts_series = sorted([t for t in fdf.index.get_level_values("timestamp").unique() if pd.notna(t)])
    if len(ts_series) < 20:
        return {"side": side, "horizon_h": horizon_h, "status": "insufficient_data", "n_ts": len(ts_series)}

    holdout_start = int(len(ts_series) * (1 - holdout_frac))
    holdout_ts = ts_series[holdout_start:]

    btc_regime = compute_btc_regime(closes) if side == "long" else None

    ics = []
    for ts in holdout_ts:
        fwd = ts + pd.Timedelta(hours=horizon_h)
        if fwd not in closes.index:
            continue
        # Long is bull-regime only
        if side == "long" and btc_regime is not None:
            if ts not in btc_regime.index or btc_regime.loc[ts, "regime_bull"] != 1.0:
                continue
        try:
            feats = fdf.xs(ts, level="timestamp").fillna(0)
        except KeyError:
            continue
        feats = feats[feats.any(axis=1)]
        if len(feats) < 20:
            continue
        scores = pd.Series(model.predict(feats), index=feats.index)
        p0 = closes.loc[ts]; p1 = closes.loc[fwd]
        ret = ((p1 - p0) / p0).reindex(scores.index).dropna()
        scores = scores.loc[ret.index]
        if len(ret) < 20:
            continue
        ic, _ = spearmanr(scores, ret)
        if not np.isnan(ic):
            ics.append(ic)

    if not ics:
        return {"side": side, "horizon_h": horizon_h, "status": "no_ic_samples"}

    arr = np.array(ics)
    mean_ic = float(arr.mean())
    t_stat = float(mean_ic / (arr.std(ddof=1) / np.sqrt(len(arr)))) if arr.std(ddof=1) > 0 else 0.0
    status = "OK"
    if mean_ic < 0.045:
        status = "WARN"
    if mean_ic < 0.035:
        status = "FREEZE"
    return {
        "side":     side,
        "horizon_h": horizon_h,
        "n_holdout_slots": len(arr),
        "mean_ic":  round(mean_ic, 4),
        "median_ic": round(float(np.median(arr)), 4),
        "std_ic":   round(float(arr.std(ddof=1)), 4),
        "t_stat":   round(t_stat, 2),
        "pos_frac": round(float((arr > 0).mean()), 3),
        "status":   status,
    }


def append_history(new_entry: dict) -> None:
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    existing = []
    if HISTORY_FILE.exists():
        try:
            existing = json.loads(HISTORY_FILE.read_text())
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    existing.append(new_entry)
    HISTORY_FILE.write_text(json.dumps(existing, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure-only", action="store_true", help="Evaluate current model (default)")
    ap.add_argument("--retrain", action="store_true", help="Not implemented yet; placeholder")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--holdout-frac", type=float, default=0.20)
    args = ap.parse_args()

    if args.retrain:
        print("Retrain mode not implemented yet — use --measure-only")
        sys.exit(1)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    short_res = measure_side_ic("short", days=args.days, holdout_frac=args.holdout_frac)
    long_res  = measure_side_ic("long",  days=args.days, holdout_frac=args.holdout_frac)

    entry = {"timestamp": now, "short": short_res, "long": long_res}
    append_history(entry)

    print(f"=== WF Holdout ({args.days}d window, {args.holdout_frac*100:.0f}% holdout) ===")
    for side, r in [("SHORT", short_res), ("LONG", long_res)]:
        if "status" in r and r["status"] in ("insufficient_data", "no_ic_samples"):
            print(f"  {side}: {r['status']} (n={r.get('n_ts', 0)})")
            continue
        print(f"  {side} h{r['horizon_h']}  n={r['n_holdout_slots']}  "
              f"mean={r['mean_ic']:+.4f}  t={r['t_stat']:+.2f}  "
              f"pos={r['pos_frac']*100:.0f}%  [{r['status']}]")
    print(f"\nAppended to {HISTORY_FILE}")


if __name__ == "__main__":
    main()
