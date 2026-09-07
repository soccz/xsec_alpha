#!/usr/bin/env python3
"""Daily holdout monitor for the production ranking contract.

The monitor deliberately reuses the live feature/model contract, while measuring
only completed, non-overlapping production signal slots:

* SHORT: ``xsec_6h.pkl`` at 05/11/17/23 UTC, next-open to open six bars later.
* LONG: ``xsec_12h.pkl`` at 11/23 UTC, next-open to open twelve bars later.

LONG also records the regime-gated top-five portfolio's gross/net return,
turnover, and hit rate. Results are appended to ``output/wf_history.json``.

Usage::

    python scripts/wf_holdout_harness.py --measure-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.run_lock import stable_data_read_lock
from utils.model_release import model_release_guard

ROOT = Path(__file__).resolve().parent.parent
HISTORY_FILE = ROOT / "output" / "wf_history.json"
_DATA_ACCESS_LOCK_WAIT_SEC = 480.0
_MIN_CROSS_SECTION = 20
_LONG_TOP_N = 5

SIDE_CONTRACTS = {
    "short": {
        "horizon_h": 6,
        "anchor_hours_utc": (5, 11, 17, 23),
        "model_file": "xsec_6h.pkl",
    },
    "long": {
        "horizon_h": 12,
        "anchor_hours_utc": (11, 23),
        "model_file": "xsec_12h.pkl",
    },
}


def _as_utc_timestamp(value) -> pd.Timestamp:
    """Normalize a timestamp to UTC without changing its instant."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def select_exact_anchor_timestamps(
    timestamps: Iterable,
    anchor_hours_utc: Iterable[int],
    horizon_h: int,
) -> list[pd.Timestamp]:
    """Return sorted exact UTC anchors whose holding windows do not overlap.

    A signal at ``t`` enters at the next bar's open and exits ``horizon_h``
    bars later. Adjacent windows may share their boundary open, but no interior
    time. Duplicate, sub-hour, and too-closely-spaced candidates are discarded.
    """
    if horizon_h <= 0:
        raise ValueError("horizon_h must be positive")

    anchors = {int(hour) for hour in anchor_hours_utc}
    if not anchors or any(hour < 0 or hour > 23 for hour in anchors):
        raise ValueError("anchor hours must be integers in [0, 23]")

    candidates: set[pd.Timestamp] = set()
    for value in timestamps:
        if pd.isna(value):
            continue
        ts = _as_utc_timestamp(value)
        if ts.hour not in anchors or ts != ts.floor("h"):
            continue
        candidates.add(ts)

    selected: list[pd.Timestamp] = []
    minimum_spacing = pd.Timedelta(hours=horizon_h)
    for ts in sorted(candidates):
        if selected and ts - selected[-1] < minimum_spacing:
            continue
        selected.append(ts)
    return selected


def compute_next_open_forward_returns(opens: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """Align next-bar-open to ``horizon_h``-bars-later-open returns to signal time."""
    from data.features import compute_forward_returns

    if horizon_h <= 0:
        raise ValueError("horizon_h must be positive")
    return compute_forward_returns(opens, horizon=horizon_h, execution_lag=1)


def long_regime_allows(regime_row, btc_7d_gate: float, btc_30d_floor: float) -> bool:
    """Match live LONG gating: both BTC returns must be finite and strictly pass."""
    if regime_row is None:
        return False
    try:
        btc_7d = float(regime_row.get("btc_ret_7d"))
        btc_30d = float(regime_row.get("btc_ret_30d"))
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(
        np.isfinite(btc_7d)
        and np.isfinite(btc_30d)
        and btc_7d > float(btc_7d_gate)
        and btc_30d > float(btc_30d_floor)
    )


def basket_turnover(current: set[str], previous: set[str]) -> float:
    """Fraction of the current equal-weight basket that must be newly bought."""
    if not current:
        return 0.0
    return len(current - previous) / len(current)


def turnover_adjusted_net_return(
    gross_return: float,
    turnover: float,
    fee_bps: float,
    slippage_bps: float,
) -> float:
    """Apply round-trip fee plus slippage to the traded basket fraction."""
    round_trip_cost = 2.0 * (float(fee_bps) + float(slippage_bps)) / 10_000.0
    return float(gross_return) - float(turnover) * round_trip_cost


def evaluate_top_n_slot(
    scores: pd.Series,
    forward_returns: pd.Series,
    previous: set[str],
    *,
    top_n: int,
    fee_bps: float,
    slippage_bps: float,
) -> dict:
    """Evaluate one equal-weight top-N LONG slot and return its next basket."""
    finite_scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite_scores) < top_n:
        raise ValueError(f"top_n={top_n} requires at least {top_n} finite scores")

    pick_index = finite_scores.nlargest(top_n).index
    picks = set(pick_index)
    selected_returns = forward_returns.reindex(pick_index).replace([np.inf, -np.inf], np.nan)
    if selected_returns.isna().any():
        # Never replace an unavailable ex-post top-five return with the sixth
        # ranked market; doing so would make portfolio selection look ahead.
        return {
            "gross_return": None,
            "net_return": None,
            "turnover": basket_turnover(picks, previous),
            "gross_hit": None,
            "net_hit": None,
            "picks": picks,
        }

    gross = float(selected_returns.mean())
    turnover = basket_turnover(picks, previous)
    net = turnover_adjusted_net_return(gross, turnover, fee_bps, slippage_bps)
    return {
        "gross_return": gross,
        "net_return": net,
        "turnover": turnover,
        "gross_hit": float(gross > 0.0),
        "net_hit": float(net > 0.0),
        "picks": picks,
    }


def _prepare_measurement_data(days: int) -> dict[str, object]:
    """Build the exact unified feature frame used by live inference."""
    from config import config
    from data.features import (
        UNIFIED_CALENDAR_COLS,
        build_top_liquidity_universe_index,
        compute_btc_regime,
        compute_unified_factors,
        crosssection_zscore,
        filter_long_frame_by_universe,
        load_and_pivot,
        load_binance_pivot,
    )

    closes, opens, highs, lows, volumes = load_and_pivot(days=days)
    # Live computes the 30d BTC gate on pre-warmup context so the 720h
    # lookback remains available after the factor warmup is discarded.
    btc_regime = compute_btc_regime(closes)
    warmup = config.Data.MIN_ROWS_PER_COIN
    closes = closes.iloc[warmup:]
    opens = opens.iloc[warmup:]
    highs = highs.iloc[warmup:]
    lows = lows.iloc[warmup:]
    volumes = volumes.iloc[warmup:]

    binance_closes = load_binance_pivot(closes.columns.tolist(), days=days)
    if not binance_closes.empty:
        binance_closes = binance_closes.iloc[warmup:]

    factor_df = compute_unified_factors(
        closes,
        opens,
        highs,
        lows,
        volumes,
        binance_closes=binance_closes,
    )
    zscore_cols = [col for col in factor_df.columns if col not in UNIFIED_CALENDAR_COLS]
    factor_df = crosssection_zscore(factor_df, cols=zscore_cols)
    selected_index, _ = build_top_liquidity_universe_index(
        closes,
        volumes,
        top_n=getattr(config.Data, "LIQUIDITY_TOP_N", 0),
    )
    factor_df = filter_long_frame_by_universe(factor_df, selected_index)

    tradable_markets = None
    if getattr(config.Portfolio, "LIVE_REQUIRE_BITGET_TRADABLE", False):
        from utils.bitget import filter_markets_to_bitget, load_bitget_usdt_perp_map

        contract_map = load_bitget_usdt_perp_map()
        tradable_markets = set(
            filter_markets_to_bitget(closes.columns.tolist(), contract_map)
        )

    return {
        "closes": closes,
        "opens": opens,
        "factors": factor_df,
        "btc_regime": btc_regime,
        "tradable_markets": tradable_markets,
    }


def _base_result(side: str, contract: dict, n_ts: int) -> dict:
    return {
        "side": side,
        "horizon_h": contract["horizon_h"],
        "model_file": contract["model_file"],
        "feature_contract": "unified",
        "anchor_hours_utc": list(contract["anchor_hours_utc"]),
        "execution_lag_bars": 1,
        "entry_price": "next_bar_open",
        "exit_price": f"open_after_{contract['horizon_h']}_bars",
        "non_overlapping": True,
        "n_ts": n_ts,
    }


def _long_metric_defaults() -> dict:
    return {
        "n_regime_active_slots": 0,
        "n_top5_slots": 0,
        "top5_gross_return": None,
        "top5_net_return": None,
        "top5_turnover": None,
        "top5_hit_rate": None,
        "top5_gross_hit_rate": None,
    }


def measure_side_ic(
    side: str,
    days: int = 60,
    holdout_frac: float = 0.20,
    prepared: dict[str, object] | None = None,
) -> dict:
    """Evaluate a production model on completed exact-anchor holdout slots."""
    from config import config
    from models.xgb_ranker import XSecRanker

    side = side.lower()
    if side not in SIDE_CONTRACTS:
        raise ValueError(f"unknown side: {side}")
    if not 0.0 < holdout_frac <= 1.0:
        raise ValueError("holdout_frac must be in (0, 1]")

    contract = SIDE_CONTRACTS[side]
    horizon_h = contract["horizon_h"]
    data = prepared if prepared is not None else _prepare_measurement_data(days)
    opens = data["opens"]
    factor_df = data["factors"]

    all_ts = select_exact_anchor_timestamps(
        factor_df.index.get_level_values("timestamp").unique(),
        range(24),
        1,
    )
    result = _base_result(side, contract, len(all_ts))
    if side == "long":
        result.update(_long_metric_defaults())
    if len(all_ts) < 20:
        result.update(status="insufficient_data", n_exact_anchor_slots=0, n_mature_anchor_slots=0)
        return result

    holdout_start = int(len(all_ts) * (1.0 - holdout_frac))
    holdout_ts = select_exact_anchor_timestamps(
        all_ts[holdout_start:],
        contract["anchor_hours_utc"],
        horizon_h,
    )
    result["n_exact_anchor_slots"] = len(holdout_ts)

    model = XSecRanker.load(str(ROOT / "models" / contract["model_file"]))
    forward_returns = compute_next_open_forward_returns(opens, horizon_h)
    btc_regime = data.get("btc_regime")
    tradable_markets = data.get("tradable_markets")
    fee_bps = float(config.Costs.ONE_WAY_FEE_BPS)
    slippage_bps = float(config.Costs.SLIPPAGE_BPS)
    btc_7d_gate = float(config.LongModel.BTC_7D_RETURN_GATE)
    btc_30d_floor = float(config.LongModel.BTC_30D_RETURN_FLOOR)

    ics: list[float] = []
    long_slots: list[dict] = []
    previous_top5: set[str] = set()
    mature_anchor_slots = 0
    regime_active_slots = 0

    for ts in holdout_ts:
        if ts not in forward_returns.index:
            continue
        slot_returns = forward_returns.loc[ts].replace([np.inf, -np.inf], np.nan)
        if int(slot_returns.notna().sum()) < _MIN_CROSS_SECTION:
            continue
        mature_anchor_slots += 1

        if side == "long":
            regime_row = None
            if btc_regime is not None and ts in btc_regime.index:
                regime_row = btc_regime.loc[ts]
            if not long_regime_allows(regime_row, btc_7d_gate, btc_30d_floor):
                # An inactive gate exits the paper basket. A later re-entry is
                # therefore treated as full turnover.
                previous_top5 = set()
                continue
            regime_active_slots += 1

        try:
            # Live inference neutralizes missing factors only after z-scoring.
            features = factor_df.xs(ts, level="timestamp").fillna(0)
        except KeyError:
            continue
        features = features[features.any(axis=1)]
        if len(features) < _MIN_CROSS_SECTION:
            continue

        scores = pd.Series(model.predict(features), index=features.index, dtype=float)
        aligned = pd.concat(
            [scores.rename("score"), slot_returns.rename("forward_return")], axis=1
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(aligned) < _MIN_CROSS_SECTION:
            continue

        ic, _ = spearmanr(aligned["score"], aligned["forward_return"])
        if np.isfinite(ic):
            ics.append(float(ic))

        if side == "long":
            portfolio_scores = scores
            if tradable_markets is not None:
                portfolio_scores = scores[scores.index.isin(tradable_markets)]
            if len(portfolio_scores) < _LONG_TOP_N:
                previous_top5 = set()
                continue
            slot = evaluate_top_n_slot(
                portfolio_scores,
                slot_returns,
                previous_top5,
                top_n=_LONG_TOP_N,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            )
            previous_top5 = slot.pop("picks")
            if slot["gross_return"] is not None:
                long_slots.append(slot)

    result["n_mature_anchor_slots"] = mature_anchor_slots
    if side == "long":
        result["n_regime_active_slots"] = regime_active_slots
        result["btc_7d_return_gate"] = btc_7d_gate
        result["btc_30d_return_floor"] = btc_30d_floor
        result["round_trip_cost_bps"] = 2.0 * (fee_bps + slippage_bps)
        result["n_top5_slots"] = len(long_slots)
        if long_slots:
            result.update(
                top5_gross_return=round(float(np.mean([s["gross_return"] for s in long_slots])), 6),
                top5_net_return=round(float(np.mean([s["net_return"] for s in long_slots])), 6),
                top5_turnover=round(float(np.mean([s["turnover"] for s in long_slots])), 4),
                top5_hit_rate=round(float(np.mean([s["net_hit"] for s in long_slots])), 3),
                top5_gross_hit_rate=round(float(np.mean([s["gross_hit"] for s in long_slots])), 3),
            )

    if not ics:
        result.update(status="no_ic_samples", n_holdout_slots=0)
        return result

    arr = np.asarray(ics, dtype=float)
    mean_ic = float(arr.mean())
    std_ic = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    t_stat = float(mean_ic / (std_ic / np.sqrt(len(arr)))) if std_ic > 0.0 else 0.0
    status = "OK"
    if mean_ic < 0.045:
        status = "WARN"
    if mean_ic < 0.035:
        status = "FREEZE"
    result.update(
        n_holdout_slots=len(arr),
        mean_ic=round(mean_ic, 4),
        median_ic=round(float(np.median(arr)), 4),
        std_ic=round(std_ic, 4),
        t_stat=round(t_stat, 2),
        pos_frac=round(float((arr > 0.0).mean()), 3),
        status=status,
    )
    return result


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
    with stable_data_read_lock(timeout_sec=_DATA_ACCESS_LOCK_WAIT_SEC):
        _main_locked()


@model_release_guard()
def _main_locked():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure-only", action="store_true", help="Evaluate current models (default)")
    ap.add_argument("--retrain", action="store_true", help="Not implemented; retained for compatibility")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--holdout-frac", type=float, default=0.20)
    args = ap.parse_args()

    if args.retrain:
        print("Retrain mode not implemented — use --measure-only")
        sys.exit(1)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prepared = _prepare_measurement_data(args.days)
    short_res = measure_side_ic(
        "short", days=args.days, holdout_frac=args.holdout_frac, prepared=prepared
    )
    long_res = measure_side_ic(
        "long", days=args.days, holdout_frac=args.holdout_frac, prepared=prepared
    )

    entry = {"timestamp": now, "short": short_res, "long": long_res}
    append_history(entry)

    print(f"=== WF Holdout ({args.days}d window, {args.holdout_frac * 100:.0f}% holdout) ===")
    for side, result in (("SHORT", short_res), ("LONG", long_res)):
        if result["status"] in ("insufficient_data", "no_ic_samples"):
            print(
                f"  {side}: {result['status']} "
                f"(anchors={result.get('n_exact_anchor_slots', 0)}, "
                f"mature={result.get('n_mature_anchor_slots', 0)})"
            )
        else:
            print(
                f"  {side} h{result['horizon_h']} n={result['n_holdout_slots']} "
                f"mean={result['mean_ic']:+.4f} t={result['t_stat']:+.2f} "
                f"pos={result['pos_frac'] * 100:.0f}% [{result['status']}]"
            )
        if side == "LONG":
            print(
                "       "
                f"regime-active={result['n_regime_active_slots']} "
                f"top5 gross={result['top5_gross_return']} "
                f"net={result['top5_net_return']} "
                f"turnover={result['top5_turnover']} "
                f"hit={result['top5_hit_rate']}"
            )
    print(f"\nAppended to {HISTORY_FILE}")


if __name__ == "__main__":
    main()
