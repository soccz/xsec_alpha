"""Per-factor IC drift detector.

For each factor, compute cross-sectional IC per timestamp over recent history.
Compare the rolling 24h mean IC against the 7d MA. If a factor's recent IC
drops >50% or flips sign, flag it. Persist state to output/drift_state.json
so health_snapshot can display it and fetch_and_rank can optionally suppress
that factor's weight.

Not a retraining trigger — just a monitoring signal. Suppression happens
only when the operator (or CLAUDE.md rule) decides.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "output" / "drift_state.json"

DROP_THRESHOLD = 0.5   # IC drop > 50% vs 7d MA → flag
SIGN_FLIP = True        # any sign flip vs 7d MA → flag


def _ic_per_slot(factor: pd.Series, realized: pd.Series) -> float:
    """Spearman IC between factor and realized return on aligned indexes."""
    df = pd.concat([factor, realized], axis=1).dropna()
    if len(df) < 10:
        return float("nan")
    # Guard against constant inputs (calendar features can be constant within
    # a single timestamp slot) — spearmanr emits ConstantInputWarning otherwise.
    if df.iloc[:, 0].nunique() < 2 or df.iloc[:, 1].nunique() < 2:
        return float("nan")
    try:
        ic, _ = spearmanr(df.iloc[:, 0], df.iloc[:, 1])
        return float(ic)
    except Exception:
        return float("nan")


def compute_factor_drift(factor_df: pd.DataFrame, closes: pd.DataFrame,
                          horizon_h: int = 6,
                          lookback_hours: int = 7 * 24) -> dict:
    """For each factor column, compute rolling IC and drift state.

    factor_df: long-format (timestamp × market) → factor columns
    closes: wide format (timestamp × market) → close price
    horizon_h: forward return horizon for IC measurement
    lookback_hours: total window to analyse

    Returns dict: {factor_name: {ic_24h, ic_7d, drop_pct, status, flag}}
    """
    ts_index = factor_df.index.get_level_values("timestamp").unique()
    ts_index = sorted([t for t in ts_index if pd.notna(t)])[-lookback_hours:]

    factor_cols = [c for c in factor_df.columns
                   if factor_df[c].dtype.kind in "fi"]

    # Precompute per-timestamp IC for each factor
    per_slot: dict[str, list[tuple[pd.Timestamp, float]]] = {c: [] for c in factor_cols}
    for ts in ts_index:
        fwd = ts + pd.Timedelta(hours=horizon_h)
        if fwd not in closes.index:
            continue
        try:
            feats = factor_df.xs(ts, level="timestamp")
        except KeyError:
            continue
        p0 = closes.loc[ts]
        p1 = closes.loc[fwd]
        ret = ((p1 - p0) / p0).reindex(feats.index).dropna()
        if len(ret) < 20:
            continue
        for col in factor_cols:
            v = feats[col].reindex(ret.index)
            ic = _ic_per_slot(v, ret)
            if not np.isnan(ic):
                per_slot[col].append((ts, ic))

    result = {}
    for col, data in per_slot.items():
        if len(data) < 10:
            result[col] = {"ic_24h": None, "ic_7d": None, "status": "insufficient_data"}
            continue
        s = pd.Series(dict(data))
        last24 = s.iloc[-24:] if len(s) >= 24 else s
        ic_24h = float(last24.mean())
        ic_7d  = float(s.mean())
        # drop pct = 1 - (|ic_24h| / |ic_7d|). If ic_7d is ~0, skip drop calc.
        if abs(ic_7d) < 1e-6:
            drop = 0.0
        else:
            drop = 1.0 - abs(ic_24h) / abs(ic_7d)
        sign_flipped = (ic_24h * ic_7d < 0) and (abs(ic_7d) > 0.02)
        status = "OK"
        flag = None
        if sign_flipped:
            status = "SIGN_FLIP"
            flag = "sign flipped vs 7d MA"
        elif drop > DROP_THRESHOLD:
            status = "DRIFT"
            flag = f"ic dropped {drop*100:.0f}% vs 7d MA"
        result[col] = {
            "ic_24h": round(ic_24h, 4),
            "ic_7d":  round(ic_7d, 4),
            "drop_pct": round(drop * 100, 1),
            "sign_flipped": sign_flipped,
            "status": status,
            "flag": flag,
            "n_slots": len(data),
        }
    return result


def persist(result: dict) -> None:
    STATE_FILE.parent.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "factors": result,
    }
    STATE_FILE.write_text(json.dumps(payload, indent=2, default=str))


def summary(result: dict) -> str:
    if not result:
        return "no factor data"
    flagged = [(c, v) for c, v in result.items() if v.get("status") in ("DRIFT", "SIGN_FLIP")]
    if not flagged:
        return "all factors OK"
    return "; ".join(f"{c}:{v['status']}({v.get('drop_pct', '?')}%)" for c, v in flagged)


def render(result: dict) -> str:
    lines = ["Factor drift state:"]
    for col, v in sorted(result.items()):
        status = v.get("status", "?")
        ic24 = v.get("ic_24h"); ic7 = v.get("ic_7d")
        ic24_s = f"{ic24:+.3f}" if ic24 is not None else "—"
        ic7_s  = f"{ic7:+.3f}" if ic7 is not None else "—"
        flag = f" ⚠ {v['flag']}" if v.get("flag") else ""
        lines.append(f"  {col:22s}  24h={ic24_s}  7d={ic7_s}  [{status}]{flag}")
    return "\n".join(lines)


def main():
    """CLI entrypoint: compute drift on current short factors, print + persist."""
    import sys
    sys.path.insert(0, str(ROOT))
    from data.features import (
        load_and_pivot, load_binance_pivot, compute_factors, crosssection_zscore,
        build_top_liquidity_universe_index, filter_long_frame_by_universe,
    )
    from config import config

    # Need enough history for warmup (480 rows) + analysis window
    closes, opens, highs, lows, volumes = load_and_pivot(days=30)
    warmup = config.Data.MIN_ROWS_PER_COIN
    closes = closes.iloc[warmup:]; opens = opens.iloc[warmup:]
    highs = highs.iloc[warmup:]; lows = lows.iloc[warmup:]; volumes = volumes.iloc[warmup:]
    binance_closes = load_binance_pivot(closes.columns.tolist(), days=15)
    if not binance_closes.empty:
        binance_closes = binance_closes.iloc[warmup:]

    sf = compute_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
    sf = crosssection_zscore(sf)
    uni, _ = build_top_liquidity_universe_index(closes, volumes, top_n=100)
    sf = filter_long_frame_by_universe(sf, uni)

    result = compute_factor_drift(sf, closes, horizon_h=6)
    persist(result)
    print(render(result))
    print(f"\nSummary: {summary(result)}")
    print(f"Wrote: {STATE_FILE}")


if __name__ == "__main__":
    main()
