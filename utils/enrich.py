"""Enrichment layer: multi-model consensus + per-coin reliability.

Two signals that improve prediction usefulness without retraining the models:

1. **Multi-model consensus** — when SHORT 6h and LONG 12h models agree on a
   coin's direction, the signal is historically more reliable. We tag these
   "⚡ consensus" and bump the tier.

2. **Per-coin reliability** — from the recommendation_ledger we can compute
   each coin's historical directional hit rate. Coins that consistently go
   the wrong way (like the ZBT/BLUR/FF cluster we found in diagnostics) get
   a ⚠ warning regardless of sigma. Coins that consistently hit get ⭐.

Both enrich predict_batch output without touching model code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "output" / "recommendation_ledger.csv"

MIN_OBS_FOR_RELIABILITY = 5
UNTRUSTED_HIT_THRESHOLD = 0.40
TRUSTED_HIT_THRESHOLD   = 0.60


def per_coin_reliability() -> pd.DataFrame:
    """Compute directional hit rate per coin from matured ledger entries.

    Returns DataFrame indexed by market with columns:
      n, hit_rate, mean_realized_pct, trust_tag, trust_note
    """
    if not LEDGER.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(LEDGER)
    except Exception:
        return pd.DataFrame()
    if df.empty or "realized_return" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["realized_return"]).copy()
    df["realized_return"] = pd.to_numeric(df["realized_return"], errors="coerce")
    df = df.dropna(subset=["realized_return"])

    # Ledger stores realized_return in "profitable" sign convention:
    #   SHORT: +ve if price dropped
    #   LONG/WATCH_LONG: +ve if price rose
    # So hit = realized_return > 0 regardless of side.
    df["hit"] = (df["realized_return"] > 0).astype(int)
    grp = df.groupby("market").agg(
        n=("hit", "count"),
        hit_rate=("hit", "mean"),
        mean_realized_pct=("realized_return", lambda s: float(s.mean() * 100)),
    )
    grp = grp[grp["n"] >= MIN_OBS_FOR_RELIABILITY].copy()
    if grp.empty:
        return grp

    def classify(row):
        h = row["hit_rate"]
        n = row["n"]
        if h >= TRUSTED_HIT_THRESHOLD:
            return pd.Series(["⭐", f"적중 {h*100:.0f}% (n={n})"], index=["trust_tag", "trust_note"])
        if h <= UNTRUSTED_HIT_THRESHOLD:
            return pd.Series(["⚠", f"적중 {h*100:.0f}% (n={n}) — 역신호 경향"], index=["trust_tag", "trust_note"])
        return pd.Series(["", f"적중 {h*100:.0f}% (n={n})"], index=["trust_tag", "trust_note"])

    cls = grp.apply(classify, axis=1)
    grp = pd.concat([grp, cls], axis=1)
    return grp


def add_consensus(short_df: pd.DataFrame | None, long_df: pd.DataFrame | None):
    """For coins predicted by both models, compute a consensus flag/boost.

    A coin is in consensus if both SHORT 6h model and LONG 12h model predict
    the same direction. Returns (short_df_enriched, long_df_enriched) with new
    columns: consensus (bool), consensus_tag ('⚡' or ''), consensus_note.

    Both input DataFrames are modified in place and returned.
    """
    if short_df is None or long_df is None:
        if short_df is not None: short_df = _add_blank_consensus(short_df)
        if long_df  is not None: long_df  = _add_blank_consensus(long_df)
        return short_df, long_df

    s_dir = short_df["direction"].astype(int)
    l_dir = long_df["direction"].astype(int)
    common = s_dir.index.intersection(l_dir.index)
    agree = common[s_dir.loc[common] == l_dir.loc[common]]
    agree_set = set(agree)

    def tag_row(idx, dir_val):
        if idx in agree_set and dir_val != 0:
            return pd.Series([True, "⚡", "6h+12h 일치"], index=["consensus", "consensus_tag", "consensus_note"])
        return pd.Series([False, "", ""], index=["consensus", "consensus_tag", "consensus_note"])

    s_enrich = short_df.index.to_series().apply(lambda i: tag_row(i, s_dir.get(i, 0)))
    l_enrich = long_df.index.to_series().apply(lambda i: tag_row(i, l_dir.get(i, 0)))
    for col in ("consensus", "consensus_tag", "consensus_note"):
        short_df[col] = s_enrich[col].values
        long_df[col]  = l_enrich[col].values
    return short_df, long_df


def _add_blank_consensus(df):
    df["consensus"] = False
    df["consensus_tag"] = ""
    df["consensus_note"] = ""
    return df


def apply_reliability(pred_df: pd.DataFrame, reliability_df: pd.DataFrame) -> pd.DataFrame:
    """Merge per-coin reliability tags into predictions."""
    if pred_df is None or len(pred_df) == 0:
        return pred_df
    if reliability_df is None or reliability_df.empty:
        pred_df["trust_tag"] = ""
        pred_df["trust_note"] = ""
        pred_df["trust_hit_rate"] = None
        return pred_df

    pred_df = pred_df.merge(
        reliability_df[["trust_tag", "trust_note", "hit_rate"]].rename(
            columns={"hit_rate": "trust_hit_rate"}
        ),
        left_index=True, right_index=True, how="left",
    )
    pred_df["trust_tag"]  = pred_df["trust_tag"].fillna("")
    pred_df["trust_note"] = pred_df["trust_note"].fillna("")
    return pred_df


def enrich_predictions(short_df, long_df):
    """Apply both consensus and reliability in one call. Returns (short, long)."""
    short_df, long_df = add_consensus(short_df, long_df)
    reliability = per_coin_reliability()
    short_df = apply_reliability(short_df, reliability)
    long_df  = apply_reliability(long_df,  reliability)
    return short_df, long_df
