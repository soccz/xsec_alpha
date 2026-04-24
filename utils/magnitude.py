"""Per-coin direction + expected-% prediction helper.

The model's raw score is trained on beta-adjusted residual returns with
cross-sectional z-score features. Its magnitude is compressed (pred_std ~0.003);
directly treating the score as "% move" under-predicts by 3-5×.

Instead we rely on a sigma-bucket calibration built from a full-universe scan
(scripts/build_calibration.py → output/calibration_sigma.json). For each bucket
we record:
  - hit_rate           (P[sign(score) == sign(ret)])
  - mean_signed_return (E[sign(score) × ret])  ← empirical direction+magnitude
  - mean_abs_return    (typical coin move size)
  - std_return         (coin variation)

At runtime, a coin's prediction is:
  direction     = sign(score)
  sigma         = |score| / std(scores in batch)
  bucket        = lookup(side, sigma)
  expected_pct  = direction × bucket.mean_signed_return
  hit_rate      = bucket.hit_rate
  typical_vol   = realized vol of this coin over the horizon
  confidence    = tier tag (🔥 / ✅ / ▫ / ·)

The user reads `expected_pct` as the historically calibrated expected return
when acting on signals of this confidence, and `typical_vol` as the coin's
own range so they can judge whether the signal is meaningful for that coin.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CALIB_FILE = ROOT / "output" / "calibration_sigma.json"

_CALIB_CACHE: dict | None = None


def _load_calibration() -> dict:
    global _CALIB_CACHE
    if _CALIB_CACHE is not None:
        return _CALIB_CACHE
    if not CALIB_FILE.exists():
        _CALIB_CACHE = {}
        return _CALIB_CACHE
    try:
        _CALIB_CACHE = json.loads(CALIB_FILE.read_text())
    except Exception:
        _CALIB_CACHE = {}
    return _CALIB_CACHE


def _find_bucket(buckets: list[dict], sigma: float) -> dict | None:
    if not buckets:
        return None
    for b in buckets:
        lo = b.get("sigma_low", 0)
        hi = b.get("sigma_high")
        if hi is None:
            if sigma >= lo:
                return b
        elif lo <= sigma < hi:
            return b
    return buckets[-1]


def realized_vol_per_horizon(closes: pd.DataFrame, horizon_h: int,
                              lookback_h: int = 72) -> pd.Series:
    """Per-coin std(hourly pct_change) × sqrt(horizon_h) over last `lookback_h` hours."""
    rets = closes.pct_change().tail(lookback_h)
    return (rets.std(ddof=0) * np.sqrt(horizon_h)).dropna()


def confidence_tier(sigma: float) -> tuple[str, str]:
    """(tag, human label) for a sigma value."""
    if sigma >= 2.0: return ("🔥", "강한 신호")
    if sigma >= 1.0: return ("✅", "보통")
    if sigma >= 0.5: return ("▫", "약함")
    return ("·", "노이즈")


# Tier-based position sizing (% of capital suggested per coin).
# Calibrated from Batch IV-IV2 consensus: 🔥 3%, ✅ 2%, ▫ 1%, · 0%.
# User trades manually; this is a suggestion only.
TIER_SIZING_PCT = {"🔥": 3.0, "✅": 2.0, "▫": 1.0, "·": 0.0}


def predict_one(side: str, score: float, batch_std: float,
                 coin_vol_h: float | None = None) -> dict:
    """Single-coin prediction bundle — probabilistic schema.

    Output fields (Tier 1 schema from 20-trader consensus):
      direction        : +1 / -1 / 0
      direction_prob   : P(realized sign matches predicted sign) from σ-bucket
      sigma            : |score| / batch_std (model confidence)
      tag / label      : 🔥/✅/▫/· with Korean label
      expected_pct     : historically calibrated signed expected return (%)
      ci_95_low/high   : 95% CI of realized return in this σ-bucket
      hit_rate         : same as direction_prob (kept for back-compat)
      typical_move_pct : E[|ret|] in this bucket
      coin_vol_pct     : this specific coin's realized vol over horizon
      position_size_pct: suggested % of capital per tier (user trades manually)
    """
    calib = _load_calibration().get(side, [])
    blank = {
        "direction": 0, "sigma": 0.0, "tag": "·", "label": "데이터 없음",
        "direction_prob": None, "expected_pct": None,
        "ci_95_low": None, "ci_95_high": None,
        "hit_rate": None, "typical_move_pct": None,
        "coin_vol_pct": None, "position_size_pct": 0.0,
    }
    if score is None or np.isnan(score) or batch_std <= 0:
        return blank

    sigma = abs(score) / batch_std
    direction = 1 if score > 0 else -1
    bucket = _find_bucket(calib, sigma)

    if bucket:
        expected_pct = direction * bucket["mean_signed_return_pct"]
        hit_rate = bucket["hit_rate"]
        typical = bucket["mean_abs_return_pct"]
        bucket_std = bucket.get("std_return_pct")
    else:
        expected_pct = hit_rate = typical = bucket_std = None

    # 95% CI using COIN-SPECIFIC realized vol (not bucket std).
    # Bucket std is cross-coin averaged (~5-6%), which hides per-coin risk differences.
    # Using each coin's own realized vol gives a tighter/wider band per coin's volatility.
    # Fallback to bucket std only if coin vol is unavailable.
    ci_low = ci_high = None
    if expected_pct is not None:
        if coin_vol_h is not None and not np.isnan(coin_vol_h):
            sd = coin_vol_h * 100
        else:
            sd = bucket_std  # fallback
        if sd is not None:
            ci_low  = expected_pct - 1.96 * sd
            ci_high = expected_pct + 1.96 * sd

    tag, label = confidence_tier(sigma)
    return {
        "direction":         direction,
        "sigma":             round(sigma, 2),
        "tag":               tag,
        "label":             label,
        "direction_prob":    round(hit_rate, 3) if hit_rate is not None else None,
        "expected_pct":      round(expected_pct, 3) if expected_pct is not None else None,
        "ci_95_low":         round(ci_low, 2) if ci_low is not None else None,
        "ci_95_high":        round(ci_high, 2) if ci_high is not None else None,
        "hit_rate":          round(hit_rate, 3) if hit_rate is not None else None,
        "typical_move_pct":  round(typical, 2) if typical is not None else None,
        "coin_vol_pct":      round(coin_vol_h * 100, 2) if coin_vol_h is not None and not np.isnan(coin_vol_h) else None,
        "position_size_pct": TIER_SIZING_PCT.get(tag, 0.0),
    }


def predict_batch(side: str, scores: pd.Series,
                   closes: pd.DataFrame | None = None,
                   horizon_h: int | None = None) -> pd.DataFrame:
    """Score DataFrame for a whole universe.

    Returns DataFrame indexed by market with columns:
      score, sigma, direction, tag, label, expected_pct, hit_rate,
      typical_move_pct, coin_vol_pct
    """
    batch_std = float(scores.std()) if len(scores) > 1 else 0.0
    if closes is not None and horizon_h is not None:
        vol = realized_vol_per_horizon(closes, horizon_h)
    else:
        vol = pd.Series(dtype=float)

    rows = []
    for mkt, s in scores.items():
        vh = float(vol.get(mkt, np.nan)) if len(vol) else None
        pred = predict_one(side, float(s), batch_std, vh)
        pred["market"] = mkt
        pred["score"] = round(float(s), 5)
        rows.append(pred)
    df = pd.DataFrame(rows).set_index("market")
    # Sort by sigma desc so strong signals float to the top
    df = df.sort_values(["sigma", "score"], ascending=[False, True])
    return df
