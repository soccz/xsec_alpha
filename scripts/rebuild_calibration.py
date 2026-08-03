#!/usr/bin/env python3
"""Rebuild sigma calibration from production-aligned temporal holdouts.

Only the 20% temporal holdout returned by ``build_dataset`` is scored. Signals
are evaluated at the live UTC anchors, filled with the live neutral-NaN policy,
and realized from the next bar's open to the open one full horizon later.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import config


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "output" / "calibration_sigma.json"
HOLDOUT_RATIO = 0.20
DATASET_DAYS = 60
SHORT_ANCHORS_UTC = (5, 11, 17, 23)
LONG_ANCHORS_UTC = (11, 23)
BUCKET_BOUNDS = (
    (0.0, 0.5),
    (0.5, 1.0),
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, None),
)
SAMPLE_COLUMNS = ("timestamp", "market", "sigma", "score", "ret")


def _is_exact_anchor(timestamp, anchors_utc: tuple[int, ...]) -> bool:
    """Return whether a timestamp is exactly one of the configured UTC hours."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return bool(
        ts.hour in anchors_utc
        and ts.minute == 0
        and ts.second == 0
        and ts.microsecond == 0
        and ts.nanosecond == 0
    )


def _next_open_returns(opens: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """Absolute return from open[t+1] to open[t+1+horizon]."""
    entry_open = opens.shift(-1)
    exit_open = opens.shift(-(horizon_h + 1))
    return exit_open / entry_open - 1.0


def _strict_long_regime_active(
    btc_context: pd.DataFrame,
    timestamp,
    *,
    btc_7d_gate: float,
    btc_30d_floor: float,
) -> bool:
    """Apply both LONG regime thresholds strictly; missing context fails closed."""
    try:
        row = btc_context.loc[timestamp]
        btc_7d = float(row["btc_ret_7d"])
        btc_30d = float(row["btc_ret_30d"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        np.isfinite(btc_7d)
        and np.isfinite(btc_30d)
        and btc_7d > btc_7d_gate
        and btc_30d > btc_30d_floor
    )


def _scan_holdout(
    model,
    X_holdout: pd.DataFrame,
    opens: pd.DataFrame,
    *,
    horizon_h: int,
    anchors_utc: tuple[int, ...],
    btc_context: pd.DataFrame | None = None,
    btc_7d_gate: float = 0.0,
    btc_30d_floor: float = -0.10,
) -> pd.DataFrame:
    """Score one temporal holdout using the exact live timing contract."""
    if not isinstance(X_holdout.index, pd.MultiIndex):
        raise ValueError("X_holdout must use a (timestamp, market) MultiIndex")

    next_open_returns = _next_open_returns(opens, horizon_h)
    timestamps = pd.DatetimeIndex(
        X_holdout.index.get_level_values("timestamp").unique()
    ).sort_values()
    rows: list[dict] = []

    for ts in timestamps:
        if not _is_exact_anchor(ts, anchors_utc):
            continue
        if btc_context is not None and not _strict_long_regime_active(
            btc_context,
            ts,
            btc_7d_gate=btc_7d_gate,
            btc_30d_floor=btc_30d_floor,
        ):
            continue

        try:
            features = X_holdout.xs(ts, level="timestamp").fillna(0.0)
            realized = next_open_returns.loc[ts]
        except KeyError:
            continue

        # Match fetch_and_rank: neutral-fill feature gaps, then discard rows
        # that contain no signal at all.
        features = features[features.any(axis=1)]
        if len(features) < 2:
            continue

        scores = pd.Series(
            model.predict(features),
            index=features.index,
            dtype=float,
            name="score",
        )
        batch_std = float(scores.std())
        if not np.isfinite(batch_std) or batch_std <= 0:
            continue

        valid_returns = realized.reindex(scores.index).dropna()
        if valid_returns.empty:
            continue
        valid_scores = scores.loc[valid_returns.index]

        for market in valid_scores.index:
            score = float(valid_scores.loc[market])
            rows.append(
                {
                    "timestamp": ts,
                    "market": market,
                    "sigma": abs(score) / batch_std,
                    "score": score,
                    "ret": float(valid_returns.loc[market]),
                }
            )

    return pd.DataFrame(rows, columns=SAMPLE_COLUMNS)


def _calibrate_buckets(samples: pd.DataFrame) -> list[dict]:
    """Aggregate scan samples without changing the established bucket schema."""
    buckets: list[dict] = []
    if samples.empty:
        return buckets

    for low, high in BUCKET_BOUNDS:
        mask = samples["sigma"] >= low
        if high is not None:
            mask &= samples["sigma"] < high
        group = samples.loc[mask]
        if group.empty:
            continue

        signed = np.sign(group["score"]) * group["ret"]
        hit = np.sign(group["score"]) == np.sign(group["ret"])
        return_std = float(group["ret"].std()) if len(group) > 1 else 0.0
        buckets.append(
            {
                "sigma_low": low,
                "sigma_high": high,
                "n": int(len(group)),
                "hit_rate": float(hit.mean()),
                "mean_signed_return_pct": float(signed.mean() * 100.0),
                "mean_abs_return_pct": float(group["ret"].abs().mean() * 100.0),
                "std_return_pct": return_std * 100.0,
            }
        )
    return buckets


def _provenance() -> dict:
    """Machine-readable declaration of the calibration data contract."""
    return {
        "oos": True,
        "split": "build_dataset temporal holdout only",
        "holdout_ratio": HOLDOUT_RATIO,
        "lag_bars": 1,
        "return_contract": "open[t+1] -> open[t+1+horizon], absolute return",
        "feature_nan_policy": "fillna(0), drop all-zero rows",
        "anchors_utc": {
            "short_6h": list(SHORT_ANCHORS_UTC),
            "long_12h": list(LONG_ANCHORS_UTC),
        },
        "long_regime": {
            "strict": True,
            "btc_ret_7d_gt": float(
                getattr(config.LongModel, "BTC_7D_RETURN_GATE", 0.0)
            ),
            "btc_ret_30d_gt": float(
                getattr(config.LongModel, "BTC_30D_RETURN_FLOOR", -0.10)
            ),
        },
        "models": {
            "short_6h": "models/xsec_6h.pkl",
            "long_12h": "models/xsec_12h.pkl",
        },
    }


def build_calibration_document(
    *,
    dataset_builder: Callable | None = None,
    model_loader: Callable | None = None,
    regime_builder: Callable | None = None,
    generated_at: str | None = None,
) -> dict:
    """Build both production calibrations with injectable dependencies for tests."""
    if dataset_builder is None:
        from data.dataset import build_dataset

        dataset_builder = build_dataset
    if model_loader is None:
        from models.xgb_ranker import XSecRanker

        model_loader = XSecRanker.load
    if regime_builder is None:
        from data.features import compute_btc_regime

        regime_builder = compute_btc_regime

    specs = (
        (
            "short_6h",
            6,
            SHORT_ANCHORS_UTC,
            ROOT / "models" / "xsec_6h.pkl",
            False,
        ),
        (
            "long_12h",
            12,
            LONG_ANCHORS_UTC,
            ROOT / "models" / "xsec_12h.pkl",
            True,
        ),
    )
    document: dict = {}

    for side_key, horizon_h, anchors_utc, model_path, is_long in specs:
        dataset = dataset_builder(
            days=DATASET_DAYS,
            holdout_ratio=HOLDOUT_RATIO,
            side="unified",
            target="absolute",
            horizon=horizon_h,
        )
        model = model_loader(str(model_path))
        btc_context = regime_builder(dataset["closes"]) if is_long else None
        samples = _scan_holdout(
            model,
            dataset["X_holdout"],
            dataset["opens"],
            horizon_h=horizon_h,
            anchors_utc=anchors_utc,
            btc_context=btc_context,
            btc_7d_gate=float(
                getattr(config.LongModel, "BTC_7D_RETURN_GATE", 0.0)
            ),
            btc_30d_floor=float(
                getattr(config.LongModel, "BTC_30D_RETURN_FLOOR", -0.10)
            ),
        )
        document[side_key] = _calibrate_buckets(samples)

    document["generated_at"] = generated_at or datetime.now(timezone.utc).isoformat()
    document["model_version"] = (
        "F1 unified production models; temporal-holdout next-open calibration"
    )
    document["_provenance"] = _provenance()
    return document


def write_calibration(document: dict, output_path: Path) -> None:
    """Atomically replace the calibration artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    os.replace(temporary, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild OOS next-open sigma calibration"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    document = build_calibration_document()
    write_calibration(document, args.output)
    print(
        "calibration rebuilt: "
        f"short_n={sum(b['n'] for b in document['short_6h'])} "
        f"long_n={sum(b['n'] for b in document['long_12h'])} "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
