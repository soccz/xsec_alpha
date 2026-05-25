#!/usr/bin/env python3
"""F1 retrain pipeline: both horizons, unified features, promotion gate, calibration rebuild.

Runs weekly via systemd (xsec-retrain.timer). Single entry point that:

  1. Pre-flight: data freshness, disk space, lock file
  2. Train candidate models for 6h and 12h (unified features, absolute target)
  3. Promotion gate per horizon:
     - new_holdout_ic >= old_holdout_ic - 0.015
     - new_holdout_ic >= 0.040 (absolute floor)
  4. Accept: archive current, promote candidate, rebuild calibration
  5. Reject: keep current model live, log reason

Outputs:
  - logs/retrain.log          : systemd log
  - output/retrain_history.json: per-run promotion decisions

Usage:
    python scripts/retrain_pipeline.py [--dry-run] [--force]
    python scripts/retrain_pipeline.py --horizons 6 --min-delta 0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
ARCHIVE_DIR = MODELS_DIR / "archive"
HISTORY_FILE = ROOT / "output" / "retrain_history.json"
LOCK_FILE = ROOT / "output" / ".retrain.lock"

# Promotion gate thresholds
IC_DELTA_THRESHOLD = -0.015   # new IC must be at most 1.5pp worse than old
IC_ABSOLUTE_FLOOR = 0.040     # new IC must be at least 0.040
MAX_DATA_AGE_HOURS = 4        # refuse retrain if DB data stale (was 3 — boundary at Sun 20:00 UTC vs 17:00 fetch tripped 3.005h)


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] {msg}", flush=True)


def preflight() -> tuple[bool, str]:
    """Return (ok, reason). ok=False blocks retrain."""
    # Lock file
    if LOCK_FILE.exists():
        return False, f"lock file present: {LOCK_FILE} (previous run?)"

    # Data freshness
    try:
        from data.database import get_latest_db_timestamp
        latest = get_latest_db_timestamp()
        if latest is None:
            return False, "no data in DB"
        age_h = (datetime.now(timezone.utc) - latest).total_seconds() / 3600
        if age_h > MAX_DATA_AGE_HOURS:
            return False, f"data stale ({age_h:.1f}h old, limit {MAX_DATA_AGE_HOURS}h)"
    except Exception as e:
        return False, f"DB check failed: {e}"

    # Disk space (need ≥500MB free for models + logs)
    try:
        stat = shutil.disk_usage(str(ROOT))
        free_mb = stat.free / (1024 * 1024)
        if free_mb < 500:
            return False, f"low disk space: {free_mb:.0f}MB free"
    except Exception:
        pass

    return True, "ok"


def measure_holdout_ic(model_path: Path, horizon: int) -> float:
    """Return mean per-slot Spearman IC on 20% holdout."""
    return measure_holdout_stats(model_path, horizon)["ic"]


def measure_holdout_stats(model_path: Path, horizon: int) -> dict:
    """Return mean IC + t-stat + n_periods on 20% holdout.

    Used by both the promotion gate (which only needs the scalar IC) and
    the dashboard export (which needs the full stats for the reconciliation
    card). Format matches output/holdout_report.json schema.
    """
    from data.dataset import build_dataset
    from models.xgb_ranker import XSecRanker

    ds = build_dataset(days=60, holdout_ratio=0.2, side="unified",
                       target="absolute", horizon=horizon)
    model = XSecRanker.load(str(model_path))
    X_h = ds["X_holdout"]
    y_h = ds["y_holdout"]
    preds = pd.Series(model.predict(X_h), index=X_h.index)

    per_slot = []
    for ts, g in preds.groupby(level="timestamp"):
        y_slice = y_h.loc[ts]
        if len(g) < 20:
            continue
        ic, _ = spearmanr(g.values, y_slice.values)
        if not np.isnan(ic):
            per_slot.append(float(ic))

    n = len(per_slot)
    if n == 0:
        return {"ic": float("nan"), "tstat": None, "n_periods": 0}
    mean = sum(per_slot) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in per_slot) / (n - 1)
        std = var ** 0.5
        tstat = mean / (std / (n ** 0.5)) if std > 0 else None
    else:
        tstat = None
    return {"ic": float(mean), "tstat": float(tstat) if tstat is not None else None, "n_periods": n}


def _write_holdout_report(per_horizon: dict[int, dict], stamp: str) -> None:
    """Update output/holdout_report.json with fresh holdout stats + provenance.

    `per_horizon[h]` should be the dict returned by measure_holdout_stats(...).
    Static fields (hit_2sigma, e_signed_pct, regime_filter) are preserved from
    the prior file when present so we do not silently zero out values not yet
    re-measured.
    """
    out_path = ROOT / "output" / "holdout_report.json"
    prior = {}
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
        except Exception:
            prior = {}

    def merge(side_key: str, h: int, stats: dict) -> dict:
        old = prior.get(side_key, {}) if isinstance(prior.get(side_key), dict) else {}
        return {
            "ic": round(float(stats["ic"]), 4) if stats["ic"] == stats["ic"] else None,  # NaN check
            "tstat": round(float(stats["tstat"]), 2) if stats.get("tstat") is not None else old.get("tstat"),
            "n_periods": int(stats.get("n_periods") or old.get("n_periods") or 0),
            "hit_2sigma": old.get("hit_2sigma"),
            "e_signed_pct": old.get("e_signed_pct"),
            "horizon_h": h,
            "regime_filter": old.get("regime_filter"),
        }

    new_doc = {
        "_provenance": {
            "asof": stamp,
            "source": f"retrain_pipeline.py auto-emit at {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "documented_in": "projects/xsec-alpha/index.html (Executive Summary)",
            "regenerate_with": "python scripts/retrain_pipeline.py",
            "note": "Auto-updated after a successful promotion. Static fields "
                    "(hit_2sigma, e_signed_pct, regime_filter) are preserved from "
                    "the prior file until re-measured by an explicit holdout pass.",
        },
    }
    if 6 in per_horizon:
        new_doc["short_h6"] = merge("short_h6", 6, per_horizon[6])
    elif "short_h6" in prior:
        new_doc["short_h6"] = prior["short_h6"]
    if 12 in per_horizon:
        new_doc["long_h12"] = merge("long_h12", 12, per_horizon[12])
    elif "long_h12" in prior:
        new_doc["long_h12"] = prior["long_h12"]

    out_path.write_text(json.dumps(new_doc, indent=2, ensure_ascii=False))
    log(f"holdout_report.json updated: {sorted(per_horizon.keys())}")


def train_candidate(horizon: int, candidate_path: Path) -> bool:
    """Run scripts/train.py for a horizon. Returns True on success."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--side", "unified",
        "--model-type", "ensemble",
        "--target", "absolute",
        "--days", "60",
        "--horizon", str(horizon),
        "--model-path", str(candidate_path),
    ]
    log(f"Training candidate {horizon}h → {candidate_path.name}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                        cwd=str(ROOT), timeout=600)
        return True
    except subprocess.CalledProcessError as e:
        log(f"FAILED {horizon}h train: {e.stderr[-500:]}")
        return False
    except subprocess.TimeoutExpired:
        log(f"FAILED {horizon}h train: timeout 600s")
        return False


def rebuild_calibration() -> bool:
    """Regenerate output/calibration_sigma.json for new models."""
    cmd = [sys.executable, "-c", """
import sys, json; sys.path.insert(0, '.')
import pandas as pd, numpy as np
from data.features import (load_and_pivot, load_binance_pivot, compute_unified_factors,
    crosssection_zscore, build_top_liquidity_universe_index, filter_long_frame_by_universe,
    UNIFIED_CALENDAR_COLS)
from models.xgb_ranker import XSecRanker
from config import config

closes, opens, highs, lows, volumes = load_and_pivot(days=60)
warmup = config.Data.MIN_ROWS_PER_COIN
closes=closes.iloc[warmup:]; opens=opens.iloc[warmup:]
highs=highs.iloc[warmup:]; lows=lows.iloc[warmup:]; volumes=volumes.iloc[warmup:]
binance_closes = load_binance_pivot(closes.columns.tolist(), days=60)
if not binance_closes.empty: binance_closes = binance_closes.iloc[warmup:]
uni, _ = build_top_liquidity_universe_index(closes, volumes, top_n=100)
uf = compute_unified_factors(closes,opens,highs,lows,volumes,binance_closes=binance_closes)
zcols = [c for c in uf.columns if c not in UNIFIED_CALENDAR_COLS]
uf = crosssection_zscore(uf, cols=zcols); uf = filter_long_frame_by_universe(uf, uni)

def scan(model, h, slots):
    ts_all = sorted([t for t in uf.index.get_level_values('timestamp').unique() if pd.notna(t)])
    rows=[]
    for ts in [t for t in ts_all if pd.Timestamp(t).hour in slots]:
        fwd = ts + pd.Timedelta(hours=h)
        if fwd not in closes.index: continue
        try: feats = uf.xs(ts, level='timestamp').fillna(0)
        except KeyError: continue
        feats = feats[feats.any(axis=1)]
        if len(feats)<30: continue
        s = pd.Series(model.predict(feats), index=feats.index)
        p0,p1 = closes.loc[ts], closes.loc[fwd]
        ret = ((p1-p0)/p0).reindex(s.index).dropna()
        s = s.loc[ret.index]; gstd = s.std()
        if gstd==0: continue
        for mkt in s.index:
            rows.append({'sigma':abs(s[mkt])/gstd,'score':s[mkt],'ret':ret[mkt]})
    return pd.DataFrame(rows)

def cal(df):
    out=[]
    for lo,hi in [(0,0.5),(0.5,1),(1,1.5),(1.5,2),(2,100)]:
        g = df[(df['sigma']>=lo)&(df['sigma']<hi)]
        if len(g)==0: continue
        signed = np.sign(g['score'])*g['ret']
        hit = (np.sign(g['score'])==np.sign(g['ret'])).mean()
        out.append({'sigma_low':lo,'sigma_high':(hi if hi<100 else None),'n':int(len(g)),
                    'hit_rate':float(hit),'mean_signed_return_pct':float(signed.mean()*100),
                    'mean_abs_return_pct':float(g['ret'].abs().mean()*100),
                    'std_return_pct':float(g['ret'].std()*100)})
    return out

short_df = scan(XSecRanker.load('models/xsec_6h.pkl'), 6, (5,11,17,23))
long_df = scan(XSecRanker.load('models/xsec_12h.pkl'), 12, (11,23))
calib = {'short_6h':cal(short_df),'long_12h':cal(long_df),
         'generated_at':pd.Timestamp.utcnow().isoformat(),
         'model_version':'F1 unified (auto-rebuild from retrain_pipeline)'}
with open('output/calibration_sigma.json','w') as f:
    json.dump(calib, f, indent=2, default=str)
print('calibration rebuilt')
"""]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                        cwd=str(ROOT), timeout=180)
        return True
    except Exception as e:
        log(f"Calibration rebuild failed: {e}")
        return False


def _refresh_dashboard_export() -> None:
    """Rebuild encrypted dashboard payloads after a successful promotion.
    Non-fatal — must not abort the retrain run.
    """
    target = Path("/home/soccz/22tb/soccz.github.io/projects/xsec-alpha/dashboard/data")
    if not target.parent.exists():
        log("Dashboard target dir absent; skipping export.")
        return
    try:
        from utils.dashboard_export import PIN_DEFAULT, export_to
        written = export_to(target, PIN_DEFAULT)
        log(f"Dashboard export refreshed: {len(written)} files")
    except Exception as e:
        log(f"Dashboard export failed (non-fatal): {e}")


def append_history(entry: dict) -> None:
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    existing = []
    if HISTORY_FILE.exists():
        try:
            existing = json.loads(HISTORY_FILE.read_text())
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    existing.append(entry)
    HISTORY_FILE.write_text(json.dumps(existing, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Train + measure, do not promote")
    ap.add_argument("--force", action="store_true", help="Skip promotion gate (dangerous)")
    ap.add_argument(
        "--horizons",
        default="6,12",
        help="Comma-separated horizons to retrain/promote (default: 6,12)",
    )
    ap.add_argument(
        "--min-delta",
        type=float,
        default=IC_DELTA_THRESHOLD,
        help=(
            "Minimum allowed new-old holdout IC delta for promotion. "
            "Default keeps the existing weekly refresh tolerance; use 0 for strict improvement."
        ),
    )
    args = ap.parse_args()

    log(f"=== retrain_pipeline START (dry_run={args.dry_run}, force={args.force}) ===")
    try:
        horizons = tuple(int(h.strip()) for h in str(args.horizons).split(",") if h.strip())
    except ValueError:
        log(f"Invalid --horizons value: {args.horizons!r}")
        sys.exit(2)
    invalid = [h for h in horizons if h not in (6, 12)]
    if invalid or not horizons:
        log(f"Invalid horizons {invalid}; supported horizons are 6 and 12")
        sys.exit(2)

    # Pre-flight
    ok, reason = preflight()
    if not ok:
        log(f"PREFLIGHT FAIL: {reason}")
        append_history({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "preflight_fail",
            "reason": reason,
        })
        sys.exit(2)

    # Lock
    LOCK_FILE.parent.mkdir(exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))

    try:
        ARCHIVE_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
        results = {}

        for horizon in horizons:
            name = f"xsec_{horizon}h"
            prod_path = MODELS_DIR / f"{name}.pkl"
            candidate_path = MODELS_DIR / f"{name}_candidate.pkl"

            # Train candidate
            if not train_candidate(horizon, candidate_path):
                results[horizon] = {"status": "train_fail"}
                continue

            # Measure both
            try:
                new_ic = measure_holdout_ic(candidate_path, horizon)
                old_ic = measure_holdout_ic(prod_path, horizon) if prod_path.exists() else float("-inf")
            except Exception as e:
                log(f"{horizon}h measurement failed: {e}")
                results[horizon] = {"status": "measure_fail", "error": str(e)}
                continue

            log(f"{horizon}h: old_ic={old_ic:+.4f}  new_ic={new_ic:+.4f}  Δ={new_ic-old_ic:+.4f}")

            # Promotion gate
            delta = new_ic - old_ic if old_ic != float("-inf") else 0.0
            pass_delta = delta >= args.min_delta
            pass_floor = new_ic >= IC_ABSOLUTE_FLOOR
            promoted = (pass_delta and pass_floor) or args.force

            results[horizon] = {
                "old_ic": round(old_ic, 4) if old_ic != float("-inf") else None,
                "new_ic": round(new_ic, 4),
                "delta": round(delta, 4),
                "pass_delta": pass_delta,
                "pass_floor": pass_floor,
                "promoted": promoted,
            }

            if args.dry_run:
                log(f"{horizon}h: dry-run — would {'promote' if promoted else 'reject'}")
                candidate_path.unlink(missing_ok=True)
                continue

            if not promoted:
                log(f"{horizon}h: REJECTED (delta={delta:+.4f}, floor={new_ic:.4f})")
                candidate_path.unlink(missing_ok=True)
                continue

            # Archive + promote
            if prod_path.exists():
                archive_path = ARCHIVE_DIR / f"{name}_{stamp}.pkl"
                shutil.move(str(prod_path), str(archive_path))
                log(f"Archived {name} → {archive_path.name}")
            shutil.move(str(candidate_path), str(prod_path))
            log(f"{horizon}h: PROMOTED")

        # Rebuild calibration if any model promoted
        any_promoted = any(r.get("promoted") for r in results.values())
        calib_rebuilt = False
        if any_promoted and not args.dry_run:
            calib_rebuilt = rebuild_calibration()

        # Auto-update holdout_report.json from the newly-promoted prod models.
        # Re-measure stats fresh (with t-stat + n_periods) so the dashboard
        # reconciliation card reflects the just-promoted weights, not the
        # frozen 2026-04-25 baseline.
        if any_promoted and not args.dry_run:
            per_h: dict[int, dict] = {}
            for h in (6, 12):
                prod = MODELS_DIR / f"xsec_{h}h.pkl"
                if not prod.exists():
                    continue
                try:
                    per_h[h] = measure_holdout_stats(prod, h)
                except Exception as e:
                    log(f"holdout stats refresh failed for h={h}: {e}")
            if per_h:
                try:
                    _write_holdout_report(per_h, stamp)
                except Exception as e:
                    log(f"holdout_report.json write failed (non-fatal): {e}")

        # Refresh the public dashboard's encrypted payloads so the new
        # holdout/calibration numbers go live immediately after promotion.
        if any_promoted and not args.dry_run:
            _refresh_dashboard_export()

        # Log run
        append_history({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "ok" if any(r.get("promoted") for r in results.values()) else "no_promotion",
            "results": results,
            "dry_run": args.dry_run,
            "calibration_rebuilt": calib_rebuilt,
        })

        log(f"=== retrain_pipeline END  results: {results}  calib_rebuilt={calib_rebuilt} ===")

    finally:
        LOCK_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
