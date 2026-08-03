#!/usr/bin/env python3
"""F1 retrain pipeline: both horizons, unified features, promotion gate, calibration rebuild.

Runs weekly via systemd (xsec-retrain.timer). Single entry point that:

  1. Pre-flight: data freshness, disk space, lock file
  2. Train candidate models for 6h and 12h (unified features, absolute target)
  3. Promotion gate per horizon:
     - new_holdout_ic >= old_holdout_ic - 0.015
     - new_holdout_ic >= 0.040 (absolute floor)
     - 12h only: at least 10 production-aligned LONG periods and mean net > 0
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

from config import config

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
ARCHIVE_DIR = MODELS_DIR / "archive"
HISTORY_FILE = ROOT / "output" / "retrain_history.json"
LOCK_FILE = ROOT / "output" / ".retrain.lock"

# Promotion gate thresholds
IC_DELTA_THRESHOLD = -0.015   # new IC must be at most 1.5pp worse than old
IC_ABSOLUTE_FLOOR = 0.040     # new IC must be at least 0.040
MIN_LONG_PERIODS = 10         # minimum eligible 12h LONG slots for promotion
MAX_DATA_AGE_HOURS = 6        # refuse retrain if DB data stale. xsec-alpha.timer fires every 6h, so any retrain run between fetches sees data at most 6h old. Hard staleness (>6h) means a fetch silently failed and we should not retrain on partial data.
_BITGET_TRADABLE_CACHE: set[str] | None = None


def _current_bitget_tradable_markets(markets) -> set[str] | None:
    """Load the live execution universe once per retrain process; fail closed."""
    if not getattr(config.Portfolio, "LIVE_REQUIRE_BITGET_TRADABLE", False):
        return None

    global _BITGET_TRADABLE_CACHE
    if _BITGET_TRADABLE_CACHE is None:
        from utils.bitget import filter_markets_to_bitget, load_bitget_usdt_perp_map

        contract_map = load_bitget_usdt_perp_map()
        _BITGET_TRADABLE_CACHE = set(
            filter_markets_to_bitget(list(markets), contract_map)
        )
    return set(_BITGET_TRADABLE_CACHE)


def _measure_long_holdout_economics(
    scores: pd.Series,
    opens: pd.DataFrame,
    btc_context: pd.DataFrame,
    *,
    horizon: int = 12,
    anchor_hour_utc: int = 11,
    long_n: int = 5,
    btc_7d_gate: float = 0.0,
    btc_30d_floor: float = -0.10,
    round_trip_cost: float = 0.002,
    tradable_markets: set[str] | None = None,
) -> dict:
    """Measure production-aligned LONG economics from already-computed scores."""
    next_open_returns = opens.shift(-(horizon + 1)) / opens.shift(-1) - 1
    timestamps = pd.DatetimeIndex(
        scores.index.get_level_values("timestamp").unique()
    ).sort_values()

    gross_returns: list[float] = []
    net_returns: list[float] = []
    hits: list[float] = []
    previous_longs: set[str] = set()

    for ts in timestamps:
        if (pd.Timestamp(ts).hour - anchor_hour_utc) % horizon != 0:
            continue

        try:
            regime = btc_context.loc[ts]
            btc_7d = float(regime["btc_ret_7d"])
            btc_30d = float(regime["btc_ret_30d"])
        except (KeyError, TypeError, ValueError):
            previous_longs = set()
            continue

        regime_active = bool(
            np.isfinite(btc_7d)
            and np.isfinite(btc_30d)
            and btc_7d > btc_7d_gate
            and btc_30d > btc_30d_floor
        )
        if not regime_active:
            previous_longs = set()
            continue

        try:
            slot_scores = scores.xs(ts, level="timestamp").dropna()
            slot_returns = next_open_returns.loc[ts]
        except KeyError:
            previous_longs = set()
            continue
        if tradable_markets is not None:
            slot_scores = slot_scores[slot_scores.index.isin(tradable_markets)]
        if len(slot_scores) < long_n:
            previous_longs = set()
            continue

        long_markets = slot_scores.nlargest(long_n).index.tolist()
        realized = slot_returns.reindex(long_markets)
        if len(realized) != long_n or realized.isna().any():
            previous_longs = set()
            continue

        current_longs = set(long_markets)
        turnover = len(current_longs - previous_longs) / long_n
        gross = float(realized.mean())
        net = gross - turnover * round_trip_cost

        gross_returns.append(gross)
        net_returns.append(net)
        hits.append(float(gross > 0))
        previous_longs = current_longs

    if not gross_returns:
        return {
            "long_gross": None,
            "long_net": None,
            "long_hit_rate": None,
            "n_long_periods": 0,
        }

    return {
        "long_gross": float(np.mean(gross_returns)),
        "long_net": float(np.mean(net_returns)),
        "long_hit_rate": float(np.mean(hits)),
        "n_long_periods": len(gross_returns),
    }


def _measure_aligned_ic(
    scores: pd.Series,
    opens: pd.DataFrame,
    *,
    horizon: int,
    anchor_hours_utc: tuple[int, ...],
    min_cross_section: int = 20,
) -> list[float]:
    """Measure non-overlapping next-open IC at exact production anchors."""
    next_open_returns = opens.shift(-(horizon + 1)) / opens.shift(-1) - 1
    anchors = set(anchor_hours_utc)
    values: list[float] = []
    timestamps = pd.DatetimeIndex(
        scores.index.get_level_values("timestamp").unique()
    ).sort_values()
    for ts in timestamps:
        if pd.Timestamp(ts).hour not in anchors or ts not in next_open_returns.index:
            continue
        try:
            slot_scores = scores.xs(ts, level="timestamp")
        except KeyError:
            continue
        aligned = pd.concat(
            [
                slot_scores.rename("score"),
                next_open_returns.loc[ts].rename("actual"),
            ],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(aligned) < min_cross_section:
            continue
        ic, _ = spearmanr(aligned["score"], aligned["actual"])
        if np.isfinite(ic):
            values.append(float(ic))
    return values


def _passes_long_economic_gate(long_net: float | None, n_long_periods: int) -> bool:
    """Return whether a 12h candidate has enough profitable LONG observations."""
    return bool(
        n_long_periods >= MIN_LONG_PERIODS
        and long_net is not None
        and np.isfinite(long_net)
        and long_net > 0
    )


def _evaluate_promotion_gate(
    *,
    horizon: int,
    new_ic: float,
    old_ic: float,
    min_delta: float,
    force: bool,
    long_net: float | None = None,
    n_long_periods: int = 0,
) -> dict:
    """Evaluate IC gates plus the 12h-only economic gate without side effects."""
    delta = new_ic - old_ic if old_ic != float("-inf") else 0.0
    pass_delta = delta >= min_delta
    pass_floor = new_ic >= IC_ABSOLUTE_FLOOR
    pass_long_economics = (
        _passes_long_economic_gate(long_net, n_long_periods)
        if horizon == 12
        else True
    )
    promoted = bool(
        force or (pass_delta and pass_floor and pass_long_economics)
    )
    return {
        "delta": delta,
        "pass_delta": bool(pass_delta),
        "pass_floor": bool(pass_floor),
        "pass_long_economics": bool(pass_long_economics),
        "promoted": promoted,
    }


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
    """Return IC statistics and, for 12h, production-aligned LONG economics.

    Used by both the promotion gate (which only needs the scalar IC) and
    the dashboard export (which needs the full stats for the reconciliation
    card). Format matches output/holdout_report.json schema.
    """
    from data.dataset import build_dataset
    from models.xgb_ranker import XSecRanker

    ds = build_dataset(days=60, holdout_ratio=0.2, side="unified",
                       target="absolute", horizon=horizon)
    model = XSecRanker.load(str(model_path))
    X_h = ds["X_holdout"].fillna(0.0)
    preds = pd.Series(model.predict(X_h), index=X_h.index)

    anchor_hours = (11, 23) if horizon == 12 else (5, 11, 17, 23)
    per_slot = _measure_aligned_ic(
        preds,
        ds["opens"],
        horizon=horizon,
        anchor_hours_utc=anchor_hours,
    )

    n = len(per_slot)
    if n == 0:
        stats = {"ic": float("nan"), "tstat": None, "n_periods": 0}
    else:
        mean = sum(per_slot) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in per_slot) / (n - 1)
            std = var ** 0.5
            tstat = mean / (std / (n ** 0.5)) if std > 0 else None
        else:
            tstat = None
        stats = {
            "ic": float(mean),
            "tstat": float(tstat) if tstat is not None else None,
            "n_periods": n,
        }

    if horizon == 12:
        from data.features import compute_btc_regime

        fee_bps = float(getattr(config.Costs, "ONE_WAY_FEE_BPS", 6.0))
        slippage_bps = float(getattr(config.Costs, "SLIPPAGE_BPS", 4.0))
        round_trip_cost = 2.0 * (fee_bps + slippage_bps) / 10000.0
        long_stats = _measure_long_holdout_economics(
            preds,
            ds["opens"],
            compute_btc_regime(ds["closes"]),
            horizon=12,
            anchor_hour_utc=int(
                getattr(config.LongModel, "REBALANCE_ANCHOR_HOUR_UTC", 11)
            ),
            long_n=int(getattr(config.LongModel, "LONG_N", 5)),
            btc_7d_gate=float(
                getattr(config.LongModel, "BTC_7D_RETURN_GATE", 0.0)
            ),
            btc_30d_floor=float(
                getattr(config.LongModel, "BTC_30D_RETURN_FLOOR", -0.10)
            ),
            round_trip_cost=round_trip_cost,
            tradable_markets=_current_bitget_tradable_markets(ds["opens"].columns),
        )
        long_stats["pass_long_economics"] = _passes_long_economic_gate(
            long_stats["long_net"],
            long_stats["n_long_periods"],
        )
        stats.update(long_stats)

    return stats


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
        merged = {
            "ic": round(float(stats["ic"]), 4) if stats["ic"] == stats["ic"] else None,  # NaN check
            "tstat": round(float(stats["tstat"]), 2) if stats.get("tstat") is not None else old.get("tstat"),
            "n_periods": int(stats.get("n_periods") or old.get("n_periods") or 0),
            "hit_2sigma": old.get("hit_2sigma"),
            "e_signed_pct": old.get("e_signed_pct"),
            "horizon_h": h,
            "regime_filter": old.get("regime_filter"),
            "target": "absolute",
            "price_source": "open",
            "execution_lag_bars": 1,
            "anchor_hours_utc": [11, 23] if h == 12 else [5, 11, 17, 23],
            "non_overlapping": True,
        }
        if h == 12:
            for key in ("long_gross", "long_net", "long_hit_rate"):
                value = stats.get(key)
                merged[key] = (
                    round(float(value), 6)
                    if value is not None and np.isfinite(value)
                    else None
                )
            merged["n_long_periods"] = int(
                stats.get("n_long_periods") or 0
            )
            merged["pass_long_economics"] = bool(
                stats.get("pass_long_economics", False)
            )
        return merged

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
    """Regenerate calibration through the audited OOS timing contract."""
    cmd = [sys.executable, str(ROOT / "scripts" / "rebuild_calibration.py")]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                        cwd=str(ROOT), timeout=180)
        return True
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or str(e))[-1000:]
        log(f"Calibration rebuild failed: {detail}")
        return False
    except subprocess.TimeoutExpired:
        log("Calibration rebuild failed: timeout 180s")
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
                new_stats = measure_holdout_stats(candidate_path, horizon)
                new_ic = float(new_stats["ic"])
                old_stats = (
                    measure_holdout_stats(prod_path, horizon)
                    if prod_path.exists()
                    else None
                )
                old_ic = (
                    float(old_stats["ic"])
                    if old_stats is not None
                    else float("-inf")
                )
            except Exception as e:
                log(f"{horizon}h measurement failed: {e}")
                results[horizon] = {"status": "measure_fail", "error": str(e)}
                continue

            log(f"{horizon}h: old_ic={old_ic:+.4f}  new_ic={new_ic:+.4f}  Δ={new_ic-old_ic:+.4f}")

            # Promotion gate
            decision = _evaluate_promotion_gate(
                horizon=horizon,
                new_ic=new_ic,
                old_ic=old_ic,
                min_delta=args.min_delta,
                force=args.force,
                long_net=new_stats.get("long_net"),
                n_long_periods=int(new_stats.get("n_long_periods", 0)),
            )
            delta = decision["delta"]
            promoted = decision["promoted"]

            results[horizon] = {
                "old_ic": round(old_ic, 4) if old_ic != float("-inf") else None,
                "new_ic": round(new_ic, 4),
                "delta": round(delta, 4),
                "pass_delta": decision["pass_delta"],
                "pass_floor": decision["pass_floor"],
                "promoted": promoted,
                "forced": bool(args.force),
            }
            if horizon == 12:
                for key in ("long_gross", "long_net", "long_hit_rate"):
                    value = new_stats.get(key)
                    results[horizon][key] = (
                        round(float(value), 6)
                        if value is not None and np.isfinite(value)
                        else None
                    )
                results[horizon]["n_long_periods"] = int(
                    new_stats.get("n_long_periods", 0)
                )
                results[horizon]["pass_long_economics"] = decision[
                    "pass_long_economics"
                ]
                log(
                    "12h LONG economics: gross=%s net=%s hit=%s n=%s pass=%s"
                    % (
                        results[horizon]["long_gross"],
                        results[horizon]["long_net"],
                        results[horizon]["long_hit_rate"],
                        results[horizon]["n_long_periods"],
                        results[horizon]["pass_long_economics"],
                    )
                )

            if args.dry_run:
                log(f"{horizon}h: dry-run — would {'promote' if promoted else 'reject'}")
                candidate_path.unlink(missing_ok=True)
                continue

            if not promoted:
                economic_reason = (
                    f", long_econ={decision['pass_long_economics']}"
                    if horizon == 12
                    else ""
                )
                log(
                    f"{horizon}h: REJECTED "
                    f"(delta={delta:+.4f}, floor={new_ic:.4f}{economic_reason})"
                )
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
            "force": args.force,
            "calibration_rebuilt": calib_rebuilt,
        })

        log(f"=== retrain_pipeline END  results: {results}  calib_rebuilt={calib_rebuilt} ===")

    finally:
        LOCK_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
