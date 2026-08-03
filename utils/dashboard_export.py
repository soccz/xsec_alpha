"""Build encrypted JSON payloads for the public xsec_alpha dashboard.

Output format matches projects/prelude/dashboard/data/*.json:
    PBKDF2-HMAC-SHA256 (250k iter, 64B) -> AES-256-CBC + HMAC-SHA256(salt|iv|ct).

Three payloads:
    summary.json   - asof, gates, latest_ic, latest_picks, drift, calibration, ops
    history.json   - asof, ic_series (per side), pick_history (last N days)
    accuracy.json  - asof, rolling IC/realized stats, calibration buckets
"""

from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import json
import math
import os
import secrets
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

PBKDF2_ITERATIONS = 250_000
DKLEN = 64  # 32B AES key + 32B HMAC key
PIN_DEFAULT = "9963"


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def encrypt_payload(plaintext: bytes, pin: str, iterations: int = PBKDF2_ITERATIONS) -> dict:
    salt = secrets.token_bytes(16)
    iv = secrets.token_bytes(16)
    keymat = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, iterations, dklen=DKLEN)
    aes_key, mac_key = keymat[:32], keymat[32:64]

    padder = PKCS7(128).padder()
    padded = padder.update(plaintext) + padder.finalize()
    encryptor = Cipher(algorithms.AES(aes_key), modes.CBC(iv)).encryptor()
    ct = encryptor.update(padded) + encryptor.finalize()
    mac = hmac.new(mac_key, salt + iv + ct, hashlib.sha256).digest()

    return {
        "encrypted": True,
        "version": 1,
        "kdf": "PBKDF2-HMAC-SHA256",
        "cipher": "AES-256-CBC-HMAC-SHA256",
        "iterations": iterations,
        "salt": _b64(salt),
        "iv": _b64(iv),
        "ct": _b64(ct),
        "mac": _b64(mac),
    }


# ---- IO helpers ----------------------------------------------------------

def _safe_load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[dashboard_export] warn: failed to read {path}: {e}", file=sys.stderr)
        return None


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x or math.isinf(x):  # NaN or +/- Infinity
        return None
    return x


def _to_int(v: Any) -> int | None:
    f = _to_float(v)
    return int(f) if f is not None else None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


# ---- payload builders ----------------------------------------------------

def _ic_series(records: list[dict], side: str | None, horizon_h: int | None,
               require_label: bool = False) -> list[dict]:
    """Filter IC records by (side, horizon_h).

    `require_label=True` drops records that lack the side/horizon fields entirely
    (legacy unlabeled rows from before the multi-side schema landed). Use this
    when the file may contain mixed schema versions; the legacy rows are kept
    out rather than misattributed.
    """
    out = []
    for r in records:
        has_side = "side" in r
        has_h = "horizon_h" in r
        if require_label and (not has_side or not has_h):
            continue
        if side is not None and has_side and r["side"] != side:
            continue
        if horizon_h is not None and has_h and r["horizon_h"] != horizon_h:
            continue
        ts = r.get("timestamp")
        ic = _to_float(r.get("ic"))
        n = _to_int(r.get("n_coins"))
        if ts is None or ic is None:
            continue
        out.append({"t": ts, "ic": round(ic, 6), "n": n})
    out.sort(key=lambda x: x["t"])
    return out


def _lag1_autocorr(vals: list[float]) -> float | None:
    """Lag-1 autocorrelation of a time series. Returns None for n<3."""
    n = len(vals)
    if n < 3:
        return None
    m = sum(vals) / n
    num = sum((vals[i] - m) * (vals[i - 1] - m) for i in range(1, n))
    den = sum((v - m) ** 2 for v in vals)
    if den <= 0:
        return None
    return num / den


def _ic_stats(series: list[dict], lookback_days: int) -> dict:
    """IC summary stats over a lookback window.

    Reports both the naive-IID t-stat and an autocorrelation-corrected t-stat
    using effective N = N(1-rho)/(1+rho). IC time series is rarely IID — the
    corrected t-stat is what an interviewer who has done time-series stats will
    expect. Both are reported so the size of the autocorr penalty is visible.
    """
    base = {"n": 0, "last": None, "mean": None, "std": None, "tstat": None,
            "tstat_ac": None, "lag1": None, "n_eff": None,
            "neg_pct": None, "lookback_days": lookback_days}
    if not series:
        return base
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    vals: list[float] = []
    for r in series:
        try:
            t = datetime.fromisoformat(r["t"])
        except Exception:
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        if t >= cutoff:
            vals.append(r["ic"])
    n = len(vals)
    if n == 0:
        return base
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = var ** 0.5
        tstat = mean / (std / (n ** 0.5)) if std > 0 else None
    else:
        std, tstat = None, None

    rho = _lag1_autocorr(vals)
    if rho is not None and -1 < rho < 1 and std and std > 0:
        n_eff = n * (1 - rho) / (1 + rho)
        # protect against the autocorr correction inflating N when rho < 0
        n_eff = min(max(n_eff, 1.0), float(n))
        tstat_ac = mean / (std / (n_eff ** 0.5))
    else:
        n_eff, tstat_ac = None, None

    neg = sum(1 for v in vals if v < 0)
    return {
        "n": n,
        "last": vals[-1],
        "mean": round(mean, 6),
        "std": round(std, 6) if std is not None else None,
        "tstat": round(tstat, 3) if tstat is not None else None,
        "tstat_ac": round(tstat_ac, 3) if tstat_ac is not None else None,
        "lag1": round(rho, 3) if rho is not None else None,
        "n_eff": round(n_eff, 1) if n_eff is not None else None,
        "neg_pct": round(100 * neg / n, 1),
        "lookback_days": lookback_days,
    }


def _drift_rows(drift: dict | None) -> list[dict]:
    if not drift or "factors" not in drift:
        return []
    rows = []
    for name, body in drift["factors"].items():
        rows.append({
            "factor": name,
            "ic_24h": _to_float(body.get("ic_24h")),
            "ic_7d": _to_float(body.get("ic_7d")),
            "drop_pct": _to_float(body.get("drop_pct")),
            "sign_flipped": bool(body.get("sign_flipped", False)),
            "status": body.get("status"),
            "flag": body.get("flag"),
            "n_slots": body.get("n_slots"),
        })
    return rows


def _picks_from_latest(rows: list[dict]) -> dict:
    out = {"short": [], "watch_long": []}
    for r in rows:
        side = (r.get("side") or "").upper()
        item = {
            "market": r.get("market"),
            "score": _to_float(r.get("score")),
            "entry_price": _to_float(r.get("entry_price")),
            "entry_time": r.get("entry_time"),
            "horizon_h": _to_int(r.get("horizon_h")),
            "bitget_symbol": r.get("bitget_symbol") or None,
            "actionable": str(r.get("actionable", "")).lower() == "true",
        }
        if side == "SHORT":
            out["short"].append(item)
        elif side == "WATCH_LONG":
            out["watch_long"].append(item)
    return out


def _pick_history(ledger: list[dict], days: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    for r in ledger:
        ts = r.get("entry_time")
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(ts)
        except Exception:
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        if t < cutoff:
            continue
        rr = _to_float(r.get("realized_return"))
        nr = _to_float(r.get("net_return"))
        cost = _to_float(r.get("estimated_cost_return"))
        out.append({
            "market": r.get("market"),
            "side": r.get("side"),
            "actionable": str(r.get("actionable", "")).lower() == "true",
            "entry_time": ts,
            "exit_time": r.get("exit_time_actual") or r.get("exit_time_target"),
            "horizon_h": _to_int(r.get("horizon_h")),
            "entry_price": _to_float(r.get("entry_price")),
            "exit_price": _to_float(r.get("exit_price")),
            "realized_pct": round(100 * rr, 4) if rr is not None else None,
            "net_pct": round(100 * nr, 4) if nr is not None else None,
            "estimated_cost_pct": round(100 * cost, 4) if cost is not None else None,
        })
    out.sort(key=lambda x: x["entry_time"], reverse=True)
    return out


# Cost assumptions (round-trip, per pick) — disclosed so reviewers can audit
# the haircut. Ledger rows carry exact estimated_cost_pct; these defaults are
# fallback values for older rows or external exports.
try:
    from config import config as _cfg
    _one_way_bps = float(getattr(_cfg.Costs, "ONE_WAY_FEE_BPS", 6.0)) + float(getattr(_cfg.Costs, "SLIPPAGE_BPS", 4.0))
    _short_extra_bps = float(getattr(_cfg.Costs, "SHORT_EXTRA_COST_BPS", 10.0))
except Exception:
    _one_way_bps = 10.0
    _short_extra_bps = 10.0
COST_PCT_BY_SIDE = {
    "SHORT": (2.0 * _one_way_bps + _short_extra_bps) / 100.0,
    "LONG": (2.0 * _one_way_bps) / 100.0,
    "WATCH_LONG": (2.0 * _one_way_bps) / 100.0,
}


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float] | tuple[None, None]:
    """Wilson 95% CI for a binomial proportion."""
    if n <= 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _clustered_t(picks: list[tuple[str, float]]) -> tuple[int, float | None]:
    """Cluster-aware t-stat: group picks by entry_time, take per-window avg,
    then compute t over those window-level means. Picks within the same
    rebalance window share the same regime and are not independent — using the
    cluster t avoids over-counting.
    Returns (n_windows, t_clustered). t is None if n<2 or std==0.
    """
    by_window: dict[str, list[float]] = {}
    for ts, v in picks:
        by_window.setdefault(ts, []).append(v)
    win_avgs = [sum(vs) / len(vs) for vs in by_window.values()]
    n = len(win_avgs)
    if n < 2:
        return (n, None)
    m = sum(win_avgs) / n
    var = sum((x - m) ** 2 for x in win_avgs) / (n - 1)
    sd = var ** 0.5
    if sd <= 0:
        return (n, None)
    return (n, m / (sd / (n ** 0.5)))


def _realized_stats(history: list[dict], side: str, days_window: list[int]) -> dict:
    """Per-side realized stats with t-stat (naive AND clustered) and Wilson CI.

    `realized_pct` in the ledger is per-trade signed return (already side-adjusted:
    SHORT positive when price drops). Sharpe is per-trade — not annualized;
    annualizing would require block resampling over rebalance buckets to avoid
    the IID assumption.

    Primary hit/t/sharpe fields use net return (legacy rows fall back to gross
    minus configured cost). Explicit ``gross_*`` fields preserve the diagnostic
    comparison. The clustered t groups by entry_time because picks from one
    rebalance window share market regime.
    """
    side_u = side.upper()
    cost = COST_PCT_BY_SIDE.get(side_u, 0.0)
    out = {}
    for d in days_window:
        cutoff = datetime.now(timezone.utc) - timedelta(days=d)
        vals: list[float] = []
        net_vals: list[float] = []
        gross_picks: list[tuple[str, float]] = []
        net_picks: list[tuple[str, float]] = []
        gross_wins = 0
        net_wins = 0
        for r in history:
            if (r.get("side") or "").upper() != side_u:
                continue
            ts = r.get("entry_time")
            if not ts:
                continue
            try:
                t = datetime.fromisoformat(ts)
            except Exception:
                continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if t < cutoff:
                continue
            rp = r.get("realized_pct")
            if rp is None:
                continue
            vals.append(rp)
            npct = r.get("net_pct")
            effective_net = npct if npct is not None else (rp - cost)
            net_vals.append(effective_net)
            gross_picks.append((ts, rp))
            net_picks.append((ts, effective_net))
            if rp > 0:
                gross_wins += 1
            if effective_net > 0:
                net_wins += 1
        n = len(vals)
        if n == 0:
            out[f"d{d}"] = {
                "n": 0, "n_windows": 0, "avg": None, "avg_net": None, "win_pct": None,
                "gross_win_pct": None, "net_win_pct": None,
                "win_ci_lo": None, "win_ci_hi": None,
                "gross_win_ci_lo": None, "gross_win_ci_hi": None,
                "net_win_ci_lo": None, "net_win_ci_hi": None,
                "std": None, "tstat": None, "tstat_clustered": None,
                "sharpe": None, "gross_std": None, "gross_tstat": None,
                "gross_tstat_clustered": None, "gross_sharpe": None,
                "metric_basis": "net_return", "cost_pct": cost,
            }
            continue
        mean = sum(vals) / n
        mean_net = (sum(net_vals) / len(net_vals)) if net_vals else (mean - cost)
        if n > 1:
            gross_var = sum((v - mean) ** 2 for v in vals) / (n - 1)
            gross_std = gross_var ** 0.5
            net_var = sum((v - mean_net) ** 2 for v in net_vals) / (n - 1)
            net_std = net_var ** 0.5
        else:
            gross_std = None
            net_std = None
        gross_tstat = (
            mean / (gross_std / (n ** 0.5))
            if (gross_std and gross_std > 0 and n > 1) else None
        )
        net_tstat = (
            mean_net / (net_std / (n ** 0.5))
            if (net_std and net_std > 0 and n > 1) else None
        )
        gross_sharpe = (mean / gross_std) if (gross_std and gross_std > 0) else None
        net_sharpe = (mean_net / net_std) if (net_std and net_std > 0) else None
        gross_n_windows, gross_tstat_clustered = _clustered_t(gross_picks)
        n_windows, net_tstat_clustered = _clustered_t(net_picks)
        gross_ci_lo, gross_ci_hi = _wilson_ci(gross_wins, n)
        net_ci_lo, net_ci_hi = _wilson_ci(net_wins, n)
        out[f"d{d}"] = {
            "n": n,
            "n_windows": n_windows,
            "avg": round(mean, 4),
            "avg_net": round(mean_net, 4),
            # Primary inference/hit fields are net of configured costs. Gross
            # counterparts stay explicit for backward-compatible analysis.
            "win_pct": round(100 * net_wins / n, 1),
            "gross_win_pct": round(100 * gross_wins / n, 1),
            "net_win_pct": round(100 * net_wins / n, 1),
            "win_ci_lo": round(100 * net_ci_lo, 1) if net_ci_lo is not None else None,
            "win_ci_hi": round(100 * net_ci_hi, 1) if net_ci_hi is not None else None,
            "gross_win_ci_lo": round(100 * gross_ci_lo, 1) if gross_ci_lo is not None else None,
            "gross_win_ci_hi": round(100 * gross_ci_hi, 1) if gross_ci_hi is not None else None,
            "net_win_ci_lo": round(100 * net_ci_lo, 1) if net_ci_lo is not None else None,
            "net_win_ci_hi": round(100 * net_ci_hi, 1) if net_ci_hi is not None else None,
            "std": round(net_std, 4) if net_std is not None else None,
            "tstat": round(net_tstat, 3) if net_tstat is not None else None,
            "tstat_clustered": round(net_tstat_clustered, 3) if net_tstat_clustered is not None else None,
            "sharpe": round(net_sharpe, 3) if net_sharpe is not None else None,
            "gross_n_windows": gross_n_windows,
            "gross_std": round(gross_std, 4) if gross_std is not None else None,
            "gross_tstat": round(gross_tstat, 3) if gross_tstat is not None else None,
            "gross_tstat_clustered": round(gross_tstat_clustered, 3) if gross_tstat_clustered is not None else None,
            "gross_sharpe": round(gross_sharpe, 3) if gross_sharpe is not None else None,
            "metric_basis": "net_return",
            "cost_pct": cost,
        }
    return out


def _model_age_days() -> dict:
    out = {}
    for name in ("xsec_6h.pkl", "xsec_12h.pkl"):
        p = PROJECT_ROOT / "models" / name
        if p.exists():
            age_days = (time.time() - p.stat().st_mtime) / 86400.0
            out[name] = round(age_days, 2)
        else:
            out[name] = None
    return out


def _ops_from_health(health: dict | None) -> dict:
    """Flatten the operational rows from health_snapshot.json into a flat dict
    the dashboard hero can read directly.
    """
    if not health:
        return {"overall": None}
    rows = (health.get("operational") or {}).get("rows") or []
    by_check = {r.get("check"): r for r in rows}
    db = by_check.get("DB", {})
    model = by_check.get("MODEL", {})
    recs = by_check.get("RECS", {})
    universe = by_check.get("UNIVERSE", {})
    return {
        "overall": health.get("overall"),
        "db": {"status": db.get("status"), "detail": db.get("detail")},
        "model": {"status": model.get("status"), "detail": model.get("detail")},
        "recs": {"status": recs.get("status"), "detail": recs.get("detail")},
        "universe": {"status": universe.get("status"), "detail": universe.get("detail")},
        "warnings": health.get("warnings") or [],
    }


def _long_paper_observation_state() -> dict | None:
    """Return paper-observation block when LIVE_LONG_TELEGRAM_SILENT is on.

    Surfaces the silence as an auditable fact on the dashboard so reviewers
    can see the LONG telegram-quiet period is deliberate, with a start date.
    """
    try:
        from config import config as _cfg
        silent = bool(getattr(_cfg.Portfolio, "LIVE_LONG_TELEGRAM_SILENT", False))
        start = getattr(_cfg.Portfolio, "LIVE_LONG_OBSERVATION_START", "") or None
    except Exception:
        return None
    if not silent:
        return None
    days = None
    if start:
        try:
            t0 = datetime.fromisoformat(start)
            if t0.tzinfo is None:
                t0 = t0.replace(tzinfo=timezone.utc)
            days = round((datetime.now(timezone.utc) - t0).total_seconds() / 86400.0, 1)
        except Exception:
            days = None
    return {
        "active": True,
        "telegram_silent": True,
        "ledger_recording": True,  # exact-entry long_shadow_ledger.csv
        "observation_start": start,
        "observation_days": days,
        "decision_after": "60 independent 12h shadow baskets and preregistered economic gates",
        "next_step": "Evaluate README §14-ter; execution requires a separate reviewed code change.",
    }


def _long_policy_state(gates: dict | None = None) -> dict:
    """Public-safe LONG execution/WATCH policy; never includes coin details."""
    from config import config as _cfg

    long_gate = (gates or {}).get("long") or {}
    execution_max = int(getattr(_cfg.Portfolio, "LIVE_WATCH_LONG_N", 0))
    model_max = int(getattr(_cfg.LongModel, "LONG_N", 5))
    alert_max = min(
        int(getattr(_cfg.Notification, "LONG_WATCH_ALERT_MAX_N", 1)),
        model_max,
    )
    horizon_h = int(getattr(_cfg.LongModel, "PREDICT_HORIZON", 12))
    anchor = int(getattr(_cfg.LongModel, "REBALANCE_ANCHOR_HOUR_UTC", 11))
    anchor_hours = sorted({hour for hour in range(24) if (hour - anchor) % horizon_h == 0})
    policy_reason = long_gate.get("policy_reason") or (
        "WATCH_LONG KILL policy: LIVE_WATCH_LONG_N=0 (README §14-bis)"
        if execution_max <= 0 else None
    )
    return {
        "contract": "strong_shadow_watch_v1",
        "status": "EXECUTION_BLOCKED_CONDITIONAL_WATCH" if execution_max <= 0 else "ACTIVE",
        "reason": policy_reason,
        "execution_max": execution_max,
        "execution_actionable": False if execution_max <= 0 else None,
        "conditional_watch_alert_max": alert_max if execution_max <= 0 else 0,
        "watch_alert_gates": {
            "anchor_hours_utc": anchor_hours,
            "bitget_tradable_required": bool(
                getattr(_cfg.Notification, "LONG_WATCH_ALERT_REQUIRE_BITGET_TRADABLE", True)
            ),
            "preflight_clear_required": True,
            "btc_7d_return_gt": float(getattr(_cfg.LongModel, "BTC_7D_RETURN_GATE", 0.0)),
            "btc_30d_return_gt": float(
                getattr(_cfg.LongModel, "BTC_30D_RETURN_FLOOR", -0.10)
            ),
            "ic_status_required": "OK",
            "ic_min_reads": int(
                getattr(_cfg.Notification, "LONG_WATCH_ALERT_MIN_IC_READS", 3)
            ),
            "ic_each_gte": float(
                getattr(_cfg.Notification, "LONG_WATCH_ALERT_IC_FLOOR", 0.05)
            ),
            "latest_ic_gte": float(
                getattr(_cfg.Notification, "LONG_WATCH_ALERT_MIN_IC", 0.10)
            ),
            "sigma_gte": float(
                getattr(_cfg.Notification, "LONG_WATCH_ALERT_MIN_SIGMA", 2.0)
            ),
            "expected_pct_gte": float(
                getattr(_cfg.Notification, "LONG_WATCH_ALERT_MIN_EXPECTED_PCT", 0.20)
            ),
            "direction_probability_gte": float(
                getattr(_cfg.Notification, "LONG_WATCH_ALERT_MIN_DIRECTION_PROB", 0.65)
            ),
            "positive_direction_required": True,
            "cross_horizon_consensus_required": bool(
                getattr(_cfg.Notification, "LONG_WATCH_ALERT_REQUIRE_CONSENSUS", True)
            ),
            "untrusted_hidden": bool(
                getattr(_cfg.Notification, "LONG_WATCH_ALERT_HIDE_UNTRUSTED", True)
            ),
            "shadow_top_n": model_max,
        },
        "economic_reactivation": {
            "automatic": False,
            "requires_separate_code_change": True,
            "min_independent_12h_baskets": 60,
            "mean_net_gt": 0.0,
            "block_bootstrap_95pct_lower_gt": 0.0,
            "three_temporal_folds_net_gt": 0.0,
            "all_allowed_regimes_net_gt": 0.0,
        },
    }


def build_summary_payload() -> dict:
    gate = _safe_load_json(OUTPUT_DIR / "gate_state.json") or {}
    drift = _safe_load_json(OUTPUT_DIR / "drift_state.json") or {}
    health = _safe_load_json(OUTPUT_DIR / "health_snapshot.json") or {}
    calib = _safe_load_json(OUTPUT_DIR / "calibration_sigma.json") or {}
    latest = _read_csv(OUTPUT_DIR / "latest.csv")

    ic_short_recs = _safe_load_json(OUTPUT_DIR / "ic_history.json") or []
    ic_long_recs = _safe_load_json(OUTPUT_DIR / "ic_history_long.json") or []
    short_series = _ic_series(ic_short_recs, side="short", horizon_h=6, require_label=True)
    long_series = _ic_series(ic_long_recs, side="long", horizon_h=12, require_label=False)

    ledger = _read_csv(OUTPUT_DIR / "recommendation_ledger.csv")
    history = _pick_history(ledger, days=60)

    holdout = _safe_load_json(OUTPUT_DIR / "holdout_report.json") or {}

    return {
        "asof": _now_utc_iso(),
        "pin_hint": "4 digits",
        "gates": gate.get("gates", {}),
        "holdout_ic": {
            "short_h6": holdout.get("short_h6"),
            "long_h12": holdout.get("long_h12"),
            "_provenance": holdout.get("_provenance"),
        } if holdout else None,
        "ic_summary": {
            "short_h6": {
                "last7": _ic_stats(short_series, 7),
                "last30": _ic_stats(short_series, 30),
            },
            "long_h12": {
                "last7": _ic_stats(long_series, 7),
                "last30": _ic_stats(long_series, 30),
            },
        },
        "realized_summary": {
            "SHORT": _realized_stats(history, "SHORT", [30]),
            "WATCH_LONG": _realized_stats(history, "WATCH_LONG", [30]),
        },
        "drift": _drift_rows(drift),
        "drift_generated_at": drift.get("generated_at"),
        "latest_picks": _picks_from_latest(latest),
        "calibration_buckets": {
            side: [
                {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in row.items()}
                for row in (calib.get(side, []) or [])
            ]
            for side in ("short_6h", "long_12h", "short_12h", "long_6h")
            if side in (calib or {})
        },
        "model_age_days": _model_age_days(),
        "ops": _ops_from_health(health),
        "cost_assumptions": COST_PCT_BY_SIDE,
        "long_paper_observation": _long_paper_observation_state(),
        "long_policy": _long_policy_state(gate.get("gates", {})),
    }


def build_history_payload(history_days: int = 60, ic_days: int = 60) -> dict:
    ic_short_recs = _safe_load_json(OUTPUT_DIR / "ic_history.json") or []
    ic_long_recs = _safe_load_json(OUTPUT_DIR / "ic_history_long.json") or []

    short_series = _ic_series(ic_short_recs, side="short", horizon_h=6, require_label=True)
    long_series = _ic_series(ic_long_recs, side="long", horizon_h=12, require_label=False)

    cutoff = datetime.now(timezone.utc) - timedelta(days=ic_days)

    def trim(series):
        out = []
        for r in series:
            try:
                t = datetime.fromisoformat(r["t"])
            except Exception:
                continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if t >= cutoff:
                out.append(r)
        return out

    ledger = _read_csv(OUTPUT_DIR / "recommendation_ledger.csv")
    history = _pick_history(ledger, history_days)

    return {
        "asof": _now_utc_iso(),
        "ic_series": {
            "short_h6": trim(short_series),
            "long_h12": trim(long_series),
        },
        "pick_history": history,
    }


def _daily_pnl_series(history: list[dict], side: str) -> list[dict]:
    """Daily aggregated stats per side: date, n_trades, hit_rate, mean_pct, cum_pct, drawdown_pct.

    Used by dashboard's cumulative PnL chart + daily heatmap. Only matured trades.
    """
    from collections import defaultdict
    daily = defaultdict(list)
    for r in history:
        if r.get("side") != side:
            continue
        rp = r.get("realized_pct")
        if rp is None:
            continue
        ts = r.get("entry_time", "")[:10]  # YYYY-MM-DD
        if not ts:
            continue
        daily[ts].append(float(rp))

    rows = []
    cum = 0.0
    peak = 0.0
    for date in sorted(daily.keys()):
        vals = daily[date]
        n = len(vals)
        mean_pct = sum(vals) / n
        hit_rate = sum(1 for v in vals if v > 0) / n
        cum += mean_pct  # equal-weight per-day (treats day as a portfolio)
        peak = max(peak, cum)
        dd = cum - peak  # ≤ 0
        rows.append({
            "date": date,
            "n_trades": n,
            "hit_rate": round(hit_rate, 4),
            "mean_pct": round(mean_pct, 4),
            "cum_pct": round(cum, 4),
            "drawdown_pct": round(dd, 4),
        })
    return rows


def _shadow_strategy_comparison(history: list[dict]) -> dict:
    """Compare two live strategies head-to-head on the same calendar.

      - current_short_only: 5 SHORT picks per rebalance, net daily mean
      - shadow_long_short5: 5 SHORT + 5 WATCH_LONG combined as a 10-name
        equal-weight basket per day (since both sides are already tracked
        in the ledger paper-mode, this needs no new data — just a different
        aggregation rule)

    portfolio_experiment.py showed shadow_long_short5 had t=+1.62 on a
    60d holdout. Live-shadowing here surfaces whether that finding holds
    forward without forcing a strategy switch on weak statistics.

    Returns: { current_short_only: [{date,n,mean_net,cum_net}], shadow_long_short5: [...] }
    Both series are NET (after costs); gross is implicit (cum_gross = cum_net + Σ cost).
    """
    from collections import defaultdict
    by_date_side = defaultdict(lambda: defaultdict(list))  # date → side → [net_pct]
    for r in history:
        npct = r.get("net_pct")
        if npct is None:
            continue
        ts = r.get("entry_time", "")[:10]
        if not ts:
            continue
        side = r.get("side")
        if side not in ("SHORT", "WATCH_LONG"):
            continue
        by_date_side[ts][side].append(float(npct))

    cur_rows, sha_rows = [], []
    cur_cum, sha_cum = 0.0, 0.0
    cur_peak, sha_peak = 0.0, 0.0
    for date in sorted(by_date_side.keys()):
        shorts = by_date_side[date].get("SHORT", [])
        longs = by_date_side[date].get("WATCH_LONG", [])

        if shorts:
            cur_mean = sum(shorts) / len(shorts)
            cur_cum += cur_mean
            cur_peak = max(cur_peak, cur_cum)
            cur_rows.append({
                "date": date, "n": len(shorts),
                "mean_net": round(cur_mean, 4),
                "cum_net": round(cur_cum, 4),
                "drawdown_net": round(cur_cum - cur_peak, 4),
            })

        combined = shorts + longs
        if combined:
            sha_mean = sum(combined) / len(combined)
            sha_cum += sha_mean
            sha_peak = max(sha_peak, sha_cum)
            sha_rows.append({
                "date": date,
                "n_short": len(shorts),
                "n_long": len(longs),
                "mean_net": round(sha_mean, 4),
                "cum_net": round(sha_cum, 4),
                "drawdown_net": round(sha_cum - sha_peak, 4),
            })

    def _summary(rows: list[dict]) -> dict:
        if not rows:
            return {"n_days": 0}
        means = [r["mean_net"] for r in rows]
        m = sum(means) / len(means)
        var = sum((v - m) ** 2 for v in means) / max(len(means) - 1, 1)
        std = var ** 0.5
        return {
            "n_days": len(rows),
            "final_cum_net": rows[-1]["cum_net"],
            "max_drawdown_net": min(r["drawdown_net"] for r in rows),
            "daily_mean_net": round(m, 4),
            "daily_sharpe_net": round(m / std, 3) if std > 0 else None,
        }

    return {
        "current_short_only": cur_rows,
        "shadow_long_short5": sha_rows,
        "summary": {
            "current_short_only": _summary(cur_rows),
            "shadow_long_short5": _summary(sha_rows),
        },
        "_note": "Live head-to-head shadow. SHORT-only is what fetch_and_rank actually emits as actionable; long_short5 is the candidate strategy from portfolio_experiment.py. Both daily series use ledger net_pct (after fees).",
    }


def _return_distribution(history: list[dict], side: str, n_bins: int = 12) -> dict:
    """Histogram + summary statistics for realized return distribution."""
    vals = [float(r["realized_pct"]) for r in history
            if r.get("side") == side and r.get("realized_pct") is not None]
    if not vals:
        return {"bins": [], "stats": {"n": 0}}

    vals_sorted = sorted(vals)
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
    std = var ** 0.5

    def pct(p):
        if n == 0:
            return None
        idx = max(0, min(n - 1, int(round((p / 100) * (n - 1)))))
        return vals_sorted[idx]

    # Clip histogram range to p2/p98 so a couple of outliers don't squash all bins.
    # Outlier counts are merged into the boundary bins.
    lo_clip = pct(2) if n >= 50 else min(vals)
    hi_clip = pct(98) if n >= 50 else max(vals)
    if hi_clip <= lo_clip:
        lo_clip, hi_clip = min(vals), max(vals)
    width = (hi_clip - lo_clip) / max(n_bins, 1) if hi_clip > lo_clip else 1.0

    bins = []
    for i in range(n_bins):
        a = lo_clip + i * width
        b = lo_clip + (i + 1) * width if i < n_bins - 1 else hi_clip + 0.001
        if i == 0:
            count = sum(1 for v in vals if v < b)
        elif i == n_bins - 1:
            count = sum(1 for v in vals if v >= a)
        else:
            count = sum(1 for v in vals if a <= v < b)
        bins.append({
            "bin_low": round(a, 3),
            "bin_high": round(b, 3),
            "count": count,
        })
    return {
        "bins": bins,
        "stats": {
            "n": n,
            "mean": round(mean, 4),
            "median": round(vals_sorted[n // 2], 4),
            "std": round(std, 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "p5": round(pct(5), 4) if pct(5) is not None else None,
            "p25": round(pct(25), 4) if pct(25) is not None else None,
            "p75": round(pct(75), 4) if pct(75) is not None else None,
            "p95": round(pct(95), 4) if pct(95) is not None else None,
            "hit_rate": round(sum(1 for v in vals if v > 0) / n, 4),
        },
    }


def _best_worst(history: list[dict], side: str, k: int = 5) -> dict:
    """Top-k wins / losses for narrative storytelling."""
    rows = [r for r in history
            if r.get("side") == side and r.get("realized_pct") is not None]
    if not rows:
        return {"best": [], "worst": []}
    rows_sorted = sorted(rows, key=lambda r: float(r["realized_pct"]))
    pick = lambda r: {
        "market": r.get("market"),
        "entry_time": r.get("entry_time"),
        "exit_time": r.get("exit_time"),
        "entry_price": r.get("entry_price"),
        "exit_price": r.get("exit_price"),
        "horizon_h": r.get("horizon_h"),
        "realized_pct": float(r["realized_pct"]),
        "actionable": bool(r.get("actionable")),
    }
    return {
        "best": [pick(r) for r in rows_sorted[-k:][::-1]],
        "worst": [pick(r) for r in rows_sorted[:k]],
    }


def _monthly_returns(history: list[dict], side: str) -> list[dict]:
    """year/month → summed mean daily return (institutional factsheet style)."""
    daily = _daily_pnl_series(history, side)
    if not daily:
        return []
    from collections import defaultdict
    bucket = defaultdict(lambda: {"sum": 0.0, "n_days": 0, "n_trades": 0})
    for row in daily:
        date = row["date"]  # YYYY-MM-DD
        try:
            y, m, _ = date.split("-")
            key = f"{y}-{m}"
        except ValueError:
            continue
        bucket[key]["sum"] += row["mean_pct"]
        bucket[key]["n_days"] += 1
        bucket[key]["n_trades"] += row["n_trades"]
    out = []
    for key in sorted(bucket.keys()):
        y, m = key.split("-")
        b = bucket[key]
        out.append({
            "year": int(y),
            "month": int(m),
            "return_pct": round(b["sum"], 4),
            "n_days": b["n_days"],
            "n_trades": b["n_trades"],
        })
    return out


def _rolling_stats(history: list[dict], side: str, window: int = 7) -> list[dict]:
    """Rolling mean / hit-rate over `window` days, anchored on each end-date."""
    daily = _daily_pnl_series(history, side)
    if len(daily) < 2:
        return []
    out = []
    for i in range(len(daily)):
        lo = max(0, i - window + 1)
        chunk = daily[lo:i + 1]
        if not chunk:
            continue
        n_trades = sum(c["n_trades"] for c in chunk)
        if n_trades == 0:
            continue
        # weighted by per-day n_trades
        weighted_mean = sum(c["mean_pct"] * c["n_trades"] for c in chunk) / n_trades
        weighted_hit = sum(c["hit_rate"] * c["n_trades"] for c in chunk) / n_trades
        out.append({
            "date": daily[i]["date"],
            "window_n_trades": n_trades,
            "rolling_mean_pct": round(weighted_mean, 4),
            "rolling_hit_rate": round(weighted_hit, 4),
        })
    return out


def _calibration_reliability(calib: dict) -> dict:
    """Convert σ-bucket calibration into reliability-diagram points.

    For each bucket, x = predicted hit probability proxy (use σ midpoint mapped to
    a soft probability via 0.5 + clip(σ_mid * scale, -0.5, 0.5)), y = empirical hit_rate.
    Simpler: just emit (sigma_mid, hit_rate) pairs and let the chart label sigma directly.
    """
    out = {"short_6h": [], "long_12h": []}
    for key in ("short_6h", "long_12h"):
        buckets = calib.get(key) or []
        for b in buckets:
            sl = b.get("sigma_low")
            sh = b.get("sigma_high")
            if sl is None:
                continue
            # For open-ended last bucket (sigma_high=None), use sl + 0.5σ as a
            # representative midpoint (typical bucket width). Pure sl
            # under-represents the bucket on a reliability x-axis.
            mid = (sl + 0.5) if sh is None else (sl + sh) / 2
            n = b.get("n", 0)
            hit = b.get("hit_rate")
            if hit is None:
                continue
            out[key].append({
                "sigma_mid": round(mid, 3),
                "sigma_low": round(sl, 3),
                "sigma_high": round(sh, 3) if sh is not None else None,
                "n": n,
                "hit_rate": round(hit, 4),
                "mean_signed_pct": b.get("mean_signed_return_pct"),
            })
    return out


def build_accuracy_payload(history_days: int = 60) -> dict:
    ledger = _read_csv(OUTPUT_DIR / "recommendation_ledger.csv")
    history = _pick_history(ledger, history_days)
    calib = _safe_load_json(OUTPUT_DIR / "calibration_sigma.json") or {}

    return {
        "asof": _now_utc_iso(),
        "window_days": history_days,
        "realized": {
            "SHORT": _realized_stats(history, "SHORT", [7, 14, 30]),
            "WATCH_LONG": _realized_stats(history, "WATCH_LONG", [7, 14, 30]),
        },
        "calibration_buckets": calib,
        # New (2026-05-08): per-side daily pnl/dd series + return distribution
        # for cumulative PnL chart, daily heatmap, and return histogram.
        "daily_series": {
            "SHORT": _daily_pnl_series(history, "SHORT"),
            "WATCH_LONG": _daily_pnl_series(history, "WATCH_LONG"),
        },
        "return_distribution": {
            "SHORT": _return_distribution(history, "SHORT"),
            "WATCH_LONG": _return_distribution(history, "WATCH_LONG"),
        },
        "best_worst": {
            "SHORT": _best_worst(history, "SHORT", k=5),
            "WATCH_LONG": _best_worst(history, "WATCH_LONG", k=5),
        },
        "monthly_returns": {
            "SHORT": _monthly_returns(history, "SHORT"),
            "WATCH_LONG": _monthly_returns(history, "WATCH_LONG"),
        },
        "rolling_stats": {
            "SHORT": _rolling_stats(history, "SHORT", window=7),
            "WATCH_LONG": _rolling_stats(history, "WATCH_LONG", window=7),
        },
        "calibration_reliability": _calibration_reliability(calib),
        # New (2026-05-25): live shadow comparison between current operational
        # strategy (SHORT-only) and the candidate long_short5 from
        # portfolio_experiment.py. Computed from the SAME ledger — no new
        # paper trades, just a different aggregation rule.
        "shadow_strategy_compare": _shadow_strategy_comparison(history),
        "n_total": len(history),
    }


# ---- top-level export ----------------------------------------------------

def build_public_summary_payload() -> dict:
    """A trimmed, PUBLIC-safe snapshot for the project narrative page.

    Strategy: include only aggregate metrics that don't expose individual
    picks, entry prices, or anything that could be reverse-engineered into a
    live trading edge. Safe to ship in plaintext alongside the encrypted
    dashboard payloads.

    Contains:
      - asof timestamp
      - IC: live 30d means + holdout snapshots (already public via the
        narrative HTML anyway)
      - realized: gross/net/win per side over 30d (aggregates, no picks)
      - model age + ops gate states
      - n_total picks tracked

    Excluded (kept inside the encrypted dashboard data):
      - latest_picks (individual coin choices)
      - drift per-factor IC (could reveal which factor is alive)
      - calibration buckets (would let outsiders reproduce the signal map)
      - daily series / heatmap / best-worst / pick history
    """
    summary = build_summary_payload()
    realized = summary.get("realized_summary") or {}

    def _side(name: str) -> dict | None:
        s = realized.get(name) or {}
        d30 = s.get("d30") or {}
        if not d30:
            return None
        return {
            "n": d30.get("n"),
            "win_pct": d30.get("net_win_pct", d30.get("win_pct")),
            "gross_win_pct": d30.get("gross_win_pct", d30.get("win_pct")),
            "net_win_pct": d30.get("net_win_pct", d30.get("win_pct")),
            "win_pct_basis": "net_return",
            "gross_pct": d30.get("avg"),
            "net_pct": d30.get("avg_net"),
            "cost_pct": d30.get("cost_pct"),
            "tstat_clustered": d30.get("tstat_clustered"),
            "sharpe": d30.get("sharpe"),
        }

    ic_sum = summary.get("ic_summary") or {}
    holdout = summary.get("holdout_ic") or {}
    gates = summary.get("gates") or {}

    return {
        "asof": summary.get("asof"),
        "ic": {
            "short_6h": {
                "live_30d_mean": (ic_sum.get("short_h6") or {}).get("last30", {}).get("mean"),
                "holdout": (holdout.get("short_h6") or {}).get("ic") if holdout else None,
                "holdout_tstat": (holdout.get("short_h6") or {}).get("tstat") if holdout else None,
                "holdout_n": (holdout.get("short_h6") or {}).get("n_periods") if holdout else None,
                "gate": (gates.get("short") or {}).get("status"),
            },
            "long_12h": {
                "live_30d_mean": (ic_sum.get("long_h12") or {}).get("last30", {}).get("mean"),
                "holdout": (holdout.get("long_h12") or {}).get("ic") if holdout else None,
                "holdout_tstat": (holdout.get("long_h12") or {}).get("tstat") if holdout else None,
                "holdout_n": (holdout.get("long_h12") or {}).get("n_periods") if holdout else None,
                "gate": (gates.get("long") or {}).get("status"),
            },
        },
        "realized_30d": {
            "SHORT": _side("SHORT"),
            "WATCH_LONG": _side("WATCH_LONG"),
        },
        "model_age_days": summary.get("model_age_days") or {},
        "cost_assumptions": summary.get("cost_assumptions") or {},
        "long_policy": summary.get("long_policy") or {},
        "_note": "public sibling of encrypted dashboard data. Aggregates only — no individual picks.",
    }


def export_to(target_dir: Path, pin: str = PIN_DEFAULT,
              history_days: int = 60, ic_days: int = 60,
              public_target: Path | None = None) -> dict[str, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "summary.json": build_summary_payload(),
        "history.json": build_history_payload(history_days=history_days, ic_days=ic_days),
        "accuracy.json": build_accuracy_payload(history_days=history_days),
    }
    written: dict[str, Path] = {}
    for name, plain in payloads.items():
        plaintext = json.dumps(plain, ensure_ascii=False, default=str).encode("utf-8")
        envelope = encrypt_payload(plaintext, pin)
        path = target_dir / name
        path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2))
        written[name] = path

    # Public sibling (unencrypted aggregates) — for narrative HTML auto-refresh.
    if public_target is not None:
        public_target.parent.mkdir(parents=True, exist_ok=True)
        public_payload = build_public_summary_payload()
        public_target.write_text(json.dumps(public_payload, ensure_ascii=False, indent=2))
        written["public_summary.json"] = public_target

    return written
