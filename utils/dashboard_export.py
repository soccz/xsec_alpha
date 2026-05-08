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
        })
    out.sort(key=lambda x: x["entry_time"], reverse=True)
    return out


# Cost assumptions (round-trip, per pick) — disclosed as a constant so reviewers
# can audit the haircut. Bitget USDT-perp taker is 0.06% per side × 2 sides
# = 0.12%, plus ~0.0075% funding per 6h hold. Upbit spot taker 0.05% per side.
COST_PCT_BY_SIDE = {
    "SHORT": 0.13,        # bitget perp 6h: 0.12 fee + ~0.01 funding
    "WATCH_LONG": 0.10,   # upbit spot 12h: 0.10 fee + 0 funding
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

    Reports both the naive per-trade t (assumes IID across all picks) and the
    clustered t (group picks by entry_time first, treat each rebalance window as
    one observation). The clustered t is the honest one because 5 SHORT picks
    in the same 6h window share market regime.
    """
    side_u = side.upper()
    cost = COST_PCT_BY_SIDE.get(side_u, 0.0)
    out = {}
    for d in days_window:
        cutoff = datetime.now(timezone.utc) - timedelta(days=d)
        vals: list[float] = []
        picks: list[tuple[str, float]] = []  # (entry_time, realized_pct) for clustering
        wins = 0
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
            picks.append((ts, rp))
            if rp > 0:
                wins += 1
        n = len(vals)
        if n == 0:
            out[f"d{d}"] = {
                "n": 0, "n_windows": 0, "avg": None, "avg_net": None, "win_pct": None,
                "win_ci_lo": None, "win_ci_hi": None,
                "std": None, "tstat": None, "tstat_clustered": None,
                "sharpe": None, "cost_pct": cost,
            }
            continue
        mean = sum(vals) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
            std = var ** 0.5
        else:
            std = None
        tstat = (mean / (std / (n ** 0.5))) if (std and std > 0 and n > 1) else None
        sharpe = (mean / std) if (std and std > 0) else None
        n_windows, tstat_clustered = _clustered_t(picks)
        ci_lo, ci_hi = _wilson_ci(wins, n)
        out[f"d{d}"] = {
            "n": n,
            "n_windows": n_windows,
            "avg": round(mean, 4),
            "avg_net": round(mean - cost, 4),
            "win_pct": round(100 * wins / n, 1),
            "win_ci_lo": round(100 * ci_lo, 1) if ci_lo is not None else None,
            "win_ci_hi": round(100 * ci_hi, 1) if ci_hi is not None else None,
            "std": round(std, 4) if std is not None else None,
            "tstat": round(tstat, 3) if tstat is not None else None,
            "tstat_clustered": round(tstat_clustered, 3) if tstat_clustered is not None else None,
            "sharpe": round(sharpe, 3) if sharpe is not None else None,
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
        "ledger_recording": True,  # WATCH_LONG already accumulates paper PnL
        "observation_start": start,
        "observation_days": days,
        "decision_after": "1-2 weeks of clean ledger data",
        "next_step": "Set LIVE_EXECUTION_MODE=balanced to wire LONG to live execution.",
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
        "n_total": len(history),
    }


# ---- top-level export ----------------------------------------------------

def export_to(target_dir: Path, pin: str = PIN_DEFAULT,
              history_days: int = 60, ic_days: int = 60) -> dict[str, Path]:
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
    return written
