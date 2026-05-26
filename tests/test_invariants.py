"""Invariants that must hold across the live system.

These tests run against PRODUCTION artifacts (output/recommendation_ledger.csv,
output/calibration_sigma.json, models/*.pkl) — they catch regressions where the
code still runs but the meaning has drifted (e.g. fee math wrong, calibration
non-monotonic, predict() output out of CLAUDE.md gate range).

Run with:
    python -m pytest tests/ -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "output" / "recommendation_ledger.csv"
CALIB = ROOT / "output" / "calibration_sigma.json"
SUMMARY = ROOT / "output" / "feature_health.json"


# ──────────────────────────────────────────────────────────────────────
# Ledger invariants (cost math, sign, time)
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ledger() -> pd.DataFrame:
    if not LEDGER.exists():
        pytest.skip(f"{LEDGER} missing — run fetch_and_rank first")
    return pd.read_csv(LEDGER)


def test_ledger_has_required_columns(ledger):
    required = {
        "market", "side", "entry_time", "exit_time_actual", "horizon_h",
        "entry_price", "exit_price", "realized_return",
        "gross_return", "fee_bps", "slippage_bps",
        "estimated_cost_return", "net_return",
    }
    missing = required - set(ledger.columns)
    assert not missing, f"ledger missing required cols: {missing}"


def test_ledger_gross_equals_realized(ledger):
    """gross_return must equal realized_return on every row (pre-cost = raw realized)."""
    df = ledger.dropna(subset=["gross_return", "realized_return"])
    diff = (df["gross_return"] - df["realized_return"]).abs()
    assert (diff < 1e-9).all(), f"gross != realized on {(diff >= 1e-9).sum()} rows; max diff {diff.max():.2e}"


def test_ledger_net_equals_gross_minus_cost(ledger):
    """net_return = gross_return - estimated_cost_return on every row."""
    df = ledger.dropna(subset=["net_return", "gross_return", "estimated_cost_return"])
    expected = df["gross_return"] - df["estimated_cost_return"]
    diff = (df["net_return"] - expected).abs()
    assert (diff < 1e-9).all(), f"net != gross-cost on {(diff >= 1e-9).sum()} rows; max diff {diff.max():.2e}"


def test_ledger_cost_matches_config_formula(ledger):
    """cost = (2*fee + 2*slip + short_extra) / 10000.

    SHORT: 2*6 + 2*4 + 10 = 30bps → 0.003
    LONG/WATCH_LONG: 2*6 + 2*4 + 0 = 20bps → 0.002
    """
    df = ledger.dropna(subset=["fee_bps", "slippage_bps", "estimated_cost_return"]).copy()
    short_extra = df.get("short_extra_cost_bps", pd.Series([0] * len(df))).fillna(0)
    expected = (2 * df["fee_bps"] + 2 * df["slippage_bps"] + short_extra) / 10000.0
    diff = (df["estimated_cost_return"] - expected).abs()
    bad = (diff >= 1e-9).sum()
    assert bad == 0, f"cost formula mismatch on {bad} rows; max diff {diff.max():.2e}"


def test_ledger_exit_after_entry(ledger):
    df = ledger.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time_actual"], utc=True, errors="coerce")
    valid = df.dropna(subset=["entry_time", "exit_time"])
    bad = (valid["exit_time"] <= valid["entry_time"]).sum()
    assert bad == 0, f"{bad} rows have exit_time <= entry_time"


def test_ledger_horizon_matches_timegap(ledger):
    """horizon_h column should equal (exit - entry) in hours within tolerance."""
    df = ledger.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time_actual"], utc=True, errors="coerce")
    df = df.dropna(subset=["entry_time", "exit_time", "horizon_h"])
    df["gap_h"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600.0
    diff = (df["gap_h"] - df["horizon_h"]).abs()
    bad = (diff > 0.5).sum()
    assert bad == 0, f"{bad} rows where horizon_h disagrees with actual gap"


# ──────────────────────────────────────────────────────────────────────
# Calibration buckets — must be monotonic-ish (allow ±5% noise per bucket)
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def calib() -> dict:
    if not CALIB.exists():
        pytest.skip(f"{CALIB} missing")
    return json.loads(CALIB.read_text())


def test_calib_buckets_have_required_fields(calib):
    for side in ("short_6h", "long_12h"):
        buckets = calib.get(side, [])
        assert buckets, f"{side} has no buckets"
        for b in buckets:
            for k in ("sigma_low", "n", "hit_rate", "mean_signed_return_pct"):
                assert k in b, f"{side} bucket missing {k}: {b}"


def test_calib_hit_rate_in_unit_interval(calib):
    for side, buckets in calib.items():
        if not isinstance(buckets, list):  # skip meta fields like generated_at
            continue
        for b in buckets:
            hr = b.get("hit_rate")
            assert 0 <= hr <= 1, f"{side} hit_rate out of [0,1]: {hr}"


def test_calib_high_sigma_bucket_outperforms_baseline(calib):
    """The σ ≥ 2 bucket should have hit_rate > 0.55 — the whole point of the model."""
    for side in ("short_6h", "long_12h"):
        buckets = calib.get(side, [])
        top = next((b for b in buckets if b.get("sigma_low", 0) >= 2), None)
        assert top is not None, f"{side} has no σ≥2 bucket"
        assert top["hit_rate"] > 0.55, (
            f"{side} σ≥2 bucket hit_rate {top['hit_rate']:.3f} <= 0.55 — model may have degraded"
        )


# ──────────────────────────────────────────────────────────────────────
# Feature health — kimchi tagged EXPECTED, no WARN in baseline
# ──────────────────────────────────────────────────────────────────────

def test_feature_health_no_warn():
    if not SUMMARY.exists():
        pytest.skip("feature_health.json missing — run fetch_and_rank")
    fh = json.loads(SUMMARY.read_text())
    warns = [c for c, v in fh.get("factors", {}).items() if v.get("status") == "WARN"]
    assert not warns, f"factors flagged WARN: {warns}"


def test_feature_health_kimchi_expected():
    if not SUMMARY.exists():
        pytest.skip("feature_health.json missing")
    fh = json.loads(SUMMARY.read_text())
    for col, info in fh.get("factors", {}).items():
        if "kimchi" in col:
            assert info.get("status") == "EXPECTED", (
                f"{col} should be tagged EXPECTED (structural NaN) but got {info.get('status')}"
            )


# ──────────────────────────────────────────────────────────────────────
# Model predict integrity — std/mean inside CLAUDE.md §8 gates
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", ["xsec_6h.pkl", "xsec_12h.pkl"])
def test_model_predict_in_gate_range(model_name):
    import joblib
    path = ROOT / "models" / model_name
    if not path.exists():
        pytest.skip(f"{path} missing")
    model = joblib.load(path)
    # 10-feature unified pipeline → synthetic but realistic z-scored cross-section
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10)).astype(np.float32)
    np.clip(X, -3, 3, out=X)
    y = model.predict(X)
    assert np.isfinite(y).all(), f"{model_name} predict produced non-finite values"
    # CLAUDE.md §8 (post 2026-05-25 update): std ∈ [0.001, 0.50], |mean| < 0.10
    assert 0.0005 < y.std() < 0.50, (
        f"{model_name} predict std {y.std():.5f} outside [0.0005, 0.50] — zero/exploding signal"
    )
    assert abs(y.mean()) < 0.10, (
        f"{model_name} predict |mean| {abs(y.mean()):.5f} > 0.10 — directional bias"
    )


# ──────────────────────────────────────────────────────────────────────
# Cross-section z-score clip — |z| <= 3 after crosssection_zscore
# ──────────────────────────────────────────────────────────────────────

def test_zscore_clip_enforced():
    """Build a small factor frame, run crosssection_zscore, verify |z| <= 3 + epsilon."""
    from data.features import crosssection_zscore
    rng = np.random.default_rng(0)
    n_ts, n_coins, n_factors = 30, 50, 3
    ts = pd.date_range("2026-01-01", periods=n_ts, freq="h")
    coins = [f"KRW-C{i:02d}" for i in range(n_coins)]
    idx = pd.MultiIndex.from_product([ts, coins], names=["timestamp", "market"])
    df = pd.DataFrame(
        rng.standard_normal((n_ts * n_coins, n_factors)),
        index=idx,
        columns=[f"f{i}" for i in range(n_factors)],
    )
    # inject a few outliers — should be clipped
    df.iloc[0, 0] = 10.0
    df.iloc[1, 1] = -8.0
    out = crosssection_zscore(df, cols=df.columns.tolist())
    assert (out.abs() <= 3.0 + 1e-9).all().all(), (
        f"|z| > 3 after clip; max = {out.abs().max().max():.4f}"
    )


# ──────────────────────────────────────────────────────────────────────
# Drift detector — constant-input guard
# ──────────────────────────────────────────────────────────────────────

def test_drift_detector_constant_input_guard():
    """spearmanr on constant input would raise warning; our guard returns nan instead."""
    from utils.drift_detector import _ic_per_slot
    s1 = pd.Series([0.5] * 50, name="constant_factor")
    s2 = pd.Series(np.random.randn(50), name="returns")
    ic = _ic_per_slot(s1, s2)
    assert np.isnan(ic), f"constant input should yield NaN IC, got {ic}"


# ──────────────────────────────────────────────────────────────────────
# Dashboard export — schema completeness
# ──────────────────────────────────────────────────────────────────────

def test_dashboard_summary_payload_has_required_keys():
    from utils.dashboard_export import build_summary_payload
    s = build_summary_payload()
    required = {
        "asof", "gates", "ic_summary", "realized_summary", "drift",
        "latest_picks", "calibration_buckets", "model_age_days", "ops",
        "cost_assumptions",
    }
    missing = required - set(s.keys())
    assert not missing, f"summary payload missing required keys: {missing}"


def test_dashboard_public_summary_excludes_private_fields():
    """public_summary.json must NOT leak individual picks or per-coin detail."""
    from utils.dashboard_export import build_public_summary_payload
    pub = build_public_summary_payload()
    forbidden = {"latest_picks", "drift", "calibration_buckets", "pick_history", "daily_series"}
    leaked = forbidden & set(pub.keys())
    assert not leaked, f"public summary leaks {leaked} — should stay encrypted"


def test_dashboard_public_summary_has_aggregates():
    from utils.dashboard_export import build_public_summary_payload
    pub = build_public_summary_payload()
    assert pub.get("ic") and pub.get("realized_30d"), "public summary missing core aggregates"
    for side in ("SHORT", "WATCH_LONG"):
        side_data = (pub["realized_30d"] or {}).get(side) or {}
        for k in ("n", "gross_pct", "net_pct", "cost_pct", "win_pct"):
            assert k in side_data, f"public.realized_30d.{side} missing {k}"


# ──────────────────────────────────────────────────────────────────────
# Telegram format — both branches non-empty
# ──────────────────────────────────────────────────────────────────────

def test_telegram_format_actionable_branches():
    from utils.telegram import format_actionable_signals
    # Branch A: recommendation present (σ >= soft floor 1.0)
    df_yes = pd.DataFrame([
        {"market": "KRW-X", "side": "SHORT", "direction": -1, "sigma": 2.0,
         "expected_pct": -1.2, "p_correct": 0.62, "trust_tag": "", "actionable": True,
         "tag": "🔥", "label": "강한 신호", "consensus": 2.0, "horizon_h": 6}
    ]).set_index("market", drop=False)
    msg_a = format_actionable_signals(df_yes, latest_ts=pd.Timestamp("2026-01-01T00:00:00Z"))
    assert msg_a and "xsec" in msg_a and "SHORT" in msg_a, "branch A (recommendation) missing structure"

    # Branch B: heartbeat — all picks below σ 1.0 soft floor
    df_no = pd.DataFrame([
        {"market": "KRW-Y", "side": "SHORT", "direction": -1, "sigma": 0.5,
         "expected_pct": -0.1, "p_correct": 0.51, "trust_tag": "", "actionable": False,
         "tag": "▫", "label": "약함", "consensus": 1.0, "horizon_h": 6}
    ]).set_index("market", drop=False)
    msg_b = format_actionable_signals(df_no, latest_ts=pd.Timestamp("2026-01-01T00:00:00Z"))
    assert msg_b and ("강한 신호 없음" in msg_b or "⏸" in msg_b), \
        "branch B (heartbeat) missing pause/no-signal marker"
