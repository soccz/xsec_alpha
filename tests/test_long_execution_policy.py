from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from config import config
from scripts import health_snapshot
from scripts import fetch_and_rank
from scripts.fetch_and_rank import (
    _effective_long_pick_limit,
    _entry_observed_timestamp,
    _long_recommendation_is_actionable,
    _long_regime_allows,
    _select_strong_long_watch_alerts,
)
from utils import ic_gate


def test_watch_long_kill_is_fixed_in_code():
    assert config.Portfolio.LIVE_WATCH_LONG_N == 0
    assert config.Notification.LONG_WATCH_ALERT_MAX_N == 1


def test_effective_long_limit_honors_policy_and_portfolio_cap():
    assert _effective_long_pick_limit(5, 0, blocked=False) == 0
    assert _effective_long_pick_limit(5, 5, blocked=True) == 0
    assert _effective_long_pick_limit(5, 3, blocked=False) == 3


@pytest.mark.parametrize(
    ("btc_7d", "btc_30d", "allowed"),
    [
        (0.01, 0.00, True),
        (0.00, 0.00, False),
        (0.01, -0.10, False),
        (-0.01, 0.20, False),
        (float("nan"), 0.20, False),
    ],
)
def test_long_regime_gate_is_enforced(btc_7d, btc_30d, allowed):
    row = {"btc_ret_7d": btc_7d, "btc_ret_30d": btc_30d}
    assert _long_regime_allows(row, 0.0, -0.10) is allowed


def test_long_actionable_fails_closed_on_prediction_quality():
    assert _long_recommendation_is_actionable("LONG", True, True)
    assert not _long_recommendation_is_actionable("WATCH_LONG", True, True)
    assert not _long_recommendation_is_actionable("LONG", False, True)
    assert not _long_recommendation_is_actionable("LONG", True, False)


def _strong_watch_predictions():
    return pd.DataFrame(
        {
            "actionable": [True, True, True, True],
            "consensus": [True, True, True, False],
            "direction": [1, 1, 1, 1],
            "direction_prob": [0.66, 0.70, 0.66, 0.66],
            "expected_pct": [1.97, 2.10, 1.97, 1.97],
            "sigma": [2.4, 2.3, 1.9, 2.5],
            "suppression": ["", "", "", ""],
            "trust_tag": ["", "", "", ""],
            "tag": ["🔥", "🔥", "✅", "🔥"],
        },
        index=["KRW-STRONG", "KRW-STRONG-2", "KRW-WEAK", "KRW-NO-CONSENSUS"],
    )


def _healthy_blocked_long_gate():
    return SimpleNamespace(
        policy_blocked=True,
        status="OK",
        last_ic=0.12,
        ic_tail=[0.06, 0.08, 0.12],
    )


def test_strong_long_watch_alert_is_non_actionable_and_shadow_only():
    alerts = _select_strong_long_watch_alerts(
        _strong_watch_predictions(),
        ["KRW-STRONG", "KRW-STRONG-2", "KRW-WEAK", "KRW-NO-CONSENSUS"],
        _healthy_blocked_long_gate(),
        regime_eligible=True,
        rebalance_due=True,
        bitget_tradable_enforced=True,
    )

    assert alerts["market"].tolist() == ["KRW-STRONG"]
    assert alerts["side"].tolist() == ["WATCH_LONG"]
    assert not alerts["actionable"].any()
    assert alerts["position_size_pct"].tolist() == [0.0]
    assert alerts["execution_policy"].tolist() == ["KILL"]
    assert _select_strong_long_watch_alerts(
        _strong_watch_predictions(),
        ["KRW-STRONG"],
        _healthy_blocked_long_gate(),
        regime_eligible=True,
        rebalance_due=True,
        bitget_tradable_enforced=False,
    ).empty


def test_strong_long_watch_alert_requires_calibrated_direction_probability():
    low_probability = _strong_watch_predictions().loc[["KRW-STRONG"]].copy()
    low_probability.loc["KRW-STRONG", "direction_prob"] = 0.64
    alerts = _select_strong_long_watch_alerts(
        low_probability,
        ["KRW-STRONG"],
        _healthy_blocked_long_gate(),
        regime_eligible=True,
        rebalance_due=True,
        bitget_tradable_enforced=True,
    )
    assert alerts.empty

    missing_probability = low_probability.drop(columns=["direction_prob"])
    assert _select_strong_long_watch_alerts(
        missing_probability,
        ["KRW-STRONG"],
        _healthy_blocked_long_gate(),
        regime_eligible=True,
        rebalance_due=True,
        bitget_tradable_enforced=True,
    ).empty


@pytest.mark.parametrize(
    ("gate", "regime_eligible", "rebalance_due"),
    [
        (SimpleNamespace(policy_blocked=True, status="OK", last_ic=0.12, ic_tail=[0.12]), True, True),
        (SimpleNamespace(policy_blocked=True, status="WARN", last_ic=0.12, ic_tail=[0.06, 0.08, 0.12]), True, True),
        (SimpleNamespace(policy_blocked=True, status="OK", last_ic=0.09, ic_tail=[0.06, 0.08, 0.09]), True, True),
        (SimpleNamespace(policy_blocked=False, status="OK", last_ic=0.12, ic_tail=[0.06, 0.08, 0.12]), True, True),
        (_healthy_blocked_long_gate(), False, True),
        (_healthy_blocked_long_gate(), True, False),
    ],
)
def test_strong_long_watch_alert_fails_closed_without_every_gate(
    gate, regime_eligible, rebalance_due
):
    alerts = _select_strong_long_watch_alerts(
        _strong_watch_predictions(),
        ["KRW-STRONG"],
        gate,
        regime_eligible=regime_eligible,
        rebalance_due=rebalance_due,
        bitget_tradable_enforced=True,
    )
    assert alerts.empty


def test_entry_observation_time_prefers_actual_and_supports_legacy_rows():
    signal = pd.Timestamp("2026-01-01T00:00:00Z")
    observed = _entry_observed_timestamp(
        {"entry_observed_at": "2026-01-01T00:08:00Z"},
        signal,
    )
    assert observed == pd.Timestamp("2026-01-01T00:08:00Z")
    assert _entry_observed_timestamp({}, signal) == signal


def test_ledger_horizon_starts_at_observed_entry(tmp_path, monkeypatch):
    project = tmp_path / "project"
    scripts_dir = project / "scripts"
    output = project / "output"
    scripts_dir.mkdir(parents=True)
    output.mkdir()
    monkeypatch.setattr(fetch_and_rank, "__file__", str(scripts_dir / "fetch_and_rank.py"))

    pd.DataFrame([
        {
            "market": "KRW-X",
            "side": "WATCH_LONG",
            "actionable": False,
            "entry_price": 100.0,
            "entry_time": "2026-01-01T00:00:00Z",
            "entry_observed_at": "2026-01-01T00:08:00Z",
            "horizon_h": 12,
        }
    ]).to_csv(output / "recommendations_20260101T0000.csv", index=False)
    closes = pd.DataFrame(
        {"KRW-X": [110.0, 111.0]},
        index=pd.to_datetime(["2026-01-01T12:00:00Z", "2026-01-01T13:00:00Z"]),
    )

    fetch_and_rank._update_realized_ledger(closes, pd.Timestamp("2026-01-01T13:00:00Z"))

    ledger = pd.read_csv(output / "recommendation_ledger.csv")
    assert ledger.loc[0, "entry_observed_at"] == "2026-01-01T00:08:00+00:00"
    assert ledger.loc[0, "exit_time_target"] == "2026-01-01T12:08:00+00:00"


def test_long_shadow_predictions_capture_point_in_time_entry(tmp_path, monkeypatch):
    project = tmp_path / "project"
    scripts_dir = project / "scripts"
    scripts_dir.mkdir(parents=True)
    monkeypatch.setattr(fetch_and_rank, "__file__", str(scripts_dir / "fetch_and_rank.py"))
    monkeypatch.setattr("data.collector.get_current_price", lambda market: 123.45)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    long_df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"] * 2,
            "direction": [1, 1],
            "sigma": [2.0, 1.0],
            "score": [0.2, 0.1],
        },
        index=pd.Index(["KRW-X", "KRW-Y"], name="market"),
    )

    fetch_and_rank._save_predictions_full(
        pd.Timestamp("2026-01-01T00:00:00Z"),
        None,
        long_df,
        long_shadow_markets=["KRW-X"],
        long_regime_eligible=True,
    )

    saved = pd.read_csv(project / "output" / "predictions_20260101T0000.csv")
    shadow = saved.set_index("market")
    assert bool(shadow.loc["KRW-X", "shadow_selected"])
    assert shadow.loc["KRW-X", "shadow_entry_price"] == pytest.approx(123.45)
    assert pd.notna(shadow.loc["KRW-X", "entry_observed_at"])
    assert not bool(shadow.loc["KRW-Y", "shadow_selected"])


def test_long_shadow_ledger_matures_observed_entries_once(tmp_path, monkeypatch):
    project = tmp_path / "project"
    scripts_dir = project / "scripts"
    output = project / "output"
    scripts_dir.mkdir(parents=True)
    output.mkdir()
    monkeypatch.setattr(fetch_and_rank, "__file__", str(scripts_dir / "fetch_and_rank.py"))
    pd.DataFrame([
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "market": "KRW-X",
            "horizon_h": 12,
            "shadow_selected": True,
            "shadow_entry_price": 100.0,
            "entry_observed_at": "2026-01-01T00:01:00Z",
            "regime_eligible": True,
            "score": 0.2,
            "sigma": 2.0,
        },
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "market": "KRW-Y",
            "horizon_h": 12,
            "shadow_selected": False,
            "shadow_entry_price": 100.0,
            "entry_observed_at": "2026-01-01T00:01:00Z",
        },
    ]).to_csv(output / "predictions_20260101T0000.csv", index=False)
    calls = []

    def price_fetcher(market):
        calls.append(market)
        return 110.0

    observed_now = pd.Timestamp("2026-01-01T12:02:00Z")
    fetch_and_rank._update_long_shadow_ledger(observed_now, price_fetcher)
    fetch_and_rank._update_long_shadow_ledger(observed_now, price_fetcher)

    ledger = pd.read_csv(output / "long_shadow_ledger.csv")
    assert calls == ["KRW-X"]
    assert len(ledger) == 1
    assert ledger.loc[0, "gross_return"] == pytest.approx(0.10)
    assert ledger.loc[0, "net_return"] == pytest.approx(0.098)
    assert ledger.loc[0, "price_source"] == "upbit_ticker_proxy"


def test_ic_gate_preserves_raw_status_while_blocking(tmp_path, monkeypatch):
    history = tmp_path / "ic_long.json"
    history.write_text(json.dumps([{"ic": 0.10}, {"ic": 0.11}, {"ic": 0.12}]))
    monkeypatch.setitem(ic_gate.HISTORY_FILES, "long", history)
    gate = ic_gate.evaluate_side("long")
    assert gate.status == "OK"
    assert gate.policy_blocked and gate.block
    assert "WATCH_LONG KILL" in gate.policy_reason


def test_ic_gate_does_not_mix_legacy_and_current_contract(tmp_path, monkeypatch):
    history = tmp_path / "ic_short.json"
    history.write_text(json.dumps([
        {"ic": -0.20},
        {"ic": -0.10},
        {"ic": 0.10, "contract_version": "absolute_open_lag1_anchors_v1"},
    ]))
    monkeypatch.setitem(ic_gate.HISTORY_FILES, "short", history)

    gate = ic_gate.evaluate_side("short")

    assert gate.status == "OK"
    assert gate.ic_tail == [0.10]


def test_health_contains_warn_but_fails_on_emitted_long(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    (output / "ic_history.json").write_text(json.dumps([
        {"timestamp": "2026-01-01T00:00:00Z", "ic": 0.10, "horizon_h": 6},
        {"timestamp": "2026-01-01T06:00:00Z", "ic": 0.11, "horizon_h": 6},
        {"timestamp": "2026-01-01T12:00:00Z", "ic": 0.12, "horizon_h": 6},
    ]))
    (output / "ic_history_long.json").write_text(json.dumps([
        {"timestamp": "2026-01-01T00:00:00Z", "ic": -0.01, "horizon_h": 12},
        {"timestamp": "2026-01-01T06:00:00Z", "ic": 0.10, "horizon_h": 12},
        {"timestamp": "2026-01-01T12:00:00Z", "ic": 0.20, "horizon_h": 12},
    ]))
    pd.DataFrame([
        {"market": "KRW-X", "side": "WATCH_LONG", "score": 0.1, "actionable": False},
    ]).to_csv(output / "latest.csv", index=False)
    monkeypatch.setattr(health_snapshot, "OUTPUT_DIR", output)

    worst, ic_info = health_snapshot.section_ic()
    assert worst == "OK"
    assert ic_info["long"]["status"] == "BLOCKED"
    assert ic_info["long"]["raw_status"] == "WARN"
    assert health_snapshot.section_current_picks()[0] == "FAIL"


def test_negative_watch_history_is_contained_not_erased(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    now = pd.Timestamp.now(tz="UTC")
    pd.DataFrame([
        {"entry_time": now.isoformat(), "side": "WATCH_LONG", "realized_return": -0.01}
        for _ in range(20)
    ]).to_csv(output / "recommendation_ledger.csv", index=False)
    monkeypatch.setattr(health_snapshot, "OUTPUT_DIR", output)
    worst, info = health_snapshot.section_realized()
    long = info["by_side"]["WATCH_LONG"]
    assert worst == "OK"
    assert long["status"] == "BLOCKED" and long["raw_status"] == "WARN"
    assert long["avg_return"] == pytest.approx(-0.01)


def test_health_prefers_cost_adjusted_net_return(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    now = pd.Timestamp.now(tz="UTC")
    pd.DataFrame([
        {
            "entry_time": now.isoformat(),
            "side": "SHORT",
            "realized_return": 0.01,
            "net_return": -0.002,
            "actionable": True,
        }
        for _ in range(20)
    ]).to_csv(output / "recommendation_ledger.csv", index=False)
    monkeypatch.setattr(health_snapshot, "OUTPUT_DIR", output)

    worst, info = health_snapshot.section_realized()

    assert worst == "WARN"
    assert info["return_metric"].startswith("net_return")
    assert info["by_side"]["SHORT"]["avg_return"] == pytest.approx(-0.002)
