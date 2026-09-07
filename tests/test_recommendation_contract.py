from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from scripts import fetch_and_rank
from utils import dashboard_export, telegram


@pytest.mark.parametrize("requested,blocked,expected", [
    (20, False, 5), (5, False, 5), (4, False, 4), (0, False, 0),
    (-1, False, 0), (5, True, 0),
])
def test_short_limit_never_expands_in_observation(requested, blocked, expected):
    assert fetch_and_rank._short_pick_limit(requested, blocked) == expected


@pytest.mark.parametrize("score,sigma,expected,basket,reason", [
    (-0.01, 2.0, -0.7, 5, ""),
    (-0.01, 0.5, -0.1, 5, "weak_signal"),
    (0.01, 2.0, -0.7, 5, "direction_mismatch"),
    (-0.01, 2.0, 0.7, 5, "direction_mismatch"),
    (-0.01, float("nan"), -0.7, 5, "prediction_unavailable"),
    (-0.01, 2.0, -0.7, 4, "insufficient_candidates"),
])
def test_short_actionability_uses_prediction_quality(score, sigma, expected, basket, reason):
    prediction = {"sigma": sigma, "expected_pct": expected}
    assert fetch_and_rank._short_suppression_reason(score, prediction, basket) == reason
    assert fetch_and_rank._short_suppression_reason(score, prediction, basket, "ic_freeze") == "ic_freeze"


def test_saved_recommendations_preserve_observation_reason(tmp_path, monkeypatch):
    from data import collector
    import time

    monkeypatch.setattr(fetch_and_rank, "__file__", str(tmp_path / "scripts" / "fetch_and_rank.py"))
    monkeypatch.setattr(collector, "get_current_price", lambda market: 100.0)
    monkeypatch.setattr(time, "sleep", lambda delay: None)
    markets = [f"KRW-T{i}" for i in range(5)]
    scores = pd.Series(-0.01, index=markets)
    predictions = pd.DataFrame({"sigma": [0.5, 2, 2, 2, 2], "expected_pct": -0.7}, index=markets)
    result = fetch_and_rank._save_recommendations(
        pd.Timestamp("2026-09-01T05:00:00Z"), pd.Series(dtype=float), scores,
        short_predictions_df=predictions,
    )
    assert not result.iloc[0]["actionable"]
    assert result.iloc[0]["suppression"] == "weak_signal"
    assert result.iloc[1:]["actionable"].all()
    frozen = fetch_and_rank._save_recommendations(
        pd.Timestamp("2026-09-01T11:00:00Z"), pd.Series(dtype=float), scores,
        short_predictions_df=predictions, short_watch_only=True, short_watch_reason="ic_freeze",
    )
    assert not frozen["actionable"].any()
    assert frozen["suppression"].eq("ic_freeze").all()
    oversized = fetch_and_rank._save_recommendations(
        pd.Timestamp("2026-09-01T17:00:00Z"), pd.Series(dtype=float),
        pd.Series(-0.01, index=[f"KRW-T{i}" for i in range(20)]),
        short_predictions_df=predictions, short_watch_only=True,
    )
    assert len(oversized) == 5
    assert not oversized["actionable"].any()


def test_short_watch_message_cannot_look_like_a_recommendation():
    row = {"market": "KRW-TEST", "side": "SHORT", "sigma": 2.1,
           "expected_pct": -0.7, "horizon_h": 6, "actionable": True}
    normal = telegram.format_actionable_signals(pd.DataFrame([row]), "2026-09-01T05:00Z")
    watch = telegram.format_actionable_signals(
        pd.DataFrame([{**row, "actionable": False, "suppression": "ic_freeze"}]),
        "2026-09-01T05:00Z",
    )
    assert "SHORT WATCH" not in normal
    assert "SHORT WATCH" in watch and "actionable=false" in watch
    assert normal != watch
    assert telegram.format_actionable_signals(pd.DataFrame(), "2026-09-01T05:00Z")
    blocked = telegram.format_actionable_signals(
        pd.DataFrame(), "2026-09-01T05:00Z", status_note="SHORT IC LIQUIDATE",
    )
    assert "SHORT IC LIQUIDATE" in blocked


@pytest.fixture
def telegram_transport(monkeypatch):
    monkeypatch.setattr(telegram, "TELEGRAM_TOKEN", "test-token")
    monkeypatch.setattr(telegram, "TELEGRAM_CHAT_ID", "test-chat")
    sleeps = []
    monkeypatch.setattr(telegram.time, "sleep", sleeps.append)
    return sleeps


def response(status, payload):
    return SimpleNamespace(status_code=status, json=lambda: payload)


def test_telegram_retries_transient_failure(telegram_transport, monkeypatch):
    outcomes = iter([requests.Timeout("not delivered"), response(200, {"ok": True, "result": {"message_id": 1}})])
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        result = next(outcomes)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(telegram.requests, "post", post)
    assert telegram.send_message("test")
    assert len(calls) == 2 and telegram_transport == [1]


def test_telegram_checks_api_ok_not_only_http_status(telegram_transport, monkeypatch):
    monkeypatch.setattr(telegram.requests, "post", lambda *a, **k: response(200, {"ok": False, "error_code": 400}))
    assert not telegram.send_message("test")
    assert not telegram_transport


def test_telegram_honors_rate_limit_and_bounds_retries(telegram_transport, monkeypatch):
    monkeypatch.setattr(telegram.requests, "post", lambda *a, **k: response(429, {"ok": False, "parameters": {"retry_after": 3}}))
    assert not telegram.send_message("test")
    assert telegram_transport == [3, 3]


def test_realized_cohorts_do_not_mix_observation_and_recommendation():
    now = datetime.now(timezone.utc).isoformat()
    base = {"side": "SHORT", "entry_time": now, "realized_pct": 1.0, "net_pct": 0.7}
    history = [{**base, "actionable": True}, {**base, "actionable": False, "net_pct": 10.0}, base]
    cohorts = dashboard_export._realized_cohorts(history, [30])
    assert cohorts["actionable"]["SHORT"]["d30"]["avg_net"] == 0.7
    assert cohorts["watch"]["SHORT"]["d30"]["avg_net"] == 10.0
    assert cohorts["unknown"]["SHORT"]["d30"]["n"] == 1
    assert cohorts["all"]["SHORT"]["d30"]["n"] == 3


def test_window_mean_does_not_overweight_large_watch_baskets():
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    base = {"side": "SHORT", "realized_pct": 1.0, "actionable": False}
    rows = [{**base, "entry_time": now.isoformat(), "net_pct": 10.0} for _ in range(20)]
    rows += [{**base, "entry_time": (now - timedelta(hours=6)).isoformat(), "net_pct": -10.0} for _ in range(5)]
    result = dashboard_export._realized_stats(rows, "SHORT", [30])["d30"]
    assert result["avg_net"] == 6.0
    assert result["avg_net_per_window"] == 0.0
    assert result["n_windows"] == 2


def test_holdout_display_uses_dated_current_calibration():
    original = {"short_h6": {"ic": 0.0882, "hit_2sigma": 0.646, "e_signed_pct": 1.29}}
    calibration = {"generated_at": "2026-09-06T20:00Z", "short_6h": [
        {"sigma_low": 2, "n": 240, "hit_rate": 0.575, "mean_signed_return_pct": 0.707},
    ]}
    result = dashboard_export._holdout_with_calibration(original, calibration)
    assert result["short_h6"]["hit_2sigma"] == 0.575
    assert result["short_h6"]["sigma_sample_n"] == 240
    assert result["short_h6"]["ic"] == 0.0882
    assert original["short_h6"]["hit_2sigma"] == 0.646
    assert dashboard_export._holdout_with_calibration(original, {})["short_h6"]["hit_2sigma"] is None


def test_run_failure_reports_error_without_hiding_failure(monkeypatch):
    from contextlib import nullcontext
    import sys
    monkeypatch.setattr(sys, "argv", ["fetch_and_rank", "--no-dashboard-export"])
    monkeypatch.setattr(fetch_and_rank, "run_lock", lambda *a, **k: nullcontext())
    def fail(args):
        raise RuntimeError("injected run failure")
    monkeypatch.setattr(fetch_and_rank, "_run", fail)
    sent = []
    from utils import operator_report, prospective
    maintained = []
    monkeypatch.setattr(prospective, "maintain_experiment", lambda: maintained.append(True))
    monkeypatch.setattr(operator_report, "publish_report",
                        lambda report, **kwargs: sent.append(operator_report.format_report(report)))
    with pytest.raises(RuntimeError, match="injected"):
        fetch_and_rank.main()
    assert len(sent) == 1
    assert "xsec" in sent[0]
    assert maintained == [True]


def test_calibration_cache_reloads_after_replacement(tmp_path, monkeypatch):
    import json
    from utils import magnitude
    path = tmp_path / "calibration.json"
    monkeypatch.setattr(magnitude, "CALIB_FILE", path)
    monkeypatch.setattr(magnitude, "_CALIB_CACHE", None)
    monkeypatch.setattr(magnitude, "_CALIB_CACHE_STAMP", None)
    path.write_text(json.dumps({"generated_at": "first"}))
    assert magnitude._load_calibration()["generated_at"] == "first"
    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps({"generated_at": "second-longer"}))
    replacement.replace(path)
    assert magnitude._load_calibration()["generated_at"] == "second-longer"


def test_dashboard_api_exposes_separate_cohorts(monkeypatch):
    import app as dashboard
    now = datetime.now(timezone.utc).isoformat()
    base = {"side": "SHORT", "market": "KRW-TEST", "entry_time": now,
            "realized_return": 0.01, "net_return": 0.007}
    frame = pd.DataFrame([{**base, "actionable": True},
                          {**base, "actionable": False, "net_return": 0.10}])
    monkeypatch.setattr(dashboard, "_load_ledger_csv", lambda: frame)
    with dashboard.app.test_client() as client:
        response = client.get("/api/ledger")
    assert response.status_code == 200
    summary = response.get_json()["summary"]
    assert summary["decision_cohorts_30d"]["actionable"]["SHORT"]["d30"]["avg_net"] == 0.7
    assert summary["decision_cohorts_30d"]["watch"]["SHORT"]["d30"]["avg_net"] == 10.0
    html = dashboard._render_ledger_block(summary)
    assert "30d SHORT Recommendations" in html
    assert "30d SHORT Observations" in html


def test_health_performance_excludes_watch_and_unknown_short(tmp_path, monkeypatch):
    from scripts import health_snapshot
    monkeypatch.setattr(health_snapshot, "OUTPUT_DIR", tmp_path)
    base = {"side": "SHORT", "entry_time": datetime.now(timezone.utc).isoformat(),
            "realized_return": 0.01, "net_return": 0.007}
    pd.DataFrame([{**base, "actionable": True},
                  {**base, "actionable": False, "net_return": 0.10}, base]).to_csv(
        tmp_path / "recommendation_ledger.csv", index=False,
    )
    _, result = health_snapshot.section_realized()
    assert result["total"] == 1
    assert result["recorded_total"] == 3
    assert result["excluded_short_observation_or_unknown"] == 2
    assert result["by_side"]["SHORT"]["avg_return"] == 0.007


def test_health_watch_only_is_not_recommendation_evidence(tmp_path, monkeypatch):
    from scripts import health_snapshot
    monkeypatch.setattr(health_snapshot, "OUTPUT_DIR", tmp_path)
    pd.DataFrame([{"side": "SHORT", "entry_time": datetime.now(timezone.utc).isoformat(),
                   "realized_return": 0.10, "actionable": False}]).to_csv(
        tmp_path / "recommendation_ledger.csv", index=False,
    )
    status, result = health_snapshot.section_realized()
    assert status == "WARN"
    assert result["total"] == 0
    assert "SHORT" not in result["by_side"]
