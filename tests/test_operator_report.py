import json

import pandas as pd
import pytest

from utils import operator_report, telegram


def latest(root):
    return json.loads((root / "output" / "latest_operator_report.json").read_text())


@pytest.mark.parametrize("reason", ["ic_liquidate", "data_stale", "model_unavailable"])
def test_blocked_run_still_has_coin_ideas_without_execution(tmp_path, reason):
    report = operator_report.build_report(
        pd.DataFrame(), pd.DataFrame({"market": ["KRW-T", "KRW-NEAR"]}),
        asof="2026-09-07T05:00Z", reason=reason, root=tmp_path,
    )
    assert not report["signals"]
    assert all(row["actionable"] is False for row in report["ideas"])
    text = operator_report.format_report(report)
    assert "NEAR" in text and "actionable=false" in text and reason in text
    assert operator_report.DASHBOARD_URL in text
    assert not list(tmp_path.glob("**/*ledger*"))


def test_analysis_failure_uses_dated_previous_coins_without_old_estimates(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    pd.DataFrame([{"market": "KRW-NEAR", "side": "SHORT", "actionable": True,
                   "expected_pct": -12, "entry_time": "2026-09-01T05:00Z"}]).to_csv(
        output / "latest.csv", index=False,
    )
    report = operator_report.build_report(error=True, reason="analysis_failed", root=tmp_path)
    assert report["ideas"][0]["reference_only"]
    assert report["ideas"][0]["asof"] == "2026-09-01T05:00Z"
    message = operator_report.format_report(report)
    assert "NEAR" in message and "2026-09-01" in message and "-12" not in message


def test_missing_all_history_names_market_references_honestly(tmp_path):
    report = operator_report.build_report(error=True, reason="no_data", root=tmp_path)
    assert report["ideas"][0]["source"] == "market_reference"
    assert "BTC" in operator_report.format_report(report)
    assert report["ideas"][0]["asof"] is None


def test_normal_picks_do_not_get_duplicate_observation_names(tmp_path):
    signals = pd.DataFrame([{"market": "KRW-T", "side": "SHORT", "actionable": True,
                             "sigma": 2, "expected_pct": -0.4, "horizon_h": 6}])
    report = operator_report.build_report(signals, root=tmp_path)
    assert not report["ideas"]
    assert report["status"] == "ok"
    assert "SHORT" in operator_report.format_report(report)


def test_failed_delivery_is_persisted_and_retried(tmp_path, monkeypatch):
    outcomes = iter([False, True])
    sent = []
    monkeypatch.setattr(telegram, "send_message", lambda message: sent.append(message) or next(outcomes))
    report = operator_report.build_report(root=tmp_path)
    assert not operator_report.publish_report(report, root=tmp_path)
    assert latest(tmp_path)["telegram"]["state"] == "pending"
    assert operator_report.retry_latest(tmp_path)
    assert latest(tmp_path)["telegram"]["state"] == "sent"
    assert latest(tmp_path)["telegram"]["attempts"] == 2
    assert operator_report.retry_latest(tmp_path)
    assert len(sent) == 2 and sent[0] == sent[1]


def test_new_report_supersedes_old_pending_message(tmp_path, monkeypatch):
    monkeypatch.setattr(telegram, "send_message", lambda message: False)
    first = operator_report.build_report(root=tmp_path)
    operator_report.publish_report(first, root=tmp_path)
    second = operator_report.build_report(root=tmp_path)
    operator_report.publish_report(second, root=tmp_path)
    prior = json.loads((tmp_path / "output" / "operator_reports" / f"{first['run_id']}.json").read_text())
    assert prior["telegram"]["state"] == "superseded"
    assert latest(tmp_path)["run_id"] == second["run_id"]


def test_suppressed_test_delivery_is_never_retried(tmp_path, monkeypatch):
    def forbidden(message):
        pytest.fail("--no-telegram report must not be delivered by background retry")
    monkeypatch.setattr(telegram, "send_message", forbidden)
    operator_report.publish_report(operator_report.build_report(root=tmp_path), send=False, root=tmp_path)
    assert operator_report.retry_latest(tmp_path)
    assert latest(tmp_path)["telegram"]["state"] == "disabled"


def test_expired_delivery_does_not_resend_an_old_actionable_signal(tmp_path, monkeypatch):
    from datetime import datetime, timedelta
    monkeypatch.setattr(telegram, "send_message", lambda message: False)
    signals = pd.DataFrame([{"market": "KRW-T", "side": "SHORT", "actionable": True,
                             "sigma": 2, "expected_pct": -9, "horizon_h": 6,
                             "entry_time": "2026-09-01T05:00Z"}])
    report = operator_report.build_report(signals, root=tmp_path)
    operator_report.publish_report(report, root=tmp_path)
    messages = []
    monkeypatch.setattr(telegram, "send_message", lambda message: messages.append(message) or True)
    assert operator_report.retry_latest(tmp_path, now=datetime.fromisoformat(report["generated_at"]) + timedelta(hours=7))
    replacement = latest(tmp_path)
    assert replacement["replaces_run_id"] == report["run_id"]
    assert not replacement["signals"]
    assert replacement["ideas"][0]["reference_only"]
    assert '-9' not in messages[0] and 'actionable=false' in messages[0]
