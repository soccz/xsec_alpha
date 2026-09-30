from contextlib import closing
from copy import deepcopy
import json

import pandas as pd
import pytest

from utils import forecast_audit as audit, prospective as trial, venue_observer as venue
from utils.experiment_supervisor import _envelope, _write

SLOT = "2026-09-30T11:00:00+00:00"


@pytest.fixture
def scenario(tmp_path):
    audit.start_audit(tmp_path, now="2026-09-30T06:00Z")
    witness = {"schema": 1, "kind": "signal", "signal_at": SLOT, "recorded_at": "2026-09-30T11:05:00+00:00",
               "horizon_h": 6, "model_sha256": "a" * 64, "source_sha256": {"test": "b" * 64},
               "changes_gate": False, "registered_trial_evidence": False, "timely_signal": True,
               "input_basis": "contemporaneous_signal_inputs", "features": ["a"],
               "rows": [{"market": f"KRW-T{i:02d}", "score": i / 100, "actual": None, "inputs": {"a": i / 100}}
                        for i in range(12)]}
    digest = trial._digest(witness)
    _write(tmp_path / "output/score_evidence/signal/20260930T1100" / (digest + ".json"), _envelope(witness))
    report = {"run_id": "20260930T110600", "generated_at": "2026-09-30T11:06:00+00:00", "data_asof": SLOT,
              "score_evidence_sha256": digest, "status": "watch", "ideas": [],
              "signals": [{"market": f"KRW-T{i:02d}", "side": "SHORT", "actionable": False} for i in range(5, 10)],
              "telegram": {"state": "sent", "acknowledged_at": "2026-09-30T11:07:00+00:00"},
              "audit_context": {"short_gate": "FREEZE", "tradable_symbols": {f"KRW-T{i:02d}": f"T{i:02d}USDT" for i in range(12)}}}
    _write(tmp_path / "output/operator_reports/20260930T110600.json", report)
    return tmp_path, witness, report


def quote_payload(now, exit_prices=False):
    return {"code": "00000", "requestTime": int(trial._utc(now).timestamp() * 1000),
            "data": [{"symbol": f"T{i:02d}USDT", "bidPr": str(100 + i if exit_prices else 100),
                      "askPr": str(100.1 + i if exit_prices else 100.1), "bidSz": "50", "askSz": "30",
                      "fundingRate": "0.0001", "ts": str(int(trial._utc(now).timestamp() * 1000))}
                     for i in range(12)]}


def raw_prices(*args):
    return pd.DataFrame([{"timestamp": trial._utc(SLOT) + pd.Timedelta(hours=h), "market": f"KRW-T{i:02d}",
                          "open": 100 if h == 1 else 100 + i} for h in (1, 7) for i in range(12)])


def contents(root):
    with closing(audit._connect(root / "output/forecast_audit/ledger.sqlite", "ro")) as conn:
        return audit._read(conn)


def test_full_observed_venue_path_separates_model_selected_and_actual_execution(scenario):
    root, witness, report = scenario
    first = audit.refresh_audit(root, now="2026-09-30T11:10Z")
    assert first["venue"]["recorded"] == 1
    for stamp, is_exit in (("2026-09-30T12:00:10Z", False), ("2026-09-30T18:00:10Z", True)):
        audit.refresh_audit(root, now=stamp, quote_loader=lambda: (quote_payload(stamp, is_exit), trial._utc(stamp)))
    result = audit.refresh_audit(root, now="2026-09-30T19:10Z", price_loader=raw_prices)
    assert result["venue"]["evaluated"] == 1
    policy, tables = contents(root)
    intent = tables["venue_intents"][SLOT]
    assert intent["model_selected"] == [f"KRW-T{i:02d}" for i in range(5)]
    assert intent["selected"] == [f"KRW-T{i:02d}" for i in range(5, 10)]
    outcome = tables["venue_outcomes"][SLOT]
    assert outcome["venue_ic"] == pytest.approx(1.) and outcome["upbit_same_pool_ic"] == pytest.approx(1.)
    assert outcome["selected_minus_model_pp"] == pytest.approx(-5.)
    assert outcome["model_short_cost_proxy_pct"] == pytest.approx(-2.1 - policy["venue"]["fee_and_extra_bps"] / 100)
    assert outcome["actionable_count"] == 0
    assert outcome["actual_fills"] is False and outcome["funding_cashflows_measured"] is False
    assert audit.backup_audit(root)["restore_verified"]
    audit.refresh_audit(root, now="2026-09-30T19:20Z", quote_loader=lambda: pytest.fail("settled quotes never fetched again"))
    assert contents(root)[1]["venue_outcomes"][SLOT] == outcome


@pytest.mark.parametrize("issue", ["missing", "duplicate", "stale", "future", "crossed", "zero", "empty_size"])
def test_bad_quotes_invalidate_full_frozen_pool_without_substitution(scenario, issue):
    _, witness, report = scenario
    intent = venue.make_intent(witness, report, "2026-09-30T11:10Z")
    stamp = "2026-09-30T12:00:10Z"
    payload = quote_payload(stamp)
    row = payload["data"][0]
    if issue == "missing":
        payload["data"] = payload["data"][1:]
    elif issue == "duplicate":
        payload["data"].append(deepcopy(row))
    elif issue in ("stale", "future"):
        row["ts"] = str(int(trial._utc(stamp).timestamp() * 1000) + (-61000 if issue == "stale" else 6000))
    elif issue == "crossed":
        row["askPr"] = "99"
    elif issue == "zero":
        row["bidPr"] = "0"
    else:
        row["bidSz"] = "0"
    quote = venue.observe_quotes(intent, "entry", payload, stamp)
    assert quote["status"] == "invalid" and len(quote["errors"]) == 1
    assert len(quote["rows"]) == 11


def test_missed_quote_time_cannot_be_filled_with_later_prices(scenario):
    root, _, _ = scenario
    audit.refresh_audit(root, now="2026-09-30T11:10Z")
    state = audit.refresh_audit(root, now="2026-09-30T12:06Z",
                                quote_loader=lambda: pytest.fail("must not query late quote"))
    assert state["venue"]["invalid"] == 1 and state["venue"]["status"] == "attention"
    assert contents(root)[1]["quotes"][SLOT + "/entry"]["status"] == "missed"


def test_quote_api_failure_is_isolated_from_saved_score_settlement(scenario):
    root, _, _ = scenario
    audit.refresh_audit(root, now="2026-09-30T11:10Z")
    def fail():
        raise OSError("public endpoint unavailable")
    state = audit.refresh_audit(root, now="2026-09-30T12:00:10Z", quote_loader=fail)
    assert state["counts"]["recorded"] == 1 and state["venue"]["invalid"] == 1
    assert "endpoint unavailable" in contents(root)[1]["quotes"][SLOT + "/entry"]["reason"]
    state = audit.refresh_audit(root, now="2026-09-30T19:10Z", price_loader=raw_prices)
    assert state["counts"]["evaluated"] == 1


def test_report_received_after_capture_window_is_not_a_preentry_decision(scenario):
    root, _, _ = scenario
    state = audit.refresh_audit(root, now="2026-09-30T11:50Z")
    assert state["counts"]["recorded"] == 1
    assert state["venue"]["recorded"] == 0 and state["venue"]["invalid"] == 1


def test_quote_windows_and_selected_membership_do_not_silently_change(scenario):
    _, witness, report = scenario
    report["signals"][0]["market"] = "KRW-OUTSIDE"
    intent = venue.make_intent(witness, report, "2026-09-30T11:10Z")
    assert not intent["selection_comparable"]
    for stamp in ("2026-09-30T11:59:59Z", "2026-09-30T12:05:01Z"):
        with pytest.raises(ValueError, match="window"):
            venue.observe_quotes(intent, "entry", quote_payload(stamp), stamp)


def test_public_quote_client_has_no_auth_or_order_endpoint(monkeypatch):
    called = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, limit):
            return json.dumps(quote_payload("2026-09-30T12:00Z")).encode()
    def fetch(request, timeout):
        called.append(request)
        assert timeout == 5 and request.get_method() == "GET"
        assert request.full_url == venue.URL and "ACCESS-KEY" not in request.headers
        return Response()
    monkeypatch.setattr(venue.urllib.request, "urlopen", fetch)
    payload, observed = venue.fetch_quotes()
    assert len(called) == 1 and len(payload["data"]) == 12 and observed.tzinfo is not None
