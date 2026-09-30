from copy import deepcopy
import json

import pytest

from test_venue_observer import scenario as scenario, SLOT, quote_payload, raw_prices, contents
from utils import execution_audit as execution, forecast_audit as audit, prospective as trial


def depth_payload(stamp, size=100):
    return {"code": "00000", "data": {"precision": "scale0", "ts": str(int(trial._utc(stamp).timestamp() * 1000)),
            "bids": [[100, size], [99, size]], "asks": [[101, size], [102, size]]}}


def funding_payload(symbol="T00USDT", rates=(.001, -.002, .003)):
    times = ("2026-09-30T08:00Z", "2026-09-30T16:00Z", "2026-10-01T00:00Z")
    return {"code": "00000", "data": [{"symbol": symbol, "fundingRate": str(rate),
            "fundingTime": str(int(trial._utc(t).timestamp() * 1000))} for t, rate in zip(times, rates)]}


@pytest.fixture
def registered_execution(scenario):
    root, _, _ = scenario
    policy = execution.register_execution(root, now="2026-09-30T07:00Z")
    return root, policy


def test_registration_is_separate_future_only_and_never_resets_audit(registered_execution):
    root, policy = registered_execution
    assert policy["first_signal_at"] == SLOT
    assert policy["notionals_usdt_per_coin"] == [100, 500, 1000]
    assert not policy["actual_fills"] and not policy["changes_gate"]
    assert trial._digest(contents(root)[0]) == policy["audit_policy_sha256"]
    with pytest.raises(FileExistsError):
        execution.register_execution(root)


def test_depth_math_keeps_same_quantity_at_exit_and_does_not_double_charge_spread(registered_execution):
    _, policy = registered_execution
    stamp = "2026-09-30T12:00:10Z"
    entry = execution.observe_depth(SLOT, "entry", "T00USDT", "2026-09-30T12:00Z", depth_payload(stamp, 1), stamp, policy)
    end = deepcopy(entry)
    end["levels"]["asks"] = [[90, 1], [91, 1]]
    small = execution.depth_cost(entry, end, 100, 22)
    big = execution.depth_cost(entry, end, 200, 22)
    assert small["net_proxy_pct"] == pytest.approx(9.78)
    assert big["entry_vwap"] == 99.5 and big["exit_vwap"] == 90.5
    assert big["net_proxy_pct"] == pytest.approx((1 - 181 / 199) * 100 - .22)
    assert big["additional_depth_bps"] > 0
    assert execution.depth_cost(entry, end, 500, 22)["status"] == "insufficient_depth"


@pytest.mark.parametrize("issue", ["stale", "future", "merged", "duplicate", "unsorted", "negative", "nan", "crossed", "late"])
def test_invalid_depth_is_rejected_without_repair(registered_execution, issue):
    _, policy = registered_execution
    now = "2026-09-30T12:00:10Z"
    payload = depth_payload(now)
    data = payload["data"]
    if issue in ("stale", "future"):
        data["ts"] = str(int(data["ts"]) + (-61000 if issue == "stale" else 6000))
    elif issue == "merged":
        data["precision"] = "scale1"
    elif issue == "duplicate":
        data["bids"][1][0] = 100
    elif issue == "unsorted":
        data["asks"].reverse()
    elif issue in ("negative", "nan"):
        data["bids"][0][1] = -1 if issue == "negative" else float("nan")
    elif issue == "crossed":
        data["bids"][0][0] = 103
    else:
        now = "2026-09-30T12:05:01Z"
    with pytest.raises(ValueError):
        execution.observe_depth(SLOT, "entry", "T00USDT", "2026-09-30T12:00Z", payload, now, policy)


def test_funding_sign_window_and_boundary_are_explicit_not_account_cashflows(registered_execution):
    _, policy = registered_execution
    args = [SLOT, "T00USDT", "2026-09-30T12:00:10Z", "2026-09-30T18:00:10Z", funding_payload(), "2026-10-01T03:10Z", policy]
    row = execution.observe_funding(*args)
    assert row["rate_sum_bps"] == -20 and len(row["settlements"]) == 1
    assert row["cashflows_measured"] is False
    args[2] = "2026-09-30T16:00:10Z"
    uncertain = execution.observe_funding(*args)
    assert uncertain["status"] == "boundary_uncertain" and uncertain["rate_sum_bps"] is None


@pytest.mark.parametrize("issue", ["empty", "unbracketed", "duplicate", "wrong_symbol", "nonfinite", "gap", "future"])
def test_funding_missing_or_ambiguous_history_cannot_be_zero_cost(registered_execution, issue):
    _, policy = registered_execution
    payload = funding_payload()
    rows = payload["data"]
    if issue == "empty":
        payload["data"] = []
    elif issue == "unbracketed":
        payload["data"] = rows[1:]
    elif issue == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif issue == "wrong_symbol":
        rows[0]["symbol"] = "OTHERUSDT"
    elif issue == "nonfinite":
        rows[0]["fundingRate"] = "nan"
    elif issue == "gap":
        payload["data"] = [rows[0], rows[-1]]
    else:
        rows[-1]["fundingTime"] = str(int(trial._utc("2026-10-02T00:00Z").timestamp() * 1000))
    with pytest.raises(ValueError):
        execution.observe_funding(SLOT, "T00USDT", "2026-09-30T12:00:10Z", "2026-09-30T18:00:10Z", payload, "2026-10-01T03:10Z", policy)


def test_end_to_end_size_and_funding_collection_survives_restart_and_backup(registered_execution):
    root, policy = registered_execution
    audit.refresh_audit(root, now="2026-09-30T11:10Z")
    calls = []
    for stamp in ("2026-09-30T12:00:10Z", "2026-09-30T18:00:10Z"):
        def loader(kind, symbol):
            calls.append((kind, symbol))
            return depth_payload(stamp), trial._utc(stamp)
        audit.refresh_audit(root, now=stamp, quote_loader=lambda: (quote_payload(stamp), trial._utc(stamp)), execution_loader=loader)
    assert len(calls) == 20
    state = audit.refresh_audit(root, now="2026-09-30T19:10Z", price_loader=raw_prices)
    assert state["execution"]["depth_observations"] == 20
    initial_row = next(r for r in state["execution"]["recent_windows"] if r["signal_at"] == SLOT)
    assert initial_row["sizes"][0]["model_net_proxy_pct"] == pytest.approx(-1.22)
    assert state["execution"]["funding_observations"] == 0
    def funded(kind, symbol):
        assert kind == "funding"
        return funding_payload(symbol), trial._utc("2026-10-01T03:10Z")
    state = audit.refresh_audit(root, now="2026-10-01T03:10Z", execution_loader=funded)
    original = contents(root)[1][execution.TABLE].copy()
    row = next(r for r in state["execution"]["recent_windows"] if r["signal_at"] == SLOT)
    assert row["model_funding_bps"] == -20 and row["selected_funding_bps"] == -20
    audit.refresh_audit(root, now="2026-10-01T03:20Z", execution_loader=lambda *a: pytest.fail("no repeat sample"))
    assert contents(root)[1][execution.TABLE] == original
    assert audit.backup_audit(root)["restore_verified"]
    assert contents(root)[0]["venue"]["funding"].endswith("not measured")


def test_depth_network_failure_is_recorded_and_cannot_block_original_ic(registered_execution):
    root, _ = registered_execution
    audit.refresh_audit(root, now="2026-09-30T11:10Z")
    def fail(*args):
        raise OSError("offline")
    audit.refresh_audit(root, now="2026-09-30T12:00:10Z", quote_loader=lambda: (quote_payload("2026-09-30T12:00:10Z"), trial._utc("2026-09-30T12:00:10Z")), execution_loader=fail)
    state = audit.refresh_audit(root, now="2026-09-30T19:10Z", price_loader=raw_prices, execution_loader=lambda *a: pytest.fail("no late quotes"))
    assert state["counts"]["evaluated"] == 1 and state["execution"]["status"] == "attention"
    assert state["execution"]["failures"] == 20


def test_rejected_depth_payload_is_retained_and_its_failure_replayed(registered_execution):
    root, _ = registered_execution
    audit.refresh_audit(root, now="2026-09-30T11:10Z")
    stamp = "2026-09-30T12:00:10Z"
    stale = depth_payload("2026-09-30T11:58Z")
    audit.refresh_audit(root, now=stamp, quote_loader=lambda: (quote_payload(stamp), trial._utc(stamp)),
                        execution_loader=lambda *args: (stale, trial._utc(stamp)))
    row = contents(root)[1][execution.TABLE][f"depth/{SLOT}/entry/T00USDT"]
    assert row["status"] == "failed" and row["response_payload"] == stale
    assert "stale" in row["reason"] and audit.backup_audit(root)["restore_verified"]


def test_non_json_finite_response_does_not_rollback_other_evidence(registered_execution):
    root, _ = registered_execution
    audit.refresh_audit(root, now="2026-09-30T11:10Z")
    stamp = "2026-09-30T12:00:10Z"
    bad = depth_payload(stamp)
    bad["data"]["bids"][0][1] = float("nan")
    state = audit.refresh_audit(root, now=stamp, quote_loader=lambda: (quote_payload(stamp), trial._utc(stamp)),
                                execution_loader=lambda *args: (bad, trial._utc(stamp)))
    assert state["counts"]["recorded"] == 1 and state["execution"]["failures"] == 10
    assert audit.backup_audit(root)["restore_verified"]


def test_public_client_uses_only_read_only_endpoints(monkeypatch):
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size): return json.dumps({"code": "00000", "data": []}).encode()
    def fetch(request, timeout):
        assert timeout == 3 and request.get_method() == "GET"
        assert all(not k.lower().startswith("access") for k in request.headers)
        calls.append(request.full_url)
        return Response()
    monkeypatch.setattr(execution.urllib.request, "urlopen", fetch)
    execution.fetch_public("depth", "BTCUSDT")
    execution.fetch_public("funding", "BTCUSDT")
    assert "precision=scale0" in calls[0] and "pageSize=100" in calls[1]
    with pytest.raises(ValueError):
        execution.fetch_public("order", "BTCUSDT")
