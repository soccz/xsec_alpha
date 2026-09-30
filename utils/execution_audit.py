"""Independent size/depth and published-funding diagnostics. No account or order API."""
from contextlib import closing
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from utils import prospective as trial
from utils.run_lock import run_lock

BASE = "https://api.bitget.com/api/v2/mix/market/"
TABLE = "execution_details"
NOTIONALS = [100, 500, 1000]


def fetch_public(kind, symbol):
    if kind not in ("depth", "funding") or not symbol.isalnum() or not symbol.endswith("USDT"):
        raise ValueError("Unknown public observation request")
    params = {"symbol": symbol, "productType": "USDT-FUTURES"}
    if kind == "depth":
        endpoint = "merge-depth"
        params.update(precision="scale0", limit="50")
    else:
        endpoint = "history-fund-rate"
        params.update(pageSize="100", pageNo="1")
    url = BASE + endpoint + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "xsec-alpha-observation/1"})
    with urllib.request.urlopen(request, timeout=3) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Oversized market response")
    payload = json.loads(raw)
    if payload.get("code") != "00000":
        raise ValueError("Public market API rejected request")
    return payload, trial._utc()


def register_execution(root=trial.ROOT, now=None):
    from utils.forecast_audit import _append, _connect, _folder, _read
    root, current = Path(root), trial._utc(now)
    with run_lock("forecast_audit", lock_dir=str(root / "logs/locks"), timeout_sec=5):
        with closing(_connect(_folder(root) / "ledger.sqlite")) as conn, conn:
            audit_policy, tables = _read(conn)
            if tables[TABLE]:
                raise FileExistsError("Execution diagnostics already registered; no reset")
            first = current.floor("h") + pd.Timedelta(hours=1)
            while first.hour not in (5, 11, 17, 23):
                first += pd.Timedelta(hours=1)
            policy = {"schema": 1, "kind": "policy", "contract": "depth_funding_diagnostics_v1",
                      "registered_at": current.isoformat(), "first_signal_at": first.isoformat(),
                      "audit_policy_sha256": trial._digest(audit_policy), "notionals_usdt_per_coin": NOTIONALS,
                      "quote_window_seconds": 300, "max_age_seconds": 60, "future_tolerance_seconds": 5,
                      "funding_delay_hours": 9, "funding_deadline_hours": 36,
                      "funding_max_gap_hours": 8, "funding_boundary_seconds": 60,
                      "fee_and_extra_bps": audit_policy["venue"]["fee_and_extra_bps"],
                      "sampling": "one first observation attempt per symbol/phase; no replacement",
                      "funding_basis": "published settled rates on fixed notional; not mark-value cashflows",
                      "actual_fills": False, "changes_gate": False}
            conn.execute(f"CREATE TABLE IF NOT EXISTS {TABLE}(key TEXT PRIMARY KEY, payload TEXT NOT NULL, sha256 TEXT NOT NULL)")
            for action in ("UPDATE", "DELETE"):
                conn.execute(f"CREATE TRIGGER IF NOT EXISTS {TABLE}_{action.lower()} BEFORE {action} ON {TABLE} "
                             "BEGIN SELECT RAISE(ABORT,'append-only audit'); END")
            _append(conn, TABLE, "policy", policy)
            return policy


def observe_depth(slot, phase, symbol, target, payload, observed_at, policy):
    current, target = trial._utc(observed_at), trial._utc(target)
    if not 0 <= (current - target).total_seconds() <= policy["quote_window_seconds"]:
        raise ValueError("Depth observation window missed")
    body = payload["data"]
    stamp = pd.to_datetime(int(body["ts"]), unit="ms", utc=True)
    age = (current - stamp).total_seconds()
    if (payload["code"] != "00000" or body["precision"] != "scale0"
            or not -policy["future_tolerance_seconds"] <= age <= policy["max_age_seconds"]):
        raise ValueError("Merged or stale depth is not eligible")
    levels = {}
    for side in ("bids", "asks"):
        values = np.asarray(body[side], dtype=float)
        if (values.ndim != 2 or values.shape[1] != 2 or not 1 <= len(values) <= 50
                or not np.isfinite(values).all() or not (values > 0).all()
                or not (np.diff(values[:, 0]) < 0 if side == "bids" else np.diff(values[:, 0]) > 0).all()):
            raise ValueError("Invalid, duplicate or unordered depth levels")
        levels[side] = values.tolist()
    if levels["bids"][0][0] > levels["asks"][0][0]:
        raise ValueError("Crossed depth")
    return {"kind": "depth", "signal_at": slot, "phase": phase, "symbol": symbol,
            "status": "observed", "observed_at": current.isoformat(), "target_at": target.isoformat(),
            "quote_at": stamp.isoformat(), "levels": levels, "payload": payload}


def consume(levels, quantity):
    if not np.isfinite(quantity) or quantity <= 0:
        raise ValueError("Invalid hypothetical quantity")
    remaining, value = quantity, 0.
    for price, size in levels:
        take = min(remaining, size)
        value += take * price
        remaining -= take
        if remaining <= quantity * 1e-12:
            return value
    return None


def depth_cost(entry, exit_quote, notional, fee_bps):
    first, last = entry["levels"], exit_quote["levels"]
    quantity = notional / first["bids"][0][0]
    proceeds, expense = consume(first["bids"], quantity), consume(last["asks"], quantity)
    if proceeds is None or expense is None:
        return {"status": "insufficient_depth", "net_proxy_pct": None}
    gross = 1 - expense / proceeds
    top = 1 - last["asks"][0][0] / first["bids"][0][0]
    values = [gross, top, proceeds, expense]
    if not np.isfinite(values).all() or proceeds <= 0:
        raise ValueError("Nonfinite depth cost")
    return {"status": "evaluated", "net_proxy_pct": 100 * (gross - fee_bps / 10000),
            "additional_depth_bps": 10000 * (top - gross), "quantity": quantity,
            "entry_vwap": proceeds / quantity, "exit_vwap": expense / quantity}


def observe_funding(slot, symbol, entry_at, exit_at, payload, observed_at, policy):
    current, entry, end = trial._utc(observed_at), trial._utc(entry_at), trial._utc(exit_at)
    if not end + pd.Timedelta(hours=policy["funding_delay_hours"]) <= current <= end + pd.Timedelta(hours=policy["funding_deadline_hours"]):
        raise ValueError("Funding collection outside declared window")
    if payload["code"] != "00000" or not isinstance(payload["data"], list) or not payload["data"]:
        raise ValueError("Missing funding history; cannot assume zero")
    rows = []
    for row in payload["data"]:
        stamp = pd.to_datetime(int(row["fundingTime"]), unit="ms", utc=True)
        rate = float(row["fundingRate"])
        if row["symbol"] != symbol or not np.isfinite(rate) or stamp > current:
            raise ValueError("Invalid funding symbol, rate or timestamp")
        rows.append({"at": stamp.isoformat(), "rate": rate})
    rows.sort(key=lambda r: r["at"])
    stamps = [trial._utc(r["at"]) for r in rows]
    if len(set(stamps)) != len(stamps) or not stamps[0] <= entry < end <= stamps[-1]:
        raise ValueError("Funding history must bracket both observation times without duplicates")
    relevant = [(a, b) for a, b in zip(stamps, stamps[1:]) if a < end and b > entry]
    if any((b - a).total_seconds() > policy["funding_max_gap_hours"] * 3600 for a, b in relevant):
        raise ValueError("Unexplained funding history gap")
    # Near settlement, actual fee eligibility cannot be inferred from quote receipt time.
    boundary = [r for r in rows if min(abs((trial._utc(r["at"]) - entry).total_seconds()),
                                     abs((trial._utc(r["at"]) - end).total_seconds())) <= policy["funding_boundary_seconds"]]
    held = [r for r in rows if entry < trial._utc(r["at"]) <= end]
    return {"kind": "funding", "signal_at": slot, "symbol": symbol, "status": "boundary_uncertain" if boundary else "observed",
            "observed_at": current.isoformat(), "entry_at": entry.isoformat(), "exit_at": end.isoformat(),
            "rate_sum_bps": None if boundary else float(sum(r["rate"] for r in held) * 10000),
            "settlements": held, "boundary_records": boundary, "payload": payload,
            "cashflows_measured": False}


def _members(intent):
    return sorted(set(intent["model_selected"]) | (set(intent["selected"]) if intent["selection_comparable"] else set()))


def _result(slot, intent, records, policy):
    books = {phase: {m: records.get(f"depth/{slot}/{phase}/{intent['symbols'][m]}") for m in _members(intent)}
             for phase in ("entry", "exit")}
    if any(value is None for phase in books.values() for value in phase.values()):
        return None
    sizes = []
    for notional in policy["notionals_usdt_per_coin"]:
        rows = {}
        for market in _members(intent):
            entry, end = books["entry"][market], books["exit"][market]
            try:
                rows[market] = depth_cost(entry, end, notional, policy["fee_and_extra_bps"]) if entry["status"] == end["status"] == "observed" else {"status": "unavailable", "net_proxy_pct": None}
            except (ValueError, ZeroDivisionError, OverflowError) as exc:
                rows[market] = {"status": "invalid", "net_proxy_pct": None, "reason": str(exc)}
        def basket(members):
            if len(members) != 5 or any(rows[m]["status"] != "evaluated" for m in members):
                return None
            return float(np.mean([rows[m]["net_proxy_pct"] for m in members]))
        model, selected = basket(intent["model_selected"]), basket(intent["selected"]) if intent["selection_comparable"] else None
        sizes.append({"notional_usdt": notional, "model_net_proxy_pct": model, "selected_net_proxy_pct": selected,
                      "selected_minus_model_pp": selected - model if selected is not None and model is not None else None,
                      "rows": rows})
    return {"kind": "result", "signal_at": slot, "sizes": sizes, "actual_fills": False,
            "status": "evaluated" if all(r["model_net_proxy_pct"] is not None for r in sizes) else "incomplete"}


def advance_execution(conn, tables, audit_policy, now=None, loader=None):
    from utils.forecast_audit import _append
    records = tables[TABLE]
    policy = records.get("policy")
    if not policy:
        return
    started = time.monotonic()
    for slot, intent in tables["venue_intents"].items():
        if trial._utc(slot) < trial._utc(policy["first_signal_at"]) or intent["status"] != "recorded":
            continue
        for phase in ("entry", "exit"):
            target = trial._utc(intent[phase + "_at"])
            for market in _members(intent):
                symbol = intent["symbols"][market]
                key = f"depth/{slot}/{phase}/{symbol}"
                current = trial._utc(now)
                if key in records or current < target:
                    continue
                base = {"kind": "depth", "signal_at": slot, "phase": phase, "symbol": symbol,
                        "target_at": target.isoformat(), "observed_at": current.isoformat()}
                response = None
                try:
                    if current > target + pd.Timedelta(seconds=policy["quote_window_seconds"]):
                        row = {**base, "status": "missed", "reason": "depth_window_missed"}
                    elif time.monotonic() - started > 90:
                        row = {**base, "status": "failed", "reason": "collection_budget_exhausted"}
                    else:
                        payload, stamp = (loader or fetch_public)("depth", symbol)
                        trial._json(payload)
                        response = {"response_payload": payload, "response_observed_at": trial._utc(stamp).isoformat()}
                        row = observe_depth(slot, phase, symbol, target, payload, stamp, policy)
                except Exception as exc:
                    row = {**base, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:300], **(response or {})}
                _append(conn, TABLE, key, row)
                records[key] = row
        result_key = "result/" + slot
        if result_key not in records:
            result = _result(slot, intent, records, policy)
            if result:
                _append(conn, TABLE, result_key, result)
                records[result_key] = result
        for market in _members(intent):
            symbol = intent["symbols"][market]
            key = f"funding/{slot}/{symbol}"
            entry = records.get(f"depth/{slot}/entry/{symbol}", {})
            end = records.get(f"depth/{slot}/exit/{symbol}", {})
            current = trial._utc(now)
            if key in records or current < trial._utc(intent["exit_at"]) + pd.Timedelta(hours=policy["funding_delay_hours"], seconds=policy["quote_window_seconds"]):
                continue
            base = {"kind": "funding", "signal_at": slot, "symbol": symbol, "observed_at": current.isoformat()}
            response = None
            try:
                if entry.get("status") != "observed" or end.get("status") != "observed":
                    row = {**base, "status": "unavailable", "reason": "depth_times_unavailable"}
                elif current > trial._utc(end["observed_at"]) + pd.Timedelta(hours=policy["funding_deadline_hours"]):
                    row = {**base, "status": "missed", "reason": "funding_deadline"}
                elif time.monotonic() - started > 90:
                    row = {**base, "status": "failed", "reason": "collection_budget_exhausted"}
                else:
                    payload, stamp = (loader or fetch_public)("funding", symbol)
                    trial._json(payload)
                    response = {"response_payload": payload, "response_observed_at": trial._utc(stamp).isoformat()}
                    row = observe_funding(slot, symbol, entry["observed_at"], end["observed_at"], payload, stamp, policy)
            except Exception as exc:
                row = {**base, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:300], **(response or {})}
            _append(conn, TABLE, key, row)
            records[key] = row


def verify_execution(tables, audit_policy):
    records = tables[TABLE]
    if not records:
        return
    policy = records["policy"]
    if (policy["contract"] != "depth_funding_diagnostics_v1" or policy["notionals_usdt_per_coin"] != NOTIONALS
            or policy["audit_policy_sha256"] != trial._digest(audit_policy)
            or trial._utc(policy["first_signal_at"]) <= trial._utc(policy["registered_at"])):
        raise ValueError("Execution diagnostic policy mismatch")
    for key, row in records.items():
        if key == "policy":
            continue
        slot = row["signal_at"]
        intent = tables["venue_intents"][slot]
        if intent["status"] != "recorded" or trial._utc(slot) < trial._utc(policy["first_signal_at"]):
            raise ValueError("Execution observation outside registered intent")
        if row["kind"] == "result":
            if key != "result/" + slot or _result(slot, intent, records, policy) != row:
                raise ValueError("Depth result replay mismatch")
            continue
        symbol = row["symbol"]
        if symbol not in [intent["symbols"][m] for m in _members(intent)]:
            raise ValueError("Unexpected execution symbol")
        if row["kind"] == "depth":
            expected_key = f"depth/{slot}/{row['phase']}/{symbol}"
            target = trial._utc(intent[row["phase"] + "_at"])
            if (trial._utc(row["target_at"]) != target or trial._utc(row["observed_at"]) < target
                    or (row["status"] == "missed" and trial._utc(row["observed_at"]) <= target + pd.Timedelta(seconds=policy["quote_window_seconds"]))):
                raise ValueError("Invalid depth failure timing")
            if row["status"] == "observed" and observe_depth(slot, row["phase"], symbol, intent[row["phase"] + "_at"], row["payload"], row["observed_at"], policy) != row:
                raise ValueError("Depth replay mismatch")
        elif row["kind"] == "funding":
            expected_key = f"funding/{slot}/{symbol}"
            if row["status"] in ("observed", "boundary_uncertain"):
                entry = records[f"depth/{slot}/entry/{symbol}"]["observed_at"]
                end = records[f"depth/{slot}/exit/{symbol}"]["observed_at"]
                if observe_funding(slot, symbol, entry, end, row["payload"], row["observed_at"], policy) != row:
                    raise ValueError("Funding replay mismatch")
        else:
            raise ValueError("Unknown execution observation kind")
        if key != expected_key or row["status"] not in ("observed", "boundary_uncertain", "failed", "missed", "unavailable"):
            raise ValueError("Invalid execution observation state")
        if row.get("response_payload") is not None:
            if row["status"] != "failed":
                raise ValueError("Rejected response on nonfailed observation")
            try:
                if row["kind"] == "depth":
                    observe_depth(slot, row["phase"], symbol, intent[row["phase"] + "_at"], row["response_payload"], row["response_observed_at"], policy)
                else:
                    observe_funding(slot, symbol, records[f"depth/{slot}/entry/{symbol}"]["observed_at"],
                                    records[f"depth/{slot}/exit/{symbol}"]["observed_at"], row["response_payload"], row["response_observed_at"], policy)
            except Exception as exc:
                if row["reason"] != f"{type(exc).__name__}: {exc}"[:300]:
                    raise ValueError("Rejected response reason changed") from exc
            else:
                raise ValueError("Valid response mislabeled as failed")


def execution_summary(tables):
    records = tables[TABLE]
    if not records:
        return None
    policy = records["policy"]
    windows = []
    for slot, intent in tables["venue_intents"].items():
        if trial._utc(slot) < trial._utc(policy["first_signal_at"]):
            continue
        result = records.get("result/" + slot, {})
        funding = [r for r in records.values() if r.get("signal_at") == slot and r["kind"] == "funding"]
        def basket(members):
            rates = {r["symbol"]: r["rate_sum_bps"] for r in funding if r["status"] == "observed"}
            if len(members) != 5 or any(intent["symbols"][m] not in rates for m in members):
                return None
            return float(np.mean([rates[intent["symbols"][m]] for m in members]))
        windows.append({"signal_at": slot, "status": result.get("status", "waiting") if intent["status"] == "recorded" else "unavailable",
                        "sizes": [{k: v for k, v in r.items() if k != "rows"} for r in result.get("sizes", [])],
                        "model_funding_bps": basket(intent["model_selected"]) if intent["status"] == "recorded" else None,
                        "selected_funding_bps": basket(intent["selected"]) if intent.get("selection_comparable") else None,
                        "funding_records": len(funding), "funding_uncertain": sum(r["status"] != "observed" for r in funding)})
    failures = [r for r in records.values() if r.get("status") in ("failed", "missed", "unavailable", "boundary_uncertain")]
    latest = windows[-1] if windows else {}
    latest_funding = max((r["signal_at"] for r in records.values() if r["kind"] == "funding"), default=None)
    attention = (latest.get("status") in ("incomplete", "unavailable")
                 or any(r["signal_at"] == latest.get("signal_at") or
                        (r["kind"] == "funding" and r["signal_at"] == latest_funding) for r in failures))
    return {"policy": policy, "status": "attention" if attention else "observing",
            "depth_observations": sum(r["kind"] == "depth" and r["status"] == "observed" for r in records.values()),
            "funding_observations": sum(r["kind"] == "funding" and r["status"] == "observed" for r in records.values()),
            "failures": len(failures), "recent_windows": list(reversed(windows[-40:])),
            "recent_failures": [{k: v for k, v in r.items() if k not in ("payload", "levels", "response_payload")} for r in failures[-20:]],
            "actual_fills": False, "funding_cashflows_measured": False}
