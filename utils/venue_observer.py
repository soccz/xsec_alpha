"""Public Bitget quote observations, never orders or claims of actual fills."""
import json
import urllib.request

import numpy as np
import pandas as pd

from utils import prospective as trial

URL = "https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES"
DOCS_URL = "https://www.bitget.com/docs/catalog/classic-contract-market/classic-contract-market"


def fetch_quotes():
    request = urllib.request.Request(URL, headers={"User-Agent": "xsec-alpha-observation/1"})
    with urllib.request.urlopen(request, timeout=5) as response:
        raw = response.read(5_000_001)
    if len(raw) > 5_000_000:
        raise ValueError("Oversized public quote response")
    body = json.loads(raw)
    if body.get("code") != "00000" or not isinstance(body.get("data"), list):
        raise ValueError("Invalid public quote API response")
    return body, trial._utc()


def make_intent(witness, report, now):
    from utils.forecast_audit import _frame

    current, signal = trial._utc(now), trial._utc(witness["signal_at"])
    frame = _frame(witness, "signal", signal, current)
    if (not signal <= trial._utc(report["generated_at"]) <= current <= signal + pd.Timedelta(minutes=45)
            or trial._utc(report["data_asof"]) != signal
            or report.get("score_evidence_sha256") != trial._digest(witness)):
        raise ValueError("Venue intent was not linked and observed before entry")
    mapping = report.get("audit_context", {}).get("tradable_symbols", {})
    symbols = {m: symbol for m, symbol in sorted(mapping.items()) if m in frame.index}
    if (len(symbols) < 10 or len(set(symbols.values())) != len(symbols)
            or any(not isinstance(s, str) or not s.endswith("USDT") or not s.isalnum() for s in symbols.values())):
        raise ValueError("Insufficient or ambiguous contemporaneous contract mapping")
    model = frame.loc[list(symbols)].score.sort_values(kind="stable").head(5).index.tolist()
    selected = [r["market"] for r in report.get("signals", []) if r.get("side") == "SHORT"]
    comparable = len(selected) == 5 and len(set(selected)) == 5 and set(selected) <= set(symbols)
    return {"status": "recorded", "signal_at": signal.isoformat(), "observed_at": current.isoformat(),
            "witness_sha256": trial._digest(witness), "report_sha256": trial._digest(report),
            "symbols": symbols, "model_selected": model, "selected": selected,
            "selection_comparable": comparable,
            "actionable_count": sum(r.get("actionable") is True for r in report.get("signals", []) if r.get("side") == "SHORT"),
            "short_gate": report.get("audit_context", {}).get("short_gate"),
            "entry_at": (signal + pd.Timedelta(hours=1)).isoformat(),
            "exit_at": (signal + pd.Timedelta(hours=7)).isoformat()}


def observe_quotes(intent, phase, payload, observed_at):
    if phase not in ("entry", "exit"):
        raise ValueError("Unknown quote phase")
    current, target = trial._utc(observed_at), trial._utc(intent[phase + "_at"])
    delay = (current - target).total_seconds()
    if not 0 <= delay <= 300:
        raise ValueError("Quotes outside the five-minute observation window")
    if payload.get("code") != "00000" or not isinstance(payload.get("data"), list):
        raise ValueError("Invalid quote payload")
    rows, errors = {}, []
    for symbol in intent["symbols"].values():
        matches = [r for r in payload["data"] if r.get("symbol") == symbol]
        if len(matches) != 1:
            errors.append({"symbol": symbol, "reason": "missing_or_duplicate"})
            continue
        try:
            raw = matches[0]
            bid, ask = float(raw["bidPr"]), float(raw["askPr"])
            bid_size, ask_size = float(raw["bidSz"]), float(raw["askSz"])
            quote_time = pd.to_datetime(int(raw["ts"]), unit="ms", utc=True)
            age = (current - quote_time).total_seconds()
            if (not np.isfinite([bid, ask, bid_size, ask_size]).all() or not 0 < bid <= ask
                    or min(bid_size, ask_size) <= 0 or not -5 <= age <= 60):
                raise ValueError("invalid_or_stale")
            rows[symbol] = {"bid": bid, "ask": ask, "bid_size": bid_size, "ask_size": ask_size,
                            "quote_at": quote_time.isoformat(), "age_seconds": age}
        except (KeyError, ValueError, TypeError, OverflowError):
            errors.append({"symbol": symbol, "reason": "invalid_or_stale"})
    selected_payload = {"code": payload["code"], "requestTime": payload.get("requestTime"),
                        "data": [r for r in payload["data"] if r.get("symbol") in intent["symbols"].values()]}
    return {"signal_at": intent["signal_at"], "phase": phase, "observed_at": current.isoformat(),
            "delay_seconds": delay, "status": "invalid" if errors else "observed",
            "rows": rows, "errors": errors, "payload": selected_payload, "source": URL}


def evaluate_venue(intent, entry, exit_quote, witness, upbit, fee_bps):
    from utils.forecast_audit import _frame, _ic

    if entry["status"] != "observed" or exit_quote["status"] != "observed":
        raise ValueError("Both venue observations must be complete")
    frame = _frame(witness, "signal", intent["signal_at"], exit_quote["observed_at"])
    costs = fee_bps / 10000
    if not np.isfinite(costs) or costs < 0:
        raise ValueError("Invalid quoted cost assumption")
    rows = []
    for market, symbol in intent["symbols"].items():
        first, last = entry["rows"][symbol], exit_quote["rows"][symbol]
        change = (last["bid"] + last["ask"]) / (first["bid"] + first["ask"]) - 1
        short = 1 - last["ask"] / first["bid"]
        if not np.isfinite([change, short]).all():
            raise ValueError("Nonfinite quoted returns")
        rows.append({"market": market, "mid_return": change, "quoted_short_gross": short,
                     "quoted_short_cost_proxy": short - costs})
    values = pd.DataFrame(rows).set_index("market")
    raw = pd.DataFrame(upbit["prices"]).set_index("market")["return"].loc[values.index]
    model, selected = intent["model_selected"], intent["selected"]
    def mean(members, column):
        return float(values.loc[members, column].mean()) * 100
    return {"status": "evaluated", "signal_at": intent["signal_at"],
            "n_coins": len(values), "venue_ic": _ic(frame.loc[values.index, "score"], values.mid_return),
            "upbit_same_pool_ic": _ic(frame.loc[values.index, "score"], raw),
            "mean_mid_return_gap_pp": float((values.mid_return - raw).mean() * 100),
            "model_short_cost_proxy_pct": mean(model, "quoted_short_cost_proxy"),
            "selected_short_cost_proxy_pct": mean(selected, "quoted_short_cost_proxy") if intent["selection_comparable"] else None,
            "selected_minus_model_pp": (mean(selected, "quoted_short_cost_proxy") - mean(model, "quoted_short_cost_proxy")) if intent["selection_comparable"] else None,
            "actionable_count": intent["actionable_count"], "rows": rows,
            "entry_observed_at": entry["observed_at"], "exit_observed_at": exit_quote["observed_at"],
            "cost_bps": fee_bps, "funding_cashflows_measured": False, "actual_fills": False}


def advance_venue(conn, tables, policy, root, now=None, quote_loader=None):
    from utils.forecast_audit import _append

    current = trial._utc(now)
    cached = None
    for slot, source in tables["signals"].items():
        if slot in tables["venue_outcomes"]:
            continue
        signal = trial._utc(slot)
        intent = tables["venue_intents"].get(slot)
        if intent is None:
            reports = [r["report"] for r in tables["deliveries"].values()
                       if r["signal_at"] == slot and r["linked"]]
            if reports and current <= signal + pd.Timedelta(minutes=45):
                chosen = min(reports, key=lambda r: (r["generated_at"], trial._digest(r)))
                try:
                    intent = make_intent(source["witness"], chosen, current)
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    intent = {"status": "invalid", "signal_at": slot, "observed_at": current.isoformat(), "reason": str(exc)}
            elif current > signal + pd.Timedelta(minutes=45):
                intent = {"status": "missed", "signal_at": slot, "observed_at": current.isoformat(), "reason": "no_preentry_venue_intent"}
            else:
                continue
            _append(conn, "venue_intents", slot, intent)
            tables["venue_intents"][slot] = intent
        if intent["status"] != "recorded":
            _append(conn, "venue_outcomes", slot, {"signal_at": slot, "status": "invalid", "reason": intent["reason"]})
            continue
        for phase in ("entry", "exit"):
            key = slot + "/" + phase
            target = trial._utc(intent[phase + "_at"])
            if key in tables["quotes"] or current < target:
                continue
            if current > target + pd.Timedelta(minutes=5):
                quote = {"signal_at": slot, "phase": phase, "status": "missed", "observed_at": current.isoformat(), "reason": "quote_window_missed"}
            else:
                try:
                    if cached is None:
                        cached = (quote_loader or fetch_quotes)()
                    payload, observed_at = cached
                    quote = observe_quotes(intent, phase, payload, observed_at)
                except Exception as exc:
                    quote = {"signal_at": slot, "phase": phase, "status": "failed", "observed_at": trial._utc(now).isoformat(),
                             "reason": f"{type(exc).__name__}: {exc}"}
            _append(conn, "quotes", key, quote)
            tables["quotes"][key] = quote
        entry, exit_quote = (tables["quotes"].get(slot + "/" + p) for p in ("entry", "exit"))
        bad = next((q for q in (entry, exit_quote) if q and q["status"] != "observed"), None)
        upbit = tables["outcomes"].get(slot)
        if bad:
            result = {"signal_at": slot, "status": "invalid", "reason": "incomplete_venue_quotes"}
        elif upbit and upbit["status"] != "evaluated":
            result = {"signal_at": slot, "status": "invalid", "reason": "invalid_upbit_comparison"}
        elif entry and exit_quote and upbit:
            result = evaluate_venue(intent, entry, exit_quote, source["witness"], upbit, policy["venue"]["fee_and_extra_bps"])
        else:
            continue
        _append(conn, "venue_outcomes", slot, result)


def verify_venue(tables, policy):
    for slot, intent in tables["venue_intents"].items():
        if intent["signal_at"] != slot or slot not in tables["signals"]:
            raise ValueError("Venue intent signal mismatch")
        if intent["status"] == "recorded":
            reports = [r["report"] for r in tables["deliveries"].values()
                       if trial._digest(r["report"]) == intent["report_sha256"]]
            if not reports or make_intent(tables["signals"][slot]["witness"], reports[0], intent["observed_at"]) != intent:
                raise ValueError("Venue intent binding mismatch")
        elif intent["status"] not in ("missed", "invalid"):
            raise ValueError("Invalid venue intent state")
    for key, quote in tables["quotes"].items():
        intent = tables["venue_intents"][quote["signal_at"]]
        if key != quote["signal_at"] + "/" + quote["phase"]:
            raise ValueError("Quote key mismatch")
        if quote["status"] in ("observed", "invalid"):
            if observe_quotes(intent, quote["phase"], quote["payload"], quote["observed_at"]) != quote:
                raise ValueError("Quote arithmetic mismatch")
        elif quote["status"] not in ("failed", "missed"):
            raise ValueError("Unknown quote status")
    for slot, result in tables["venue_outcomes"].items():
        intent = tables["venue_intents"][slot]
        if result["signal_at"] != slot:
            raise ValueError("Venue outcome signal mismatch")
        if result["status"] == "evaluated":
            expected = evaluate_venue(intent, tables["quotes"][slot + "/entry"], tables["quotes"][slot + "/exit"],
                                      tables["signals"][slot]["witness"], tables["outcomes"][slot], policy["venue"]["fee_and_extra_bps"])
            if result != expected:
                raise ValueError("Venue outcome arithmetic mismatch")
        elif result["status"] != "invalid":
            raise ValueError("Invalid venue outcome status")


def venue_summary(tables):
    outcomes = tables["venue_outcomes"]
    rows = []
    for slot, intent in tables["venue_intents"].items():
        result = outcomes.get(slot, {})
        rows.append({"signal_at": slot, "status": result.get("status", "observing"), "reason": result.get("reason"),
                     "entry_status": tables["quotes"].get(slot + "/entry", {}).get("status", "waiting"),
                     "exit_status": tables["quotes"].get(slot + "/exit", {}).get("status", "waiting"),
                     **{key: result.get(key) for key in ("n_coins", "venue_ic", "upbit_same_pool_ic", "mean_mid_return_gap_pp",
                                                       "model_short_cost_proxy_pct", "selected_short_cost_proxy_pct", "selected_minus_model_pp")}})
    return {"status": "attention" if rows and rows[-1]["status"] == "invalid" else "observing",
            "recorded": sum(r["status"] == "recorded" for r in tables["venue_intents"].values()),
            "evaluated": sum(r["status"] == "evaluated" for r in outcomes.values()),
            "invalid": sum(r["status"] == "invalid" for r in outcomes.values()),
            "quotes": len(tables["quotes"]), "recent_windows": list(reversed(rows[-40:])),
            "actual_fills": False, "funding_cashflows_measured": False}
