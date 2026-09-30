#!/usr/bin/env python3
"""Offline synthetic load/replay drill in a disposable ledger, never live evidence."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from utils import execution_audit as execution, forecast_audit as audit, prospective as trial, venue_observer as venue
from utils.experiment_supervisor import _write
from utils.selection_trace import attach_selection_trace


def seed(root, windows):
    policy = audit.start_audit(root, now="2026-09-30T06:00Z")
    ep = execution.register_execution(root, now="2026-09-30T07:00Z")
    markets = [f"KRW-T{i:03d}" for i in range(100)]
    symbols = {m: m[4:] + "USDT" for m in markets}
    scores = pd.Series({m: -.02 + i * .0004 for i, m in enumerate(markets)})
    with closing(audit._connect(root / "output/forecast_audit/ledger.sqlite")) as conn, conn:
        for i in range(windows):
            signal = trial._utc(policy["first_signal_at"]) + pd.Timedelta(hours=6 * i)
            slot, recorded = signal.isoformat(), signal + pd.Timedelta(minutes=10)
            witness = {"schema": 1, "kind": "signal", "signal_at": slot, "recorded_at": recorded.isoformat(),
                       "horizon_h": 6, "model_sha256": "a" * 64, "source_sha256": {"synthetic": "b" * 64},
                       "changes_gate": False, "registered_trial_evidence": False, "timely_signal": True,
                       "input_basis": "contemporaneous_signal_inputs", "features": ["synthetic"],
                       "rows": [{"market": m, "score": float(scores[m]), "actual": None, "inputs": {"synthetic": float(scores[m])}} for m in markets]}
            report = {"run_id": f"synthetic-{i}", "generated_at": recorded.isoformat(), "data_asof": slot,
                      "score_evidence_sha256": trial._digest(witness), "status": "watch", "ideas": [],
                      "signals": [{"market": m, "side": "SHORT", "actionable": False} for m in markets[5:10]],
                      "telegram": {"state": "sent", "acknowledged_at": recorded.isoformat()},
                      "audit_context": {"short_gate": "FREEZE", "tradable_symbols": symbols}}
            attach_selection_trace(report, scores, scores.sort_values(ascending=False), scores.loc[markets[5:10]], set(markets[5:10]), set(), 5, 0, 10)
            intent = venue.make_intent(witness, report, recorded)
            prices = pd.DataFrame([{"timestamp": signal + pd.Timedelta(hours=h), "market": m,
                                    "open": 100 if h == 1 else 99 + j / 100} for h in (1, 7) for j, m in enumerate(markets)])
            outcome, _ = audit.evaluate(witness, prices, signal + pd.Timedelta(hours=8))
            audit._append(conn, "signals", slot, {"signal_at": slot, "observed_at": recorded.isoformat(), "status": "recorded", "witness": witness})
            audit._append(conn, "deliveries", slot + "/" + trial._digest(report), audit._delivery(report, slot, trial._digest(witness), recorded))
            audit._append(conn, "outcomes", slot, outcome)
            audit._append(conn, "venue_intents", slot, intent)
            books, quotes = {}, {}
            for phase, hours in (("entry", 1), ("exit", 7)):
                target = signal + pd.Timedelta(hours=hours)
                observed = target + pd.Timedelta(seconds=10)
                millis = str(int(observed.timestamp() * 1000))
                payload = {"code": "00000", "data": [{"symbol": s, "bidPr": str(100 + (j / 100 if phase == "exit" else 0)),
                            "askPr": str(100.1 + (j / 100 if phase == "exit" else 0)), "bidSz": "50", "askSz": "50", "ts": millis}
                           for j, s in enumerate(symbols.values())]}
                quotes[phase] = venue.observe_quotes(intent, phase, payload, observed)
                audit._append(conn, "quotes", slot + "/" + phase, quotes[phase])
                for m in execution._members(intent):
                    depth = {"code": "00000", "data": {"precision": "scale0", "ts": millis,
                             "bids": [[100 - j * .01, 10] for j in range(50)],
                             "asks": [[100.1 + j * .01, 10] for j in range(50)]}}
                    key = f"depth/{slot}/{phase}/{symbols[m]}"
                    books[key] = execution.observe_depth(slot, phase, symbols[m], target, depth, observed, ep)
                    audit._append(conn, execution.TABLE, key, books[key])
            audit._append(conn, "venue_outcomes", slot, venue.evaluate_venue(intent, quotes["entry"], quotes["exit"], witness, outcome, policy["venue"]["fee_and_extra_bps"]))
            audit._append(conn, execution.TABLE, "result/" + slot, execution._result(slot, intent, books, ep))
            for m in execution._members(intent):
                funding = {"code": "00000", "data": [{"symbol": symbols[m], "fundingRate": ".0001",
                           "fundingTime": str(int((signal + pd.Timedelta(hours=h)).timestamp() * 1000))} for h in (0, 4, 8)]}
                row = execution.observe_funding(slot, symbols[m], quotes["entry"]["observed_at"], quotes["exit"]["observed_at"], funding, signal + pd.Timedelta(hours=17), ep)
                audit._append(conn, execution.TABLE, f"funding/{slot}/{symbols[m]}", row)
    return signal + pd.Timedelta(hours=17)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, default=336, choices=range(1, 337), metavar="1..336")
    args = parser.parse_args()
    output = trial.ROOT / "output/operating_checks"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output, prefix="synthetic-load-") as folder:
        root = Path(folder)
        seeded_at = time.monotonic()
        now = seed(root, args.windows)
        seeded_seconds = time.monotonic() - seeded_at
        print(f"Synthetic {args.windows}-window ledger seeded in {seeded_seconds:.2f}s; starting replay", flush=True)
        def offline(*args):
            raise AssertionError("Synthetic load drill must not fetch live data")
        started = time.monotonic()
        state = audit.refresh_audit(root, now=now, price_loader=offline, quote_loader=offline, execution_loader=offline)
        refreshed = time.monotonic()
        print(f"Synthetic refresh took {refreshed - started:.2f}s; starting backup/restore", flush=True)
        backup = audit.backup_audit(root)
        finished = time.monotonic()
        result = {"synthetic": True, "checked_at": trial._utc().isoformat(), "windows": args.windows,
                  "coins_per_window": 100, "depth_levels": 50, "seed_seconds": round(seeded_seconds, 3),
                  "refresh_seconds": round(refreshed - started, 3), "backup_seconds": round(finished - refreshed, 3),
                  "audit_and_backup_seconds": round(finished - started, 3), "capture_seconds": state["timings"]["capture_seconds"],
                  "row_counts": backup["row_counts"], "restore_verified": backup["restore_verified"],
                  "within_180_second_service_budget": finished - started < 180,
                  "within_60_second_capture_budget": state["timings"]["capture_seconds"] < 60,
                  "network_latency_included": False, "full_supervisor_measured": False}
        _write(output / f"load-{args.windows}.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["within_180_second_service_budget"] and result["within_60_second_capture_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
