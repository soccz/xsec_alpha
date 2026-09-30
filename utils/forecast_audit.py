"""Evaluate saved live scores without replaying a model or modifying an IC gate."""
from contextlib import closing
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import config
from utils import prospective as trial
from utils.experiment_supervisor import _envelope, _sync, _verified_document, _write
from utils.model_release import artifact_sha256
from utils.run_lock import run_lock

ROOT = trial.ROOT
TABLES = ("signals", "outcomes", "comparisons", "deliveries", "venue_intents", "quotes", "venue_outcomes", "ingestion_events", "execution_details")
LEGACY_CONTRACT = "absolute_open_lag1_anchors_v1"


def _folder(root):
    return Path(root) / "output/forecast_audit"


def _connect(path, mode="rw"):
    return sqlite3.connect(Path(path).resolve().as_uri() + f"?mode={mode}", uri=True, timeout=5)


def _event_table(conn):
    # Additive operational journal; old evidence and the registered policy stay intact.
    conn.execute("CREATE TABLE IF NOT EXISTS ingestion_events(key TEXT PRIMARY KEY, payload TEXT NOT NULL, sha256 TEXT NOT NULL)")
    for action in ("UPDATE", "DELETE"):
        conn.execute(f"CREATE TRIGGER IF NOT EXISTS ingestion_events_{action.lower()} BEFORE {action} ON ingestion_events "
                     "BEGIN SELECT RAISE(ABORT,'append-only audit'); END")


def _input_case(stage, source, slot):
    return trial._digest([stage, source, slot])


def _try_input(conn, tables, stage, source, slot, now, reader):
    """Isolate external input errors, never ledger writes or integrity failures."""
    error, result = None, None
    try:
        result = reader()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
    case = _input_case(stage, source, slot)
    events = tables["ingestion_events"]
    previous = max((r for r in events.values() if r["case"] == case),
                   key=lambda r: r["sequence"], default=None)
    state = "error" if error else "recovered"
    if (error or previous) and (previous is None or (previous["state"], previous["error"]) != (state, error)):
        sequence = previous["sequence"] + 1 if previous else 1
        body = {"schema": 1, "case": case, "sequence": sequence, "stage": stage,
                "source": source, "signal_at": slot, "observed_at": trial._utc(now).isoformat(),
                "state": state, "error": error}
        key = case + f"/{sequence:08d}"
        _append(conn, "ingestion_events", key, body)
        events[key] = body
    return error is None, result


def _ingestion_summary(tables):
    events = sorted(tables["ingestion_events"].values(), key=lambda r: (r["observed_at"], r["case"], r["sequence"]))
    latest = {}
    for row in events:
        if row["sequence"] > latest.get(row["case"], {}).get("sequence", 0):
            latest[row["case"]] = row
    return {"active": [r for r in latest.values() if r["state"] == "error"],
            "recovered": sum(r["state"] == "recovered" for r in events),
            "event_count": len(events), "recent_events": list(reversed(events[-40:]))}


def start_audit(root=ROOT, now=None):
    root, current = Path(root), trial._utc(now)
    with run_lock("forecast_audit", lock_dir=str(root / "logs/locks"), timeout_sec=5):
        folder = _folder(root)
        if folder.exists():
            raise FileExistsError("Forecast audit already exists; no in-place reset")
        first = current.floor("h") + pd.Timedelta(hours=1)
        while first.hour not in (5, 11, 17, 23):
            first += pd.Timedelta(hours=1)
        policy = {
            "schema": 1, "contract": "saved_full_pool_raw_open_6h_v1",
            "registered_at": current.isoformat(), "first_signal_at": first.isoformat(),
            "capture_minutes": 45, "horizon_h": 6, "price_grace_hours": 48,
            "minimum_coins": 10, "ic_difference_warning": .03,
            "selection": "first observed timely witness; recorded_at/hash order within a batch; never replace",
            "evaluation": "exact raw Upbit opens at signal+1h and +7h; evaluate after +8h, by +55h",
            "missing": "entire original pool required; no filling, exclusions or historical backfill",
            "purpose": "ongoing descriptive measurement audit, not a superiority experiment",
            "legacy_contract": LEGACY_CONTRACT, "changes_gate": False,
            "automatic_promotion": False, "executable_profitability": "not_tested",
            "venue": {"quote_window_seconds": 300, "max_quote_age_seconds": 60,
                      "fee_and_extra_bps": 2 * config.Costs.ONE_WAY_FEE_BPS + config.Costs.SHORT_EXTRA_COST_BPS,
                      "price_basis": "observed Bitget USDT perpetual bid/ask; not actual fills",
                      "comparison": "same frozen tradable pool, model bottom5 vs reported five; descriptive only",
                      "funding": "current quote field retained; realized funding cashflows not measured",
                      "alignment": "receipt within five minutes of Upbit reference hour; clock/currency/basis differences remain"},
        }
        if not np.isfinite(policy["venue"]["fee_and_extra_bps"]) or policy["venue"]["fee_and_extra_bps"] < 0:
            raise ValueError("Invalid venue cost assumption")
        folder.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder.parent, prefix=".forecast-audit-") as temporary:
            stage = Path(temporary) / "audit"
            stage.mkdir(mode=0o700)
            with closing(sqlite3.connect(stage / "ledger.sqlite")) as conn, conn:
                conn.execute("CREATE TABLE protocol(payload TEXT NOT NULL, sha256 TEXT NOT NULL)")
                conn.execute("INSERT INTO protocol VALUES (?,?)", (trial._json(policy), trial._digest(policy)))
                for table in TABLES:
                    conn.execute(f"CREATE TABLE {table}(key TEXT PRIMARY KEY, payload TEXT NOT NULL, sha256 TEXT NOT NULL)")
                for table in ("protocol", *TABLES):
                    for action in ("UPDATE", "DELETE"):
                        conn.execute(f"CREATE TRIGGER {table}_{action.lower()} BEFORE {action} ON {table} "
                                     "BEGIN SELECT RAISE(ABORT,'append-only audit'); END")
            _write(stage / "policy.json", _envelope(policy), immutable=True)
            stage.rename(folder)
        return policy


def _append(conn, table, key, body):
    if table not in TABLES:
        raise ValueError("Unknown audit table")
    if not conn.execute(f"SELECT 1 FROM {table} WHERE key=?", (key,)).fetchone():
        sources = (Path(__file__), Path(__file__).with_name("venue_observer.py"), Path(trial.__file__),
                   Path(__file__).with_name("execution_audit.py"), Path(__file__).with_name("selection_trace.py"))
        document = {**body, "_writer": {
            "sources": {p.name: artifact_sha256(p) for p in sources},
            "packages": {name: version(name) for name in ("numpy", "pandas", "scipy")},
        }}
        conn.execute(f"INSERT INTO {table} VALUES (?,?,?)", (key, trial._json(document), trial._digest(document)))


def _frame(witness, kind, slot, now):
    signal, recorded = trial._utc(witness["signal_at"]), trial._utc(witness["recorded_at"])
    if (witness["schema"] != 1 or witness["kind"] != kind or signal != trial._utc(slot)
            or witness["horizon_h"] != 6 or not signal <= recorded <= trial._utc(now)
            or witness.get("changes_gate") is not False or witness.get("registered_trial_evidence") is not False):
        raise ValueError("Invalid witness contract or timestamp")
    digest = witness["model_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid witness model hash")
    frame = pd.DataFrame(witness["rows"]).set_index("market")
    if (len(frame) < (10 if kind == "signal" else 5) or not frame.index.is_unique or not np.isfinite(frame.score).all()
            or not all(isinstance(m, str) and m.startswith("KRW-") for m in frame.index)):
        raise ValueError("Invalid full-pool witness membership or scores")
    features = witness["features"]
    if not features or len(set(features)) != len(features) or not witness["source_sha256"]:
        raise ValueError("Missing input/source provenance")
    for inputs in frame.inputs:
        if set(inputs) != set(features) or any(v is not None and not np.isfinite(v) for v in inputs.values()):
            raise ValueError("Original input mask or values changed")
    if kind == "signal":
        if frame.actual.notna().any() or witness["input_basis"] != "contemporaneous_signal_inputs":
            raise ValueError("Signal witness contains outcomes or replayed inputs")
        timely = recorded <= signal + pd.Timedelta(minutes=45)
        if witness["timely_signal"] is not timely:
            raise ValueError("Signal witness timing label mismatch")
    elif (recorded < signal + pd.Timedelta(hours=7) or not np.isfinite(frame.actual).all()
          or witness["input_basis"] != "recomputed_historical_inputs_at_measurement"):
        raise ValueError("Invalid recomputed measurement witness")
    return frame.sort_index()


def _witnesses(root, kind, slot, now):
    folder = Path(root) / "output/score_evidence" / kind / trial._utc(slot).strftime("%Y%m%dT%H%M")
    records = []
    for path in sorted(folder.glob("*.json")):
        body = _verified_document(path)
        if trial._digest(body) != path.stem:
            raise ValueError("Witness filename checksum mismatch")
        # A writer may finish after this pass's cutoff; consume it next pass.
        if trial._utc(body["recorded_at"]) > trial._utc(now):
            continue
        _frame(body, kind, slot, now)
        records.append(body)
    return sorted(records, key=lambda body: (body["recorded_at"], trial._digest(body)))


def _ic(scores, actual, minimum=10):
    if len(scores) < minimum or len(set(scores)) < 2 or len(set(actual)) < 2:
        return None
    return float(spearmanr(scores, actual).statistic)


def evaluate(witness, prices, now):
    """Return one terminal outcome or a visible pending reason, with no pool shrinkage."""
    current, signal = trial._utc(now), trial._utc(witness["signal_at"])
    frame = _frame(witness, "signal", signal, current)
    if not witness["timely_signal"]:
        raise ValueError("Late signal is not eligible for prospective evaluation")
    base = {"signal_at": signal.isoformat(), "evaluated_at": current.isoformat(),
            "witness_sha256": trial._digest(witness), "model_sha256": witness["model_sha256"],
            "n_coins": len(frame), "saved_ic": None, "price_basis": "upbit_raw_open_proxy_not_bitget_fills"}
    if current < signal + pd.Timedelta(hours=8):
        return None, "waiting_maturity"
    if current > signal + pd.Timedelta(hours=55):
        return {**base, "status": "invalid", "reason": "evaluation_deadline"}, None
    times = [signal + pd.Timedelta(hours=h) for h in (1, 7)]
    raw = prices.copy()
    raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True, errors="coerce")
    raw = raw.loc[raw.timestamp.isin(times) & raw.market.isin(frame.index)]
    raw["open"] = pd.to_numeric(raw.open, errors="coerce")
    reason = "duplicate_prices" if raw.duplicated(["timestamp", "market"]).any() else (
        "missing_prices" if len(raw) != 2 * len(frame) else
        "invalid_prices" if not (np.isfinite(raw.open).all() and (raw.open > 0).all()) else None)
    if reason:
        if current < signal + pd.Timedelta(hours=55):
            return None, reason
        return {**base, "status": "invalid", "reason": reason}, None
    wide = raw.pivot(index="timestamp", columns="market", values="open").reindex(index=times, columns=frame.index)
    with np.errstate(over="ignore", invalid="ignore"):
        returns = wide.iloc[1] / wide.iloc[0] - 1
    if not np.isfinite(returns).all():
        if current < signal + pd.Timedelta(hours=55):
            return None, "invalid_prices"
        return {**base, "status": "invalid", "reason": "invalid_prices"}, None
    value = _ic(frame.score, returns)
    rows = [{"market": m, "entry_open": float(wide.iloc[0][m]), "exit_open": float(wide.iloc[1][m]),
             "return": float(returns[m])} for m in frame.index]
    return {**base, "status": "evaluated" if value is not None else "invalid",
            "reason": None if value is not None else "constant_ranks", "saved_ic": value,
            "entry_at": times[0].isoformat(), "exit_at": times[1].isoformat(), "prices": rows}, None


def compare(witness, outcome, history, measurement, now):
    original = _frame(witness, "signal", witness["signal_at"], now)
    legacy_ic = history["ic"] if history else None
    if history and (history.get("contract_version") != LEGACY_CONTRACT
                    or history.get("side") != "short" or history.get("horizon_h") != 6
                    or trial._utc(history["timestamp"]) != trial._utc(witness["signal_at"])
                    or not np.isfinite(legacy_ic) or not -1 <= legacy_ic <= 1):
        raise ValueError("Incompatible legacy IC history")
    gap = outcome["saved_ic"] - legacy_ic if legacy_ic is not None else None
    result = {"signal_at": witness["signal_at"], "compared_at": trial._utc(now).isoformat(),
              "saved_ic": outcome["saved_ic"], "legacy_ic": legacy_ic, "difference": gap,
              "warning": abs(gap) >= .03 if gap is not None else False,
              "legacy_history": history, "measurement": measurement,
              "measurement_matches_history": False, "same_model": None, "same_pool": None,
              "measurement_raw_ic": None, "measurement_target_ic": None,
              "score_rank_correlation": None, "max_target_difference": None}
    if measurement:
        measured = _frame(measurement, "measurement", witness["signal_at"], now)
        measured_ic = _ic(measured.score, measured.actual, minimum=5)
        same_pool = measured.index.equals(original.index)
        common = original.index.intersection(measured.index)
        result.update(same_model=measurement["model_sha256"] == witness["model_sha256"],
                      same_pool=same_pool, measurement_target_ic=measured_ic,
                      score_rank_correlation=_ic(original.loc[common, "score"], measured.loc[common, "score"]))
        if history and measured_ic is not None:
            result["measurement_matches_history"] = (len(measured) == history.get("n_coins")
                                                      and abs(measured_ic - legacy_ic) <= .00000051)
        if same_pool:
            actual = pd.DataFrame(outcome["prices"]).set_index("market")["return"].reindex(measured.index)
            result["measurement_raw_ic"] = _ic(measured.score, actual)
            result["max_target_difference"] = float((measured.actual - actual).abs().max())
    return result


def _delivery(report, signal, witness_sha, now):
    current, signal = trial._utc(now), trial._utc(signal)
    generated = trial._utc(report["generated_at"])
    if not signal <= generated <= current or trial._utc(report["data_asof"]) != signal:
        raise ValueError("Report does not belong to this signal")
    telegram = report.get("telegram", {})
    acknowledged = telegram.get("state") == "sent" and bool(telegram.get("acknowledged_at"))
    if acknowledged and not generated <= trial._utc(telegram["acknowledged_at"]) <= current:
        raise ValueError("Invalid Telegram acknowledgment time")
    linked = witness_sha is not None and report.get("score_evidence_sha256") == witness_sha
    return {"signal_at": signal.isoformat(), "observed_at": current.isoformat(), "report": report,
            "acknowledged": acknowledged, "linked": linked,
            "complete": bool(acknowledged and linked and report.get("status") != "error"
                             and (report.get("signals") or report.get("ideas")))}


def _read(conn):
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError("Forecast audit SQLite integrity failure")
    policy = trial._manifest(conn)
    if policy["contract"] != "saved_full_pool_raw_open_6h_v1":
        raise ValueError("Unknown forecast audit contract")
    tables = {}
    for table in TABLES:
        records = {}
        if table in ("ingestion_events", "execution_details") and not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
            tables[table] = records
            continue
        for key, raw, digest in conn.execute(f"SELECT key,payload,sha256 FROM {table} ORDER BY key"):
            body = json.loads(raw)
            if trial._digest(body) != digest:
                raise ValueError(f"Audit {table} checksum mismatch")
            writer = body.pop("_writer")
            if not writer["packages"] or any(len(digest) != 64 for digest in writer["sources"].values()):
                raise ValueError("Missing audit writer provenance")
            records[key] = body
        tables[table] = records
    previous_events = {}
    for key, event in tables["ingestion_events"].items():
        case = _input_case(event["stage"], event["source"], event["signal_at"])
        previous = previous_events.get(case)
        if (event["schema"] != 1 or event["case"] != case
                or event["stage"] not in ("signal", "prices", "outcome", "history", "history_row", "measurement", "comparison", "report")
                or event["sequence"] != (previous["sequence"] + 1 if previous else 1)
                or key != case + f'/{event["sequence"]:08d}'
                or event["state"] not in ("error", "recovered")
                or (event["state"] == "error") != bool(event["error"])
                or (event["state"] == "recovered" and (not previous or previous["state"] != "error"))
                or (previous and trial._utc(event["observed_at"]) < trial._utc(previous["observed_at"]))):
            raise ValueError("Invalid ingestion event sequence")
        previous_events[case] = event
    for slot, record in tables["signals"].items():
        signal = trial._utc(slot)
        first = trial._utc(policy["first_signal_at"])
        if signal < first or (signal - first).total_seconds() % 21600 or record["signal_at"] != slot:
            raise ValueError("Audit signal outside schedule")
        if record["status"] == "recorded":
            witness = record["witness"]
            _frame(witness, "signal", slot, record["observed_at"])
            if not witness["timely_signal"]:
                raise ValueError("Late witness in recorded signals")
        elif record["status"] != "missed" or trial._utc(record["observed_at"]) <= signal + pd.Timedelta(minutes=45):
            raise ValueError("Invalid missed-signal record")
    for slot, result in tables["outcomes"].items():
        source = tables["signals"][slot]
        if source["status"] != "recorded" or result["signal_at"] != slot:
            raise ValueError("Outcome lacks original signal")
        witness = source["witness"]
        if result["witness_sha256"] != trial._digest(witness):
            raise ValueError("Outcome witness binding mismatch")
        prices = pd.DataFrame([{"timestamp": result[time], "market": row["market"], "open": row[column]}
                               for row in result.get("prices", [])
                               for time, column in (("entry_at", "entry_open"), ("exit_at", "exit_open"))],
                              columns=["timestamp", "market", "open"])
        if result.get("prices") or result.get("reason") == "evaluation_deadline":
            expected, _ = evaluate(witness, prices, result["evaluated_at"])
            if expected != result:
                raise ValueError("Saved outcome arithmetic mismatch")
        elif (result["status"] != "invalid" or result.get("reason") not in ("missing_prices", "duplicate_prices", "invalid_prices")
              or result.get("saved_ic") is not None
              or trial._utc(result["evaluated_at"]) < trial._utc(slot) + pd.Timedelta(hours=55)):
            raise ValueError("Invalid terminal price failure")
    for key, result in tables["comparisons"].items():
        slot = result["signal_at"]
        expected = compare(tables["signals"][slot]["witness"], tables["outcomes"][slot],
                           result["legacy_history"], result["measurement"], result["compared_at"])
        if expected != result or key != slot + "/" + trial._digest([result["legacy_history"], result["measurement"]]):
            raise ValueError("Comparison arithmetic or key mismatch")
    for key, result in tables["deliveries"].items():
        slot = result["signal_at"]
        source = tables["signals"].get(slot, {})
        witness_sha = trial._digest(source["witness"]) if source.get("witness") else None
        if (_delivery(result["report"], slot, witness_sha, result["observed_at"]) != result
                or key != slot + "/" + trial._digest(result["report"])):
            raise ValueError("Delivery binding mismatch")
        trace = result["report"].get("selection_trace")
        if trace and trace["status"] == "recorded":
            from utils.selection_trace import verify_selection_trace
            verify_selection_trace(trace, result["report"], source.get("witness") if result["linked"] else None)
    from utils.venue_observer import verify_venue
    verify_venue(tables, policy)
    from utils.execution_audit import verify_execution
    verify_execution(tables, policy)
    return policy, tables


def _summary(policy, tables, now, pending, current_comparisons):
    signals, outcomes = tables["signals"], tables["outcomes"]
    ingestion = _ingestion_summary(tables)
    first, current = trial._utc(policy["first_signal_at"]), trial._utc(now)
    expected = list(pd.date_range(first, current, freq="6h")) if current >= first else []
    rows = []
    for signal in expected:
        slot = signal.isoformat()
        source, outcome = signals.get(slot, {}), outcomes.get(slot, {})
        latest = tables["comparisons"].get(current_comparisons.get(slot))
        deliveries = [r for r in tables["deliveries"].values() if r["signal_at"] == slot]
        complete = any(r["complete"] for r in deliveries)
        problems = [r for r in ingestion["active"] if r["signal_at"] in (None, slot)]
        stages = {r["stage"] for r in problems}
        overdue = current > signal + pd.Timedelta(hours=55)
        legacy_status = ("error" if stages & {"history", "history_row", "comparison"} else
                         "available" if latest and latest["legacy_ic"] is not None else
                         "missing" if overdue else "waiting")
        measurement_status = ("error" if "measurement" in stages else
                              "available" if latest and latest["measurement"] else
                              "missing" if overdue else "waiting")
        rows.append({"signal_at": slot, "status": outcome.get("status") or source.get("status") or
                     ("source_error" if "signal" in stages else "awaiting_signal"),
                     "reason": outcome.get("reason") or pending.get(slot), "saved_ic": outcome.get("saved_ic"),
                     "n_coins": outcome.get("n_coins") or len(source.get("witness", {}).get("rows", [])),
                     "model_sha256": source.get("witness", {}).get("model_sha256"),
                     "legacy_ic": latest["legacy_ic"] if latest else None,
                     "difference": latest["difference"] if latest else None,
                     "comparison_warning": latest["warning"] if latest else False,
                     "measurement_matches_history": latest["measurement_matches_history"] if latest else False,
                     "same_model": latest["same_model"] if latest else None,
                     "same_pool": latest["same_pool"] if latest else None,
                     "measurement_raw_ic": latest["measurement_raw_ic"] if latest else None,
                     "legacy_status": legacy_status, "measurement_status": measurement_status,
                     "comparison_status": "error" if "error" in (legacy_status, measurement_status) else legacy_status,
                     "delivery": "verified" if complete else "waiting" if current <= signal + pd.Timedelta(minutes=45) else "unverified"})
    valid = [row for row in rows if row["status"] == "evaluated"]
    counts = {"scheduled": len(rows), "recorded": sum(r["status"] == "recorded" for r in signals.values()),
              "evaluated": len(valid), "missed": sum(r["status"] == "missed" for r in signals.values()),
              "invalid": sum(r["status"] == "invalid" for r in outcomes.values()),
              "pending": sum(r["status"] in ("recorded", "awaiting_signal", "source_error") for r in rows),
              "compared": sum(r["legacy_ic"] is not None for r in valid),
              "differences": sum(r["comparison_warning"] for r in rows),
              "delivery_verified": sum(r["delivery"] == "verified" for r in rows),
              "source_errors": sum(r["status"] == "source_error" for r in rows),
              "comparison_errors": sum(r["comparison_status"] == "error" for r in valid),
              "legacy_missing": sum(r["legacy_status"] == "missing" for r in valid)}
    issues = []
    closed = [r for r in rows if trial._utc(r["signal_at"]) + pd.Timedelta(minutes=45) < current]
    if closed and closed[-1]["status"] == "missed":
        issues.append("latest_signal_missing")
    if closed and closed[-1]["delivery"] == "unverified":
        issues.append("latest_delivery_unverified")
    finished = [r for r in rows if r["status"] in ("evaluated", "invalid")]
    if finished and finished[-1]["status"] == "invalid":
        issues.append("latest_outcome_invalid")
    if ingestion["active"]:
        issues.append("input_errors_active")
    if counts["legacy_missing"]:
        issues.append("legacy_comparison_missing")
    return {"status": "attention" if issues else "monitoring", "checked_at": current.isoformat(), "policy": policy,
            "counts": counts, "mean_saved_ic": float(np.mean([r["saved_ic"] for r in valid])) if valid else None,
            "first_cycle": rows[0] if rows else None, "recent_windows": list(reversed(rows[-40:])),
            "changes_gate": False, "automatic_promotion": False, "operational_issues": issues,
            "ingestion": ingestion, "errors": []}


def _history_rows(path):
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError("Legacy history must be a list of rows")
    grouped = {}
    for row in rows:
        if row.get("contract_version") == LEGACY_CONTRACT and row.get("side") == "short" and row.get("horizon_h") == 6:
            slot = trial._utc(row["timestamp"]).isoformat()
            grouped.setdefault(slot, []).append(row)
    return grouped


def _legacy_row(rows):
    if len(rows) > 1:
        raise ValueError("Ambiguous legacy IC history")
    if rows and (not np.isfinite(rows[0]["ic"]) or not -1 <= rows[0]["ic"] <= 1):
        raise ValueError("Invalid legacy IC value")
    return rows[0] if rows else None


def _advance_comparisons(conn, tables, root, current):
    selected = {}
    path = root / "output/ic_history.json"
    previously_failed = any(r["stage"] == "history" for r in tables["ingestion_events"].values())
    history_ok, history = _try_input(conn, tables, "history", "output/ic_history.json", None, current,
                                    lambda: _history_rows(path)) if path.exists() or previously_failed else (True, {})
    for slot, outcome in tables["outcomes"].items():
        if outcome["status"] != "evaluated":
            continue
        legacy = None
        if history_ok:
            _, legacy = _try_input(conn, tables, "history_row", "output/ic_history.json", slot, current,
                                   lambda: _legacy_row(history.get(slot, [])))
        _, measured = _try_input(conn, tables, "measurement", "output/score_evidence/measurement", slot, current,
                                 lambda: _witnesses(root, "measurement", slot, current))
        ok, results = _try_input(conn, tables, "comparison", "saved_scores_vs_legacy", slot, current,
                                 lambda: [(slot + "/" + trial._digest([legacy, measurement]),
                                           compare(tables["signals"][slot]["witness"], outcome, legacy, measurement, current))
                                          for measurement in measured or [None] if legacy is not None or measurement is not None])
        if ok:
            for key, result in results:
                _append(conn, "comparisons", key, result)
                selected[slot] = key
    return selected


def _report_delivery(path, tables, current):
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError("Operator report must be an object")
    if not report.get("data_asof"):
        return None
    slot = trial._utc(report["data_asof"]).isoformat()
    source = tables["signals"].get(slot)
    if source is None:
        return None
    ack = report.get("telegram", {}).get("acknowledged_at")
    if trial._utc(report["generated_at"]) > current or (ack and trial._utc(ack) > current):
        return None
    digest = trial._digest(source["witness"]) if source.get("witness") else None
    trace = report.get("selection_trace")
    if trace and trace["status"] == "recorded":
        from utils.selection_trace import verify_selection_trace
        verify_selection_trace(trace, report, source.get("witness") if report.get("score_evidence_sha256") == digest else None)
    return _delivery(report, slot, digest, current)


def _advance_deliveries(conn, tables, root, current):
    complete = {r["signal_at"] for r in tables["deliveries"].values() if r["complete"]}
    days = {trial._utc(slot).strftime("%Y%m%d") for slot in tables["signals"] if slot not in complete}
    paths = {path for day in days for path in (root / "output/operator_reports").glob(day + "*.json")}
    paths.update(root / r["source"] for r in _ingestion_summary(tables)["active"] if r["stage"] == "report")
    for path in sorted(paths):
        ok, result = _try_input(conn, tables, "report", str(path.relative_to(root)), None, current,
                                lambda: _report_delivery(path, tables, current))
        if ok and result:
            _append(conn, "deliveries", result["signal_at"] + "/" + trial._digest(result["report"]), result)


def refresh_audit(root=ROOT, now=None, price_loader=None, quote_loader=None, execution_loader=None):
    root, current = Path(root), trial._utc(now)
    folder = _folder(root)
    if not (folder / "ledger.sqlite").exists():
        return None
    try:
        with run_lock("forecast_audit", lock_dir=str(root / "logs/locks"), timeout_sec=5):
            with closing(_connect(folder / "ledger.sqlite")) as conn, conn:
                policy, tables = _read(conn)
                if _verified_document(folder / "policy.json") != policy:
                    raise ValueError("Audit policy copy mismatch")
                _event_table(conn)
                first = trial._utc(policy["first_signal_at"])
                slots = pd.date_range(first, current, freq="6h") if current >= first else []
                for signal in slots:
                    slot = signal.isoformat()
                    if slot in tables["signals"]:
                        continue
                    ok, candidates = _try_input(conn, tables, "signal", "output/score_evidence/signal", slot, current,
                                                lambda: _witnesses(root, "signal", slot, current))
                    if not ok:
                        continue
                    eligible = [r for r in candidates if r["timely_signal"]]
                    row = {"signal_at": slot, "observed_at": current.isoformat()}
                    if eligible:
                        row.update(status="recorded", witness=eligible[0])
                    elif current > signal + pd.Timedelta(minutes=45):
                        row.update(status="missed", reason="no_timely_signal", late_witnesses=len(candidates))
                    else:
                        continue
                    _append(conn, "signals", slot, row)
                    tables["signals"][slot] = row
                pending = {}
                mature = [r["witness"] for slot, r in tables["signals"].items()
                          if r["status"] == "recorded" and slot not in tables["outcomes"]
                          and trial._utc(slot) + pd.Timedelta(hours=8) <= current
                          <= trial._utc(slot) + pd.Timedelta(hours=55)]
                prices = pd.DataFrame(columns=["timestamp", "market", "open"])
                prices_ok = True
                if mature:
                    prices_ok, prices = _try_input(conn, tables, "prices", "raw_upbit_database", None, current,
                                                  lambda: (price_loader or trial._raw_prices)(
                                                      min(trial._utc(w["signal_at"]) + pd.Timedelta(hours=1) for w in mature),
                                                      max(trial._utc(w["signal_at"]) + pd.Timedelta(hours=7) for w in mature)))
                for slot, source in tables["signals"].items():
                    if source["status"] != "recorded" or slot in tables["outcomes"]:
                        continue
                    if not prices_ok and trial._utc(slot) + pd.Timedelta(hours=8) <= current <= trial._utc(slot) + pd.Timedelta(hours=55):
                        pending[slot] = "price_source_error"
                        continue
                    ok, evaluated = _try_input(conn, tables, "outcome", "raw_upbit_database", slot, current,
                                               lambda: evaluate(source["witness"], prices, trial._utc(now)))
                    if not ok:
                        pending[slot] = "price_source_error"
                        continue
                    result, reason = evaluated
                    if result:
                        _append(conn, "outcomes", slot, result)
                        tables["outcomes"][slot] = result
                    else:
                        pending[slot] = reason
                current_comparisons = _advance_comparisons(conn, tables, root, current)
                _advance_deliveries(conn, tables, root, current)
                policy, tables = _read(conn)
                from utils.venue_observer import advance_venue, venue_summary
                advance_venue(conn, tables, policy, root, now=now, quote_loader=quote_loader)
                policy, tables = _read(conn)
                from utils.execution_audit import advance_execution, execution_summary
                from utils.selection_trace import selection_summary
                advance_execution(conn, tables, policy, now=now, loader=execution_loader)
                policy, tables = _read(conn)
                state = _summary(policy, tables, trial._utc(now), pending, current_comparisons)
                state["venue"] = venue_summary(tables)
                state["execution"] = execution_summary(tables)
                state["selection"] = selection_summary(tables)
            _write(folder / "summary.json", state)
            return state
    except (Exception, SystemExit) as exc:
        _write(folder / "summary.json", {"status": "error", "checked_at": current.isoformat(),
                                         "errors": [f"{type(exc).__name__}: {exc}"], "changes_gate": False})
        raise


def backup_audit(root=ROOT):
    root = Path(root)
    source = _folder(root) / "ledger.sqlite"
    if not source.exists():
        return None
    folder = root / "output/forecast_audit_backups"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    if shutil.disk_usage(folder).free < 2 * source.stat().st_size + 5 * 1024**3:
        raise OSError("Forecast audit backup requires 5 GiB reserve")
    with tempfile.TemporaryDirectory(dir=folder, prefix=".restore-") as temporary:
        snapshot = Path(temporary) / "ledger.sqlite"
        started = time.monotonic()
        def bounded_backup(status, remaining, total):
            if time.monotonic() - started > 20:
                raise TimeoutError("Forecast audit backup exceeded 20 seconds")
        with closing(_connect(source, "ro")) as src, closing(sqlite3.connect(snapshot)) as dst:
            src.backup(dst, pages=128, progress=bounded_backup, sleep=.1)
        with closing(_connect(snapshot, "ro")) as conn:
            policy, tables = _read(conn)
        digest = artifact_sha256(snapshot)
        destination = folder / (digest + ".sqlite")
        if destination.exists():
            if artifact_sha256(destination) != digest:
                raise ValueError("Forecast audit recovery copy changed")
        else:
            snapshot.chmod(0o600)
            _sync(snapshot)
            snapshot.rename(destination)
            _sync(folder)
        restored = Path(temporary) / "restored.sqlite"
        shutil.copy2(destination, restored)
        with closing(_connect(restored, "ro")) as conn:
            if artifact_sha256(restored) != digest or _read(conn) != (policy, tables):
                raise ValueError("Forecast audit restore verification failed")
    return {"status": "verified", "checkpoint": digest, "restore_verified": True,
            "mode": "same_disk", "disk_failure_protected": False,
            "row_counts": {key: len(value) for key, value in tables.items()}}
