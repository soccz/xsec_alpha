"""Append-only prospective rotation feasibility pilot, isolated from live recommendations."""
from contextlib import closing
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

import numpy as np
import pandas as pd

from utils import prospective as trial
from utils.experiment_supervisor import _envelope, _sync, _sync_tree, _verified_document, _write
from utils.model_release import artifact_sha256
from utils.rotation_research import RotationRules, shadow_net_pct, signal_weights, weekly_review
from utils.run_lock import run_lock

ROOT = trial.ROOT
SOURCES = ("utils/rotation_pilot.py", "utils/rotation_research.py", "utils/prospective.py",
           "utils/regime_observer.py", "utils/eval_metrics.py", "scripts/rotation_pilot.py")
TABLES = ("reviews", "observations", "outcomes", "terminal")
RESERVE_BYTES = 5 * 1024**3


def _folder(root):
    return Path(root) / "output/rotation_pilot"


def _connection(path, writable=False):
    return sqlite3.connect(Path(path).resolve().as_uri() + ("?mode=rw" if writable else "?mode=ro"),
                           uri=True, timeout=5)


def _append(conn, table, key, body):
    if table not in TABLES:
        raise ValueError("Unexpected pilot table")
    conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?)", (key, trial._json(body), trial._digest(body)))


def _read(path):
    with closing(_connection(path)) as conn:
        conn.execute("BEGIN")
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Pilot SQLite integrity failure")
        protocol = trial._manifest(conn)
        tables = {}
        for table in TABLES:
            tables[table] = {}
            for key, raw, digest in conn.execute(f"SELECT slot, payload, sha256 FROM {table} ORDER BY slot"):
                body = json.loads(raw)
                if trial._digest(body) != digest:
                    raise ValueError(f"Pilot {table} checksum mismatch")
                tables[table][key] = body
    return protocol, tables


def _observer(root):
    path = Path(root) / "output/regime_observation/ledger.sqlite"
    with closing(_connection(path)) as conn:
        conn.execute("BEGIN")
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Observer SQLite integrity failure")
        protocol = trial._manifest(conn)
        observations = {k: json.loads(v) for k, v in conn.execute("SELECT slot, payload FROM observations ORDER BY slot")}
        outcomes = {k: json.loads(v) for k, v in conn.execute("SELECT slot, payload FROM outcomes ORDER BY slot")}
    if any(artifact_sha256(Path(root) / name) != digest for name, digest in protocol["sources"].items()):
        raise ValueError("Registered observer sources changed")
    return protocol, observations, outcomes


def start_pilot(root=ROOT, now=None):
    root = Path(root)
    target = _folder(root)
    with run_lock("rotation_pilot", lock_dir=str(root / "logs/locks"), timeout_sec=5):
        if target.exists():
            raise FileExistsError("Rotation pilot already registered; no reset or in-place replacement")
        observer, _, _ = _observer(root)
        current = trial._utc(now)
        if current < trial._utc(observer["registered_at"]):
            raise ValueError("Cannot register before the source observer")
        first = current.floor("h") + pd.Timedelta(hours=1)
        while first.hour not in (5, 11, 17, 23):
            first += pd.Timedelta(hours=1)
        protocol = {
            "schema": 1, "experiment_id": current.strftime("rotation_pilot_%Y%m%dT%H%M%SZ"),
            "registered_at": current.isoformat(), "first_signal_at": first.isoformat(), "n_slots": 336,
            "horizon_h": 6, "basket_size": 5, "capture_deadline_minutes": 45, "price_grace_hours": 48,
            "last_signal_at": (first + pd.Timedelta(hours=6 * 335)).isoformat(),
            "hard_review_at": (first + pd.Timedelta(hours=6 * 335 + 7 + 48)).isoformat(),
            "purpose": "feasibility pilot; no confirmatory superiority or automatic promotion",
            "primary_endpoint": "mean paired mixed-policy minus current-model shadow net per 6h window",
            "price_basis": "upbit_raw_hourly_open_proxy_not_bitget_fills",
            "observer_protocol_sha256": trial._digest(observer), "cost_bps": observer["cost_bps"],
            "rules": asdict(RotationRules()), "weight_change_bps": 10., "cost_sensitivity_bps": [0., 10., 20.],
            "min_exercised_windows": 30, "min_exercised_dates": 8,
            "review_rule": "current Monday UTC only, known outcomes strictly before boundary; no backfilled reviews",
            "capture_rule": "persist by signal+45min and before next-open; missed slots never reconstructed",
            "gap_cost_rule": "previous scheduled captured weight; zero if prior slot missed, no implied trade",
            "sources": {name: artifact_sha256(root / name) for name in SOURCES},
            "packages": {name: version(name) for name in ("numpy", "pandas", "scipy")},
            "automatic_live_switching": False, "automatic_promotion": False,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent, prefix=".rotation-register-") as temporary:
            stage = Path(temporary) / "bundle"
            stage.mkdir(mode=0o700)
            for name, digest in protocol["sources"].items():
                dest = stage / "sources" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / name, dest)
                if artifact_sha256(dest) != digest:
                    raise ValueError("Source changed during registration")
            _write(stage / "protocol.json", _envelope(protocol), immutable=True)
            _write(stage / "observer_protocol.json", _envelope(observer), immutable=True)
            with closing(sqlite3.connect(stage / "ledger.sqlite")) as conn, conn:
                conn.execute("CREATE TABLE protocol(payload TEXT NOT NULL, sha256 TEXT NOT NULL)")
                conn.execute("INSERT INTO protocol VALUES (?, ?)", (trial._json(protocol), trial._digest(protocol)))
                for table in TABLES:
                    conn.execute(f"CREATE TABLE {table}(slot TEXT PRIMARY KEY, payload TEXT NOT NULL, sha256 TEXT NOT NULL)")
                for table in ("protocol", *TABLES):
                    for operation in ("UPDATE", "DELETE"):
                        conn.execute(f"CREATE TRIGGER {table}_{operation} BEFORE {operation} ON {table} "
                                     "BEGIN SELECT RAISE(ABORT, 'append-only pilot'); END")
            _sync_tree(stage)
            if trial._utc(now) >= first:
                raise ValueError("Registration crossed first signal; retry with a future slot")
            stage.rename(target)
            _sync(target.parent)
        return protocol


def _snapshot(row, observer):
    signal = trial._utc(row["signal_at"])
    evaluation = row["evaluation"]
    if (row["status"] != "recorded" or signal.hour not in (5, 11, 17, 23) or signal != signal.floor("h")
            or signal < trial._utc(observer["first_signal_at"])
            or not signal <= trial._utc(row["recorded_at"]) <= signal + pd.Timedelta(minutes=45)
            or trial._utc(row["entry_at"]) != signal + pd.Timedelta(hours=1)
            or trial._utc(row["exit_at"]) != signal + pd.Timedelta(hours=7)
            or not row["model_sha256"] or row["model_sha256"] != evaluation["model_sha256"]
            or row["protocol_sha256"] != trial._digest(evaluation)
            or row["observation_protocol_sha256"] != trial._digest(observer)
            or evaluation["cost_bps"] != observer["cost_bps"] or evaluation["price_grace_hours"] != 48):
        raise ValueError("Observer snapshot timing or provenance mismatch")
    members = row["rows"]
    if len(members) < 10 or len({r["market"] for r in members}) != len(members):
        raise ValueError("Observer snapshot has incomplete membership")
    if np.std([r["score"] for r in members], ddof=1) < 0.0005:
        raise ValueError("Observer scores collapsed")
    for score, selected in (("score", "model_selected"), ("baseline_score", "baseline_selected")):
        if not all(np.isfinite(r[score]) and isinstance(r[selected], bool) for r in members):
            raise ValueError("Invalid scores or membership flags")
        expected = {r["market"] for r in sorted(members, key=lambda r: (r[score], r["market"]))[:5]}
        if {r["market"] for r in members if r[selected]} != expected:
            raise ValueError("Observer selection does not match saved scores")
    context = row["regime"]
    if context["regime"] != "unknown":
        if trial._utc(context["data_asof"]) != signal - pd.Timedelta(hours=1):
            raise ValueError("Regime label is not from the previous closed candle")
        values = [context[k] for k in ("btc_ret_7d", "btc_vol_7d", "vol_reference")]
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Invalid regime context")
        label = ("bull" if values[0] >= 0 else "bear") + ("_highvol" if values[1] >= values[2] else "_lowvol")
        if label != context["regime"]:
            raise ValueError("Regime label disagrees with saved context")


def _outcome(snapshot, outcome):
    if outcome["status"] != "matured":
        if (outcome["status"] != "invalid_prices" or trial._utc(outcome["evaluated_at"])
                < trial._utc(snapshot["exit_at"]) + pd.Timedelta(hours=48)):
            raise ValueError("Premature or invalid source outcome")
        return
    prices = pd.DataFrame([{"timestamp": snapshot[key], "market": r["market"], "open": r[field]}
                           for r in outcome["prices"] for key, field in (
                               ("entry_at", "entry_price"), ("exit_at", "exit_price"))])
    expected = trial.evaluate_snapshot(snapshot, prices, snapshot["evaluation"], now=outcome["evaluated_at"])
    if trial._digest(expected) != trial._digest(outcome):
        raise ValueError("Source outcome arithmetic or membership mismatch")


def _training(evidence):
    return [{"status": "matured", "signal_at": s["signal_at"], "recorded_at": s["recorded_at"],
             "exit_at": s["exit_at"], "evaluated_at": o["evaluated_at"],
             "model_sha256": s["model_sha256"], "regime": s["regime"]["regime"],
             "model_net_pct": 100 * o["strategies"]["model_short5"]["net_return"],
             "baseline_net_pct": 100 * o["strategies"]["reversal_short5"]["net_return"]}
            for s, o in evidence]


def _settled(snapshot, outcome, protocol, now):
    if outcome["status"] != "matured":
        return {"status": "invalid_prices", "evaluated_at": now.isoformat(), "source_outcome": outcome}
    model = 100 * outcome["strategies"]["model_short5"]["net_return"]
    baseline = 100 * outcome["strategies"]["reversal_short5"]["net_return"]
    costs = {str(int(bps)): shadow_net_pct(snapshot["weights"], model, baseline,
                                         previous_reversal_weight=snapshot["previous_reversal_weight"], switch_bps=bps)
             for bps in protocol["cost_sensitivity_bps"]}
    mixed = costs[str(int(protocol["weight_change_bps"]))]
    return {"status": "matured", "evaluated_at": now.isoformat(), "source_outcome": outcome,
            "model_net_pct": model, "mixed_net_pct": mixed, "paired_pp": mixed - model,
            "cost_sensitivity_net_pct": costs}


def _decision(protocol, tables, now):
    obs, results = tables["observations"], tables["outcomes"]
    counts = {"recorded": sum(r["status"] == "recorded" for r in obs.values()),
              "missed": sum(r["status"] == "missed" for r in obs.values()),
              "matured": sum(r["status"] == "matured" for r in results.values()),
              "invalid": sum(r["status"] != "matured" for r in results.values())}
    counts["pending"] = counts["recorded"] - len(results)
    exercised = [key for key, r in obs.items() if r["status"] == "recorded" and r["weights"]["reversal"] > 0
                 and results.get(key, {}).get("status") == "matured"]
    counts["exercised"] = len(exercised)
    counts["exercised_dates"] = len({k[:10] for k in exercised})
    terminal = len(obs) == protocol["n_slots"] and len(results) == counts["recorded"]
    terminal = terminal or now >= trial._utc(protocol["hard_review_at"])
    code = "observing"
    if terminal:
        if counts["matured"] != protocol["n_slots"]:
            code = "insufficient_evidence"
        elif counts["exercised"] < protocol["min_exercised_windows"] or counts["exercised_dates"] < protocol["min_exercised_dates"]:
            code = "policy_not_exercised"
        else:
            code = "pilot_complete_not_confirmatory"
    matured = [r for r in results.values() if r["status"] == "matured"]
    stats = {key: float(np.mean([r[key] for r in matured])) if matured else None
             for key in ("model_net_pct", "mixed_net_pct", "paired_pp")}
    values = [r["paired_pp"] for r in matured]
    stats["paired_std_pp"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return {"code": code, "terminal": terminal, "counts": counts, "statistics": stats,
            "confirmatory_evidence": False, "automatic_promotion": False, "actual_fills": False}


def _audit(folder, protocol, tables):
    if _verified_document(folder / "protocol.json") != protocol:
        raise ValueError("Pilot protocol file/ledger mismatch")
    observer = _verified_document(folder / "observer_protocol.json")
    if trial._digest(observer) != protocol["observer_protocol_sha256"]:
        raise ValueError("Pilot observer contract mismatch")
    for name, digest in protocol["sources"].items():
        if artifact_sha256(folder / "sources" / name) != digest:
            raise ValueError("Archived pilot source changed")
    known_snapshots, known_outcomes = {}, {}

    def evidence(snapshot, outcome=None):
        key, digest = snapshot["signal_at"], trial._digest(snapshot)
        if key in known_snapshots and known_snapshots[key] != digest:
            raise ValueError("Previously used source snapshot changed")
        if key not in known_snapshots:
            _snapshot(snapshot, observer)
            known_snapshots[key] = digest
        if outcome is not None:
            digest = trial._digest(outcome)
            if key in known_outcomes and known_outcomes[key] != digest:
                raise ValueError("Previously used source outcome changed")
            if key not in known_outcomes:
                _outcome(snapshot, outcome)
                known_outcomes[key] = digest

    previous = None
    for key, review in tables["reviews"].items():
        recorded = trial._utc(review["recorded_at"])
        if (key != review["state"]["review_at"] or not trial._utc(protocol["registered_at"]) <= recorded
                or not trial._utc(key) <= recorded < trial._utc(key) + pd.Timedelta(days=7)):
            raise ValueError("Backfilled or invalid weekly review")
        for snapshot, outcome in review["evidence"]:
            evidence(snapshot, outcome)
            if (trial._utc(outcome["evaluated_at"]) >= trial._utc(key)
                    or not trial._utc(key) - pd.Timedelta(days=protocol["rules"]["lookback_days"])
                    <= trial._utc(snapshot["signal_at"]) < trial._utc(key)):
                raise ValueError("Weekly evidence outside its knowledge cutoff")
        expected = weekly_review(_training(review["evidence"]), key, previous, RotationRules(**protocol["rules"]))
        if trial._digest(expected) != trial._digest(review["state"]):
            raise ValueError("Weekly decision differs from stored training evidence")
        previous = review["state"]
    allowed = {slot.isoformat() for slot in trial._slots(protocol)}
    for key, snapshot in tables["observations"].items():
        if key not in allowed or snapshot["signal_at"] != key:
            raise ValueError("Pilot observation outside schedule")
        signal, recorded = trial._utc(key), trial._utc(snapshot["recorded_at"])
        if snapshot["status"] == "missed":
            if recorded <= signal + pd.Timedelta(minutes=45):
                raise ValueError("Premature missed slot")
            continue
        if snapshot["status"] != "recorded" or not signal <= recorded <= signal + pd.Timedelta(minutes=45):
            raise ValueError("Pilot capture after deadline")
        source = snapshot["source_snapshot"]
        evidence(source)
        if source["signal_at"] != key or trial._utc(source["recorded_at"]) > recorded:
            raise ValueError("Pilot consumed future source forecast")
        review = tables["reviews"].get(snapshot["review_key"])
        if snapshot["review_key"] is not None and (review is None or trial._utc(review["recorded_at"]) > recorded):
            raise ValueError("Pilot consumed future weekly decision")
        weights = signal_weights(review["state"] if review else None, source["regime"]["regime"], key)
        if weights != snapshot["weights"]:
            raise ValueError("Recorded weights differ from prior decision")
        prior = tables["observations"].get((signal - pd.Timedelta(hours=6)).isoformat(), {})
        prior_weight = prior.get("weights", {}).get("reversal", 0.)
        if snapshot["previous_reversal_weight"] != prior_weight:
            raise ValueError("Previous scheduled weight mismatch")
    for key, outcome in tables["outcomes"].items():
        snapshot = tables["observations"].get(key)
        if snapshot is None or snapshot["status"] != "recorded":
            raise ValueError("Orphan pilot outcome")
        at = trial._utc(outcome["evaluated_at"])
        if outcome["status"] == "missing_source_outcome":
            if at < trial._utc(key) + pd.Timedelta(hours=55):
                raise ValueError("Premature missing outcome")
            continue
        source_result = outcome["source_outcome"]
        evidence(snapshot["source_snapshot"], source_result)
        if trial._utc(source_result["evaluated_at"]) > at:
            raise ValueError("Pilot consumed future source outcome")
        if trial._digest(_settled(snapshot, source_result, protocol, at)) != trial._digest(outcome):
            raise ValueError("Pilot outcome cost or return mismatch")
    if tables["terminal"]:
        if list(tables["terminal"]) != ["final"]:
            raise ValueError("Unexpected terminal records")
        final = tables["terminal"]["final"]
        if final["decision"] != _decision(protocol, tables, trial._utc(final["generated_at"])):
            raise ValueError("Final pilot decision changed")
    return observer, known_snapshots, known_outcomes


def refresh_pilot(root=ROOT, now=None):
    root, folder = Path(root), _folder(root)
    if not folder.exists():
        return {"status": "not_started"}
    with run_lock("rotation_pilot", lock_dir=str(root / "logs/locks"), timeout_sec=5):
        try:
            if (folder / "integrity_failure.json").exists():
                _verified_document(folder / "integrity_failure.json")
                raise ValueError("Pilot stopped on integrity failure; no automatic restart")
            return _refresh(root, now)
        except Exception as exc:
            integrity = isinstance(exc, (ValueError, KeyError, TypeError, sqlite3.IntegrityError))
            integrity = integrity or (isinstance(exc, sqlite3.DatabaseError) and not isinstance(exc, sqlite3.OperationalError))
            failure = {"status": "integrity_failure" if integrity else "attention",
                       "checked_at": trial._utc(now).isoformat(), "error_type": type(exc).__name__,
                       "automatic_live_switching": False, "confirmatory_evidence": False}
            if integrity:
                target = folder / "integrity_failure.json"
                if not target.exists():
                    _write(target, _envelope({"generated_at": failure["checked_at"], "decision": {
                        "code": "integrity_failure", "terminal": True, "confirmatory_evidence": False,
                        "automatic_promotion": False}, "error": str(exc)}), immutable=True)
                failure["decision"] = _verified_document(target)["decision"]
            _write(folder / "summary.json", failure)
            raise


def _refresh(root, now):
    folder = _folder(root)
    protocol, tables = _read(folder / "ledger.sqlite")
    current = trial._utc(now)
    for table, rows in tables.items():
        stamp = "generated_at" if table == "terminal" else "evaluated_at" if table == "outcomes" else "recorded_at"
        if any(trial._utc(row[stamp]) > current for row in rows.values()):
            raise ValueError("Pilot contains records from the future")
    observer, known_s, known_o = _audit(folder, protocol, tables)
    if not tables["terminal"]:
        if (any(artifact_sha256(root / name) != digest for name, digest in protocol["sources"].items())
                or any(version(name) != v for name, v in protocol["packages"].items())):
            raise ValueError("Pilot runtime changed; register a separately reviewed version")
        source_p, sources, outcomes = _observer(root)
        if source_p != observer:
            raise ValueError("Observer protocol changed after pilot registration")
        if any(key not in sources or trial._digest(sources[key]) != value for key, value in known_s.items()):
            raise ValueError("Consumed source snapshot changed or disappeared")
        if any(key not in outcomes or trial._digest(outcomes[key]) != value for key, value in known_o.items()):
            raise ValueError("Consumed source outcome changed or disappeared")
        with closing(_connection(folder / "ledger.sqlite", writable=True)) as conn, conn:
            review_at = current.normalize() - pd.Timedelta(days=current.weekday())
            review_key = review_at.isoformat()
            if (review_at > trial._utc(protocol["registered_at"]) and review_key not in tables["reviews"]
                    and current <= trial._utc(protocol["last_signal_at"]) + pd.Timedelta(minutes=45)):
                training = []
                for key, result in outcomes.items():
                    if (result["status"] == "matured" and trial._utc(result["evaluated_at"]) < review_at
                            and review_at - pd.Timedelta(days=protocol["rules"]["lookback_days"]) <= trial._utc(key) < review_at):
                        source = sources[key]
                        _snapshot(source, observer)
                        _outcome(source, result)
                        training.append((source, result))
                previous = next(reversed(tables["reviews"].values()))["state"] if tables["reviews"] else None
                state = weekly_review(_training(training), review_at, previous, RotationRules(**protocol["rules"]))
                reviewed = trial._utc(now)
                if reviewed >= review_at + pd.Timedelta(days=7):
                    raise TimeoutError("Weekly review crossed the next boundary")
                review = {"recorded_at": reviewed.isoformat(), "state": state, "evidence": training}
                _append(conn, "reviews", review_key, review)
                tables["reviews"][review_key] = review
            for slot in trial._slots(protocol):
                key = slot.isoformat()
                recorded = trial._utc(now)
                if key in tables["observations"] or slot > recorded:
                    continue
                row = {"signal_at": key, "recorded_at": recorded.isoformat()}
                if recorded > slot + pd.Timedelta(minutes=45):
                    row.update(status="missed", reason="pilot_not_captured_before_deadline")
                else:
                    source = sources.get(key)
                    if not source or source["status"] != "recorded":
                        continue
                    _snapshot(source, observer)
                    if trial._utc(source["recorded_at"]) > recorded:
                        raise ValueError("Source forecast timestamp is in the future")
                    reviews = [(k, r) for k, r in tables["reviews"].items()
                               if trial._utc(k) <= slot and trial._utc(r["recorded_at"]) <= recorded]
                    key_review, review = reviews[-1] if reviews else (None, None)
                    weights = signal_weights(review["state"] if review else None, source["regime"]["regime"], slot)
                    prior = tables["observations"].get((slot - pd.Timedelta(hours=6)).isoformat(), {})
                    recorded = trial._utc(now)
                    if recorded > slot + pd.Timedelta(minutes=45):
                        continue
                    row.update(status="recorded", recorded_at=recorded.isoformat(), source_snapshot=source,
                               review_key=key_review, weights=weights,
                               previous_reversal_weight=prior.get("weights", {}).get("reversal", 0.))
                _append(conn, "observations", key, row)
                tables["observations"][key] = row
            for key, snapshot in tables["observations"].items():
                if snapshot["status"] != "recorded" or key in tables["outcomes"]:
                    continue
                result = outcomes.get(key)
                at = trial._utc(now)
                if result and trial._utc(result["evaluated_at"]) <= at:
                    _outcome(snapshot["source_snapshot"], result)
                    outcome = _settled(snapshot, result, protocol, at)
                elif at >= trial._utc(key) + pd.Timedelta(hours=55):
                    outcome = {"status": "missing_source_outcome", "evaluated_at": at.isoformat()}
                else:
                    continue
                _append(conn, "outcomes", key, outcome)
                tables["outcomes"][key] = outcome
            decision = _decision(protocol, tables, trial._utc(now))
            if decision["terminal"]:
                final = {"generated_at": trial._utc(now).isoformat(), "protocol_sha256": trial._digest(protocol),
                         "decision": decision}
                _append(conn, "terminal", "final", final)
                tables["terminal"]["final"] = final
    _audit(folder, protocol, tables)
    final = tables["terminal"].get("final")
    if final:
        final_path = folder / "final_review.json"
        if final_path.exists():
            if _verified_document(final_path) != final:
                raise ValueError("Final pilot report changed")
        else:
            _write(final_path, _envelope(final), immutable=True)
    decision = final["decision"] if final else _decision(protocol, tables, trial._utc(now))
    recorded = [r for r in tables["observations"].values() if r["status"] == "recorded"]
    latest = recorded[-1] if recorded else None
    summary = {"status": decision["code"], "checked_at": trial._utc(now).isoformat(),
               "protocol": protocol, "protocol_sha256": trial._digest(protocol),
               "decision": decision, "n_reviews": len(tables["reviews"]),
               "latest": {"signal_at": latest["signal_at"], "recorded_at": latest["recorded_at"],
                          "weights": latest["weights"], "regime": latest["source_snapshot"]["regime"]["regime"],
                          "model_sha256": latest["source_snapshot"]["model_sha256"]} if latest else None,
               "automatic_live_switching": False, "confirmatory_evidence": False}
    valid = [r for r in tables["outcomes"].values() if r["status"] == "matured"]
    summary["cost_sensitivity"] = [
        {"weight_change_bps": bps, "mean_mixed_net_pct": float(np.mean([
            r["cost_sensitivity_net_pct"][str(int(bps))] for r in valid])) if valid else None}
        for bps in protocol["cost_sensitivity_bps"]]
    summary["recent_windows"] = [
        {"signal_at": key, "recorded_at": row["recorded_at"],
         "status": tables["outcomes"].get(key, {}).get("status", row["status"]),
         "reversal_weight": row.get("weights", {}).get("reversal"),
         "paired_pp": tables["outcomes"].get(key, {}).get("paired_pp")}
        for key, row in list(tables["observations"].items())[-12:]]
    _write(folder / "summary.json", summary)
    return summary


def verify_backup(folder):
    folder = Path(folder)
    protocol, tables = _read(folder / "ledger.sqlite")
    _audit(folder, protocol, tables)
    final = tables["terminal"].get("final")
    if final and _verified_document(folder / "final_review.json") != final:
        raise ValueError("Backup lacks the final pilot report")
    if (folder / "integrity_failure.json").exists():
        _verified_document(folder / "integrity_failure.json")
    return {"protocol_sha256": trial._digest(protocol),
            "row_counts": {key: len(rows) for key, rows in tables.items()}}


def backup_pilot(root=ROOT):
    root = Path(root)
    folder = _folder(root)
    if not folder.exists():
        return None
    primary = root / "output/rotation_backups"
    primary.mkdir(parents=True, exist_ok=True, mode=0o700)
    with run_lock("rotation_pilot", lock_dir=str(root / "logs/locks"), timeout_sec=5):
        needed = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        if shutil.disk_usage(primary).free < 3 * needed + RESERVE_BYTES:
            raise OSError("Local pilot backup requires 5 GiB reserve")
        with tempfile.TemporaryDirectory(dir=primary, prefix=".staging-") as temporary:
            stage = Path(temporary) / "bundle"
            stage.mkdir(mode=0o700)
            started = time.monotonic()
            def bounded(status, remaining, total):
                if time.monotonic() - started > 20:
                    raise TimeoutError("Pilot SQLite backup exceeded 20 seconds")
            with closing(_connection(folder / "ledger.sqlite")) as src, closing(sqlite3.connect(stage / "ledger.sqlite")) as dst:
                src.backup(dst, pages=128, progress=bounded, sleep=0.1)
            shutil.copytree(folder / "sources", stage / "sources")
            for name in ("protocol.json", "observer_protocol.json", "final_review.json", "integrity_failure.json"):
                if (folder / name).exists():
                    shutil.copy2(folder / name, stage / name)
            details = verify_backup(stage)
            files = {str(p.relative_to(stage)): artifact_sha256(p) for p in stage.rglob("*") if p.is_file()}
            checksum = trial._digest(files)
            target = primary / checksum
            _write(stage / "manifest.json", _envelope(files), immutable=True)
            _sync_tree(stage)
            if target.exists():
                saved = _verified_document(target / "manifest.json")
                if saved != files or any(artifact_sha256(target / name) != digest for name, digest in saved.items()):
                    raise ValueError("Existing pilot backup changed")
            else:
                stage.rename(target)
                _sync(primary)
            restored = Path(temporary) / "restored"
            shutil.copytree(target, restored)
            if verify_backup(restored) != details or any(artifact_sha256(restored / name) != digest for name, digest in files.items()):
                raise ValueError("Pilot restore verification failed")
    return {"status": "verified", "mode": "same_disk", "disk_failure_protected": False,
            "restore_verified": True, "checkpoint": checksum, "primary": str(target), **details}
