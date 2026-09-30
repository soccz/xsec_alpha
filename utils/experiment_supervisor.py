"""Supervise the frozen trial without changing its registered scoring contract."""
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

import numpy as np
import pandas as pd

from utils import prospective as trial
from utils.model_release import artifact_sha256
from utils.run_lock import run_lock

ROOT = trial.ROOT
SECONDARY = Path.home() / "xsec_prospective_offdisk"
RESERVE_BYTES = 5 * 1024**3


def _folder(root):
    return Path(root) / "output/experiment_supervision"


def _load(path):
    return json.loads(Path(path).read_text())


def _write(path, document, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(trial._json(document))
            handle.flush()
            os.fsync(handle.fileno())
            if immutable:
                os.link(temporary, path)
            else:
                os.replace(temporary, path)
            _sync(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _sync(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sync_tree(path):
    paths = list(Path(path).rglob("*"))
    for item in sorted(paths, key=lambda p: len(p.parts), reverse=True):
        _sync(item)
    _sync(path)


def _envelope(body):
    return {"document": body, "sha256": trial._digest(body)}


def _verified_document(path):
    envelope = _load(path)
    body = envelope["document"]
    if trial._digest(body) != envelope["sha256"]:
        raise ValueError(f"Document checksum mismatch: {Path(path).name}")
    return body


def register_review(root=ROOT, now=None):
    """Declare endpoint interpretation before any first-slot evidence exists."""
    with run_lock("experiment_supervisor", lock_dir=str(Path(root) / "logs/locks")):
        target = _folder(root) / "review_plan.json"
        if target.exists():
            raise FileExistsError("Review plan already registered; refusing to overwrite")
        summary = trial.experiment_summary(root)
        protocol = summary["protocol"]
        current = trial._utc(now)
        if current >= trial._utc(protocol["first_signal_at"]) or summary["n_recorded"]:
            raise ValueError("Cannot preregister this review after the first slot or observations")
        if not (summary["runtime_matches"] and summary["frozen_model_matches"]):
            raise ValueError("Trial integrity must pass before review registration")
        last = trial._slots(protocol)[-1]
        document = {
            "schema": 1, "experiment_id": protocol["experiment_id"],
            "registered_at": current.isoformat(), "protocol_sha256": trial._digest(protocol),
            "scheduled_windows": protocol["n_slots"],
            "first_signal_at": protocol["first_signal_at"], "last_signal_at": last.isoformat(),
            "earliest_final_at": (last + pd.Timedelta(hours=8)).isoformat(),
            "hard_review_at": (last + pd.Timedelta(hours=7 + protocol["price_grace_hours"])).isoformat(),
            "relative_edge": "all 60 valid paired windows and registered CI95 lower > 0",
            "baseline_better": "all 60 valid paired windows and registered CI95 upper < 0",
            "inconclusive": "all 60 valid windows but CI includes zero; do not extend this trial",
            "insufficient_evidence": "missing/invalid windows or unresolved evidence at hard deadline",
            "profitability": "mean model net is descriptive only; executable profitability remains untested",
            "next_relative_edge": "preregister Bitget price/time alignment and live gate/buffer shadow comparison",
            "next_baseline_better": "audit ranking failure; retain baseline as research control, no automatic live switch",
            "next_inconclusive": "design a separately registered, adequately powered follow-up; no repeated significance peeking",
            "next_insufficient": "repair evidence collection before changing models",
            "automatic_promotion": False,
        }
        _write(target, _envelope(document), immutable=True)
        _sync(target.parent)
        return document


def decide(summary, plan, now=None):
    """A terminal classification need not be evidence of an investment edge."""
    current = trial._utc(now)
    deadline = current >= trial._utc(plan["hard_review_at"])
    intact = summary.get("runtime_matches") is True and summary.get("frozen_model_matches") is True
    terminal = (current >= trial._utc(plan["earliest_final_at"])
                and summary.get("status") in ("ready_for_review", "incomplete_evidence")) or deadline
    base = {"terminal": terminal, "automatic_promotion": False,
            "executable_profitability": "not_tested"}
    if not intact or summary.get("status") == "error":
        return {**base, "code": "integrity_failure" if terminal else "integrity_blocked",
                "next_step": "repair evidence integrity; do not interpret performance"}
    if not terminal:
        return {**base, "code": "collecting", "next_step": "complete the fixed schedule without tuning"}
    if summary.get("n_matured") != plan["scheduled_windows"] or summary.get("n_missed") or summary.get("n_invalid"):
        return {**base, "code": "insufficient_evidence", "next_step": plan["next_insufficient"]}
    interval = summary.get("paired_ci95_pct")
    if (not isinstance(interval, list) or len(interval) != 2
            or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in interval)
            or interval[0] > interval[1]):
        raise ValueError("Final paired interval unavailable or invalid")
    code = "relative_edge" if interval[0] > 0 else "baseline_better" if interval[1] < 0 else "inconclusive"
    model_mean = next((s["mean_net_pct"] for s in summary["strategies"] if s["name"] == "model_short5"), None)
    return {**base, "code": code, "paired_ci95_pct": interval,
            "paired_excess_pct": summary.get("paired_excess_pct"),
            "model_mean_net_pct": model_mean,
            "paper_mean": "positive_only" if model_mean is not None and model_mean > 0 else "not_positive",
            "next_step": plan[f"next_{code}"]}


def _read_ledger(path):
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)) as conn:
        conn.execute("BEGIN")
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
        protocol = trial._manifest(conn)
        observations = list(conn.execute("SELECT slot, payload FROM observations ORDER BY slot"))
        outcomes = list(conn.execute("SELECT slot, payload FROM outcomes ORDER BY slot"))
    _audit_ledger(protocol, observations, outcomes)
    return protocol, observations, outcomes


def _audit_ledger(protocol, observations, outcomes):
    """Check saved timing, membership and return arithmetic, never regenerate scores."""
    allowed = {s.isoformat() for s in trial._slots(protocol)}
    snapshots = {}
    for key, raw in observations:
        row = json.loads(raw)
        if key not in allowed or row["signal_at"] != key:
            raise ValueError("Observation outside registered schedule")
        if row["status"] == "missed":
            continue
        if row["status"] != "recorded" or row["protocol_sha256"] != trial._digest(protocol):
            raise ValueError("Observation protocol mismatch")
        signal, recorded, entry, exit_at = [trial._utc(row[name]) for name in (
            "signal_at", "recorded_at", "entry_at", "exit_at")]
        if (not signal <= recorded <= signal + pd.Timedelta(minutes=45)
                or entry != signal + pd.Timedelta(hours=1)
                or exit_at != entry + pd.Timedelta(hours=6)
                or row["model_sha256"] != protocol["model_sha256"]):
            raise ValueError("Observation timing or model mismatch")
        members = row["rows"]
        markets = [m["market"] for m in members]
        if len(markets) < 10 or len(set(markets)) != len(markets) or row["universe_n"] != len(markets):
            raise ValueError("Observation universe is incomplete")
        for field in ("model_selected", "baseline_selected"):
            if sum(m[field] is True for m in members) != protocol["basket_size"]:
                raise ValueError("Observation basket size mismatch")
        snapshots[key] = row
    costs = protocol["cost_bps"]
    cost = (2 * (costs["one_way_fee"] + costs["one_way_slippage"]) + costs["short_extra"]) / 10000
    for key, raw in outcomes:
        if key not in snapshots:
            raise ValueError("Outcome has no recorded forecast")
        result, snapshot = json.loads(raw), snapshots[key]
        exit_at = trial._utc(snapshot["exit_at"])
        if result["status"] == "invalid_prices":
            if trial._utc(result["evaluated_at"]) < exit_at + pd.Timedelta(hours=protocol["price_grace_hours"]):
                raise ValueError("Price invalidation before grace deadline")
            continue
        if result["status"] != "matured" or trial._utc(result["evaluated_at"]) < exit_at + pd.Timedelta(hours=1):
            raise ValueError("Outcome matured before its horizon")
        expected = {r["market"] for r in snapshot["rows"]}
        actual = {r["market"]: r for r in result["prices"]}
        if set(actual) != expected or len(result["prices"]) != len(expected):
            raise ValueError("Outcome changed frozen universe membership")
        for prices in actual.values():
            pair = [prices["entry_price"], prices["exit_price"]]
            if not all(np.isfinite(v) and v > 0 for v in pair):
                raise ValueError("Outcome contains invalid prices")
            raw_return = pair[1] / pair[0] - 1
            if not np.isfinite(raw_return) or not np.isclose(raw_return, prices["raw_return"], rtol=1e-8, atol=1e-10):
                raise ValueError("Outcome raw return mismatch")
        for name, field in (("model_short5", "model_selected"), ("reversal_short5", "baseline_selected"),
                            ("universe_short", None)):
            selected = [r["market"] for r in snapshot["rows"] if field is None or r[field]]
            gross = -float(np.mean([actual[m]["raw_return"] for m in selected]))
            saved = result["strategies"][name]
            if not all(np.isclose(saved[k], v, rtol=1e-8, atol=1e-10) for k, v in (
                    ("gross_return", gross), ("cost_return", cost), ("net_return", gross - cost))):
                raise ValueError("Outcome basket return or cost mismatch")


def verify_checkpoint(path):
    """Verify an independently readable checkpoint, not just that a copy exists."""
    path = Path(path)
    manifest = _verified_document(path / "manifest.json")
    for name, expected in manifest["files"].items():
        target = path / name
        target.resolve().relative_to(path.resolve())
        if target.is_symlink() or artifact_sha256(target) != expected:
            raise ValueError(f"Checkpoint file mismatch: {name}")
    protocol, observations, outcomes = _read_ledger(path / "output/prospective/ledger.sqlite")
    if trial._digest(protocol) != manifest["protocol_sha256"]:
        raise ValueError("Checkpoint protocol mismatch")
    if artifact_sha256(path / "output/prospective/model.pkl") != protocol["model_sha256"]:
        raise ValueError("Checkpoint model mismatch")
    for name, expected in protocol["runtime"]["sources"].items():
        if artifact_sha256(path / name) != expected:
            raise ValueError(f"Registered source changed: {name}")
    if [len(observations), len(outcomes)] != manifest["row_counts"]:
        raise ValueError("Checkpoint row count mismatch")
    return manifest


def _sealed_archive(root):
    """Validate the completed experiment independently of later live-code edits."""
    root = Path(root)
    seal = _verified_document(_folder(root) / "completed_archive.json")
    archive = root / "output/prospective_backups" / seal["checkpoint"]
    archive.resolve().relative_to((root / "output/prospective_backups").resolve())
    manifest = verify_checkpoint(archive)
    if trial._digest(manifest) != seal["checkpoint"]:
        raise ValueError("Completed archive identity mismatch")
    if trial._digest(_read_ledger(root / "output/prospective/ledger.sqlite")) != seal["ledger_sha256"]:
        raise ValueError("Completed ledger changed after sealing")
    for name in ("output/prospective/model.pkl", "output/experiment_supervision/final_review.json",
                 "output/experiment_supervision/review_plan.json"):
        if artifact_sha256(root / name) != manifest["files"][name]:
            raise ValueError(f"Completed evidence changed: {name}")
    return seal, archive


def review_summary(root=ROOT):
    if not (_folder(root) / "completed_archive.json").exists():
        return trial.experiment_summary(root)
    seal, _ = _sealed_archive(root)
    summary = seal["summary"]
    summary["integrity_basis"] = "sealed_completed_archive"
    summary["live_runtime_matches"] = summary["protocol"]["runtime"] == trial._runtime_contract(root)
    return summary


def seal_completed(root=ROOT):
    """Seal once, before changing registered source; never rewrite the final verdict."""
    root = Path(root)
    with run_lock("experiment_supervisor", lock_dir=str(root / "logs/locks")):
        path = _folder(root) / "completed_archive.json"
        if path.exists():
            return _sealed_archive(root)[0]
        final = _verified_document(_folder(root) / "final_review.json")
        summary = trial.experiment_summary(root)
        if not (final["decision"]["terminal"] and summary["runtime_matches"] and summary["frozen_model_matches"]):
            raise ValueError("Cannot seal an unfinished or changed experiment")
        target, manifest = _primary_checkpoint(root)
        plan = _verified_document(_folder(root) / "review_plan.json")
        if (final["review_plan_sha256"] != trial._digest(plan)
                or plan["protocol_sha256"] != manifest["protocol_sha256"]):
            raise ValueError("Final review protocol mismatch")
        document = {"sealed_at": trial._utc().isoformat(), "checkpoint": target.name,
                    "ledger_sha256": trial._digest(_read_ledger(root / "output/prospective/ledger.sqlite")),
                    "summary": summary}
        _write(path, _envelope(document), immutable=True)
        _sealed_archive(root)
        return document


def _primary_checkpoint(root):
    root = Path(root)
    if (_folder(root) / "completed_archive.json").exists():
        _, target = _sealed_archive(root)
        return target, verify_checkpoint(target)
    source_dir = root / "output/prospective"
    plan_path = _folder(root) / "review_plan.json"
    plan = _verified_document(plan_path)
    primary = root / "output/prospective_backups"
    primary.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(dir=primary, prefix=".staging-") as temp:
        stage = Path(temp) / "checkpoint"
        db_target = stage / "output/prospective/ledger.sqlite"
        db_target.parent.mkdir(parents=True, mode=0o700)
        started = time.monotonic()
        def bounded_backup(status, remaining, total):
            if time.monotonic() - started > 20:
                raise TimeoutError("Experiment backup exceeded 20 seconds")
        with closing(sqlite3.connect((source_dir / "ledger.sqlite").resolve().as_uri() + "?mode=ro", uri=True)) as src:
            with closing(sqlite3.connect(db_target)) as dst:
                src.backup(dst, pages=128, progress=bounded_backup, sleep=0.1)
        protocol, observations, outcomes = _read_ledger(db_target)
        if trial._digest(protocol) != plan["protocol_sha256"]:
            raise ValueError("Review plan is bound to a different protocol")
        names = ["output/prospective/model.pkl", *protocol["runtime"]["sources"],
                 "output/experiment_supervision/review_plan.json"]
        review_path = _folder(root) / "final_review.json"
        if review_path.exists():
            names.append("output/experiment_supervision/final_review.json")
        for name in names:
            target = stage / name
            target.resolve().relative_to(stage.resolve())
            (root / name).resolve().relative_to(root.resolve())
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copy2(root / name, target)
        all_names = ["output/prospective/ledger.sqlite", *names]
        files = {name: artifact_sha256(stage / name) for name in all_names}
        manifest = {"schema": 1, "experiment_id": protocol["experiment_id"],
                    "protocol_sha256": trial._digest(protocol), "files": files,
                    "row_counts": [len(observations), len(outcomes)]}
        key = trial._digest(manifest)
        _write(stage / "manifest.json", _envelope(manifest))
        verify_checkpoint(stage)
        for name in all_names + ["manifest.json"]:
            os.chmod(stage / name, 0o600)
        _sync_tree(stage)
        target = primary / key
        if target.exists():
            verify_checkpoint(target)
        else:
            stage.rename(target)
            _sync(primary)
    return target, manifest


def backup_checkpoint(root=ROOT, secondary=SECONDARY, require_separate=True):
    """SQLite online backup plus immutable model/code copies on two filesystems."""
    root, secondary = Path(root), Path(secondary)
    target, manifest = _primary_checkpoint(root)
    key, files = target.name, manifest["files"]
    # Keep the on-disk snapshot even if the second filesystem is temporarily unavailable.
    secondary.mkdir(parents=True, exist_ok=True, mode=0o700)
    separate = (root / "output/prospective").stat().st_dev != secondary.stat().st_dev
    if require_separate and not separate:
        raise ValueError("Secondary checkpoint must be on a different filesystem")
    offdisk = secondary / key
    if not offdisk.exists():
        size = sum((target / name).stat().st_size for name in files)
        if shutil.disk_usage(secondary).free < size + RESERVE_BYTES:
            raise OSError("Secondary backup requires at least 5 GiB reserve")
        with tempfile.TemporaryDirectory(dir=secondary, prefix=".staging-") as temp:
            copied = Path(temp) / "checkpoint"
            shutil.copytree(target, copied)
            verify_checkpoint(copied)
            _sync_tree(copied)
            copied.rename(offdisk)
            _sync(secondary)
    verify_checkpoint(offdisk)
    # Restore the whole bundle into a temporary location, never over production.
    with tempfile.TemporaryDirectory(dir=root / "output/prospective_backups", prefix=".restore-") as temp:
        restored = Path(temp) / "checkpoint"
        shutil.copytree(offdisk, restored)
        if verify_checkpoint(restored) != manifest:
            raise ValueError("Restore drill failed")
    return {"status": "verified", "checkpoint": key, "primary": str(target),
            "secondary": str(offdisk), "separate_filesystem": separate,
            "restore_verified": True, "row_counts": manifest["row_counts"]}


def _milestones(summary, plan, now):
    windows = summary.get("recent_windows", [])
    first = next((w for w in windows if trial._utc(w["signal_at"]) == trial._utc(plan["first_signal_at"])), None)
    elapsed = now > trial._utc(plan["first_signal_at"]) + pd.Timedelta(minutes=45)
    return {"first_snapshot": "recorded" if first and first["status"] != "missed" else "missing" if elapsed else "waiting",
            "first_outcome": first["status"] if first else "waiting",
            "recorded": summary.get("n_recorded"), "matured": summary.get("n_matured"),
            "missed": summary.get("n_missed"), "invalid": summary.get("n_invalid"),
            "pending": summary.get("n_pending"), "target": plan["scheduled_windows"]}


def supervise(root=ROOT, secondary=SECONDARY, now=None, publisher=None, require_separate=True):
    """One bounded, restartable pass. Only final evidence is classified as terminal."""
    root = Path(root)
    current = trial._utc(now)
    folder = _folder(root)
    with run_lock("experiment_supervisor", lock_dir=str(root / "logs/locks"), timeout_sec=1, exit_code=75):
        plan = _verified_document(folder / "review_plan.json")
        errors = []
        try:
            from utils.regime_observer import refresh_observation
            refresh_observation(root=root, now=current)
        except Exception as exc:
            errors.append(f"regime_observation: {type(exc).__name__}: {exc}")
        final_path = folder / "final_review.json"
        if not final_path.exists():
            try:
                with run_lock("prospective", lock_dir=str(root / "logs/locks"), timeout_sec=1, exit_code=75):
                    trial.advance_experiment(root=root, now=current)
            except (Exception, SystemExit) as exc:
                errors.append(f"settlement: {type(exc).__name__}: {exc}")
        try:
            _read_ledger(root / "output/prospective/ledger.sqlite")
            summary = review_summary(root)
            if trial._digest(summary["protocol"]) != plan["protocol_sha256"]:
                raise ValueError("Live protocol changed since review registration")
            if not (summary["runtime_matches"] and summary["frozen_model_matches"]):
                errors.append("integrity: registered runtime or frozen model no longer matches")
        except Exception as exc:
            errors.append(f"evidence: {type(exc).__name__}: {exc}")
            summary = {"status": "error"}
        decision = decide(summary, plan, now=current)
        backup = None
        try:
            backup = backup_checkpoint(root, secondary, require_separate=require_separate)
        except Exception as exc:
            errors.append(f"backup: {type(exc).__name__}: {exc}")
        if final_path.exists():
            final = _verified_document(final_path)
            if final["review_plan_sha256"] != trial._digest(plan):
                raise ValueError("Final report belongs to a different review plan")
            decision = final["decision"]
        elif decision["terminal"]:
            final = {"generated_at": current.isoformat(), "experiment_id": plan["experiment_id"],
                     "review_plan_sha256": trial._digest(plan), "decision": decision,
                     "evidence": {key: summary.get(key) for key in (
                         "n_recorded", "n_matured", "n_missed", "n_invalid", "n_pending",
                         "paired_excess_pct", "paired_ci95_pct", "model_ic_mean", "baseline_ic_mean", "strategies")},
                     "checkpoint": backup["checkpoint"] if backup else None,
                     "errors_at_review": errors.copy()}
            _write(final_path, _envelope(final), immutable=True)
            # The terminal report itself must be recoverable along with its evidence.
            try:
                backup = backup_checkpoint(root, secondary, require_separate=require_separate)
            except Exception as exc:
                errors.append(f"final_backup: {type(exc).__name__}: {exc}")
        state = {"checked_at": current.isoformat(), "experiment_id": plan["experiment_id"],
                 "status": "attention" if errors else "complete" if decision["terminal"] else "monitoring",
                 "plan": plan, "milestones": _milestones(summary, plan, current),
                 "runtime_matches": summary.get("runtime_matches"),
                 "integrity_basis": summary.get("integrity_basis", "registered_live_runtime"),
                 "live_runtime_matches": summary.get("live_runtime_matches", summary.get("runtime_matches")),
                 "frozen_model_matches": summary.get("frozen_model_matches"),
                 "backup": backup, "decision": decision, "errors": errors}
        _write(folder / "status.json", state)
        publication_path = folder / "publication.json"
        previous = _load(publication_path) if publication_path.exists() else {}
        signature = trial._digest({key: value for key, value in state.items() if key != "checked_at"})
        elapsed = (current - trial._utc(previous["published_at"])).total_seconds() if previous else float("inf")
        if publisher is not None and (signature != previous.get("signature") or elapsed >= 3600):
            try:
                if publisher():
                    _write(publication_path, {"published_at": current.isoformat(), "signature": signature})
                else:
                    raise RuntimeError("publisher returned False; public site not refreshed")
            except Exception as exc:
                state["errors"].append(f"publication: {type(exc).__name__}: {exc}")
                state["status"] = "attention"
                _write(folder / "status.json", state)
        return state
