"""Operational first-cycle evidence, never a profitability or promotion decision."""
import hashlib
import json
import math
from pathlib import Path
import urllib.request

import pandas as pd

from utils import prospective as trial
from utils.experiment_supervisor import _envelope, _verified_document, _write
from utils.model_release import artifact_sha256

PUBLIC_ROOT = "https://soccz.github.io/projects/xsec-alpha/"


def first_evidence(policy, tables):
    execution = tables["execution_details"].get("policy", {})
    slot = execution.get("first_signal_at", policy["first_signal_at"])
    records = {name: {key: row for key, row in rows.items() if row.get("signal_at") == slot}
               for name, rows in tables.items() if name != "ingestion_events"}
    return slot, trial._digest({"policy": policy, "execution_policy": execution, "records": records})


def cycle_summary(policy, tables, now):
    slot, digest = first_evidence(policy, tables)
    signal, current = trial._utc(slot), trial._utc(now)
    source = tables["signals"].get(slot, {})
    intent = tables["venue_intents"].get(slot, {})
    execution = tables["execution_details"]
    stages = []
    def stage(name, passed, terminal, deadline, reason):
        status = "passed" if passed else "failed" if terminal or current > deadline else "waiting"
        stages.append({"name": name, "status": status, "reason": reason})
    capture_end = signal + pd.Timedelta(minutes=45)
    stage("signal", source.get("status") == "recorded", bool(source), capture_end, source.get("status", "awaiting_signal"))
    deliveries = [r for r in tables["deliveries"].values() if r["signal_at"] == slot and r["complete"]]
    stage("telegram", bool(deliveries), False, capture_end, "api_ack_and_score_link" if deliveries else "no_linked_ack")
    traces = [r["report"].get("selection_trace", {}) for r in deliveries]
    trace_ok = any(r.get("status") == "recorded" and r.get("replay_matches") and r.get("report_matches") for r in traces)
    stage("selection", trace_ok, False, capture_end, "replayed" if trace_ok else "no_matching_trace")
    stage("venue_intent", intent.get("status") == "recorded", bool(intent), capture_end, intent.get("status", "awaiting_intent"))
    for phase, hours in (("entry", 1), ("exit", 7)):
        row = tables["quotes"].get(slot + "/" + phase, {})
        stage(phase + "_quote", row.get("status") == "observed", bool(row), signal + pd.Timedelta(hours=hours, minutes=5), row.get("status", "waiting"))
    outcome = tables["outcomes"].get(slot, {})
    stage("saved_score", outcome.get("status") == "evaluated", bool(outcome), signal + pd.Timedelta(hours=55), outcome.get("reason", outcome.get("status", "waiting")))
    result = execution.get("result/" + slot, {})
    sizes_ok = bool(result.get("sizes")) and all(
        row.get("status") == "evaluated" for size in result["sizes"] for row in size["rows"].values())
    stage("depth_sizes", sizes_ok, bool(result), signal + pd.Timedelta(hours=7, minutes=5), result.get("status", "waiting"))
    members = set(intent.get("model_selected", [])) | (set(intent.get("selected", [])) if intent.get("selection_comparable") else set())
    funding = [execution.get(f"funding/{slot}/{intent['symbols'][market]}", {}) for market in sorted(members)]
    funding_ok = bool(funding) and all(r.get("status") == "observed" for r in funding)
    stage("funding_history", funding_ok, bool(funding) and all(funding), signal + pd.Timedelta(hours=43, minutes=5),
          "published_rates_only" if funding_ok else "missing_failed_or_boundary_uncertain")
    pending = any(r["status"] == "waiting" for r in stages)
    status = ("awaiting_future" if current < signal else "observing" if pending else
              "complete" if all(r["status"] == "passed" for r in stages) else "closed_with_gaps")
    return {"signal_at": slot, "evidence_sha256": digest, "status": status, "terminal": not pending,
            "stages": stages, "changes_gate": False, "profitability_proven": False}


def gate_shadow(summary, now):
    """Compare identical matured slots; missing data never becomes a healthy gate."""
    from utils.ic_gate import _decide, FREEZE_THRESHOLD, WARN_THRESHOLD

    current = trial._utc(now)
    first = trial._utc(summary["policy"]["first_signal_at"])
    last = current.floor("h") - pd.Timedelta(hours=8)
    scheduled = list(pd.date_range(first, last, freq="6h")) if last >= first else []
    slots = [r.isoformat() for r in scheduled[-3:]]
    by_slot = {r["signal_at"]: r for r in summary["recent_windows"]}
    rows = [by_slot.get(slot, {"signal_at": slot, "status": "missing"}) for slot in slots]
    reasons = []
    if len(rows) < 3:
        reasons.append("fewer_than_three_matured_slots")
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if any(r.get("status") != "evaluated" or not finite(r.get("saved_ic")) for r in rows):
        reasons.append("incomplete_consecutive_scores")
    if rows and (len({r.get("model_sha256") for r in rows}) != 1 or not rows[0].get("model_sha256")):
        reasons.append("model_cohort_not_constant")
    ready = not reasons
    legacy_ready = ready and all(finite(r.get("legacy_ic")) and r.get("comparison_status") == "available" for r in rows)
    return {"status": "comparable" if ready else "unavailable", "reasons": reasons,
            "saved_status": _decide([r["saved_ic"] for r in rows])[0] if ready else None,
            "matched_legacy_status": _decide([r["legacy_ic"] for r in rows])[0] if legacy_ready else None,
            "slots": [{k: r.get(k) for k in ("signal_at", "saved_ic", "legacy_ic", "model_sha256", "same_model", "same_pool")} for r in rows],
            "freeze_threshold": FREEZE_THRESHOLD, "warn_threshold": WARN_THRESHOLD,
            "live_source": "legacy_recomputed_ic", "automatic_switch": False}


def _download(relative):
    request = urllib.request.Request(PUBLIC_ROOT + relative, headers={"Cache-Control": "no-cache", "User-Agent": "xsec-acceptance/1"})
    with urllib.request.urlopen(request, timeout=3) as response:
        payload = response.read(4 * 1024**2 + 1)
    if len(payload) > 4 * 1024**2:
        raise ValueError("Published payload exceeds verification limit")
    return payload


def check_publication(expected, loader=None):
    loader = loader or _download
    public = json.loads(loader("public_summary.json?acceptance=" + expected))
    receipt = public.get("operating_receipt", {})
    if receipt.get("closed_sha256") != expected:
        raise ValueError("Pages does not yet contain this first-cycle closure")
    summary = loader("dashboard/data/summary.json?acceptance=" + expected)
    digest = hashlib.sha256(summary).hexdigest()
    if receipt.get("summary_sha256") != digest:
        raise ValueError("Pages summary and receipt belong to different publications")
    return {"closed_sha256": expected, "summary_sha256": digest, "public_sha256": trial._digest(public)}


def _saved_document(root, name):
    paths = [root / "output" / folder / name for folder in ("operating_acceptance", "operating_acceptance_backup")]
    documents = [_verified_document(path) for path in paths if path.exists()]
    if not documents:
        return None
    if any(doc != documents[0] for doc in documents):
        raise ValueError("Operating closure and recovery copy disagree")
    for path in paths:
        if not path.exists():
            _write(path, _envelope(documents[0]), immutable=True)
        if _verified_document(path) != documents[0]:
            raise ValueError("Operating closure copy verification failed")
    return documents[0]


def refresh_acceptance(root, audit, backup, now=None, public_loader=None):
    if not audit or not audit.get("lifecycle"):
        return None
    root, current = Path(root), trial._utc(now)
    folder = root / "output/operating_acceptance"
    lifecycle = audit["lifecycle"]
    backup_ok = bool(backup and backup.get("restore_verified") and
                     backup.get("first_cycle_evidence_sha256") == lifecycle["evidence_sha256"])
    closed_path = folder / "closed.json"
    closed = _saved_document(root, "closed.json")
    if closed is None and lifecycle["terminal"] and backup_ok:
        closed = {"schema": 1, "closed_at": current.isoformat(), "cycle": lifecycle,
                  "backup_checkpoint": backup["checkpoint"], "changes_gate": False}
        _write(closed_path, _envelope(closed), immutable=True)
        closed = _saved_document(root, "closed.json")
    publication = None
    if closed:
        checkpoint = root / "output/forecast_audit_backups" / (closed["backup_checkpoint"] + ".sqlite")
        if artifact_sha256(checkpoint) != closed["backup_checkpoint"]:
            raise ValueError("First-cycle backup changed or disappeared")
        digest = trial._digest(closed)
        receipt_path = folder / "publication.json"
        publication = _saved_document(root, "publication.json")
        if publication:
            if publication["closed_sha256"] != digest:
                raise ValueError("Publication receipt belongs to another first cycle")
        else:
            attempt = {"checked_at": current.isoformat(), "closed_sha256": digest}
            try:
                publication = {**attempt, **check_publication(digest, public_loader), "status": "verified"}
                _write(receipt_path, _envelope(publication), immutable=True)
                publication = _saved_document(root, "publication.json")
                attempt = publication
            except Exception as exc:
                attempt.update(status="waiting", error=f"{type(exc).__name__}: {exc}"[:300])
            attempt_path = folder / "publication_attempts" / (trial._digest(attempt) + ".json")
            if not attempt_path.exists():
                _write(attempt_path, _envelope(attempt), immutable=True)
            elif _verified_document(attempt_path) != attempt:
                raise ValueError("Publication attempt changed")
            publication = attempt
    return {"status": (closed["cycle"]["status"] if publication and publication["status"] == "verified" else
                       "awaiting_publication" if closed else lifecycle["status"] if not lifecycle["terminal"] else "awaiting_backup"),
            "cycle": closed["cycle"] if closed else lifecycle, "backup_verified": backup_ok,
            "closed_sha256": trial._digest(closed) if closed else None, "closed": closed,
            "publication": publication, "changes_gate": False, "profitability_proven": False}
