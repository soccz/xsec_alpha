from copy import deepcopy
import hashlib
import json

import pandas as pd
import pytest

from test_execution_audit import registered_execution as registered_execution, scenario as scenario, depth_payload, funding_payload
from test_venue_observer import SLOT, quote_payload, raw_prices
from utils import forecast_audit as audit, operating_acceptance as acceptance, prospective as trial
from utils.experiment_supervisor import _write
from utils.selection_trace import attach_selection_trace


def complete_cycle(root):
    path = root / "output/operator_reports/20260930T110600.json"
    report = json.loads(path.read_text())
    scores = pd.Series({f"KRW-T{i:02d}": i / 100 for i in range(12)})
    selection = scores.sort_values(ascending=False)
    picks = selection.tail(5)
    report["signals"] = [{"market": m, "side": "SHORT", "actionable": False} for m in picks.index]
    attach_selection_trace(report, scores, selection, picks, set(), set(), 5, 0, 0)
    _write(path, report)
    audit.refresh_audit(root, now="2026-09-30T11:10Z")
    for stamp in ("2026-09-30T12:00:10Z", "2026-09-30T18:00:10Z"):
        audit.refresh_audit(root, now=stamp, quote_loader=lambda: (quote_payload(stamp), trial._utc(stamp)),
                            execution_loader=lambda *a: (depth_payload(stamp), trial._utc(stamp)))
    audit.refresh_audit(root, now="2026-09-30T19:10Z", price_loader=raw_prices)
    return audit.refresh_audit(root, now="2026-10-01T03:10Z",
                               execution_loader=lambda kind, symbol: (funding_payload(symbol), trial._utc("2026-10-01T03:10Z")))


def test_future_and_partial_cycle_never_claim_completion_or_query_pages(registered_execution):
    root, _ = registered_execution
    state = audit.refresh_audit(root, now="2026-09-30T10:00Z")
    result = acceptance.refresh_acceptance(root, state, audit.backup_audit(root), now="2026-09-30T10:00Z",
                                           public_loader=lambda *a: pytest.fail("no future publication"))
    assert result["status"] == "awaiting_future" and not result["closed_sha256"]
    state = audit.refresh_audit(root, now="2026-09-30T11:10Z")
    assert state["lifecycle"]["status"] == "observing" and not state["lifecycle"]["terminal"]


def test_complete_cycle_requires_matching_backup_and_published_encrypted_payload(registered_execution):
    root, _ = registered_execution
    state = complete_cycle(root)
    assert state["lifecycle"]["status"] == "complete"
    backup = audit.backup_audit(root)
    bad = {**backup, "first_cycle_evidence_sha256": "0" * 64}
    assert acceptance.refresh_acceptance(root, state, bad, now="2026-10-01T03:20Z")["status"] == "awaiting_backup"
    def offline(*args):
        raise OSError("network unavailable")
    pending = acceptance.refresh_acceptance(root, state, backup, now="2026-10-01T03:20Z", public_loader=offline)
    assert pending["status"] == "awaiting_publication"
    original = (root / "output/operating_acceptance/closed.json").read_bytes()
    payload = b'{"encrypted":true,"ct":"test-only"}'
    receipt = {"closed_sha256": pending["closed_sha256"], "summary_sha256": hashlib.sha256(payload).hexdigest()}
    def pages(path):
        return json.dumps({"operating_receipt": receipt}).encode() if path.startswith("public_") else payload
    done = acceptance.refresh_acceptance(root, state, backup, now="2026-10-01T03:30Z", public_loader=pages)
    assert done["status"] == "complete" and not done["profitability_proven"]
    again = acceptance.refresh_acceptance(root, state, backup, now="2026-10-01T03:40Z", public_loader=lambda *a: pytest.fail("receipt already saved"))
    assert again == done and (root / "output/operating_acceptance/closed.json").read_bytes() == original
    assert len(list((root / "output/operating_acceptance/publication_attempts").glob("*.json"))) == 2
    (root / "output/operating_acceptance/closed.json").unlink()
    (root / "output/operating_acceptance/publication.json").unlink()
    restored = acceptance.refresh_acceptance(root, state, backup, now="2026-10-01T03:50Z", public_loader=lambda *a: pytest.fail("recover receipt without refetch"))
    assert restored == done and (root / "output/operating_acceptance/closed.json").read_bytes() == original
    (root / "output/operating_acceptance_backup/closed.json").write_text("corrupt")
    with pytest.raises((ValueError, KeyError)):
        acceptance.refresh_acceptance(root, state, backup, now="2026-10-01T04:00Z")


def test_failed_first_cycle_is_closed_honestly_never_replaced_by_a_later_slot(registered_execution):
    root, _ = registered_execution
    state = audit.refresh_audit(root, now="2026-10-03T11:00Z", price_loader=raw_prices)
    assert state["lifecycle"]["status"] == "closed_with_gaps"
    backup = audit.backup_audit(root)
    result = acceptance.refresh_acceptance(root, state, backup, now="2026-10-03T11:00Z", public_loader=lambda *a: b'{}')
    closed = result["closed"]
    assert closed["cycle"]["signal_at"] == SLOT
    changed = deepcopy(state)
    changed["lifecycle"]["signal_at"] = "2026-10-03T11:00:00+00:00"
    again = acceptance.refresh_acceptance(root, changed, backup, now="2026-10-03T11:10Z", public_loader=lambda *a: b'{}')
    assert again["closed"] == closed
    (root / "output/forecast_audit_backups" / (backup["checkpoint"] + ".sqlite")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="backup"):
        acceptance.refresh_acceptance(root, state, backup, now="2026-10-03T11:20Z")


@pytest.mark.parametrize("problem", ["old_closure", "mixed_payload", "offline"])
def test_push_or_old_pages_is_not_proof_of_this_publication(problem):
    def loader(path):
        if problem == "offline":
            raise OSError("offline")
        if path.startswith("public_"):
            return json.dumps({"operating_receipt": {"closed_sha256": "old" if problem == "old_closure" else "current",
                                                      "summary_sha256": "0" * 64}}).encode()
        return b"not the matching encrypted payload"
    with pytest.raises((ValueError, OSError)):
        acceptance.check_publication("current", loader)


def shadow_state():
    return {"policy": {"first_signal_at": SLOT}, "recent_windows": [
        {"signal_at": (trial._utc(SLOT) + pd.Timedelta(hours=6 * i)).isoformat(), "status": "evaluated",
         "saved_ic": -.1, "legacy_ic": .1, "comparison_status": "available", "model_sha256": "a" * 64}
        for i in range(3)]}


def test_shadow_gate_uses_same_slots_and_never_changes_production():
    result = acceptance.gate_shadow(shadow_state(), "2026-10-01T07:00Z")
    assert result["saved_status"] == "LIQUIDATE" and result["matched_legacy_status"] == "OK"
    assert result["live_source"] == "legacy_recomputed_ic" and not result["automatic_switch"]


@pytest.mark.parametrize("problem", ["missing", "model_change", "immature", "next_slot_missing", "nan", "inf"])
def test_shadow_gate_never_skips_gaps_mixes_models_or_uses_unmatured_scores(problem):
    state, now = shadow_state(), "2026-10-01T07:00Z"
    if problem == "missing":
        state["recent_windows"].pop(1)
    elif problem == "model_change":
        state["recent_windows"][1]["model_sha256"] = "b" * 64
    elif problem == "immature":
        now = "2026-10-01T06:59Z"
    elif problem in ("nan", "inf"):
        state["recent_windows"][1]["saved_ic"] = float(problem)
    else:
        now = "2026-10-01T13:00Z"
    result = acceptance.gate_shadow(state, now)
    assert result["status"] == "unavailable" and result["saved_status"] is None


def test_export_receipt_binds_only_closure_hash_and_exact_private_bytes(tmp_path, monkeypatch):
    from utils import dashboard_export as export
    closed = {"schema": 1, "cycle": {"signal_at": SLOT}}
    digest = trial._digest(closed)
    monkeypatch.setattr(export, "build_summary_payload", lambda: {"experiment_supervision": {
        "operating_acceptance": {"closed": closed, "closed_sha256": digest}}})
    monkeypatch.setattr(export, "build_history_payload", lambda **k: {})
    monkeypatch.setattr(export, "build_accuracy_payload", lambda **k: {})
    monkeypatch.setattr(export, "build_public_summary_payload", lambda: {"public": "aggregate"})
    files = export.export_to(tmp_path / "data", pin="test-only", public_target=tmp_path / "public_summary.json")
    receipt = json.loads(files["public_summary.json"].read_text())["operating_receipt"]
    assert set(receipt) == {"closed_sha256", "summary_sha256"}
    assert receipt == {"closed_sha256": digest, "summary_sha256": hashlib.sha256(files["summary.json"].read_bytes()).hexdigest()}
