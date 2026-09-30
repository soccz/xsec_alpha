from copy import deepcopy

import pandas as pd
import pytest
from test_venue_observer import scenario as scenario

from utils import selection_trace as trace


def example(blocked=False):
    scores = pd.Series({f"KRW-T{i:02d}": i / 100 for i in range(12)})
    eligible = scores.iloc[1:].sort_values(ascending=False)
    previous = {"KRW-T06", "KRW-T07"}
    selected = pd.Series(dtype=float) if blocked else eligible.loc[["KRW-T06", "KRW-T07", "KRW-T01", "KRW-T02", "KRW-T03"]]
    report = {"data_asof": "2026-09-30T11:00:00+00:00", "signals": [
        {"market": m, "side": "SHORT", "actionable": False, "suppression": "ic_freeze"} for m in selected.index]}
    quality = pd.DataFrame({"actionable": True, "sigma": 1.4}, index=scores.index)
    trace.attach_selection_trace(report, scores, eligible, selected, previous, set(), 0 if blocked else 5, 0, 2, quality)
    return report, scores


def test_trace_explains_filter_buffer_and_gate_without_mutating_picks():
    report, scores = example()
    original = deepcopy(report["signals"])
    result = trace.verify_selection_trace(report["selection_trace"], report,
                                         {"rows": [{"market": m, "score": s} for m, s in scores.items()]})
    assert result["replay_matches"] and result["report_matches"]
    assert result["counts"] == {"scored": 12, "eligible": 11, "selected": 5, "actionable": 0}
    rows = {r["market"]: r for r in result["rows"]}
    assert rows["KRW-T00"]["reason"] == "not_tradable"
    assert rows["KRW-T06"]["reason"] == "buffer_retained"
    assert rows["KRW-T01"]["quality"]["actionable"] and not rows["KRW-T01"]["actionable"]
    assert report["signals"] == original


def test_blocked_live_side_retains_reference_basket_without_fabricating_trades():
    report, _ = example(blocked=True)
    result = report["selection_trace"]
    assert result["replay_matches"] and result["counts"]["selected"] == 0
    assert len(result["model_five"]) == len(result["buffer_reference"]) == 5
    assert all(r["reason"] == "gate_blocked" for r in result["rows"] if r["tradable"])


def test_trace_error_is_metadata_only_and_cannot_silence_report():
    report = {"signals": [{"market": "KRW-BTC"}], "data_asof": "invalid"}
    trace.attach_selection_trace(report, None, None, None, set(), set(), 5, 0, 2)
    assert report["selection_trace"]["status"] == "error"
    assert report["signals"] == [{"market": "KRW-BTC"}]


def test_trace_detects_changed_reason_and_different_original_scores():
    report, scores = example()
    report["selection_trace"]["rows"][0]["reason"] = "selected_rank"
    with pytest.raises(ValueError, match="replay"):
        trace.verify_selection_trace(report["selection_trace"], report)
    report, scores = example()
    with pytest.raises(ValueError, match="original scores"):
        trace.verify_selection_trace(report["selection_trace"], report, {"rows": []})


def test_trace_preserves_real_tie_order_for_buffer_replay():
    scores = pd.Series(0., index=[f"KRW-T{i:02d}" for i in reversed(range(12))])
    selected = trace.select_positions_with_buffer(scores, 0, 5, set(), {"KRW-T06"}, 2)[1]
    report = {"data_asof": "2026-09-30T11:00Z", "signals": [{"market": m,"side": "SHORT"} for m in selected.index]}
    trace.attach_selection_trace(report, scores, scores, selected, {"KRW-T06"}, set(), 5, 0, 2)
    assert report["selection_trace"]["replay_matches"]


def test_selection_report_is_verified_preserved_and_exported_by_audit(scenario):
    from utils import forecast_audit as audit
    from utils.experiment_supervisor import _write
    root, witness, report = scenario
    scores = pd.Series({r["market"]: r["score"] for r in witness["rows"]})
    picked = [r["market"] for r in report["signals"]]
    trace.attach_selection_trace(report, scores, scores.sort_values(ascending=False), scores.loc[picked],
                                 set(picked), set(), 5, 0, 5)
    _write(root / "output/operator_reports/20260930T110600.json", report)
    state = audit.refresh_audit(root, now="2026-09-30T11:10Z")
    assert state["selection"]["replay_matches"] and state["selection"]["report_matches"]
    assert len(state["selection"]["rows"]) == 12
    assert state["first_cycle"]["delivery"] == "verified"
    assert audit.backup_audit(root)["restore_verified"]
