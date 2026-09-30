import copy
from dataclasses import asdict

import pandas as pd
import pytest

from utils.rotation_research import RotationRules, followup_plan, shadow_net_pct, signal_weights, weekly_review


def records(difference=-1.):
    result = []
    for slot in pd.date_range("2026-09-30T05:00Z", periods=40, freq="6h"):
        result.append({"status": "matured", "signal_at": slot.isoformat(),
                       "recorded_at": (slot + pd.Timedelta(minutes=5)).isoformat(),
                       "exit_at": (slot + pd.Timedelta(hours=7)).isoformat(),
                       "evaluated_at": (slot + pd.Timedelta(hours=9)).isoformat(),
                       "model_sha256": "model-v1", "regime": "bull_lowvol",
                       "model_net_pct": difference, "baseline_net_pct": 0.})
    return result


def test_two_distinct_weeks_required_and_only_quarter_challenger():
    first = weekly_review(records(), "2026-10-12")
    assert first["regimes"]["bull_lowvol"]["reason"] == "await_confirmation"
    second = weekly_review(records(), "2026-10-19", first)
    assert second["regimes"]["bull_lowvol"]["policy"] == "reversal"
    assert second["automatic_live_switching"] is False
    weights = signal_weights(second, "bull_lowvol", "2026-10-19T05:00Z")
    assert weights["model"] == .75 and weights["reversal"] == .25
    assert shadow_net_pct(weights, 1., 2.) == pytest.approx(1.225)
    with pytest.raises(ValueError, match="Duplicate"):
        weekly_review(records(), "2026-10-19", second)


def test_future_or_late_evaluated_outcomes_do_not_leak():
    data = records()
    original = weekly_review(data, "2026-10-05")
    for r in data:
        if pd.Timestamp(r["evaluated_at"]) >= pd.Timestamp("2026-10-05", tz="UTC"):
            r["model_net_pct"] = -9999
    assert weekly_review(data, "2026-10-05") == original
    row = data[0]
    row["evaluated_at"] = "2026-10-05T00:00:00+00:00"
    newer = weekly_review(data, "2026-10-05")
    assert newer["regimes"]["bull_lowvol"]["n_windows"] == original["regimes"]["bull_lowvol"]["n_windows"] - 1


def test_skipped_week_resets_confirmation_and_future_state_rejected():
    first = weekly_review(records(), "2026-10-12")
    second = weekly_review(records(), "2026-10-26", first)
    assert second["regimes"]["bull_lowvol"]["streak"] == 1
    with pytest.raises(ValueError, match="Future"):
        signal_weights(second, "bull_lowvol", "2026-10-25")


def test_sparse_unknown_stale_and_empty_use_model():
    first = weekly_review(records()[:20], "2026-10-12")
    assert first["regimes"]["bull_lowvol"]["reason"] == "insufficient_data"
    assert signal_weights(first, "unknown", "2026-10-12")["reversal"] == 0
    assert signal_weights(first, "bull_lowvol", "2026-10-19")["reason"] == "stale_review_fallback"
    assert all(r["n_windows"] == 0 for r in weekly_review([], "2026-10-05")["regimes"].values())
    assert signal_weights(None, "bull_lowvol", "2026-10-12")["reason"] == "missing_review_fallback"
    first["regimes"]["bull_lowvol"]["policy"] = "reversal"
    assert signal_weights(first, "bull_lowvol", "2026-10-12")["reversal"] == 0


def test_cooldown_and_changed_rules():
    first = weekly_review(records(), "2026-10-12")
    second = weekly_review(records(), "2026-10-19", first)
    altered = copy.deepcopy(second)
    altered["regimes"]["bull_lowvol"].update(candidate="model", streak=1)
    third = weekly_review(records(1.), "2026-10-26", altered)
    assert third["regimes"]["bull_lowvol"]["reason"] == "cooldown"
    assert third["regimes"]["bull_lowvol"]["policy"] == "reversal"
    with pytest.raises(ValueError, match="Changed rules"):
        weekly_review(records(), "2026-10-26", second, RotationRules(min_windows=40))


def test_no_partial_outcome_or_duplicate_or_post_entry_evidence():
    data = records()
    data[0]["recorded_at"] = data[0]["exit_at"]
    with pytest.raises(ValueError, match="timing"):
        weekly_review(data, "2026-10-12")
    data = records()
    with pytest.raises(ValueError, match="Duplicate"):
        weekly_review(data + data[:1], "2026-10-12")
    with pytest.raises(ValueError, match="Monday"):
        weekly_review(data, "2026-10-13")


def test_cost_on_each_weight_change_even_when_regime_changes():
    assert shadow_net_pct({"model": .75, "reversal": .25}, 1., 2., switch_bps=10) == pytest.approx(1.225)
    assert shadow_net_pct({"model": 1., "reversal": 0.}, 1., 2., previous_reversal_weight=.25) == pytest.approx(.975)
    assert shadow_net_pct({"model": .75, "reversal": .25}, 1., 2., previous_reversal_weight=.25) == 1.25
    with pytest.raises(ValueError):
        shadow_net_pct({"model": 1., "reversal": .25}, 1., 2.)


def test_default_rules_are_explicit_not_optimized_from_old_trial():
    assert asdict(RotationRules()) == {"lookback_days": 90, "min_windows": 30, "min_days": 8,
                                      "min_advantage_pp": .25, "consecutive_reviews": 2,
                                      "cooldown_days": 14, "challenger_weight": .25}


def test_plan_is_not_preregistration_or_powered_adaptive_claim():
    plan = followup_plan({"source_checkpoint": "test", "generated_at": "2026-09-30"})
    assert plan["status"] == "designed_not_registered" and plan["first_signal_at"] is None
    assert plan["fixed_model_power_is_not_adaptive_power"]
    assert plan["scheduled_windows"] == 336 and not plan["automatic_live_switching"]
