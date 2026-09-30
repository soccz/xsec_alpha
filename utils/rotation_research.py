"""Pure research candidate. Not imported by live ranking or the regime observer."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from utils import prospective as trial

REGIMES = ("bull_highvol", "bull_lowvol", "bear_highvol", "bear_lowvol")


@dataclass(frozen=True)
class RotationRules:
    lookback_days: int = 90
    min_windows: int = 30
    min_days: int = 8
    min_advantage_pp: float = 0.25
    consecutive_reviews: int = 2
    cooldown_days: int = 14
    challenger_weight: float = 0.25

    def __post_init__(self):
        if (self.lookback_days < 1 or self.min_windows < 2 or self.min_days < 2
                or not np.isfinite(self.min_advantage_pp) or self.min_advantage_pp <= 0
                or self.consecutive_reviews < 2 or self.cooldown_days < 7
                or not 0 < self.challenger_weight <= 1):
            raise ValueError("Invalid research rotation rules")


def weekly_review(records, review_at, previous=None, rules=RotationRules()):
    """Use only timely recorded, fully evaluated outcomes known before Monday UTC."""
    now = trial._utc(review_at)
    if now.weekday() != 0 or now != now.normalize():
        raise ValueError("Review boundary must be Monday 00:00 UTC")
    if previous:
        last = trial._utc(previous["review_at"])
        if now <= last:
            raise ValueError("Duplicate or backwards weekly review")
        if previous["rules"] != asdict(rules):
            raise ValueError("Changed rules require a new research version")
        consecutive = now - last == pd.Timedelta(days=7)
    else:
        consecutive = False
    rows, seen = [], set()
    for row in records:
        signal = trial._utc(row["signal_at"])
        if signal >= now:
            continue
        if signal.isoformat() in seen:
            raise ValueError("Duplicate historical slot")
        seen.add(signal.isoformat())
        if row["status"] != "matured":
            continue
        recorded, evaluated, exit_at = [trial._utc(row[k]) for k in ("recorded_at", "evaluated_at", "exit_at")]
        if (not signal <= recorded <= signal + pd.Timedelta(minutes=45)
                or exit_at != signal + pd.Timedelta(hours=7)
                or evaluated < exit_at + pd.Timedelta(hours=1)):
            raise ValueError("Invalid prospective observation timing")
        if evaluated >= now or signal < now - pd.Timedelta(days=rules.lookback_days):
            continue
        if (not all(np.isfinite(row[k]) for k in ("model_net_pct", "baseline_net_pct"))
                or not row.get("model_sha256")):
            raise ValueError("Invalid outcome or missing model version")
        rows.append(row)
    states = {}
    for regime in REGIMES:
        old = (previous or {}).get("regimes", {}).get(regime, {})
        current = old.get("policy", "model")
        subset = [r for r in rows if r["regime"] == regime]
        days = len({r["signal_at"][:10] for r in subset})
        ready = len(subset) >= rules.min_windows and days >= rules.min_days
        difference = float(np.mean([r["model_net_pct"] - r["baseline_net_pct"] for r in subset])) if subset else None
        candidate = current
        if ready and current == "model" and difference <= -rules.min_advantage_pp:
            candidate = "reversal"
        elif ready and current == "reversal" and difference >= rules.min_advantage_pp:
            candidate = "model"
        wants_change = candidate != current
        streak = ((old.get("streak", 0) if consecutive and old.get("candidate") == candidate else 0) + 1
                  if wants_change else 0)
        last_change = old.get("last_change_at")
        cooldown = last_change and now - trial._utc(last_change) < pd.Timedelta(days=rules.cooldown_days)
        change = wants_change and streak >= rules.consecutive_reviews and not cooldown
        states[regime] = {
            "policy": candidate if change else current,
            "candidate": candidate, "streak": 0 if change else streak,
            "last_change_at": now.isoformat() if change else last_change,
            "n_windows": len(subset), "n_days": days,
            "model_versions": len({r["model_sha256"] for r in subset}),
            "model_minus_baseline_pp": difference,
            "reason": ("insufficient_data" if not ready else "changed" if change else
                       "cooldown" if cooldown and wants_change else "await_confirmation" if wants_change else "retain"),
        }
    return {"status": "research_candidate_not_live", "review_at": now.isoformat(),
            "rules": asdict(rules), "regimes": states, "automatic_live_switching": False}


def signal_weights(state, regime, signal_at):
    signal = trial._utc(signal_at)
    if state is None:
        return {"model": 1., "reversal": 0., "reason": "missing_review_fallback"}
    review = trial._utc(state["review_at"])
    if review > signal:
        raise ValueError("Future weekly decision cannot select a past signal")
    fresh = signal < review + pd.Timedelta(days=7)
    row = state["regimes"].get(regime, {})
    eligible = row.get("reason") != "insufficient_data"
    selected = row.get("policy", "model") if fresh and eligible else "model"
    weight = state["rules"]["challenger_weight"] if selected == "reversal" else 0.
    return {"model": 1. - weight, "reversal": weight,
            "reason": "stale_review_fallback" if not fresh else "unknown_regime_fallback" if regime not in REGIMES
            else "insufficient_data_fallback" if not eligible else selected}


def shadow_net_pct(weights, model_net_pct, baseline_net_pct, *, previous_reversal_weight=0., switch_bps=10.):
    """Conservative additional policy-weight cost; neither actual turnover nor fills."""
    numeric_weights = (weights["model"], weights["reversal"])
    if (not all(np.isfinite(v) for v in (*numeric_weights, model_net_pct, baseline_net_pct,
                                        previous_reversal_weight, switch_bps))
            or not np.isclose(weights["model"] + weights["reversal"], 1.)
            or not all(0 <= v <= 1 for v in numeric_weights)
            or not 0 <= previous_reversal_weight <= 1 or switch_bps < 0):
        raise ValueError("Invalid weights, returns or cost")
    cost_pct = abs(weights["reversal"] - previous_reversal_weight) * switch_bps / 100
    return weights["model"] * model_net_pct + weights["reversal"] * baseline_net_pct - cost_pct


def followup_plan(report):
    """An executable candidate's design record, not an activated/preregistered trial."""
    return {
        "version": "regime_rotation_candidate_v1", "status": "designed_not_registered",
        "source_checkpoint": report["source_checkpoint"],
        "designed_at": report["generated_at"], "first_signal_at": None,
        "purpose": "bounded prospective feasibility pilot, not a powered superiority claim",
        "scheduled_windows": 336, "calendar_days": 84, "price_grace_hours": 48,
        "rules": asdict(RotationRules()),
        "comparator": "unchanged current-model bottom-five shadow ranking, not gated Telegram portfolio",
        "challenger": "75% model + 25% reversal only when the prior weekly policy qualifies",
        "primary_endpoint": "paired challenger minus comparator mean net per scheduled 6h window",
        "costs": {"base": "same registered regime-observer costs for both baskets",
                  "additional_weight_change_bps": 10, "sensitivity_bps": [0, 10, 20],
                  "real_fills": False},
        "training_evidence": "new regime-observer outcomes available before each review only; exclude old 60 windows",
        "confirmation": "two consecutive eligible Monday reviews, no duplicate or skipped-week confirmation",
        "fallback": "100% model for unknown regimes, insufficient data, stale or unavailable review",
        "outcomes": ["integrity_failure", "insufficient_evidence", "policy_not_exercised", "pilot_complete_not_confirmatory"],
        "policy_exercise_minimums": {"nonzero_challenger_windows": 30, "distinct_utc_dates": 8,
                                     "are_power_thresholds": False},
        "stopping": "fixed schedule plus final horizon and 48h grace; no extension until significance",
        "registration_requirements": ["separate immutable registry and scoring-source fingerprints",
                                      "pre-entry persistence of weights, review identity and model version",
                                      "append-only outcomes; missing windows never backfilled or treated as zero",
                                      "actual pilot variance required before powering a new confirmatory endpoint"],
        "automatic_live_switching": False, "automatic_promotion": False,
        "telegram_policy_changes": False,
        "fixed_model_power_is_not_adaptive_power": True,
    }
