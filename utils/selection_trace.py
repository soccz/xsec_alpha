"""Explain the observed selection path without changing picks or execution gates."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.eval_metrics import select_positions_with_buffer
from utils.model_release import artifact_sha256


def _derive(inputs, report):
    scores = pd.Series(inputs["scores"], dtype=float)
    eligible = inputs["eligible"]
    if (not scores.index.is_unique or not np.isfinite(scores).all()
            or len(eligible) != len(set(eligible)) or not set(eligible) <= set(scores.index)):
        raise ValueError("Invalid selection input membership")
    selection = scores.loc[eligible]
    previous = set(inputs["previous_short"])
    long_previous = set(inputs["previous_long"])
    count, buffer = inputs["short_limit"], inputs["buffer"]
    if min(count, buffer, inputs["long_limit"]) < 0:
        raise ValueError("Invalid selection limits")
    if previous or long_previous:
        _, replay = select_positions_with_buffer(selection, inputs["long_limit"], count,
                                                 long_previous, previous, buffer)
    else:
        replay = selection.tail(count) if count else pd.Series(dtype=float)
    observed = inputs["selected_before_save"]
    pure = selection.sort_index().sort_values(kind="stable").head(5).index.tolist()
    reference = select_positions_with_buffer(selection, 0, 5, set(), previous, buffer)[1].index.tolist()
    ranked = scores.sort_index().sort_values(kind="stable")
    eligible_ranks = selection.rank(method="min", ascending=True)
    full_ranks = scores.rank(method="min", ascending=True)
    actual = {r["market"]: r for r in report.get("signals", []) if r.get("side") == "SHORT"}
    rows = []
    for market in ranked.index:
        selected = market in observed
        quality = inputs["quality"].get(market, {})
        reason = ("not_tradable" if market not in eligible else "gate_blocked" if count == 0 else
                  "buffer_retained" if selected and buffer > 0 and market in previous and market not in pure and eligible_ranks[market] > 5 else
                  "selected_tie" if selected and market not in pure and eligible_ranks[market] <= 5 else
                  "selected_rank" if selected else "outside_selection")
        rows.append({"market": market, "score": float(scores[market]), "rank": int(full_ranks[market]),
                     "eligible_rank": int(eligible_ranks[market]) if market in eligible else None,
                     "tradable": market in eligible, "previous_recommendation": market in previous,
                     "model_five": market in pure, "buffer_reference": market in reference,
                     "selected": selected, "reported": market in actual,
                     "actionable": actual.get(market, {}).get("actionable") is True,
                     "reason": reason, "suppression": actual.get(market, {}).get("suppression"),
                     "quality": quality})
    return {"replay_matches": set(replay.index) == set(observed),
            "report_matches": set(actual) == set(observed), "model_five": pure,
            "buffer_reference": reference, "selected": observed,
            "buffer_replacements": sorted(set(reference) - set(pure)),
            "counts": {"scored": len(scores), "eligible": len(eligible), "selected": len(observed),
                       "actionable": sum(r["actionable"] for r in rows)}, "rows": rows}


def attach_selection_trace(report, scores, selection_scores, shorts, prev_shorts, prev_longs,
                           short_n, long_n, rebal_buffer, quality=None):
    """Reporting metadata failures must never prevent the mandatory coin report."""
    try:
        columns = [c for c in ("actionable", "suppression", "direction_prob", "sigma", "calibration_generated_at")
                   if quality is not None and c in quality.columns]
        quality_rows = json.loads(quality[columns].to_json(orient="index")) if columns else {}
        inputs = {"scores": {m: float(v) for m, v in scores.items()},
                  "eligible": selection_scores.index.tolist(), "previous_short": sorted(prev_shorts),
                  "previous_long": sorted(prev_longs), "short_limit": int(short_n), "long_limit": int(long_n),
                  "buffer": int(rebal_buffer), "selected_before_save": shorts.index.tolist(), "quality": quality_rows}
        report["selection_trace"] = {"schema": 1, "status": "recorded", "signal_at": report["data_asof"],
                                     "source_sha256": artifact_sha256(Path(__file__)), "inputs": inputs,
                                     **_derive(inputs, report)}
    except Exception as exc:
        report["selection_trace"] = {"schema": 1, "status": "error", "reason": f"{type(exc).__name__}: {exc}"[:300]}


def verify_selection_trace(trace, report, witness=None):
    if trace["schema"] != 1 or trace["status"] != "recorded":
        raise ValueError("Selection trace unavailable")
    derived = _derive(trace["inputs"], report)
    if (trace["signal_at"] != report["data_asof"] or len(trace["source_sha256"]) != 64
            or any(trace[k] != v for k, v in derived.items())):
        raise ValueError("Selection trace replay mismatch")
    if witness is not None and trace["inputs"]["scores"] != {r["market"]: r["score"] for r in witness["rows"]}:
        raise ValueError("Selection trace not bound to original scores")
    return derived


def selection_summary(tables):
    reports = [r["report"] for r in tables["deliveries"].values() if r["linked"]]
    latest = max(reports, key=lambda r: r["generated_at"], default={})
    trace = latest.get("selection_trace")
    if not trace:
        return {"status": "waiting", "rows": []}
    if trace["status"] != "recorded":
        return {"status": "error", "reason": trace.get("reason"), "rows": []}
    return {k: v for k, v in trace.items() if k not in ("inputs", "source_sha256")}
