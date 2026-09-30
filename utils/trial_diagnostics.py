"""Read-only, post-hoc diagnostics of a sealed trial; never a promotion gate."""
from contextlib import closing
from importlib.metadata import version
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import config
from utils import prospective as trial
from utils.eval_metrics import build_btc_context
from utils.experiment_supervisor import _read_ledger, _sealed_archive, _verified_document
from utils.model_release import artifact_sha256

STRATEGIES = ("model_short5", "reversal_short5", "universe_short")
SOURCES = {
    "sample_size": "https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm",
    "time_blocks": "https://bashtage.github.io/arch/bootstrap/timeseries-bootstraps.html",
    "regime_hypothesis": "https://www.nber.org/papers/w17182",
}


def _values(values):
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("Need at least two finite ordered windows")
    return data


def block_means(values, block_size=4, draws=10000, seed=42):
    """Circular moving blocks, matching the original trial's sampling convention."""
    data = _values(values)
    if not 1 <= block_size <= len(data) or draws < 2:
        raise ValueError("Invalid block size or draw count")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data), size=(draws, math.ceil(len(data) / block_size)))
    indices = ((starts[:, :, None] + np.arange(block_size)) % len(data)).reshape(draws, -1)
    return data[indices[:, :len(data)]].mean(axis=1)


def distribution(values):
    data = _values(values)
    ordered = np.sort(data)
    tail_n = max(1, math.ceil(len(data) * 0.1))
    return {
        "n_windows": len(data), "mean_pct": float(data.mean()),
        "median_pct": float(np.median(data)), "std_pct": float(data.std(ddof=1)),
        "positive_windows": int((data > 0).sum()), "negative_windows": int((data < 0).sum()),
        "min_pct": float(data.min()), "max_pct": float(data.max()),
        "worst_decile_mean_pct": float(ordered[:tail_n].mean()),
        "worst_decile_n": tail_n,
    }


def power_scenarios(values, effects_pp=(0.10, 0.25, 0.50), alpha=0.05, power=0.8):
    """Approximate planning sensitivity, NOT the power of an adaptive selector."""
    data = _values(values)
    if len(data) < 8 or not 0 < alpha < 1 or not 0.5 < power < 1:
        raise ValueError("Insufficient pilot or invalid alpha/power")
    if not all(np.isfinite(x) and x > 0 for x in effects_pp):
        raise ValueError("Effects must be positive finite percentage points")
    # Var(circular block sum)/block length estimates long-run per-window variance.
    # Keep IID as a floor: a short, negatively correlated pilot must not shrink N.
    variances = {"iid": float(data.var(ddof=1))}
    for length in (4, 8):
        indices = (np.arange(len(data))[:, None] + np.arange(length)) % len(data)
        variances[f"block_{length}"] = float(data[indices].sum(axis=1).var(ddof=1) / length)
    variance = max(variances.values())
    if variance <= 0:
        raise ValueError("Zero pilot variance cannot justify a sample-size estimate")
    z_alpha, z_power = norm.ppf(1 - alpha / 2), norm.ppf(power)
    rows = []
    for effect in effects_pp:
        for variance_multiplier in (1.0, 1.5, 2.0):
            adjusted = variance * variance_multiplier
            n = max(8, math.ceil((z_alpha + z_power) ** 2 * adjusted / effect**2))
            n = 4 * math.ceil(n / 4)
            rows.append({"effect_pp": effect, "variance_multiplier": variance_multiplier,
                         "n_windows": n, "ideal_days_at_four_per_day": n / 4,
                         "calendar_days_at_90pct_valid": math.ceil(n / 3.6)})
    return {
        "status": "planning_only", "endpoint": "fixed_model_minus_reversal",
        "adaptive_policy_power_established": False,
        "alpha_two_sided": alpha, "target_power": power,
        "pilot_n": len(data), "variance_estimates_pp2": variances,
        "planning_variance_pp2": variance, "scenarios": rows,
        "formula": "ceil_to_4((z(1-alpha/2)+z(power))^2 * variance / effect_pp^2)",
        "assumptions": ["future variance/dependence resembles the short pilot",
                        "normal approximation, not a calibrated block-bootstrap test",
                        "days assume four usable windows daily, not regime-specific arrivals",
                        "effects are planning assumptions, not the observed mean or guaranteed gains",
                        "adaptive-minus-unchanged is a different endpoint requiring its own pilot"],
    }


def reconstruct_regimes(signals, db_path=None):
    """Historical raw candles, acquired now; not contemporaneously saved labels."""
    times = pd.to_datetime(signals, utc=True)
    start, end = times.min() - pd.Timedelta(days=45), times.max()
    path = Path(db_path or config.General.DB_PATH)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)) as conn:
        frame = pd.read_sql_query(
            "SELECT timestamp, close FROM crypto_data WHERE market=? AND timestamp>=? "
            "AND timestamp<? ORDER BY timestamp", conn,
            params=["KRW-BTC", start.strftime("%Y-%m-%d"),
                    (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame[(frame.timestamp >= start) & (frame.timestamp < end)]
    if frame.empty:
        raise ValueError("No historical raw BTC candles; do not invent regime labels")
    closes = frame.set_index("timestamp").rename(columns={"close": "KRW-BTC"})
    panel = build_btc_context(closes)
    labels = {}
    for signal in times:
        at = signal - pd.Timedelta(hours=1)
        labels[signal.isoformat()] = str(panel.at[at, "regime"]) if at in panel.index else "unknown"
    return labels, {"status": "post_hoc_reconstructed_not_saved_at_signal",
                    "raw_rows": len(frame), "raw_sha256": trial._digest(frame.to_json(date_format="iso")),
                    "data_before": end.isoformat(), "future_candles_used": False,
                    "candle_revision_history_available": False}


def analyze_rows(protocol, observations, outcomes, regimes):
    """Input rows must have passed the sealed archive's semantic ledger audit."""
    snapshots = {s: json.loads(raw) for s, raw in observations}
    results = {s: json.loads(raw) for s, raw in outcomes}
    slots = [s.isoformat() for s in trial._slots(protocol)]
    if (set(snapshots) != set(slots) or set(results) != set(slots)
            or any(snapshots[s]["status"] != "recorded" or results[s]["status"] != "matured" for s in slots)):
        raise ValueError("Diagnostics require a complete trial; never silently drop windows")
    n, size = len(slots), protocol["basket_size"]
    windows, coins = [], {}
    selected_missing = 0
    for slot in slots:
        snapshot, result = snapshots[slot], results[slot]
        members = snapshot["rows"]
        prices = {p["market"]: p["raw_return"] for p in result["prices"]}
        model = {r["market"] for r in members if r["model_selected"]}
        baseline = {r["market"] for r in members if r["baseline_selected"]}
        live = {r["market"] for r in snapshot.get("live_recommendations", []) if r.get("actionable")}
        nets = {name: 100 * result["strategies"][name]["net_return"] for name in STRATEGIES}
        cost_pct = 100 * result["strategies"]["model_short5"]["cost_return"]
        window = {"signal_at": slot, "regime": regimes.get(slot, "unknown"),
                  "gate": snapshot.get("context", {}).get("short_gate", "unknown"),
                  "live_model_same": snapshot.get("context", {}).get("live_model_sha256") == protocol["model_sha256"],
                  "cutoff_gap_sigma": snapshot["cutoff_gap_sigma"],
                  "overlap_n": len(model & baseline), "live_overlap_n": len(model & live),
                  **nets, "paired_pp": nets["model_short5"] - nets["reversal_short5"]}
        windows.append(window)
        for member in members:
            market = member["market"]
            if market not in model | baseline:
                continue
            row = coins.setdefault(market, {"market": market, "model_count": 0, "baseline_count": 0,
                                           "model_mean_contribution_pp": 0., "paired_mean_contribution_pp": 0.})
            a, b = int(market in model), int(market in baseline)
            net = -100 * prices[market] - cost_pct
            row["model_count"] += a
            row["baseline_count"] += b
            row["model_mean_contribution_pp"] += a * net / size / n
            row["paired_mean_contribution_pp"] += (a - b) * net / size / n
            selected_missing += int(a and bool(member.get("missing_inputs")))
    model = np.array([r["model_short5"] for r in windows])
    paired = np.array([r["paired_pp"] for r in windows])

    def group(rows, label):
        return {"label": label, "n_windows": len(rows), "n_utc_dates": len({r["signal_at"][:10] for r in rows}),
                **{name + "_mean_pct": float(np.mean([r[name] for r in rows])) for name in STRATEGIES},
                "paired_mean_pp": float(np.mean([r["paired_pp"] for r in rows]))}

    grouped = {}
    for field in ("regime", "gate", "live_model_same"):
        grouped[field] = [group([r for r in windows if r[field] == key], str(key))
                          for key in sorted({r[field] for r in windows})]
    grouped["chronological_thirds"] = [group([windows[i] for i in ids], f"third_{j+1}")
                                       for j, ids in enumerate(np.array_split(np.arange(n), 3))]
    grouped["utc_signal_hour"] = [group([r for r in windows if pd.Timestamp(r["signal_at"]).hour == h], str(h))
                                  for h in (5, 11, 17, 23)]
    top3 = np.argsort(model)[-3:]
    positive = model[model > 0].sum()
    contributions = sorted(coins.values(), key=lambda r: r["model_mean_contribution_pp"], reverse=True)
    # Removing a contribution leaves that slot's weight in cash; never re-normalize winners.
    if not np.isclose(sum(c["model_mean_contribution_pp"] for c in contributions), model.mean()):
        raise ValueError("Coin contributions do not reconcile with basket mean")
    if not np.isclose(sum(c["paired_mean_contribution_pp"] for c in contributions), paired.mean()):
        raise ValueError("Paired contributions do not reconcile")
    return {
        "n_windows": n, "first_signal_at": slots[0], "last_signal_at": slots[-1],
        "strategies": {name: distribution([r[name] for r in windows]) for name in STRATEGIES},
        "paired": distribution(paired), "groups": grouped,
        "sensitivity": {
            "model_mean_without_best_three_windows_pct": float(np.delete(model, top3).mean()),
            "paired_mean_without_best_three_paired_windows_pp": float(np.sort(paired)[:-3].mean()),
            "best_three_share_positive_model_pnl": float(model[top3].clip(0).sum() / positive) if positive else None,
            "model_mean_minus_top_coin_contribution_pct": float(model.mean() - contributions[0]["model_mean_contribution_pp"]),
            "warning": "post-hoc influence checks, not an executable filtered strategy",
        },
        "selection": {"unique_model_coins": sum(r["model_count"] > 0 for r in contributions),
                      "mean_baseline_overlap_n": float(np.mean([r["overlap_n"] for r in windows])),
                      "mean_live_actionable_overlap_n": float(np.mean([r["live_overlap_n"] for r in windows])),
                      "selected_missing_input_rows": selected_missing, "selected_coin_rows": n * size},
        "cost_sensitivity": [{"additional_common_cost_bps": bps,
                              "model_mean_net_pct": float(model.mean() - bps / 100),
                              "paired_mean_pp": float(paired.mean())} for bps in (0, 10, 20, 30)],
        "uncertainty_sensitivity": [{"block_windows": length,
                                     "paired_ci95_pp": np.quantile(block_means(paired, length), [.025, .975]).tolist()}
                                    for length in (4, 8)],
        "coin_attribution": contributions,
        "windows": windows, "power": power_scenarios(paired),
    }


def build_diagnostics(root=trial.ROOT, db_path=None):
    root = Path(root)
    seal, archive = _sealed_archive(root)
    protocol, observations, outcomes = _read_ledger(archive / "output/prospective/ledger.sqlite")
    labels, provenance = reconstruct_regimes([s for s, _ in observations], db_path)
    report = analyze_rows(protocol, observations, outcomes, labels)
    final = _verified_document(archive / "output/experiment_supervision/final_review.json")
    report.update({"schema": 1, "status": "post_hoc_diagnostic", "confirmatory_evidence": False,
                   "generated_at": trial._utc().isoformat(), "experiment_id": protocol["experiment_id"],
                   "source_checkpoint": seal["checkpoint"], "source_ledger_sha256": seal["ledger_sha256"],
                   "protocol_sha256": trial._digest(protocol),
                   "analysis_source_sha256": artifact_sha256(Path(__file__)),
                   "analysis_sources_sha256": {
                       name: artifact_sha256(trial.ROOT / name) for name in (
                           "utils/trial_diagnostics.py", "utils/rotation_research.py", "scripts/analyze_completed_trial.py")},
                   "analysis_packages": {name: version(name) for name in ("numpy", "pandas", "scipy")},
                   "original_decision": final["decision"], "regime_provenance": provenance,
                   "method_sources": SOURCES,
                   "boundaries": ["no original protocol, outcome, forecast or final-review edits",
                                  "all subgroup and influence analyses are post-hoc, not causal",
                                  "Upbit paper proxy, not Bitget fills or account return",
                                  "no automatic live switching or promotion"]})
    return report
