"""Prospective regime diagnostics for the live ranking system, never a switching gate."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from config import config
from utils import prospective as trial
from utils.eval_metrics import build_btc_context
from utils.experiment_supervisor import _write
from utils.model_release import artifact_sha256
from utils.run_lock import run_lock

ROOT = trial.ROOT
SOURCES = ("utils/regime_observer.py", "utils/eval_metrics.py", "utils/prospective.py")


def _path(root):
    return Path(root) / "output/regime_observation"


def _connect(root):
    return sqlite3.connect((_path(root) / "ledger.sqlite").resolve().as_uri() + "?mode=rw", uri=True, timeout=5)


def _contract(conn):
    payload, digest = conn.execute("SELECT payload, sha256 FROM protocol").fetchone()
    protocol = json.loads(payload)
    if trial._digest(protocol) != digest:
        raise ValueError("Regime observation protocol changed")
    return protocol


def load_btc_history(signal_at):
    """Read raw closed BTC candles; live feature pivots have already forward-filled gaps."""
    signal = trial._utc(signal_at)
    uri = Path(config.General.DB_PATH).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=5)) as conn:
        data = pd.read_sql_query(
            "SELECT timestamp, close FROM crypto_data WHERE market = ? AND timestamp >= ? AND timestamp < ? ORDER BY timestamp",
            conn, params=["KRW-BTC", (signal - pd.Timedelta(days=45)).strftime("%Y-%m-%d"),
                          (signal + pd.Timedelta(days=1)).strftime("%Y-%m-%d")])
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data.set_index("timestamp").rename(columns={"close": "KRW-BTC"}).loc[lambda df: df.index < signal]


def start_observation(root=ROOT, now=None):
    root = Path(root)
    folder = _path(root)
    with run_lock("regime_observation", lock_dir=str(root / "logs/locks")):
        if folder.exists():
            raise FileExistsError("Regime observation already registered; no in-place reset")
        current = trial._utc(now)
        first = current.floor("h") + pd.Timedelta(hours=1)
        while first.hour not in (5, 11, 17, 23):
            first += pd.Timedelta(hours=1)
        protocol = {
            "schema": 1, "registered_at": current.isoformat(), "first_signal_at": first.isoformat(),
            "purpose": "prospective descriptive diagnostics; not a confirmatory superiority test",
            "sources": {name: artifact_sha256(root / name) for name in SOURCES},
            "regimes": "BTC 7d return sign x 7d hourly volatility vs prior 30d median (min 7d)",
            "labels": "closed candles strictly before signal; unknown on gaps or insufficient warmup",
            "model_policy": "live ranking system; model hash saved every slot; no pooled single-model claim",
            "price_basis": "upbit_raw_hourly_open_proxy", "horizon_h": 6, "price_grace_hours": 48,
            "cost_bps": {"one_way_fee": config.Costs.ONE_WAY_FEE_BPS,
                         "one_way_slippage": config.Costs.SLIPPAGE_BPS,
                         "short_extra": config.Costs.SHORT_EXTRA_COST_BPS},
            "review": "weekly, using only outcomes recorded before Monday 00:00 UTC; last 90d",
            "review_min_windows_per_regime": 30, "review_min_days_per_regime": 8,
            "minimums_are_power_calculation": False,
            "automatic_switching": False, "automatic_promotion": False,
        }
        folder.mkdir(parents=True, mode=0o700)
        with closing(sqlite3.connect(folder / "ledger.sqlite")) as conn, conn:
            conn.executescript("CREATE TABLE protocol(payload TEXT NOT NULL, sha256 TEXT NOT NULL);"
                               "CREATE TABLE observations(slot TEXT PRIMARY KEY, payload TEXT NOT NULL);"
                               "CREATE TABLE outcomes(slot TEXT PRIMARY KEY, payload TEXT NOT NULL);")
            conn.execute("INSERT INTO protocol VALUES (?, ?)", (trial._json(protocol), trial._digest(protocol)))
        _write(folder / "protocol.json", protocol, immutable=True)
        return protocol


def record_observation(signal_at, closes, scores, baseline, model_sha256, *, root=ROOT, now=None):
    if not (_path(root) / "ledger.sqlite").exists():
        return "not_started"
    signal, current = trial._utc(signal_at), trial._utc(now)
    if (signal != signal.floor("h") or signal.hour not in (5, 11, 17, 23)
            or not signal <= current <= signal + pd.Timedelta(minutes=45)):
        return "outside_capture_window"
    if not model_sha256 or not scores.index.is_unique or not baseline.index.is_unique:
        raise ValueError("Missing model provenance or duplicate markets")
    frame = pd.DataFrame({"score": scores, "baseline_score": baseline}).sort_index()
    frame = frame[np.isfinite(frame).all(axis=1)]
    if len(frame) < 10 or frame["score"].std() < 0.0005:
        raise ValueError("Insufficient comparable scores")
    selected = frame["score"].sort_values(kind="stable").head(5).index
    control = frame["baseline_score"].sort_values(kind="stable").head(5).index
    frame["model_selected"] = frame.index.isin(selected)
    frame["baseline_selected"] = frame.index.isin(control)
    history = closes.loc[closes.index < signal]
    context = {"regime": "unknown", "data_asof": None}
    if not history.empty:
        panel = build_btc_context(history)
        label_at = signal - pd.Timedelta(hours=1)
        if label_at in panel.index:
            row = panel.loc[label_at]
            context = {"regime": row["regime"], "data_asof": label_at.isoformat(),
                       **{name: trial._number(row[name]) for name in (
                           "btc_ret_7d", "btc_ret_30d", "btc_vol_7d", "vol_reference")}}
    with run_lock("regime_observation", lock_dir=str(Path(root) / "logs/locks")):
        with closing(_connect(root)) as conn, conn:
            protocol = _contract(conn)
            if signal < trial._utc(protocol["first_signal_at"]):
                return "before_registration"
            if conn.execute("SELECT 1 FROM observations WHERE slot=?", (signal.isoformat(),)).fetchone():
                return "already_recorded"
            if any(artifact_sha256(Path(root) / name) != digest for name, digest in protocol["sources"].items()):
                raise ValueError("Regime observation code changed; review a new version")
            recorded = trial._utc(now)
            if recorded > signal + pd.Timedelta(minutes=45):
                return "outside_capture_window"
            entry = signal + pd.Timedelta(hours=1)
            evaluation = {"model_sha256": model_sha256, "cost_bps": protocol["cost_bps"],
                          "price_grace_hours": protocol["price_grace_hours"]}
            snapshot = {"status": "recorded", "signal_at": signal.isoformat(),
                        "recorded_at": recorded.isoformat(), "entry_at": entry.isoformat(),
                        "exit_at": (entry + pd.Timedelta(hours=6)).isoformat(),
                        "model_sha256": model_sha256, "evaluation": evaluation,
                        "protocol_sha256": trial._digest(evaluation), "regime": context,
                        "observation_protocol_sha256": trial._digest(protocol),
                        "rows": frame.rename_axis("market").reset_index().to_dict("records")}
            trial._append(conn, "observations", signal.isoformat(), snapshot)
    return "recorded"


def refresh_observation(root=ROOT, now=None, price_loader=None):
    if not (_path(root) / "ledger.sqlite").exists():
        return {"status": "not_started", "automatic_switching": False}
    current = trial._utc(now)
    with run_lock("regime_observation", lock_dir=str(Path(root) / "logs/locks")):
        with closing(_connect(root)) as conn, conn:
            protocol = _contract(conn)
            if any(artifact_sha256(Path(root) / name) != digest for name, digest in protocol["sources"].items()):
                raise ValueError("Regime observation code changed; review a new version")
            # Missed slots stay visible; no historical reconstruction of recommendations.
            for slot in pd.date_range(protocol["first_signal_at"], current, freq="6h"):
                if current > slot + pd.Timedelta(minutes=45):
                    trial._append(conn, "observations", slot.isoformat(), {"status": "missed"})
            observations = {slot: json.loads(raw) for slot, raw in conn.execute("SELECT slot, payload FROM observations ORDER BY slot")}
            outcomes = {slot: json.loads(raw) for slot, raw in conn.execute("SELECT slot, payload FROM outcomes ORDER BY slot")}
            pending = [(slot, row) for slot, row in observations.items() if row["status"] == "recorded"
                       and slot not in outcomes and trial._utc(row["exit_at"]) + pd.Timedelta(hours=1) <= current]
            if pending:
                prices = (price_loader or trial._raw_prices)(min(row["entry_at"] for _, row in pending),
                                                           max(row["exit_at"] for _, row in pending))
                for slot, row in pending:
                    if row["observation_protocol_sha256"] != trial._digest(protocol):
                        raise ValueError("Regime snapshot protocol mismatch")
                    result = trial.evaluate_snapshot(row, prices, row["evaluation"], now=current)
                    if result is not None:
                        trial._append(conn, "outcomes", slot, result)
                        outcomes[slot] = result
            summary = summarize(protocol, observations, outcomes, current)
        _write(_path(root) / "summary.json", summary)
        return summary


def summarize(protocol, observations, outcomes, now):
    current = trial._utc(now)
    review_at = current.normalize() - pd.Timedelta(days=current.weekday())
    groups = {}
    for slot, result in outcomes.items():
        if (result["status"] != "matured" or trial._utc(result["evaluated_at"]) > review_at
                or not review_at - pd.Timedelta(days=90) <= trial._utc(slot) < review_at):
            continue
        snapshot = observations[slot]
        name = snapshot["regime"]["regime"]
        groups.setdefault(name, []).append((slot, snapshot, result))
    regimes = []
    for name, rows in sorted(groups.items()):
        model = [r["strategies"]["model_short5"]["net_return"] for _, _, r in rows]
        baseline = [r["strategies"]["reversal_short5"]["net_return"] for _, _, r in rows]
        days = len({s[:10] for s, _, _ in rows})
        ready = (name != "unknown" and len(rows) >= protocol["review_min_windows_per_regime"]
                 and days >= protocol["review_min_days_per_regime"])
        regimes.append({"regime": name, "n_windows": len(rows), "n_days": days,
                        "model_versions": len({s["model_sha256"] for _, s, _ in rows}),
                        "model_net_pct": 100 * float(np.mean(model)),
                        "baseline_net_pct": 100 * float(np.mean(baseline)),
                        "paired_excess_pct": 100 * float(np.mean(np.array(model) - np.array(baseline))),
                        "review_status": "descriptive_review_only" if ready else "insufficient_data"})
    recorded = [row for row in observations.values() if row["status"] == "recorded"]
    return {"status": "observing", "checked_at": current.isoformat(), "protocol": protocol,
            "first_signal_at": protocol["first_signal_at"], "weekly_review_at": review_at.isoformat(),
            "automatic_switching": False, "confirmatory_evidence": False,
            "n_recorded": len(recorded), "n_matured": sum(r["status"] == "matured" for r in outcomes.values()),
            "n_missed": sum(r["status"] == "missed" for r in observations.values()),
            "n_invalid": sum(r["status"] != "matured" for r in outcomes.values()),
            "n_pending": len(recorded) - len(outcomes),
            "latest": {"signal_at": recorded[-1]["signal_at"], **recorded[-1]["regime"]} if recorded else None,
            "regimes": regimes}
