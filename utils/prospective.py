"""Preregistered, append-only SHORT ranking experiment; never places orders."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import config
from models.xgb_ranker import XSecRanker
from utils.logger import logger
from utils.model_release import artifact_sha256, model_release_guard
from utils.run_lock import run_lock

ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = (
    "config.py", "data/features.py", "data/database.py", "models/xgb_ranker.py",
    "scripts/fetch_and_rank.py", "utils/prospective.py",
)
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost")
STRATEGIES = ("model_short5", "reversal_short5", "universe_short")


def _utc(value=None):
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Missing timestamp")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _directory(root):
    return Path(root) / "output" / "prospective"


def _runtime_contract(root):
    return {
        "sources": {name: artifact_sha256(Path(root) / name) for name in SOURCE_FILES},
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "liquidity_top_n": config.Data.LIQUIDITY_TOP_N,
        "require_bitget": config.Portfolio.LIVE_REQUIRE_BITGET_TRADABLE,
    }


@contextmanager
def _db(root):
    path = _directory(root) / "ledger.sqlite"
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=rw", uri=True, timeout=5)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _manifest(conn):
    raw, checksum = conn.execute("SELECT payload, sha256 FROM protocol").fetchone()
    document = json.loads(raw)
    if _digest(document) != checksum:
        raise ValueError("Experiment protocol checksum mismatch")
    return document


def start_experiment(root=ROOT, now=None):
    """Freeze a copy, not production artifacts. Refuse to reset an existing trial."""
    root = Path(root)
    with model_release_guard(root), run_lock(
        "prospective", lock_dir=str(root / "logs/locks"), timeout_sec=5,
    ):
        destination = _directory(root)
        if destination.exists():
            raise FileExistsError("Experiment already exists; it cannot be restarted in place")
        started = _utc(now)
        first = started.floor("h") + pd.Timedelta(hours=1)
        while first.hour not in (5, 11, 17, 23):
            first += pd.Timedelta(hours=1)
        model_path = root / "models/xsec_6h.pkl"
        model = XSecRanker.load(str(model_path))
        features = list(model._feature_names)
        protocol = {
            "schema": 1, "experiment_id": started.strftime("short_rank_%Y%m%dT%H%M%SZ"),
            "registered_at": started.isoformat(), "first_signal_at": first.isoformat(),
            "n_slots": 60, "horizon_h": 6, "basket_size": 5,
            "capture_deadline_minutes": 45, "price_grace_hours": 48,
            "model_sha256": artifact_sha256(model_path), "features": features,
            "runtime": _runtime_contract(root),
            "cost_bps": {
                "one_way_fee": config.Costs.ONE_WAY_FEE_BPS,
                "one_way_slippage": config.Costs.SLIPPAGE_BPS,
                "short_extra": config.Costs.SHORT_EXTRA_COST_BPS,
            },
            "selection": "ascending score; alphabetical ties; equal-weight bottom five; no buffer or gates",
            "baseline": "ascending reversal_4h; same observed Bitget-tradable active universe",
            "missing_inputs": "existing live model inputs; original missing columns recorded per coin",
            "price_basis": "upbit_raw_hourly_open_proxy",
            "entry_rule": "signal hour + 1h, strictly after snapshot persistence; exit entry + 6h",
            "missing_prices": "require every frozen universe member; no filling, substitutions or renormalization",
            "primary_metric": "paired model_short5 minus reversal_short5 mean net return per window",
            "review_rule": "one review after 60 scheduled windows; all 60 paired windows required; no automatic promotion",
            "uncertainty": "circular moving-block bootstrap, four adjacent 6h windows, 2000 draws, seed 42; final review only",
            "sensitivity": "score minus score with one input set to contemporaneous universe median; not causal or additive attribution",
            "scope": "ranking-only shadow experiment, not live buffered/gated recommendations or Bitget fills",
        }
        if not all(np.isfinite(v) and v >= 0 for v in protocol["cost_bps"].values()):
            raise ValueError("Invalid experiment costs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".prospective-") as temp:
            stage = Path(temp) / "trial"
            stage.mkdir()
            shutil.copy2(model_path, stage / "model.pkl")
            if artifact_sha256(stage / "model.pkl") != protocol["model_sha256"]:
                raise ValueError("Frozen model copy mismatch")
            conn = sqlite3.connect(stage / "ledger.sqlite")
            try:
                with conn:
                    conn.execute("CREATE TABLE protocol(payload TEXT NOT NULL, sha256 TEXT NOT NULL)")
                    conn.execute("CREATE TABLE observations(slot TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                    conn.execute("CREATE TABLE outcomes(slot TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                    conn.execute("INSERT INTO protocol VALUES (?, ?)", (_json(protocol), _digest(protocol)))
            finally:
                conn.close()
            stage.rename(destination)
        return protocol


def _slots(protocol):
    return pd.date_range(protocol["first_signal_at"], periods=protocol["n_slots"], freq="6h")


def _append(conn, table, slot, document):
    if table not in ("observations", "outcomes"):
        raise ValueError("Invalid ledger table")
    conn.execute(f"INSERT OR IGNORE INTO {table} VALUES (?, ?)", (slot, _json(document)))


def _number(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def record_snapshot(
    signal_at, model_inputs, raw_inputs, tradable_markets, live_rows=None,
    context=None, root=ROOT, now=None,
):
    """Record predictions without reading outcomes. A duplicate slot is immutable."""
    if not _directory(root).exists():
        return "not_started"
    signal = _utc(signal_at)
    with _db(root) as conn:
        protocol = _manifest(conn)
        if signal not in _slots(protocol):
            return "outside_protocol"
        key = signal.isoformat()
        if conn.execute("SELECT 1 FROM observations WHERE slot=?", (key,)).fetchone():
            return "already_recorded"
        observed = _utc(now)
        if observed < signal or observed > signal + pd.Timedelta(minutes=protocol["capture_deadline_minutes"]):
            return "outside_capture_window"
        if protocol["runtime"] != _runtime_contract(root):
            raise ValueError("Experiment runtime changed; register a separately reviewed trial")
        model_path = _directory(root) / "model.pkl"
        if artifact_sha256(model_path) != protocol["model_sha256"]:
            raise ValueError("Frozen model changed")
        if not model_inputs.index.is_unique or not raw_inputs.index.is_unique:
            raise ValueError("Duplicate markets in snapshot")
        markets = sorted(set(tradable_markets) & set(model_inputs.index) & set(raw_inputs.index))
        frame = model_inputs.loc[markets, protocol["features"]].copy()
        baseline = raw_inputs.loc[markets, "reversal_4h"]
        # Exclusions depend only on information available now, never future returns.
        valid = np.isfinite(frame).all(axis=1) & np.isfinite(baseline)
        frame = frame.loc[valid]
        baseline = baseline.loc[valid]
        if len(frame) < 10:
            raise ValueError("Fewer than ten comparable markets")
        model = XSecRanker.load(str(model_path))
        scores = pd.Series(model.predict(frame), index=frame.index, dtype=float)
        if not np.isfinite(scores).all() or scores.std() < 0.0005:
            raise ValueError("Invalid or collapsed frozen scores")
        selected = scores.sort_values(kind="stable").head(5).index.tolist()
        control = baseline.sort_values(kind="stable").head(5).index.tolist()
        ranks = scores.rank(method="first", ascending=True)
        medians = frame.median()
        sensitivity = {}
        for feature in frame.columns:
            perturbed = frame.loc[selected].copy()
            perturbed[feature] = medians[feature]
            deltas = scores.loc[selected].to_numpy() - model.predict(perturbed)
            if not np.isfinite(deltas).all():
                raise ValueError("Invalid model sensitivity")
            sensitivity[feature] = dict(zip(selected, deltas))
        rows = []
        for market in frame.index:
            effects = sorted(
                ({"feature": f, "score_delta": float(sensitivity[f][market])}
                 for f in frame.columns), key=lambda row: abs(row["score_delta"]), reverse=True,
            ) if market in selected else []
            rows.append({
                "market": market, "score": float(scores[market]), "rank": int(ranks[market]),
                "baseline_score": float(baseline[market]),
                "model_selected": market in selected, "baseline_selected": market in control,
                "inputs": {f: float(frame.at[market, f]) for f in frame.columns},
                "missing_inputs": [f for f in frame.columns if pd.isna(raw_inputs.at[market, f])],
                "sensitivity": effects[:3],
            })
        observed = _utc(now)
        entry = signal + pd.Timedelta(hours=1)
        if observed > signal + pd.Timedelta(minutes=protocol["capture_deadline_minutes"]):
            return "outside_capture_window"
        ordered = scores.sort_values(kind="stable")
        document = {
            "status": "recorded", "signal_at": key, "recorded_at": observed.isoformat(),
            "entry_at": entry.isoformat(), "exit_at": (entry + pd.Timedelta(hours=6)).isoformat(),
            "model_sha256": protocol["model_sha256"], "protocol_sha256": _digest(protocol),
            "universe_n": len(rows), "excluded_input_n": len(markets) - len(rows),
            "rows": rows, "context": context or {},
            "cutoff_gap_sigma": float((ordered.iloc[5] - ordered.iloc[4]) / scores.std()),
            "live_recommendations": live_rows or [],
        }
        _append(conn, "observations", key, document)
    return "recorded"


def _raw_prices(start, end):
    """Read exact raw candles, including delisted names; no live-universe filter."""
    uri = Path(config.General.DB_PATH).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    try:
        # DB timestamps are ISO strings without timezone; accept the legacy space form too.
        return pd.read_sql_query(
            "SELECT timestamp, market, open FROM crypto_data "
            "WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp, market",
            conn, params=[_utc(start).strftime("%Y-%m-%d"),
                          (_utc(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")],
        )
    finally:
        conn.close()


def evaluate_snapshot(snapshot, prices, protocol, now=None):
    """No partial baskets: missing/duplicate/invalid candles make the pair ineligible."""
    if snapshot["protocol_sha256"] != _digest(protocol) or snapshot["model_sha256"] != protocol["model_sha256"]:
        raise ValueError("Snapshot provenance mismatch")
    if not (_utc(snapshot["signal_at"]) <= _utc(snapshot["recorded_at"]) < _utc(snapshot["entry_at"])):
        raise ValueError("Snapshot was not recorded before entry")
    current = _utc(now)
    exit_at = _utc(snapshot["exit_at"])
    if current < exit_at + pd.Timedelta(hours=1):
        return None
    rows = snapshot["rows"]
    markets = [row["market"] for row in rows]
    frame = prices.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame[frame["market"].isin(markets) & frame["timestamp"].isin([
        _utc(snapshot["entry_at"]), exit_at,
    ])]
    duplicate = frame.duplicated(["timestamp", "market"]).any()
    invalid = duplicate or len(frame) != 2 * len(markets)
    if not invalid:
        values = pd.to_numeric(frame["open"], errors="coerce")
        invalid = not (np.isfinite(values).all() and (values > 0).all())
    if invalid:
        if current < exit_at + pd.Timedelta(hours=protocol["price_grace_hours"]):
            return None
        return {"status": "invalid_prices", "evaluated_at": current.isoformat(),
                "observed_price_rows": len(frame), "required_price_rows": 2 * len(markets),
                "duplicate_prices": bool(duplicate)}
    matrix = frame.pivot(index="timestamp", columns="market", values="open").astype(float)
    entry = matrix.loc[_utc(snapshot["entry_at"]), markets]
    exit_prices = matrix.loc[exit_at, markets]
    returns = exit_prices / entry - 1.0
    costs = protocol["cost_bps"]
    cost = (2 * (costs["one_way_fee"] + costs["one_way_slippage"]) + costs["short_extra"]) / 10000
    selections = {
        "model_short5": [row["market"] for row in rows if row["model_selected"]],
        "reversal_short5": [row["market"] for row in rows if row["baseline_selected"]],
        "universe_short": markets,
    }
    results = {}
    for name, selected in selections.items():
        gross = -float(returns.loc[selected].mean())
        results[name] = {"gross_return": gross, "cost_return": cost, "net_return": gross - cost}
    actual = returns.loc[markets].to_numpy()
    model_scores = [row["score"] for row in rows]
    baseline_scores = [row["baseline_score"] for row in rows]
    has_variance = np.std(actual) > 0
    return {
        "status": "matured", "evaluated_at": current.isoformat(), "strategies": results,
        "model_ic": _number(spearmanr(model_scores, actual).statistic) if has_variance else None,
        "baseline_ic": _number(spearmanr(baseline_scores, actual).statistic)
        if has_variance and np.std(baseline_scores) > 0 else None,
        "prices": [{"market": m, "entry_price": float(entry[m]), "exit_price": float(exit_prices[m]),
                    "raw_return": float(returns[m])} for m in markets],
    }


def advance_experiment(root=ROOT, now=None, price_loader=None):
    """Account for missed slots and mature saved forecasts; never reconstruct them."""
    if not _directory(root).exists():
        return
    current = _utc(now)
    with _db(root) as conn:
        protocol = _manifest(conn)
        for slot in _slots(protocol):
            if current > slot + pd.Timedelta(minutes=protocol["capture_deadline_minutes"]):
                _append(conn, "observations", slot.isoformat(), {
                    "status": "missed", "signal_at": slot.isoformat(),
                    "recorded_at": current.isoformat(), "reason": "no timely valid snapshot",
                })
        pending = conn.execute(
            "SELECT o.slot, o.payload FROM observations o LEFT JOIN outcomes r ON o.slot=r.slot "
            "WHERE r.slot IS NULL ORDER BY o.slot"
        ).fetchall()
        snapshots = [(slot, json.loads(raw)) for slot, raw in pending]
        snapshots = [(slot, row) for slot, row in snapshots if row["status"] == "recorded"
                     and _utc(row["exit_at"]) + pd.Timedelta(hours=1) <= current]
        if not snapshots:
            return
        loader = price_loader or _raw_prices
        prices = loader(min(_utc(s["entry_at"]) for _, s in snapshots),
                        max(_utc(s["exit_at"]) for _, s in snapshots))
        for slot, snapshot in snapshots:
            result = evaluate_snapshot(snapshot, prices, protocol, now=current)
            if result is not None:
                _append(conn, "outcomes", slot, result)


def _block_interval(values):
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(data), size=(2000, int(np.ceil(len(data) / 4))))
    indices = ((starts[:, :, None] + np.arange(4)) % len(data)).reshape(2000, -1)[:, :len(data)]
    return np.quantile(data[indices].mean(axis=1), [0.025, 0.975]).tolist()


def experiment_summary(root=ROOT):
    if not _directory(root).exists():
        return {"status": "not_started"}
    with _db(root) as conn:
        protocol = _manifest(conn)
        observations = {slot: json.loads(raw) for slot, raw in conn.execute(
            "SELECT slot, payload FROM observations ORDER BY slot")}
        outcomes = {slot: json.loads(raw) for slot, raw in conn.execute(
            "SELECT slot, payload FROM outcomes ORDER BY slot")}
    valid = [(slot, row) for slot, row in outcomes.items() if row["status"] == "matured"]
    missed = sum(row["status"] == "missed" for row in observations.values())
    invalid = sum(row["status"] != "matured" for row in outcomes.values())
    terminal = len(outcomes) + missed
    final = terminal == protocol["n_slots"]
    complete = len(valid) == protocol["n_slots"]
    strategies = []
    for name in STRATEGIES:
        values = [row["strategies"][name]["net_return"] for _, row in valid]
        strategies.append({"name": name, "n_windows": len(values),
                           "mean_net_pct": 100 * float(np.mean(values)) if values else None})
    paired = [row["strategies"]["model_short5"]["net_return"]
              - row["strategies"]["reversal_short5"]["net_return"] for _, row in valid]
    recorded = [row for row in observations.values() if row["status"] == "recorded"]
    latest = recorded[-1] if recorded else None
    by_context = {}
    for slot, outcome in valid:
        gate = str(observations[slot].get("context", {}).get("short_gate", "unknown"))
        by_context.setdefault(gate, []).append(outcome["strategies"]["model_short5"]["net_return"])
    recent_windows = []
    for slot, observation in reversed(list(observations.items())):
        outcome = outcomes.get(slot, {})
        realized = {r["market"]: r["raw_return"] for r in outcome.get("prices", [])}
        model_result = outcome.get("strategies", {}).get("model_short5", {})
        control_result = outcome.get("strategies", {}).get("reversal_short5", {})
        recent_windows.append({
            "signal_at": slot, "status": outcome.get("status", observation["status"]),
            "short_gate": observation.get("context", {}).get("short_gate"),
            "model_net_pct": 100 * model_result["net_return"] if model_result else None,
            "baseline_net_pct": 100 * control_result["net_return"] if control_result else None,
            "evidence": [{"market": row["market"], "sensitivity": row["sensitivity"],
                          "net_pct": 100 * (-realized[row["market"]] - model_result["cost_return"])
                          if row["market"] in realized else None}
                         for row in observation.get("rows", []) if row["model_selected"]],
        })
    return {
        "status": "ready_for_review" if complete else "incomplete_evidence" if final else "collecting",
        "runtime_matches": protocol["runtime"] == _runtime_contract(root),
        "frozen_model_matches": artifact_sha256(_directory(root) / "model.pkl") == protocol["model_sha256"],
        "protocol": protocol, "n_scheduled": protocol["n_slots"], "n_recorded": len(recorded),
        "n_matured": len(valid), "n_missed": missed, "n_invalid": invalid,
        "n_pending": len(recorded) - len(outcomes), "strategies": strategies,
        "paired_excess_pct": 100 * float(np.mean(paired)) if paired else None,
        "paired_ci95_pct": [100 * v for v in _block_interval(paired)] if complete else None,
        "model_ic_mean": _number(pd.Series([r["model_ic"] for _, r in valid], dtype=float).mean()),
        "baseline_ic_mean": _number(pd.Series([r["baseline_ic"] for _, r in valid], dtype=float).mean()),
        "context_breakdown": [{"short_gate": gate, "n_windows": len(values),
                               "mean_net_pct": 100 * float(np.mean(values))}
                              for gate, values in sorted(by_context.items())],
        "recent_windows": recent_windows,
        "latest": {"signal_at": latest["signal_at"], "recorded_at": latest["recorded_at"],
                   "entry_at": latest["entry_at"], "exit_at": latest["exit_at"],
                   "cutoff_gap_sigma": latest["cutoff_gap_sigma"],
                   "rows": [row for row in latest["rows"] if row["model_selected"]]} if latest else None,
    }


def maintain_experiment(root=ROOT):
    """An experiment failure must never prevent Telegram or dashboard publication."""
    try:
        with run_lock("prospective", lock_dir=str(Path(root) / "logs/locks"), timeout_sec=1):
            advance_experiment(root=root)
    except (Exception, SystemExit):
        logger.exception("Prospective maintenance failed; recommendation delivery unchanged")
