"""Read-only incident reconstruction, never a replacement for registered IC evidence."""
from contextlib import closing
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import config
from utils import prospective as trial
from utils.eval_metrics import build_btc_context
from utils.model_release import artifact_sha256

ROOT = trial.ROOT
COST_RETURN = .003
SOURCES = ("utils/ic_incident.py", "scripts/diagnose_ic.py", "scripts/track_ic.py", "utils/score_evidence.py",
           "scripts/fetch_and_rank.py", "data/features.py", "data/database.py",
           "data/collector.py", "models/xgb_ranker.py", "utils/magnitude.py", "config.py")


def ic(scores, returns):
    frame = pd.concat([scores.rename("score"), returns.rename("return")], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 10 or frame.score.nunique() < 2 or frame["return"].nunique() < 2:
        return None
    return float(spearmanr(frame.score, frame["return"]).statistic)


def exact_returns(prices, signal, markets, horizon=6):
    signal = trial._utc(signal)
    times = [signal + pd.Timedelta(hours=h) for h in (1, horizon + 1)]
    frame = prices[prices.timestamp.isin(times) & prices.market.isin(markets)]
    if frame.duplicated(["timestamp", "market"]).any():
        raise ValueError("Duplicate raw entry/exit prices")
    wide = frame.pivot(index="timestamp", columns="market", values="open").reindex(index=times, columns=markets)
    if not (np.isfinite(wide.to_numpy()) & (wide.to_numpy() > 0)).all():
        raise ValueError("Incomplete or invalid raw prices; do not shrink the pool")
    return wide.iloc[1] / wide.iloc[0] - 1, wide


def saved_frame(frame, signal):
    frame = frame.loc[frame.horizon_h == 6].copy()
    if (len(frame) < 10 or frame.market.duplicated().any()
            or not (pd.to_datetime(frame.timestamp, utc=True) == trial._utc(signal)).all()
            or not np.isfinite(frame.score).all()):
        raise ValueError("Invalid saved forecast membership, timestamp or score")
    if frame.score.nunique() < 2:
        raise ValueError("Constant saved scores")
    if "selected" in frame and not frame.selected.isin([True, False]).all():
        raise ValueError("Invalid saved selection flags")
    return frame.set_index("market").sort_index()


def rank_contributions(scores, returns):
    if not scores.index.equals(returns.index) or not np.isfinite(returns).all():
        raise ValueError("Rank contribution membership mismatch")
    a, b = scores.rank(method="average"), returns.rank(method="average")
    a, b = a - a.mean(), b - b.mean()
    scale = float(np.sqrt((a*a).sum() * (b*b).sum()))
    if scale <= 0:
        raise ValueError("Constant ranks")
    return a * b / scale


def describe_saved(frame, prices, signal, recorded_ic=None):
    frame = saved_frame(frame, signal)
    returns, matrix = exact_returns(prices, signal, frame.index)
    score = frame.score
    contributions = rank_contributions(score, returns)
    bottom = score.nsmallest(5).index
    selected = frame.index[frame.selected] if "selected" in frame else pd.Index([])
    worst = contributions.nsmallest(3).index
    selected_contributions = 100*(-returns.loc[selected]-COST_RETURN)/len(selected) if len(selected) else pd.Series(dtype=float)
    model_hashes = sorted(frame.model_sha256.dropna().unique()) if "model_sha256" in frame else []
    if len(model_hashes) > 1:
        raise ValueError("Mixed model hashes in one saved forecast")
    return {
        "status": "evaluated_saved_archive", "signal_at": trial._utc(signal).isoformat(),
        "n_coins": len(frame), "recorded_ic": recorded_ic, "saved_ic": ic(score, returns),
        "model_sha256": model_hashes[0] if model_hashes else None,
        "score_mean": float(score.mean()), "score_std": float(score.std()),
        "pool_return_pct": float(100 * returns.mean()), "pool_up_fraction": float((returns > 0).mean()),
        "bottom5_net_pct": float(100 * (-returns.loc[bottom].mean() - COST_RETURN)),
        "selected_count": len(selected),
        "selected_proxy_net_pct": float(100 * (-returns.loc[selected].mean() - COST_RETURN)) if len(selected) else None,
        "worst_selected_contribution_pp": float(selected_contributions.min()) if len(selected) else None,
        "other_selected_contribution_pp": float(selected_contributions.sum()-selected_contributions.min()) if len(selected) else None,
        "ic_without_three_worst_contributors": ic(score.drop(worst), returns.drop(worst)),
        "worst_rank_contributors": contributions.nsmallest(5).index.tolist(),
        "rows": [{"market": m, "score": float(score[m]), "entry_open": float(matrix.iloc[0][m]),
                  "exit_open": float(matrix.iloc[1][m]), "return_pct": float(100*returns[m]),
                  "rank_product": float(contributions[m]), "bottom5": m in bottom, "selected": m in selected,
                  "selected_net_contribution_pp": float(100*(-returns[m]-COST_RETURN)/len(selected)) if m in selected else 0.}
                 for m in frame.index],
    }


def compare_replay(saved, inputs, current_scores, old_scores, prices, signal):
    """Separate input/universe drift from model replacement on IDENTICAL reconstructed inputs."""
    if not inputs.index.equals(current_scores.index) or not inputs.index.equals(old_scores.index):
        raise ValueError("Counterfactual models must use identical membership")
    if not np.isfinite(current_scores).all() or not np.isfinite(old_scores).all():
        raise ValueError("Invalid replay scores")
    returns, _ = exact_returns(prices, signal, inputs.index)
    common = saved.index.intersection(inputs.index)
    if len(common) < 10:
        raise ValueError("Insufficient common saved/replay membership")
    current_common = current_scores.loc[common]
    old_common = old_scores.loc[common]
    actual = returns.loc[common]
    missing = inputs.isna().any(axis=1)
    groups = {}
    for name, members in (("any_missing", inputs.index[missing]), ("complete_inputs", inputs.index[~missing])):
        groups[name] = {"n": len(members), "replay_ic": ic(current_scores.loc[members], returns.loc[members])}
    return {
        "replay_ic": ic(current_scores, returns), "previous_model_ic": ic(old_scores, returns),
        "n_replay": len(inputs), "n_common": len(common),
        "added_to_replay": sorted(set(inputs.index) - set(saved.index)),
        "removed_from_replay": sorted(set(saved.index) - set(inputs.index)),
        "saved_common_ic": ic(saved.score.loc[common], actual),
        "replay_common_ic": ic(current_common, actual), "previous_common_ic": ic(old_common, actual),
        "saved_replay_rank_correlation": ic(saved.score.loc[common], current_common),
        "current_previous_rank_correlation": ic(current_common, old_common),
        "saved_vs_rounded_replay_max_error": float((saved.score.loc[common] - current_common.round(5)).abs().max()),
        "replay_rounding_ic_delta": (ic(current_scores.round(5), returns) - ic(current_scores, returns)
                                     if ic(current_scores.round(5), returns) is not None and ic(current_scores, returns) is not None else None),
        "current_bottom5_net_pct": float(100*(-returns.loc[current_scores.nsmallest(5).index].mean()-COST_RETURN)),
        "previous_bottom5_net_pct": float(100*(-returns.loc[old_scores.nsmallest(5).index].mean()-COST_RETURN)),
        "missing_input_groups": groups,
        "feature_missing_counts": {c: int(inputs[c].isna().sum()) for c in inputs},
        "input_basis": "reconstructed_now_not_saved_at_signal",
        "rows": [{"market": m, "current_score": float(current_scores[m]), "previous_score": float(old_scores[m]),
                  "return_pct": float(returns[m]*100),
                  "inputs": {c: float(inputs.at[m,c]) if pd.notna(inputs.at[m,c]) else None for c in inputs}}
                 for m in inputs.index],
    }


def _raw(db_path, start, end, btc_only=False):
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri()+"?mode=ro", uri=True, timeout=5)) as conn:
        query = "SELECT timestamp,market,open,close FROM crypto_data WHERE timestamp>=? AND timestamp<?"
        if btc_only:
            query += " AND market='KRW-BTC'"
        frame = pd.read_sql_query(query+" ORDER BY timestamp,market", conn,
                                  params=[start.strftime("%Y-%m-%d"), (end+pd.Timedelta(days=1)).strftime("%Y-%m-%d")])
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    return frame[(frame.timestamp >= start) & (frame.timestamp <= end)].copy()


def _inputs():
    from data.features import load_and_pivot, load_binance_pivot
    panels = load_and_pivot(days=45)
    warmup = config.Data.MIN_ROWS_PER_COIN
    panels = tuple(frame.iloc[warmup:] for frame in panels)
    bn = load_binance_pivot(panels[0].columns.tolist(), days=45)
    if not bn.empty:
        bn = bn.iloc[warmup:]
    return _factors(panels, bn)


def _factors(panels, bn):
    from data.features import (compute_unified_factors, crosssection_zscore,
                              build_top_liquidity_universe_index, filter_long_frame_by_universe,
                              UNIFIED_CALENDAR_COLS)
    factors = compute_unified_factors(*panels, binance_closes=bn)
    factors = crosssection_zscore(factors, cols=[c for c in factors if c not in UNIFIED_CALENDAR_COLS])
    universe, _ = build_top_liquidity_universe_index(panels[0], panels[4])
    return filter_long_frame_by_universe(factors, universe)


def cutoff_inputs(db_path, asof):
    """Mirror the loader with a fixed historical cutoff; still current-DB reconstruction."""
    asof = trial._utc(asof)
    start = asof-pd.Timedelta(days=45)
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri()+"?mode=ro",uri=True)) as conn:
        conn.execute("BEGIN")
        raw = {}
        for table in ("crypto_data","binance_data"):
            frame = pd.read_sql_query(
                f"SELECT timestamp,market,open,high,low,close,volume FROM {table} WHERE timestamp>=? AND timestamp<? ORDER BY timestamp,market",
                conn,params=[start.strftime("%Y-%m-%d"),(asof+pd.Timedelta(days=1)).strftime("%Y-%m-%d")])
            frame["timestamp"] = pd.to_datetime(frame.timestamp,utc=True)
            raw[table] = frame[(frame.timestamp>=start)&(frame.timestamp<=asof)]
    frame = raw["crypto_data"]
    last = frame.groupby("market").timestamp.max()
    markets = sorted(m for m in last.index if m.startswith("KRW-") and last[m]>=asof-pd.Timedelta(days=7)
                     and not any(ex in m[4:] for ex in config.Data.DYNAMIC_UNIVERSE_EXCLUDE))
    frame = frame[frame.market.isin(markets)]
    panels = tuple(frame.pivot(index="timestamp",columns="market",values=c).ffill()
                   for c in ("close","open","high","low","volume"))
    valid = panels[0].columns[panels[0].notna().all()]
    warmup = config.Data.MIN_ROWS_PER_COIN
    panels = tuple(panel.loc[:,valid].iloc[warmup:] for panel in panels)
    bn = raw["binance_data"]
    bn = bn[bn.market.isin(valid)].drop_duplicates(["timestamp","market"])
    bn = bn.pivot(index="timestamp",columns="market",values="close").ffill().iloc[warmup:]
    return _factors(panels,bn), {
        "asof":asof.isoformat(),"fresh_markets":len(markets),"complete_history_markets":len(valid),
        "removed_by_first_row_completeness":sorted(set(markets)-set(valid)),
        "raw_sha256":{name:trial._digest(data.to_json(date_format="iso")) for name,data in raw.items()},
        "point_in_time_archive":False,
    }


def build_incident(root=ROOT, control_days=28, now=None):
    """Call under the existing stable-data and model-release locks."""
    from models.xgb_ranker import XSecRanker

    root = Path(root)
    if not isinstance(control_days,int) or control_days<7:
        raise ValueError("Controls must cover at least the 7-day factor diagnostic")
    history_path = root / "output/ic_history.json"
    history_hash = artifact_sha256(history_path)
    history = json.loads(history_path.read_text())
    contract = next(r["contract_version"] for r in reversed(history) if r.get("contract_version"))
    aligned = [r for r in history if r.get("contract_version") == contract]
    if contract != "absolute_open_lag1_anchors_v1":
        raise ValueError("Unsupported IC contract")
    if any(trial._utc(b["timestamp"]) <= trial._utc(a["timestamp"]) for a,b in zip(aligned, aligned[1:])):
        raise ValueError("Duplicate or unordered IC history")
    events = aligned[-2:]
    if len(events) != 2 or any(not np.isfinite(r["ic"]) or r["ic"] >= .03 for r in events):
        raise ValueError("Expected two latest low-IC windows; no arbitrary event selection")
    first, last = [trial._utc(r["timestamp"]) for r in events]
    if last-first != pd.Timedelta(hours=6) or trial._utc(now) < last+pd.Timedelta(hours=8):
        raise ValueError("Incident anchors must be consecutive and fully matured")
    start = first-pd.Timedelta(days=control_days)
    end = last+pd.Timedelta(hours=7)
    prices = _raw(config.General.DB_PATH, start, end)
    btc = _raw(config.General.DB_PATH, start-pd.Timedelta(days=45), last-pd.Timedelta(hours=1), True)
    context = build_btc_context(btc.set_index("timestamp")[["close"]].rename(columns={"close":"KRW-BTC"}))
    recorded = {trial._utc(r["timestamp"]):r["ic"] for r in aligned}
    windows, hashes = [], {}
    for signal in pd.date_range(start, last, freq="6h"):
        path = root / "output" / ("predictions_"+signal.strftime("%Y%m%dT%H%M")+".csv")
        row = {"signal_at":signal.isoformat(), "status":"missing_forecast", "recorded_ic":recorded.get(signal)}
        if path.exists():
            hashes[str(path.relative_to(root))] = artifact_sha256(path)
            try:
                row = describe_saved(pd.read_csv(path), prices, signal, recorded.get(signal))
            except ValueError as exc:
                row.update(status="invalid_evidence", reason=str(exc))
        at = signal-pd.Timedelta(hours=1)
        row["regime"] = str(context.at[at,"regime"]) if at in context.index else "unknown"
        row["btc_ret_7d_pct"] = float(100*context.at[at,"btc_ret_7d"]) if at in context.index and pd.notna(context.at[at,"btc_ret_7d"]) else None
        row["role"] = "event" if signal >= first else "control"
        windows.append(row)
    focal = [r for r in windows if r["role"] == "event"]
    if any(r["status"] != "evaluated_saved_archive" for r in focal):
        raise ValueError("Focal forecasts/prices incomplete; do not infer a cause")
    for prior,row in zip(windows, windows[1:]):
        if "rows" in prior and "rows" in row:
            current = pd.DataFrame(row["rows"]).set_index("market")
            previous = {r["market"] for r in prior["rows"]}
            common = current.index.intersection(sorted(previous))
            row["pool_turnover"] = {"n_common":len(common), "n_added":len(current)-len(common),
                                     "n_removed":len(previous)-len(common),
                                     "current_ic_on_common":ic(current.loc[common,"score"],current.loc[common,"return_pct"])}
    factors = _inputs()
    model_path = root / config.Model.MODEL_PATH
    model_hash = artifact_sha256(model_path)
    if any(row["model_sha256"] != model_hash for row in focal):
        raise ValueError("Current model differs from event model; use archived event artifact explicitly")
    releases = sorted((root / "models/archive").glob("release_*/0_xsec_6h.pkl"))
    if not releases:
        raise ValueError("No archived predecessor model")
    previous_path = releases[-1]
    if pd.Timestamp(previous_path.parent.name.replace("release_", ""), tz="UTC") >= first:
        raise ValueError("Predecessor release must predate the incident")
    current, previous = XSecRanker.load(str(model_path)), XSecRanker.load(str(previous_path))
    for path in (model_path, previous_path):
        hashes[str(path.relative_to(root))] = artifact_sha256(path)
    factor_history = {name: [] for name in current._feature_names}
    for signal in pd.date_range(first-pd.Timedelta(days=7), first-pd.Timedelta(hours=6), freq="6h"):
        inputs = factors.xs(signal,level="timestamp")[current._feature_names]
        returns,_ = exact_returns(prices,signal,inputs.index)
        for name in inputs:
            value = ic(inputs[name],returns)
            if value is not None:
                factor_history[name].append(value)
    for event in focal:
        signal = trial._utc(event["signal_at"])
        inputs = factors.xs(signal,level="timestamp")[current._feature_names]
        inputs = inputs.loc[inputs.fillna(0.).any(axis=1)]
        neutral = inputs.fillna(0.)
        new_score = pd.Series(current.predict(neutral),index=inputs.index)
        old_score = pd.Series(previous.predict(neutral),index=inputs.index)
        saved = pd.DataFrame(event["rows"]).set_index("market")
        event["replay"] = compare_replay(saved,inputs,new_score,old_score,prices,signal)
        returns,_ = exact_returns(prices,signal,inputs.index)
        event["replay"]["recorded_ic_difference"] = event["replay"]["replay_ic"]-event["recorded_ic"]
        event["replay"]["feature_checks"] = [{
            "feature":name, "valid_coins":int(inputs[name].notna().sum()),
            "event_factor_ic":ic(inputs[name],returns),
            "prior_7d_mean_factor_ic":float(np.mean(factor_history[name])) if factor_history[name] else None,
            "prior_slots":len(factor_history[name]),
            "neutralized_model_ic":ic(pd.Series(current.predict(neutral.assign(**{name:0.})),index=inputs.index),returns),
        } for name in inputs if name not in ("dow_bull","hour_vol")]
        event["replay"]["components"] = {
            name:ic(pd.Series(getattr(current,attr).predict(neutral),index=inputs.index),returns)
            for name,attr in (("ridge","_ridge"),("lgbm","_lgbm")) if hasattr(current,attr)}
        for name,cutoff in (("signal_cutoff",signal),("tracker_cutoff",signal+pd.Timedelta(hours=12))):
            bounded,metadata = cutoff_inputs(config.General.DB_PATH,cutoff)
            bounded = bounded.xs(signal,level="timestamp")[current._feature_names]
            bounded = bounded.loc[bounded.fillna(0.).any(axis=1)]
            now_scores = pd.Series(current.predict(bounded.fillna(0.)),index=bounded.index)
            old_scores = pd.Series(previous.predict(bounded.fillna(0.)),index=bounded.index)
            result = compare_replay(saved,bounded,now_scores,old_scores,prices,signal)
            result["loader"] = metadata
            result["recorded_ic_difference"] = result["replay_ic"]-event["recorded_ic"]
            event[name] = result
    controls = [r for r in windows if r["role"] == "control" and r["status"] == "evaluated_saved_archive"]
    def summarize(rows):
        return {"n":len(rows), "mean_saved_ic":float(np.mean([r["saved_ic"] for r in rows])) if rows else None,
                "mean_bottom5_net_pct":float(np.mean([r["bottom5_net_pct"] for r in rows])) if rows else None,
                "n_low_ic":sum(r["saved_ic"] < .03 for r in rows)}
    by_regime = {name:summarize([r for r in controls if r["regime"] == name])
                 for name in sorted({r["regime"] for r in controls})}
    by_model = {name:summarize([r for r in controls if (r["model_sha256"] or "unknown") == name])
                for name in sorted({r["model_sha256"] or "unknown" for r in controls})}
    if artifact_sha256(history_path) != history_hash or any(artifact_sha256(root/name) != value for name,value in hashes.items()):
        raise ValueError("Incident evidence changed during diagnosis")
    return {"schema":1, "generated_at":trial._utc(now).isoformat(), "status":"post_hoc_diagnosis",
            "contract":contract, "event_start":first.isoformat(), "event_end":last.isoformat(),
            "controls":{**summarize(controls),"scheduled":len(windows)-2,"missing_or_invalid":len(windows)-2-len(controls)},
            "by_regime":by_regime, "by_model":by_model, "events":focal, "windows":windows,
            "current_model_sha256":model_hash, "previous_model_path":str(previous_path.relative_to(root)),
            "previous_model_sha256":hashes[str(previous_path.relative_to(root))],
            "provenance":{"history_sha256":history_hash,"inputs":hashes,
                          "raw_prices_sha256":trial._digest(prices.to_json(date_format="iso")),
                          "raw_btc_sha256":trial._digest(btc.to_json(date_format="iso")),
                          "sources":{name:artifact_sha256(root/name) for name in SOURCES},
                          "packages":{name:version(name) for name in ("numpy","pandas","scipy","scikit-learn","lightgbm")}},
            "limitations":["Post-hoc event-selected diagnosis; two windows do not establish an economic cause.",
                           "Saved CSVs are not append-only; scores rounded to 5 decimals; no original input snapshots for these events.",
                           "Reconstructed inputs/regimes use the current DB, not a point-in-time raw-data archive.",
                           "Counterfactual old/new models share reconstructed inputs; not a prospective challenger test.",
                           "Subset ICs and leave-three-out results are descriptive, not additive causal effects or deployable filters.",
                           "Selected basket uses next-open Upbit proxy and 30bp assumed cost, not realized Bitget fills.",
                           "No IC history, live gates, models, recommendations or registered experiments changed."],
            "automatic_promotion":False,"live_policy_changed":False}


def summary(report):
    result = {k:v for k,v in report.items() if k not in ("windows","events","provenance")}
    result["events"] = []
    for event in report["events"]:
        row = {k:v for k,v in event.items() if k not in ("rows","worst_rank_contributors","replay","signal_cutoff","tracker_cutoff")}
        for name in ("replay","signal_cutoff","tracker_cutoff"):
            row[name] = {k:v for k,v in event[name].items() if k not in ("rows","added_to_replay","removed_from_replay","loader")}
            if "loader" in event[name]:
                row[name]["loader"] = {k:v for k,v in event[name]["loader"].items() if k != "removed_by_first_row_completeness"}
        result["events"].append(row)
    result["report_sha256"] = trial._digest(report)
    return result
