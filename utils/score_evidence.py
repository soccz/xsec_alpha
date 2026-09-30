"""Immutable diagnostic witnesses; never change scoring, gates or registered trials."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils import prospective as trial
from utils.experiment_supervisor import _envelope, _write
from utils.model_release import artifact_sha256
from utils.run_lock import run_lock


def record_score_evidence(kind, signal_at, raw_inputs, scores, model_sha256, *,
                          actual=None, root=trial.ROOT, now=None):
    if kind not in ("signal", "measurement"):
        raise ValueError("Unknown score evidence kind")
    signal, recorded = trial._utc(signal_at), trial._utc(now)
    if signal > recorded or not raw_inputs.index.is_unique or not scores.index.is_unique:
        raise ValueError("Invalid evidence time or membership")
    if not raw_inputs.index.equals(scores.index) or not np.isfinite(scores).all():
        raise ValueError("Inputs and scores must have identical finite membership")
    if np.isinf(raw_inputs.to_numpy(dtype=float)).any():
        raise ValueError("Infinite original inputs")
    if not isinstance(model_sha256,str) or len(model_sha256) != 64 or any(c not in "0123456789abcdef" for c in model_sha256):
        raise ValueError("Missing model fingerprint")
    if kind == "measurement":
        if (actual is None or not actual.index.equals(scores.index) or not np.isfinite(actual).all()
                or recorded < signal+pd.Timedelta(hours=7)):
            raise ValueError("Invalid measurement outcome")
    elif actual is not None:
        raise ValueError("A signal witness cannot contain future outcomes")
    root = Path(root)
    caller = "scripts/fetch_and_rank.py" if kind == "signal" else "scripts/track_ic.py"
    sources = (caller,"utils/score_evidence.py","data/features.py","data/database.py",
               "models/xgb_ranker.py","config.py")
    body = {
        "schema":1,"kind":kind,"signal_at":signal.isoformat(),"recorded_at":recorded.isoformat(),
        "model_sha256":model_sha256,"horizon_h":6,"source_sha256":{p:artifact_sha256(root/p) for p in sources},
        "timely_signal":kind == "signal" and signal <= recorded <= signal+pd.Timedelta(minutes=45),
        "input_basis":"contemporaneous_signal_inputs" if kind == "signal" else "recomputed_historical_inputs_at_measurement",
        "target_basis":None if kind == "signal" else "legacy_forward_filled_open_loader_lag1_not_raw_price_proof",
        "features":list(raw_inputs.columns),
        "rows":[{"market":str(m),"score":float(scores[m]),
                 "inputs":{c:float(raw_inputs.at[m,c]) if pd.notna(raw_inputs.at[m,c]) else None for c in raw_inputs},
                 "actual":float(actual[m]) if actual is not None else None} for m in raw_inputs.index],
        "changes_gate":False,"registered_trial_evidence":False,
    }
    digest = trial._digest(body)
    folder = root/"output/score_evidence"
    with run_lock("score_evidence",lock_dir=str(root/"logs/locks"),timeout_sec=5):
        path = folder/kind/signal.strftime("%Y%m%dT%H%M")/(digest+".json")
        if not path.exists():
            _write(path,_envelope(body),immutable=True)
        else:
            from utils.experiment_supervisor import _verified_document
            if _verified_document(path) != body:
                raise ValueError("Existing score evidence changed")
        status_path = folder/"status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        status[kind] = {"signal_at":body["signal_at"],"recorded_at":body["recorded_at"],
                        "n_coins":len(scores),"model_sha256":model_sha256,"sha256":digest,
                        "timely_signal":body["timely_signal"]}
        _write(status_path,status)
    return path
