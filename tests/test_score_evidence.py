import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils import score_evidence as evidence
from utils.experiment_supervisor import _verified_document


@pytest.fixture
def inputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in ("scripts/fetch_and_rank.py","scripts/track_ic.py","utils/score_evidence.py",
                 "data/features.py","data/database.py","models/xgb_ranker.py","config.py"):
        target = tmp_path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((root/name).read_bytes())
    x = pd.DataFrame({"a":[np.nan,.3],"b":[.2,.1]},index=["KRW-A","KRW-B"])
    scores = pd.Series([-.0123456789,.123456789],index=x.index)
    return tmp_path,x,scores


def test_original_missing_mask_full_precision_and_reruns_are_preserved(inputs):
    root,x,scores = inputs
    before = x.copy(deep=True)
    path = evidence.record_score_evidence("signal","2026-09-30T11:00Z",x,scores,"a"*64,
                                         root=root,now="2026-09-30T11:05Z")
    original = path.read_bytes()
    same = evidence.record_score_evidence("signal","2026-09-30T11:00Z",x,scores,"a"*64,
                                         root=root,now="2026-09-30T11:05Z")
    assert same == path and path.read_bytes() == original
    doc = _verified_document(path)
    assert doc["timely_signal"] and not doc["registered_trial_evidence"]
    assert doc["rows"][0]["inputs"]["a"] is None
    assert doc["rows"][0]["score"] == scores.iloc[0]
    later = evidence.record_score_evidence("signal","2026-09-30T11:00Z",x,scores*2,"a"*64,
                                          root=root,now="2026-09-30T11:07Z")
    assert later != path and path.read_bytes() == original
    pd.testing.assert_frame_equal(x,before)
    assert not (root/"output/ic_history.json").exists()
    assert not (root/"output/prospective").exists()
    assert not (root/"output/rotation_pilot").exists()


def test_measurement_is_explicitly_recomputed_not_original_signal_or_raw_price(inputs):
    root,x,scores = inputs
    path = evidence.record_score_evidence("measurement","2026-09-30T11:00Z",x,scores,"a"*64,
                                         actual=scores/2,root=root,now="2026-09-30T23:05Z")
    doc = _verified_document(path)
    assert not doc["timely_signal"]
    assert "recomputed" in doc["input_basis"]
    assert "not_raw_price_proof" in doc["target_basis"]
    assert doc["rows"][0]["actual"] == scores.iloc[0]/2


@pytest.mark.parametrize("change", ["future","outcome_leak","early_outcome","mismatch","infinite","fingerprint"])
def test_invalid_evidence_does_not_create_a_witness(inputs,change):
    root,x,scores = inputs
    kind,now,actual,digest = "signal","2026-09-30T11:05Z",None,"a"*64
    if change == "future":
        now = "2026-09-30T10:59Z"
    elif change == "outcome_leak":
        actual = scores/2
    elif change == "early_outcome":
        kind,actual = "measurement",scores/2
    elif change == "mismatch":
        scores = scores.iloc[::-1]
    elif change == "infinite":
        x.iloc[0,0] = float("inf")
    else:
        digest = "x"*64
    with pytest.raises(ValueError):
        evidence.record_score_evidence(kind,"2026-09-30T11:00Z",x,scores,digest,actual=actual,root=root,now=now)
    assert not (root/"output/score_evidence").exists()


def test_late_signal_is_labeled_late_and_never_backdated(inputs):
    root,x,scores = inputs
    path = evidence.record_score_evidence("signal","2026-09-30T11:00Z",x,scores,"a"*64,
                                         root=root,now="2026-09-30T12:00Z")
    assert not _verified_document(path)["timely_signal"]


def test_corrupt_existing_witness_is_not_silently_overwritten(inputs):
    root,x,scores = inputs
    path = evidence.record_score_evidence("signal","2026-09-30T11:00Z",x,scores,"a"*64,
                                         root=root,now="2026-09-30T11:05Z")
    path.write_text('{"document":{},"sha256":"bad"}')
    with pytest.raises(ValueError,match="checksum"):
        evidence.record_score_evidence("signal","2026-09-30T11:00Z",x,scores,"a"*64,
                                       root=root,now="2026-09-30T11:05Z")
    assert json.loads(path.read_text())["sha256"] == "bad"


def test_diagnostic_exports_are_private_only(tmp_path,monkeypatch):
    from utils import dashboard_export
    for name,body in (("ic_incident/summary.json",{"status":"post_hoc_diagnosis"}),
                      ("score_evidence/status.json",{"signal":{"n_coins":100}})):
        path = tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(body))
    monkeypatch.setattr(dashboard_export,"OUTPUT_DIR",tmp_path)
    private = dashboard_export.build_summary_payload()
    assert private["ic_incident"]["status"] == "post_hoc_diagnosis"
    assert private["score_evidence"]["signal"]["n_coins"] == 100
    public = dashboard_export.build_public_summary_payload()
    assert "ic_incident" not in public and "score_evidence" not in public


@pytest.mark.parametrize("fail_witness",[False,True])
def test_tracker_witness_failure_cannot_change_ic_or_duplicate_history(tmp_path,monkeypatch,fail_witness):
    from scripts import track_ic
    from models.xgb_ranker import XSecRanker

    model = tmp_path/"models/xsec_6h.pkl"
    model.parent.mkdir()
    model.write_bytes(b"test model")
    monkeypatch.setattr(track_ic,"__file__",str(tmp_path/"scripts/track_ic.py"))
    monkeypatch.setattr(track_ic.config.Data,"MIN_ROWS_PER_COIN",0)
    times = pd.date_range("2026-01-01T05:00Z",periods=25,freq="h")
    markets = [f"KRW-T{i:02d}" for i in range(12)]
    prices = pd.DataFrame(np.exp(np.arange(25)[:,None]*np.arange(1,13)[None,:]*.001),index=times,columns=markets)
    factors = pd.DataFrame([{"timestamp":t,"market":m,"a":i*.01,"b":np.nan if i==0 else .1}
                            for t in times for i,m in enumerate(markets)]).set_index(["timestamp","market"])
    class TestRanker:
        def predict(self,frame):
            assert not frame.isna().any().any()
            return frame.a.to_numpy()
    monkeypatch.setattr(XSecRanker,"load",lambda path:TestRanker())
    monkeypatch.setattr(track_ic,"load_and_pivot",lambda **kwargs:(prices,prices,prices,prices,prices))
    monkeypatch.setattr(track_ic,"load_binance_pivot",lambda *args,**kwargs:pd.DataFrame())
    monkeypatch.setattr(track_ic,"compute_unified_factors",lambda *args,**kwargs:factors)
    monkeypatch.setattr(track_ic,"crosssection_zscore",lambda frame,**kwargs:frame)
    monkeypatch.setattr(track_ic,"build_top_liquidity_universe_index",lambda *args,**kwargs:(
        factors.index,{"avg_selected":12.,"min_selected":12,"max_selected":12}))
    history = tmp_path/"history.json"
    monkeypatch.setattr(track_ic,"_history_path_for_side",lambda side:str(history))
    called = []
    def record(kind,signal_at,raw_inputs,scores,model_sha256,**kwargs):
        called.append(True)
        assert kind == "measurement" and raw_inputs.isna().any().any()
        assert scores.index.equals(kwargs["actual"].index)
        if fail_witness:
            raise OSError("simulated evidence disk failure")
    monkeypatch.setattr(evidence,"record_score_evidence",record)
    track_ic._track_side("short",45)
    first = history.read_bytes()
    assert json.loads(first)[0]["ic"] == 1.
    track_ic._track_side("short",45)
    assert history.read_bytes() == first and len(called) == 2
