import json
from pathlib import Path
import pickle
import sqlite3

import numpy as np
import pandas as pd
import pytest

from utils import ic_incident as audit
from utils import prospective as trial
from utils.model_release import artifact_sha256


@pytest.fixture
def window():
    signal = trial._utc("2026-09-29T11:00Z")
    markets = [f"KRW-T{i:02d}" for i in range(12)]
    scores = np.array([-.02,-.02,-.01,-.009,-.008,-.005,0,0,.01,.02,.03,.05])
    returns = np.array([.1,-.05,.04,.01,-.02,.02,0,.03,-.03,-.04,.08,.06])
    saved = pd.DataFrame({"timestamp":signal.isoformat(),"market":markets,"score":scores,
                          "selected":[i<5 for i in range(12)],"horizon_h":6,"model_sha256":"m"})
    prices = pd.DataFrame([{"timestamp":signal+pd.Timedelta(hours=h),"market":market,
                            "open":100. if h == 1 else 100*(1+returns[i])}
                           for i,market in enumerate(markets) for h in (1,7)])
    return signal,saved,prices


def test_saved_ties_and_rank_products_reconcile_exactly(window):
    signal,saved,prices = window
    row = audit.describe_saved(saved,prices,signal,-.1)
    assert sum(r["rank_product"] for r in row["rows"]) == pytest.approx(row["saved_ic"])
    assert sum(r["selected_net_contribution_pp"] for r in row["rows"]) == pytest.approx(row["selected_proxy_net_pct"])
    assert row["selected_count"] == 5 and row["n_coins"] == 12
    assert row["recorded_ic"] == -.1
    pd.testing.assert_frame_equal(saved,window[1])


@pytest.mark.parametrize("change", ["missing","duplicate","zero","negative","infinite"])
def test_prices_are_never_filled_and_membership_is_never_silently_shrunk(window,change):
    signal,saved,prices = window
    prices = prices.copy()
    if change == "missing":
        prices = prices.iloc[1:]
    elif change == "duplicate":
        prices = pd.concat([prices,prices.iloc[:1]])
    else:
        prices.loc[0,"open"] = {"zero":0.,"negative":-1.,"infinite":float("inf")}[change]
    with pytest.raises(ValueError,match="prices"):
        audit.describe_saved(saved,prices,signal)


@pytest.mark.parametrize("change", ["timestamp","duplicate","nonfinite","constant","selection","model"])
def test_invalid_saved_forecasts_are_not_reinterpreted(window,change):
    signal,saved,prices = window
    saved = saved.copy()
    if change == "duplicate":
        saved = pd.concat([saved,saved.iloc[:1]])
    elif change == "timestamp":
        saved.loc[0,"timestamp"] = (signal+pd.Timedelta(hours=1)).isoformat()
    elif change == "nonfinite":
        saved.loc[0,"score"] = float("nan")
    elif change == "constant":
        saved["score"] = 0.
    elif change == "selection":
        saved["selected"] = saved.selected.astype(object)
        saved.loc[0,"selected"] = "false"
    else:
        saved.loc[0,"model_sha256"] = "other"
    with pytest.raises(ValueError):
        audit.describe_saved(saved,prices,signal)


def test_replay_keeps_identical_model_inputs_and_separates_groups(window):
    signal,saved,prices = window
    saved = audit.saved_frame(saved,signal)
    inputs = pd.DataFrame({"a":saved.score,"b":1.},index=saved.index)
    inputs.loc[inputs.index[:2],"b"] = np.nan
    original = inputs.copy(deep=True)
    now,old = saved.score,saved.score * -1
    row = audit.compare_replay(saved,inputs,now,old,prices,signal)
    assert row["n_common"] == 12
    assert row["replay_ic"] == pytest.approx(-row["previous_model_ic"])
    assert row["current_previous_rank_correlation"] == pytest.approx(-1.)
    assert row["missing_input_groups"]["any_missing"] == {"n":2,"replay_ic":None}
    assert row["feature_missing_counts"]["b"] == 2
    assert row["rows"][0]["inputs"]["b"] is None
    pd.testing.assert_frame_equal(inputs,original)
    with pytest.raises(ValueError,match="identical membership"):
        audit.compare_replay(saved,inputs,now,old.iloc[::-1],prices,signal)


def test_constant_or_tiny_replay_is_undefined_not_a_zero_ic(window):
    signal,saved,prices = window
    saved = audit.saved_frame(saved,signal)
    inputs = pd.DataFrame({"a":saved.score},index=saved.index)
    row = audit.compare_replay(saved,inputs,saved.score*0,saved.score,prices,signal)
    assert row["replay_ic"] is None and row["replay_rounding_ic_delta"] is None
    row = audit.compare_replay(saved,inputs,saved.score*1e-8,saved.score,prices,signal)
    assert row["replay_ic"] is not None and row["replay_rounding_ic_delta"] is None


class IncidentRanker:
    _feature_names = ["a","b"]
    def __init__(self,scale=1.):
        self.scale = scale
    def predict(self,frame):
        return self.scale*(frame.a+frame.b).to_numpy()


def test_complete_diagnostic_is_read_only_and_summary_contains_no_coin_rows(tmp_path,monkeypatch):
    signal = trial._utc("2026-09-29T11:00Z")
    start = signal-pd.Timedelta(days=7)
    output = tmp_path/"output"
    output.mkdir()
    model = tmp_path/"models/xsec_6h.pkl"
    old = tmp_path/"models/archive/release_2026-09-27T200257/0_xsec_6h.pkl"
    old.parent.mkdir(parents=True)
    model.write_bytes(pickle.dumps(IncidentRanker()))
    old.write_bytes(pickle.dumps(IncidentRanker(.9)))
    for name in audit.SOURCES:
        target = tmp_path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
    db = tmp_path/"raw.sqlite"
    times = pd.date_range(start-pd.Timedelta(days=45),signal+pd.Timedelta(hours=13),freq="h")
    markets = ["KRW-BTC"]+[f"KRW-T{i:02d}" for i in range(11)]
    raw = pd.DataFrame([{"timestamp":ts.isoformat(),"market":m,
                          "open":100+np.sin(j/10+i)*10+i,"close":101+np.sin(j/10+i)*10+i}
                         for j,ts in enumerate(times) for i,m in enumerate(markets)])
    with sqlite3.connect(db) as conn:
        raw.to_sql("crypto_data",conn,index=False)
    anchors = pd.date_range(start,signal+pd.Timedelta(hours=6),freq="6h")
    history = []
    for ts in anchors:
        saved = pd.DataFrame({"timestamp":ts.isoformat(),"market":markets,"score":np.arange(12)*.003,
                              "horizon_h":6,"selected":[i<5 for i in range(12)],"model_sha256":artifact_sha256(model)})
        saved.to_csv(output/("predictions_"+ts.strftime("%Y%m%dT%H%M")+".csv"),index=False)
        history.append({"timestamp":ts.isoformat(),"ic":-.1,"contract_version":"absolute_open_lag1_anchors_v1"})
    history_path = output/"ic_history.json"
    history_path.write_text(json.dumps(history))
    factors = pd.DataFrame([{"timestamp":ts,"market":m,"a":i*.003,"b":0.001}
                             for ts in anchors for i,m in enumerate(markets)]).set_index(["timestamp","market"])
    monkeypatch.setattr(audit,"_inputs",lambda:factors)
    monkeypatch.setattr(audit,"cutoff_inputs",lambda *args:(factors,{"point_in_time_archive":False}))
    monkeypatch.setattr(audit.config.General,"DB_PATH",str(db))
    before = {p:artifact_sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    report = audit.build_incident(tmp_path,control_days=7,now=signal+pd.Timedelta(hours=15))
    assert report["controls"]["n"] == 28 and report["controls"]["missing_or_invalid"] == 0
    assert len(report["events"]) == 2 and not report["automatic_promotion"]
    assert {p:artifact_sha256(p) for p in before} == before
    private = audit.summary(report)
    assert "windows" not in private and "provenance" not in private
    assert "KRW-T" not in json.dumps(private)
    json.dumps(report,allow_nan=False)
    with pytest.raises(ValueError,match="fully matured"):
        audit.build_incident(tmp_path,control_days=7,now=signal)
    history_path.write_text(json.dumps(history+[history[-1]]))
    with pytest.raises(ValueError,match="unordered"):
        audit.build_incident(tmp_path,control_days=7,now=signal+pd.Timedelta(hours=15))


def test_cutoff_loader_excludes_future_rows_and_reveals_first_row_selection(tmp_path,monkeypatch):
    now = trial._utc("2026-09-29T11:00Z")
    times = pd.date_range(now-pd.Timedelta(days=45),now+pd.Timedelta(hours=1),freq="h")
    raw = pd.DataFrame([{"timestamp":ts.isoformat(),"market":m,"open":1.,"high":2.,"low":.5,"close":1.,"volume":1.}
                         for ts in times for m in ("KRW-BTC","KRW-TEST")])
    raw = raw[~((raw.market=="KRW-TEST")&(raw.timestamp==times[0].isoformat()))]
    db = tmp_path/"raw.sqlite"
    with sqlite3.connect(db) as conn:
        for table in ("crypto_data","binance_data"):
            raw.to_sql(table,conn,index=False)
    monkeypatch.setattr(audit,"_factors",lambda panels,bn:panels[0])
    first,meta = audit.cutoff_inputs(db,now)
    assert first.index.max() == now and list(first.columns) == ["KRW-BTC"]
    assert meta["complete_history_markets"] == 1
    later,later_meta = audit.cutoff_inputs(db,now+pd.Timedelta(hours=1))
    assert set(later.columns) == {"KRW-BTC","KRW-TEST"}
    assert later_meta["complete_history_markets"] == 2
