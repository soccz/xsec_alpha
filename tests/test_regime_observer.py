import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.eval_metrics import build_btc_context
from utils import regime_observer as observer


def btc_history(end="2026-09-07T05:00Z"):
    index = pd.date_range(end=end, periods=24 * 45, freq="h")
    prices = 100 * np.exp(np.cumsum(0.001 * np.sin(np.arange(len(index)) / 7)))
    return pd.DataFrame({"KRW-BTC": prices}, index=index)


def test_labels_cannot_change_when_future_candles_are_appended():
    closes = btc_history()
    prefix = closes.iloc[:800]
    pd.testing.assert_frame_equal(build_btc_context(prefix), build_btc_context(closes).loc[prefix.index])


def test_gaps_are_unknown_not_implicitly_filled_or_compressed():
    closes = btc_history().drop(btc_history().index[-4])
    context = build_btc_context(closes)
    assert context.iloc[-1]["regime"] == "unknown"
    assert pd.isna(context.iloc[-1]["btc_vol_7d"])


@pytest.fixture
def registered(tmp_path):
    source = Path(__file__).resolve().parents[1]
    for name in observer.SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / name).read_bytes())
    protocol = observer.start_observation(tmp_path, now="2026-09-07T04:00Z")
    return tmp_path, protocol


def record(root, now="2026-09-07T05:05Z", signal="2026-09-07T05:00Z"):
    markets = [f"KRW-T{i:02d}" for i in range(10)]
    scores = pd.Series(np.arange(10) * 0.01, index=markets)
    return observer.record_observation(signal, btc_history(end=signal), scores, -scores,
                                       "model-v1", root=root, now=now)


def prices(start, end):
    return pd.DataFrame([{"timestamp": stamp, "market": f"KRW-T{i:02d}",
                          "open": 100 if stamp == start else 90 + i}
                         for stamp in (start, end) for i in range(10)])


def test_registration_and_slots_cannot_be_reset_or_backfilled(registered):
    root, _ = registered
    with pytest.raises(FileExistsError):
        observer.start_observation(root)
    assert record(root, now="2026-09-07T06:00Z") == "outside_capture_window"
    assert record(root) == "recorded"
    assert record(root) == "already_recorded"
    with observer._connect(root) as conn:
        saved = json.loads(conn.execute("SELECT payload FROM observations").fetchone()[0])
    assert saved["regime"]["data_asof"] == "2026-09-07T04:00:00+00:00"
    assert saved["regime"]["regime"] != "unknown"


def test_code_change_blocks_new_observations(registered):
    root, _ = registered
    (root / observer.SOURCES[0]).write_text("changed")
    with pytest.raises(ValueError, match="code changed"):
        record(root)


def test_maturity_and_weekly_review_never_use_future_outcomes(registered):
    root, _ = registered
    record(root)
    early = observer.refresh_observation(root, now="2026-09-07T12:59Z", price_loader=prices)
    assert early["n_matured"] == 0
    settled = observer.refresh_observation(root, now="2026-09-07T13:00Z", price_loader=prices)
    assert settled["n_matured"] == 1 and settled["n_missed"] == 1
    assert settled["regimes"] == []
    later = observer.refresh_observation(root, now="2026-09-14T00:00Z", price_loader=prices)
    assert later["regimes"][0]["n_windows"] == 1
    assert later["regimes"][0]["paired_excess_pct"] == pytest.approx(5.0)
    assert later["regimes"][0]["review_status"] == "insufficient_data"
    assert later["automatic_switching"] is False and later["confirmatory_evidence"] is False


def test_missing_prices_never_shrink_baskets(registered):
    root, _ = registered
    record(root)
    def missing(start, end):
        return prices(start, end).iloc[1:]
    pending = observer.refresh_observation(root, now="2026-09-07T13:00Z", price_loader=missing)
    assert pending["n_pending"] == 1 and pending["n_matured"] == 0
    final = observer.refresh_observation(root, now="2026-09-09T12:00Z", price_loader=missing)
    assert final["n_invalid"] == 1 and final["n_matured"] == 0
