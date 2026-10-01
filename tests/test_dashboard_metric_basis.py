from datetime import datetime, timedelta, timezone

import pytest

from utils import dashboard_export as export


def test_basket_gross_cost_net_share_weighting():
    now = datetime.now(timezone.utc)
    first, second = now.isoformat(), (now - timedelta(hours=6)).isoformat()
    rows = [
        {"side": "SHORT", "entry_time": first, "realized_pct": 2, "net_pct": 1.8},
        {"side": "SHORT", "entry_time": first, "realized_pct": 4, "net_pct": 3.6},
        {"side": "SHORT", "entry_time": second, "realized_pct": -1, "net_pct": -1.5},
    ]
    stats = export._realized_stats(rows, "SHORT", [30])["d30"]
    assert stats["n"] == 3 and stats["n_windows"] == 2
    assert stats["avg_gross_per_window"] == 1
    assert stats["avg_cost_per_window"] == .4
    assert stats["avg_net_per_window"] == .6
    assert stats["avg"] != stats["avg_gross_per_window"]
    assert stats["avg_gross_per_window"] - stats["avg_cost_per_window"] == pytest.approx(stats["avg_net_per_window"])


def test_missing_and_legacy_cost_basis():
    empty = export._realized_stats([], "SHORT", [30])["d30"]
    assert empty["n"] == 0
    for key in ("avg_net_per_window", "avg_gross_per_window", "avg_cost_per_window"):
        assert empty[key] is None
    stats = export._realized_stats([
        {"side": "SHORT", "entry_time": datetime.now(timezone.utc).isoformat(), "realized_pct": 0},
    ], "SHORT", [30])["d30"]
    assert stats["avg_gross_per_window"] == 0
    assert stats["avg_cost_per_window"] == export.COST_PCT_BY_SIDE["SHORT"]
    assert stats["avg_net_per_window"] == -stats["avg_cost_per_window"]


def test_export_generation_is_shared_but_unique(monkeypatch):
    monkeypatch.setattr(export, "build_summary_payload", lambda: {"asof": "summary"})
    monkeypatch.setattr(export, "build_history_payload", lambda **kwargs: {"asof": "history", **kwargs})
    monkeypatch.setattr(export, "build_accuracy_payload", lambda **kwargs: {"asof": "accuracy", **kwargs})
    first = export.build_dashboard_payloads(history_days=12, ic_days=7)
    second = export.build_dashboard_payloads()
    assert len({p["export_id"] for p in first.values()}) == 1
    assert first["summary.json"]["export_id"] != second["summary.json"]["export_id"]
    assert first["history.json"]["history_days"] == 12
    assert first["history.json"]["ic_days"] == 7
