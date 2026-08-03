"""Regression tests for collection pacing and source ordering."""

import sys

import pytest


def test_upbit_market_collection_default_pacing_is_below_rate_limit():
    from config import config

    assert config.Data.COLLECTOR_MARKET_SLEEP_SEC == pytest.approx(0.15)
    assert 1 / config.Data.COLLECTOR_MARKET_SLEEP_SEC < 10


def test_upbit_market_collection_uses_configured_pacing(monkeypatch):
    from data import collector

    calls = []
    monkeypatch.setattr(collector, "init_db", lambda: None)
    monkeypatch.setattr(
        collector,
        "get_all_krw_markets",
        lambda: ["KRW-BTC", "KRW-ETH"],
    )
    monkeypatch.setattr(
        collector,
        "collect_market_data",
        lambda market, days: calls.append((market, days)),
    )
    monkeypatch.setattr(
        collector.config.Data,
        "COLLECTOR_MARKET_SLEEP_SEC",
        0.15,
    )
    sleeps = []
    monkeypatch.setattr(collector.time, "sleep", sleeps.append)

    collector.run_all(days=7)

    assert calls == [("KRW-BTC", 7), ("KRW-ETH", 7)]
    assert sleeps == [0.15, 0.15]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["update_data.py"], ["init", "binance", "upbit"]),
        (["update_data.py", "--upbit-only"], ["init", "upbit"]),
        (["update_data.py", "--binance-only"], ["init", "binance"]),
    ],
)
def test_update_data_preserves_single_modes_and_orders_full_update_last_upbit(
    monkeypatch,
    tmp_path,
    argv,
    expected,
):
    from scripts import update_data

    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(update_data, "init_db", lambda: calls.append("init"))
    monkeypatch.setattr(
        update_data,
        "update_binance",
        lambda days: calls.append("binance"),
    )
    monkeypatch.setattr(
        update_data,
        "upbit_run_all",
        lambda days: calls.append("upbit"),
    )

    update_data.main()

    assert calls == expected
