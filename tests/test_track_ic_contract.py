"""Production-contract regressions for the live IC tracker."""

import pandas as pd

from scripts import track_ic


def test_live_anchor_hours_match_installed_timers():
    assert track_ic._anchor_hours_for_side("short") == (5, 11, 17, 23)
    assert track_ic._anchor_hours_for_side("long") == (11, 23)


def test_latest_complete_anchor_ignores_off_cycle_and_incomplete_rows():
    index = pd.to_datetime([
        "2026-01-01T11:00:00Z",
        "2026-01-01T17:00:00Z",
        "2026-01-01T23:00:00Z",
        "2026-01-02T11:00:00Z",
    ])
    returns = pd.DataFrame(
        {
            "A": [0.1, 0.2, 0.3, float("nan")],
            "B": [0.1, 0.2, 0.3, float("nan")],
            "C": [0.1, 0.2, 0.3, float("nan")],
            "D": [0.1, 0.2, 0.3, float("nan")],
            "E": [0.1, 0.2, 0.3, float("nan")],
        },
        index=index,
    )

    assert track_ic._latest_complete_anchor(returns, "long") == index[2]
    assert track_ic._latest_complete_anchor(returns, "short") == index[2]


def test_long_regime_contract_requires_both_strict_thresholds(monkeypatch):
    monkeypatch.setattr(track_ic.config.LongModel, "BTC_7D_RETURN_GATE", 0.0)
    monkeypatch.setattr(track_ic.config.LongModel, "BTC_30D_RETURN_FLOOR", -0.10)

    assert track_ic._long_regime_eligible({"btc_ret_7d": 0.01, "btc_ret_30d": 0.0})
    assert not track_ic._long_regime_eligible({"btc_ret_7d": 0.0, "btc_ret_30d": 0.0})
    assert not track_ic._long_regime_eligible({"btc_ret_7d": 0.01, "btc_ret_30d": -0.10})


def test_alerts_do_not_mix_legacy_and_new_contracts(capsys):
    history = [
        {"ic": -0.5},
        {"ic": -0.4},
        {
            "ic": 0.10,
            "contract_version": track_ic.IC_CONTRACT_VERSION,
        },
    ]

    track_ic._check_alerts(history, "long")

    assert capsys.readouterr().out == ""
