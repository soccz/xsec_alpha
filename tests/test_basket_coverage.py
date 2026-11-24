from contextlib import closing
from datetime import timedelta
from io import BytesIO
import json
import sqlite3
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts import basket_coverage as audit


def fixture(tmp_path):
    output = tmp_path / "output"
    backups = tmp_path / "backups"
    backups.mkdir()
    # The pilot is already empty/closed. The next observer slot must still be audited.
    for folder in ("regime_observation", "rotation_pilot"):
        path = output / folder
        path.mkdir(parents=True)
        with closing(sqlite3.connect(path / "ledger.sqlite")) as db:
            for table in ("observations", "outcomes"):
                db.execute(f"CREATE TABLE {table}(slot TEXT PRIMARY KEY,payload TEXT)")
            db.execute("CREATE TABLE protocol(payload TEXT)")
            db.execute(
                "INSERT INTO protocol VALUES (?)",
                (json.dumps({"weight_change_bps": 10}),),
            )
            db.commit()
    slot = "2027-01-01T00:00:00+00:00"
    entry, exit_at = "2027-01-01T01:00:00", "2027-01-01T07:00:00"
    rows = [
        {"market": str(i), "model_selected": i < 5, "baseline_selected": 4 <= i < 9}
        for i in range(10)
    ]
    obs = dict(
        rows=rows,
        status="recorded",
        entry_at=entry,
        exit_at=exit_at,
        recorded_at=slot,
        regime={"regime": "bull_lowvol"},
        evaluation={
            "cost_bps": {"one_way_fee": 5, "one_way_slippage": 5, "short_extra": 10}
        },
    )
    outcome = dict(
        status="invalid_prices",
        evaluated_at="2027-01-03T07:00:00Z",
        observed_price_rows=19,
        required_price_rows=20,
    )
    with (
        closing(sqlite3.connect(output / "regime_observation/ledger.sqlite")) as db,
        db,
    ):
        db.execute("INSERT INTO observations VALUES (?,?)", (slot, json.dumps(obs)))
        db.execute("INSERT INTO outcomes VALUES (?,?)", (slot, json.dumps(outcome)))
        db.execute(
            "INSERT INTO observations VALUES (?,?)",
            ("2027-01-01T06:00:00+00:00", json.dumps({"status": "missed"})),
        )
    path = backups / "crypto_data_20270104T000000Z_001.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(
            "CREATE TABLE crypto_data(timestamp TEXT,market TEXT,open REAL, PRIMARY KEY(timestamp,market))"
        )
        db.executemany(
            "INSERT INTO crypto_data VALUES (?,?,?)",
            [
                (stamp, str(i), 100 if stamp == entry else 90)
                for stamp in (entry, exit_at)
                for i in range(10)
                if not (stamp == exit_at and i == 9)
            ],
        )
        db.execute("INSERT INTO crypto_data VALUES ('2027-01-04T00:00:00','0',95)")
    return backups, path, audit.utc("2027-01-04T01:00:00Z"), slot, exit_at


@pytest.mark.parametrize(
    "damage,expected",
    [
        (None, "basket_only"),
        ("selected", "basket_incomplete"),
        ("duplicate", "basket_incomplete"),
        ("invalid", "basket_incomplete"),
    ],
)
def test_independent_prices_and_idempotence(tmp_path, damage, expected):
    backups, path, now, slot, end = fixture(tmp_path)
    with closing(sqlite3.connect(path)) as raw, raw:
        if damage == "selected":
            raw.execute(
                "DELETE FROM crypto_data WHERE timestamp=? AND market='4'", (end,)
            )
        elif damage == "duplicate":
            raw.execute(
                "INSERT INTO crypto_data VALUES (?,'4',90)", (end.replace("T", " "),)
            )
        elif damage == "invalid":
            raw.execute(
                "UPDATE crypto_data SET open=0 WHERE timestamp=? AND market='4'", (end,)
            )
    for _ in range(2):
        report = audit.run(tmp_path, backups, 0, now)
        assert report["observer"] == {expected: 1, "missed": 1}
        assert report["pilot_invalid"] == {}
        assert report["pilot_backup_comparison"]["mean_difference_pp"] is None
    with closing(
        sqlite3.connect(tmp_path / "output/basket_coverage/audit.sqlite")
    ) as db:
        assert db.execute("SELECT count(*) FROM prices").fetchone()[0] == 20
        ret, official = db.execute(
            "SELECT model_net_pct,official FROM slots WHERE slot=?", (slot,)
        ).fetchone()
        assert official == "invalid_prices"
        if damage:
            assert ret is None
        else:
            assert ret == pytest.approx(9.7)


def test_lag_and_stale_backup_do_not_freeze_false_gaps(tmp_path):
    backups, path, now, _, _ = fixture(tmp_path)
    with closing(sqlite3.connect(path)) as raw, raw:
        raw.execute("DELETE FROM crypto_data WHERE timestamp > '2027-01-01T06:00:00'")
    report = audit.run(tmp_path, backups, 0, now)
    assert report["pending"] == 1 and report["observer"] == {"missed": 1}
    with pytest.raises(ValueError, match="backup timestamp"):
        audit.run(tmp_path, backups, 0, now + timedelta(days=3))


def test_corrupt_slots_are_isolated_even_after_partial_price_writes(tmp_path):
    backups, _, now, slot, _ = fixture(tmp_path)
    with (
        closing(
            sqlite3.connect(tmp_path / "output/regime_observation/ledger.sqlite")
        ) as db,
        db,
    ):
        row = json.loads(
            db.execute(
                "SELECT payload FROM observations WHERE slot=?", (slot,)
            ).fetchone()[0]
        )
        del row["evaluation"]
        db.execute(
            "UPDATE observations SET payload=? WHERE slot=?", (json.dumps(row), slot)
        )
        db.execute("INSERT INTO observations VALUES ('2026-12-31T18:00:00Z','{')")
    for _ in range(2):
        report = audit.run(tmp_path, backups, 0, now)
        assert report["observer"] == {"quarantined": 2, "missed": 1}
        assert len(report["quarantine_recent"]) == 2
    with closing(
        sqlite3.connect(tmp_path / "output/basket_coverage/audit.sqlite")
    ) as db:
        assert db.execute("SELECT count(*) FROM prices").fetchone()[0] == 0
        assert db.execute(
            "SELECT length(observation_sha256) FROM quarantine"
        ).fetchall() == [(64,), (64,)]


def test_database_failure_is_not_a_bad_input_slot(tmp_path, monkeypatch):
    backups, _, now, _, _ = fixture(tmp_path)

    def storage_failure(*args):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(audit, "inspect_slot", storage_failure)
    with pytest.raises(sqlite3.OperationalError, match="I/O"):
        audit.run(tmp_path, backups, 0, now)
    assert not (tmp_path / "output/basket_coverage/summary.json").exists()


def test_probe_never_backfills_prices(tmp_path, monkeypatch):
    backups, _, now, _, _ = fixture(tmp_path)
    audit.run(tmp_path, backups, 0, now)
    with closing(
        sqlite3.connect(tmp_path / "output/basket_coverage/audit.sqlite")
    ) as db:
        monkeypatch.setattr(audit.time, "sleep", lambda _: None)
        audit.probe_gaps(db, 5, now, lambda *_: ("available_now", 88))
        audit.probe_gaps(db, 5, now, lambda *_: pytest.fail("duplicate probe"))
        assert db.execute("SELECT result,price,tries FROM probes").fetchone() == (
            "available_now",
            88,
            1,
        )
        assert db.execute(
            "SELECT price FROM prices WHERE status='absent'"
        ).fetchone() == (None,)


def test_pilot_absent_does_not_stop_audit_or_claim_zero(tmp_path):
    backups, _, now, _, _ = fixture(tmp_path)
    pilot = tmp_path / "output/rotation_pilot/ledger.sqlite"
    pilot.rename(pilot.with_suffix(".archived"))
    report = audit.run(tmp_path, backups, 0, now)
    assert report["pilot_invalid"] is None and report["pilot_backup_comparison"] is None
    assert report["observer"]["basket_only"] == 1
    with closing(
        sqlite3.connect(tmp_path / "output/basket_coverage/recovery.sqlite")
    ) as copy:
        assert copy.execute("SELECT count(*) FROM prices").fetchone() == (20,)


def test_shadow_difference_uses_saved_weights_and_change_cost(tmp_path):
    backups, _, now, slot, _ = fixture(tmp_path)
    with (
        closing(
            sqlite3.connect(tmp_path / "output/rotation_pilot/ledger.sqlite")
        ) as pilot,
        pilot,
    ):
        pilot.execute(
            "INSERT INTO observations VALUES (?,?)",
            (
                slot,
                json.dumps(
                    {
                        "status": "recorded",
                        "weights": {"reversal": 0.25},
                        "previous_reversal_weight": 0,
                    }
                ),
            ),
        )
        pilot.execute(
            "INSERT INTO outcomes VALUES (?,?)", (slot, '{"status":"invalid_prices"}')
        )
    report = audit.run(tmp_path, backups, 0, now)
    assert report["pilot_invalid"] == {"basket_only": 1}
    pair = report["pilot_backup_comparison"]
    assert pair["paired"] == pair["exercised"] == 1
    assert pair["mean_difference_pp"] == pytest.approx(-0.025)


def test_rate_limit_is_unknown_and_does_not_repair_or_spin(tmp_path):
    backups, _, now, _, _ = fixture(tmp_path)
    audit.run(tmp_path, backups, 0, now)

    def limited(*_):
        raise HTTPError("https://example.invalid", 429, "limit", None, None)

    with closing(
        sqlite3.connect(tmp_path / "output/basket_coverage/audit.sqlite")
    ) as db:
        audit.probe_gaps(db, 30, now, limited)
        audit.probe_gaps(db, 30, now, lambda *_: pytest.fail("same-day retry"))
        assert db.execute("SELECT result,detail FROM probe_attempts").fetchone() == (
            "unknown",
            "http_429",
        )


@pytest.mark.parametrize(
    "returned,expected",
    [
        ("2026-11-01T10:00:00", ("available_now", 42)),
        ("2026-11-01T09:00:00", ("not_returned_now", None)),
        (None, ("not_returned_now", None)),
    ],
)
def test_exchange_probe_checks_exact_utc_hour(monkeypatch, returned, expected):
    def download(url, timeout):
        query = parse_qs(urlsplit(url).query)
        assert query == {
            "market": ["KRW-X"],
            "count": ["1"],
            "to": ["2026-11-01T11:00:00+00:00"],
        }
        assert timeout == 4
        data = (
            []
            if returned is None
            else [
                {
                    "market": "KRW-X",
                    "candle_date_time_utc": returned,
                    "opening_price": 42,
                }
            ]
        )
        return BytesIO(json.dumps(data).encode())

    monkeypatch.setattr(audit, "urlopen", download)
    assert audit.probe("KRW-X", "2026-11-01T10:00:00") == expected


def test_daily_exchange_budget_survives_process_restart(tmp_path):
    backups, _, now, _, _ = fixture(tmp_path)
    audit.run(tmp_path, backups, 0, now)
    path = tmp_path / "output/basket_coverage/audit.sqlite"
    with closing(sqlite3.connect(path)) as db, db:
        db.executemany(
            "INSERT INTO probes VALUES (?,?,?,'unknown',NULL,1)",
            [(str(i), "old", now.isoformat()) for i in range(30)],
        )
    with closing(sqlite3.connect(path)) as db:
        audit.probe_gaps(db, 30, now, lambda *_: pytest.fail("exceeded daily limit"))
