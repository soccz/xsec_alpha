"""Regression tests for update/read serialization in the systemd pipeline."""

import os
import sqlite3
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


def test_update_data_lock_conflict_is_nonzero(monkeypatch, tmp_path):
    """An overlapping updater must stop ExecStartPre, not look successful."""
    from scripts import update_data

    lock_dir = tmp_path / "logs" / "locks"
    lock_dir.mkdir(parents=True)
    (lock_dir / "update_data.lock").write_text(str(os.getpid()))

    def must_not_run(*_args, **_kwargs):
        pytest.fail("collection ran despite a live update_data lock")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["update_data.py"])
    monkeypatch.setattr(update_data, "init_db", must_not_run)
    monkeypatch.setattr(update_data, "upbit_run_all", must_not_run)
    monkeypatch.setattr(update_data, "update_binance", must_not_run)
    monkeypatch.setattr(update_data, "_UPDATE_LOCK_WAIT_SEC", 0.01)

    with pytest.raises(SystemExit) as exc:
        update_data.main()

    assert exc.value.code == os.EX_TEMPFAIL


@pytest.mark.parametrize("peer_ready", [True, False])
def test_update_data_waits_for_peer_then_validates(
    monkeypatch, tmp_path, peer_ready
):
    """A released peer lock is reused only after its DB passes validation."""
    from scripts import update_data

    lock_dir = tmp_path / "logs" / "locks"
    lock_dir.mkdir(parents=True)
    lock_path = lock_dir / "update_data.lock"
    lock_path.write_text(str(os.getpid()))

    def must_not_run(*_args, **_kwargs):
        pytest.fail("duplicate collection ran after waiting for a peer")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["update_data.py"])
    monkeypatch.setattr(update_data, "_UPDATE_LOCK_WAIT_SEC", 1.0)
    monkeypatch.setattr(
        update_data,
        "_peer_update_is_complete",
        lambda: (peer_ready, "fixture result"),
    )
    monkeypatch.setattr(update_data, "init_db", must_not_run)
    monkeypatch.setattr(update_data, "upbit_run_all", must_not_run)
    monkeypatch.setattr(update_data, "update_binance", must_not_run)

    release = threading.Timer(0.05, lock_path.unlink)
    release.start()
    try:
        if peer_ready:
            update_data.main()
        else:
            with pytest.raises(SystemExit) as exc:
                update_data.main()
            assert exc.value.code == os.EX_TEMPFAIL
    finally:
        release.join()


def test_peer_update_validation_rejects_partial_binance_markets(
    monkeypatch, tmp_path
):
    """A fresh table-wide MAX is insufficient when most symbols are stale."""
    from scripts import update_data

    db_path = tmp_path / "peer.db"
    now = datetime.now(timezone.utc).replace(microsecond=0)
    latest = now.strftime("%Y-%m-%dT%H:%M:%S")
    stale = (now - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%S")

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE crypto_data (timestamp TEXT, market TEXT)")
    conn.execute("CREATE TABLE binance_data (timestamp TEXT, market TEXT)")
    conn.execute("INSERT INTO crypto_data VALUES (?, ?)", (latest, "KRW-BTC"))
    for index in range(10):
        timestamp = latest if index == 0 else stale
        conn.execute(
            "INSERT INTO binance_data VALUES (?, ?)",
            (timestamp, f"KRW-C{index}"),
        )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        update_data,
        "get_db_connection",
        lambda: sqlite3.connect(db_path),
    )

    ready, reason = update_data._peer_update_is_complete()

    assert not ready
    assert "1/10 Binance markets" in reason


def test_updater_holds_shared_data_access_lock(monkeypatch, tmp_path):
    """DB writes and IC reads must coordinate on the same exclusive lock."""
    from scripts import update_data

    lock_path = tmp_path / "logs" / "locks" / "data_access.lock"
    calls = []

    def assert_locked(*_args, **_kwargs):
        assert lock_path.read_text() == str(os.getpid())
        calls.append("called")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["update_data.py", "--upbit-only"])
    monkeypatch.setattr(update_data, "init_db", assert_locked)
    monkeypatch.setattr(update_data, "upbit_run_all", assert_locked)

    update_data.main()

    assert calls == ["called", "called"]
    assert not lock_path.exists()


def test_updater_data_access_timeout_fails_closed(monkeypatch, tmp_path):
    """An IC reader already in progress must prevent DB mutation."""
    from scripts import update_data

    lock_dir = tmp_path / "logs" / "locks"
    lock_dir.mkdir(parents=True)
    (lock_dir / "data_access.lock").write_text(str(os.getpid()))

    def must_not_run(*_args, **_kwargs):
        pytest.fail("DB mutation ran while data_access lock was held")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["update_data.py", "--upbit-only"])
    monkeypatch.setattr(update_data, "_UPDATE_LOCK_WAIT_SEC", 0.01)
    monkeypatch.setattr(update_data, "init_db", must_not_run)
    monkeypatch.setattr(update_data, "upbit_run_all", must_not_run)

    with pytest.raises(SystemExit) as exc:
        update_data.main()

    assert exc.value.code == os.EX_TEMPFAIL


def test_systemd_units_serialize_catch_up_and_retry_lock_conflicts():
    """Persistent timers caught up at boot must not consume a partial DB."""
    alpha = (ROOT / "deploy" / "xsec-alpha.service").read_text()
    retrain = (ROOT / "deploy" / "xsec-retrain.service").read_text()
    measure = (ROOT / "deploy" / "xsec-measure.service").read_text()

    assert "Before=xsec-retrain.service xsec-measure.service" in alpha
    assert "After=network.target xsec-alpha.service" in retrain
    assert "Before=xsec-measure.service" in retrain
    assert "After=network.target xsec-alpha.service xsec-retrain.service" in measure

    for unit in (alpha, retrain):
        assert "Restart=on-failure" in unit
        assert "RestartSec=60" in unit
