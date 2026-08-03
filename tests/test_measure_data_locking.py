"""Concurrency regression tests for IC readers versus the DB updater."""

import importlib
import os
import threading
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "scripts.measure_ic",
        "scripts.wf_holdout_harness",
        "utils.drift_detector",
    ],
)
def test_measurement_holds_shared_data_lock(monkeypatch, tmp_path, module_name):
    module = importlib.import_module(module_name)
    lock_path = tmp_path / "logs" / "locks" / "data_access.lock"
    calls = []

    def assert_locked():
        assert lock_path.read_text() == str(os.getpid())
        calls.append("measured")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "_main_locked", assert_locked)

    module.main()

    assert calls == ["measured"]
    assert not lock_path.exists()


def test_measurement_waits_for_legacy_updater_then_reads(monkeypatch, tmp_path):
    """A pre-deployment updater without data_access is drained safely."""
    module = importlib.import_module("scripts.wf_holdout_harness")
    lock_dir = tmp_path / "logs" / "locks"
    lock_dir.mkdir(parents=True)
    update_lock = lock_dir / "update_data.lock"
    data_lock = lock_dir / "data_access.lock"
    update_lock.write_text(str(os.getpid()))
    calls = []

    def assert_locked():
        assert data_lock.read_text() == str(os.getpid())
        calls.append("measured")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "_DATA_ACCESS_LOCK_WAIT_SEC", 1.0)
    monkeypatch.setattr(module, "_main_locked", assert_locked)

    release = threading.Timer(0.05, update_lock.unlink)
    release.start()
    try:
        module.main()
    finally:
        release.join()

    assert calls == ["measured"]
    assert not data_lock.exists()


@pytest.mark.parametrize(
    "module_name",
    [
        "scripts.measure_ic",
        "scripts.wf_holdout_harness",
        "utils.drift_detector",
    ],
)
@pytest.mark.parametrize("lock_name", ["update_data", "data_access"])
def test_measurement_lock_timeout_fails_closed(
    monkeypatch, tmp_path, module_name, lock_name
):
    module = importlib.import_module(module_name)
    lock_dir = tmp_path / "logs" / "locks"
    lock_dir.mkdir(parents=True)
    (lock_dir / f"{lock_name}.lock").write_text(str(os.getpid()))

    def must_not_measure():
        pytest.fail("measurement ran while data_access lock was held")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "_DATA_ACCESS_LOCK_WAIT_SEC", 0.01)
    monkeypatch.setattr(module, "_main_locked", must_not_measure)

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == os.EX_TEMPFAIL
