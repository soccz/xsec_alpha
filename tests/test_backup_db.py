"""Regression tests for the xsec SQLite backup command."""

import os
from pathlib import Path
import sqlite3
import subprocess
import time


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backup_db.sh"


def create_source(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE crypto_data (market TEXT, close REAL)")
        conn.execute("INSERT INTO crypto_data VALUES ('KRW-BTC', 123.45)")


def backup_env(source: Path, backup_dir: Path, **overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "XSEC_DB_PATH": str(source),
            "XSEC_BACKUP_DIR": str(backup_dir),
            "XSEC_SECONDARY_BACKUP_DIR": str(backup_dir.parent / "secondary"),
            "XSEC_BACKUP_KEEP_DAYS": "30",
            "XSEC_SECONDARY_BACKUP_KEEP_COUNT": "3",
            "XSEC_SECONDARY_BACKUP_RESERVE_KB": "0",
            "XSEC_REQUIRE_SEPARATE_DEVICE": "0",
            "XSEC_BACKUP_BUSY_TIMEOUT_MS": "100",
            "XSEC_BACKUP_ATTEMPTS": "3",
            "XSEC_BACKUP_RETRY_DELAY_SECONDS": "0",
        }
    )
    env.update(overrides)
    return env


def run_backup(source: Path, backup_dir: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=backup_env(source, backup_dir, **overrides),
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )


def snapshots(backup_dir: Path) -> list[Path]:
    return sorted(backup_dir.glob("crypto_data_*.db")) if backup_dir.exists() else []


def partials(backup_dir: Path) -> list[Path]:
    return sorted(backup_dir.glob("*.partial")) if backup_dir.exists() else []


def test_creates_verified_snapshot_without_partial(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    backup_dir = tmp_path / "backups"
    create_source(source)

    result = run_backup(source, backup_dir)

    assert result.returncode == 0, result.stderr
    assert len(snapshots(backup_dir)) == 1
    assert len(snapshots(tmp_path / "secondary")) == 1
    assert partials(backup_dir) == []
    assert partials(tmp_path / "secondary") == []
    assert snapshots(backup_dir)[0].read_bytes() == snapshots(tmp_path / "secondary")[0].read_bytes()
    with sqlite3.connect(snapshots(backup_dir)[0]) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute("SELECT * FROM crypto_data").fetchone() == (
            "KRW-BTC",
            123.45,
        )


def test_missing_source_is_not_created_or_backed_up(tmp_path: Path) -> None:
    source = tmp_path / "missing.db"
    backup_dir = tmp_path / "backups"

    result = run_backup(source, backup_dir)

    assert result.returncode != 0
    assert "source DB is missing, unreadable, or empty" in result.stderr
    assert not source.exists()
    assert snapshots(backup_dir) == []
    assert snapshots(tmp_path / "secondary") == []
    assert partials(backup_dir) == []


def test_empty_source_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "empty.db"
    backup_dir = tmp_path / "backups"
    source.touch()

    result = run_backup(source, backup_dir)

    assert result.returncode != 0
    assert snapshots(backup_dir) == []
    assert snapshots(tmp_path / "secondary") == []
    assert partials(backup_dir) == []


def test_database_without_crypto_data_is_not_published(tmp_path: Path) -> None:
    source = tmp_path / "wrong.db"
    backup_dir = tmp_path / "backups"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE unrelated (id INTEGER)")

    result = run_backup(source, backup_dir)

    assert result.returncode == 2
    assert "required crypto_data table is missing" in result.stderr
    assert snapshots(backup_dir) == []
    assert snapshots(tmp_path / "secondary") == []
    assert partials(backup_dir) == []


def test_retries_until_exclusive_lock_is_released(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    backup_dir = tmp_path / "backups"
    create_source(source)
    locker = sqlite3.connect(source, isolation_level=None)
    locker.execute("BEGIN EXCLUSIVE")

    process = subprocess.Popen(
        ["bash", str(SCRIPT)],
        env=backup_env(
            source,
            backup_dir,
            XSEC_BACKUP_ATTEMPTS="5",
            XSEC_BACKUP_RETRY_DELAY_SECONDS="1",
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(0.35)
        locker.execute("COMMIT")
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        locker.close()

    assert process.returncode == 0, stderr
    assert "retrying" in stderr
    assert "[backup_db] ok:" in stdout
    assert len(snapshots(backup_dir)) == 1
    assert len(snapshots(tmp_path / "secondary")) == 1
    assert partials(backup_dir) == []


def test_lock_exhaustion_leaves_no_snapshot_or_partial(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    backup_dir = tmp_path / "backups"
    create_source(source)
    locker = sqlite3.connect(source, isolation_level=None)
    locker.execute("BEGIN EXCLUSIVE")
    try:
        result = run_backup(
            source,
            backup_dir,
            XSEC_BACKUP_ATTEMPTS="2",
        )
    finally:
        locker.execute("ROLLBACK")
        locker.close()

    assert result.returncode != 0
    assert "backup failed after 2 attempts" in result.stderr
    assert snapshots(backup_dir) == []
    assert snapshots(tmp_path / "secondary") == []
    assert partials(backup_dir) == []
    assert partials(tmp_path / "secondary") == []


def test_separate_device_requirement_rejects_same_filesystem(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    backup_dir = tmp_path / "backups"
    create_source(source)

    result = run_backup(
        source,
        backup_dir,
        XSEC_REQUIRE_SEPARATE_DEVICE="1",
    )

    assert result.returncode != 0
    assert "different physical filesystem" in result.stderr
    assert snapshots(backup_dir) == []
    assert snapshots(tmp_path / "secondary") == []


def test_secondary_retention_keeps_newest_three(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    backup_dir = tmp_path / "backups"
    create_source(source)

    for _ in range(4):
        result = run_backup(source, backup_dir)
        assert result.returncode == 0, result.stderr

    assert len(snapshots(backup_dir)) == 4
    assert len(snapshots(tmp_path / "secondary")) == 3
