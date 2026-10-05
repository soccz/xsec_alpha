import sqlite3

from scripts import backup_evidence_offhost as be


def _fixture(tmp_path):
    out, models, dest = tmp_path / "output", tmp_path / "models", tmp_path / "repo"
    (out / "forecast_audit").mkdir(parents=True)
    models.mkdir()
    (out / "recommendation_ledger.csv").write_text("a,b\n1,2\n")
    con = sqlite3.connect(out / "forecast_audit" / "ledger.sqlite")
    con.execute("create table t(k text primary key, v real)")
    con.executemany("insert into t values (?,?)", [("x", 1.5), ("y", -2.0)])
    con.commit()
    con.close()
    (out / "forecast_audit" / "ledger.sqlite-wal").write_text("ignored")
    (out / "forecast_audit" / "summary.json").write_text("{}")
    (models / "xsec_6h.pkl").write_bytes(b"model")
    (models / "other.bin").write_bytes(b"not included")
    return out, models, dest


def test_snapshot_dumps_sqlite_as_restorable_sql_and_skips_transients(tmp_path):
    out, models, dest = _fixture(tmp_path)
    manifest = be.build_snapshot(out=out, models=models, dest=dest)
    sql = dest / "output/forecast_audit/ledger.sqlite.sql"
    restored = sqlite3.connect(":memory:")
    restored.executescript(sql.read_text())
    assert restored.execute("select k, v from t order by k").fetchall() == [("x", 1.5), ("y", -2.0)]
    assert not (dest / "output/forecast_audit/ledger.sqlite-wal").exists()
    assert (dest / "output/recommendation_ledger.csv").read_text() == "a,b\n1,2\n"
    assert (dest / "models/xsec_6h.pkl").exists() and not (dest / "models/other.bin").exists()
    assert manifest["sqlite_dumps"] == 1 and manifest["skipped_large"] == []


def test_snapshot_replaces_previous_content_but_keeps_git_dir(tmp_path):
    out, models, dest = _fixture(tmp_path)
    (dest / ".git").mkdir(parents=True)
    (dest / ".git" / "HEAD").write_text("ref")
    (dest / "stale.txt").write_text("old")
    be.build_snapshot(out=out, models=models, dest=dest)
    assert (dest / ".git" / "HEAD").read_text() == "ref" and not (dest / "stale.txt").exists()


def test_oversized_files_are_skipped_and_reported(tmp_path, monkeypatch):
    out, models, dest = _fixture(tmp_path)
    monkeypatch.setattr(be, "MAX_FILE_BYTES", 3)
    manifest = be.build_snapshot(out=out, models=models, dest=dest)
    assert "output/recommendation_ledger.csv" in manifest["skipped_large"]
    assert not (dest / "output/recommendation_ledger.csv").exists()
