#!/usr/bin/env python3
"""Daily off-host copy of irreplaceable evidence to a PRIVATE GitHub repo.

Everything else in xsec_alpha either lives in git (code, docs) or can be
re-collected (price DB, research backfill). The ledgers below cannot: if the
single HDD fails, months of pre-registered evidence would be lost. This script
snapshots them into a local clone of soccz/xsec_alpha-evidence (private) and
pushes one commit per day. History is kept; SQLite ledgers are exported as
SQL text (consistent snapshot via the backup API) so git stores only the daily
growth. Status: output/evidence_offhost/status.json.

Usage: python scripts/backup_evidence_offhost.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
WORK = OUT / "evidence_offhost"
REPO = WORK / "repo"
REMOTE = os.environ.get("XSEC_EVIDENCE_REMOTE", "https://github.com/soccz/xsec_alpha-evidence.git")
MAX_FILE_BYTES = 95 * 1024 * 1024  # GitHub rejects files over 100 MB

FILES = [
    "recommendation_ledger.csv", "long_shadow_ledger.csv", "ic_history.json", "ic_history_long.json",
    "gate_state.json", "retrain_history.json", "wf_history.json", "drift_state.json",
    "feature_health.json", "latest_operator_report.json", "calibration_sigma.json",
]
DIRS = [
    "operator_reports", "score_evidence", "experiment_supervision", "rotation_pilot", "regime_observation",
    "operating_acceptance", "prospective", "deadlines", "forecast_audit", "research/bitget_funding",
]
MODEL_GLOBS = ["xsec_6h.pkl", "xsec_12h.pkl", "*.json"]
SKIP_SUFFIXES = (".sqlite-wal", ".sqlite-shm", ".db-wal", ".db-shm", ".lock", ".tmp", ".log")


def _sqlite_dump(src: Path, dst: Path) -> None:
    """Consistent snapshot of a live SQLite DB, written as deterministic SQL text."""
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=60)
    memory = sqlite3.connect(":memory:")
    try:
        source.backup(memory)
    finally:
        source.close()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for line in memory.iterdump():
            f.write(line + "\n")
    memory.close()


def build_snapshot(out: Path = OUT, models: Path = ROOT / "models", dest: Path = REPO) -> dict:
    """Mirror the evidence set into dest (replacing previous snapshot content). Returns a manifest."""
    for child in dest.iterdir() if dest.exists() else []:
        if child.name != ".git":
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"files": 0, "bytes": 0, "sqlite_dumps": 0, "skipped_large": []}

    def put(src: Path, rel: Path) -> None:
        target = dest / rel
        if src.suffix in (".sqlite", ".db"):
            target = target.with_suffix(target.suffix + ".sql")
            _sqlite_dump(src, target)
            manifest["sqlite_dumps"] += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        size = target.stat().st_size
        if size > MAX_FILE_BYTES:
            target.unlink()
            manifest["skipped_large"].append(str(rel))
            return
        manifest["files"] += 1
        manifest["bytes"] += size

    for name in FILES:
        if (out / name).is_file():
            put(out / name, Path("output") / name)
    for name in DIRS:
        base = out / name
        if not base.is_dir():
            continue
        for src in sorted(base.rglob("*")):
            if src.is_file() and not src.name.endswith(SKIP_SUFFIXES):
                put(src, Path("output") / src.relative_to(out))
    for pattern in MODEL_GLOBS:
        for src in sorted(models.glob(pattern)):
            if src.is_file():
                put(src, Path("models") / src.name)
    (dest / "README.md").write_text(
        "# xsec_alpha evidence (private)\n\nDaily automated snapshot of irreplaceable ledgers from "
        "soccz/xsec_alpha (scripts/backup_evidence_offhost.py). SQLite ledgers are SQL text dumps "
        "(restore: `sqlite3 new.sqlite < file.sql`). Not for publication: contains coin-level picks.\n")
    return manifest


def _git(*args, check=True):
    return subprocess.run(["git", "-C", str(REPO), *args], check=check, capture_output=True, text=True, timeout=600)


def ensure_repo() -> None:
    if not (REPO / ".git").exists():
        REPO.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", "-b", "main", str(REPO)], check=True)
        _git("remote", "add", "origin", REMOTE)
        _git("config", "user.name", "xsec evidence backup")
        _git("config", "user.email", "soccz@users.noreply.github.com")
        _git("config", "gc.auto", "0")


def _write_status(payload: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = WORK / "status.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    os.replace(tmp, WORK / "status.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="build the snapshot in a scratch dir, no git")
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    if args.dry_run:
        scratch = Path(os.environ.get("TMPDIR", "/home/soccz/22tb/tmp")) / "xsec_evidence_dryrun"
        manifest = build_snapshot(dest=scratch)
        print(json.dumps(manifest, indent=1))
        shutil.rmtree(scratch, ignore_errors=True)
        return 0
    status = {"checked_at": now.isoformat(), "remote": REMOTE.split("@")[-1], "ok": False}
    try:
        ensure_repo()
        status.update(build_snapshot())
        _git("add", "-A")
        if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
            status.update(ok=True, committed=False, note="no changes")
        else:
            _git("commit", "-q", "-m", f"evidence snapshot {now:%Y-%m-%d %H:%MZ}")
            status["committed"] = True
        push = _git("push", "-q", "origin", "main", check=False)
        if push.returncode != 0:
            raise RuntimeError("push failed: " + push.stderr.strip()[-300:])
        status.update(ok=True, commit=_git("rev-parse", "--short", "HEAD").stdout.strip())
    except Exception as exc:  # recorded for monitoring; never raises into systemd silently
        status["error"] = f"{type(exc).__name__}: {exc}"[:500]
    _write_status(status)
    print(json.dumps(status, ensure_ascii=False))
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
