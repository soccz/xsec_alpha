#!/usr/bin/env bash
# xsec_alpha DB backup
#
# Why: 1.5M+ row sqlite. If corrupted, ~1 year of data lost. SQLite's
# .backup command is hot-safe (uses copy-on-write semantics; no need to
# stop fetch_and_rank).
#
# Run via xsec-backup.timer (daily 18:00 UTC = KST 03:00, off-peak vs
# fetch cadence). Keeps last 30 snapshots, then prunes by mtime.

set -euo pipefail

DB="/mnt/20t/main/gan_t/data/crypto_data.db"
BACKUP_DIR="/home/soccz/22tb/backups/xsec_db"
TS=$(date -u +%Y%m%dT%H%M)
TARGET="${BACKUP_DIR}/crypto_data_${TS}.db"
KEEP_DAYS=30

mkdir -p "${BACKUP_DIR}"

# SQLite hot backup — atomic, doesn't block writers
sqlite3 "${DB}" ".backup '${TARGET}'"

# Verify the backup is readable
sqlite3 "${TARGET}" "PRAGMA integrity_check;" | head -1 | grep -q '^ok$' || {
  echo "[backup_db] integrity_check failed on ${TARGET}" >&2
  rm -f "${TARGET}"
  exit 2
}

# Prune older than KEEP_DAYS
find "${BACKUP_DIR}" -name "crypto_data_*.db" -mtime "+${KEEP_DAYS}" -delete

# Report
COUNT=$(find "${BACKUP_DIR}" -name "crypto_data_*.db" | wc -l)
SIZE=$(du -sh "${BACKUP_DIR}" | cut -f1)
echo "[backup_db] ok: ${TARGET} (kept ${COUNT} snapshots, total ${SIZE})"
