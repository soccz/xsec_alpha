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
umask 077

DB="${XSEC_DB_PATH:-/mnt/20t/main/gan_t/data/crypto_data.db}"
BACKUP_DIR="${XSEC_BACKUP_DIR:-/home/soccz/22tb/backups/xsec_db}"
# Same-disk recovery is the approved default; a second filesystem is optional.
SECONDARY_BACKUP_DIR="${XSEC_SECONDARY_BACKUP_DIR:-}"
KEEP_DAYS="${XSEC_BACKUP_KEEP_DAYS:-30}"
SECONDARY_KEEP_COUNT="${XSEC_SECONDARY_BACKUP_KEEP_COUNT:-3}"
SECONDARY_RESERVE_KB="${XSEC_SECONDARY_BACKUP_RESERVE_KB:-5242880}"
PRIMARY_RESERVE_KB="${XSEC_PRIMARY_BACKUP_RESERVE_KB:-5242880}"
REQUIRE_SEPARATE_DEVICE="${XSEC_REQUIRE_SEPARATE_DEVICE:-1}"
BUSY_TIMEOUT_MS="${XSEC_BACKUP_BUSY_TIMEOUT_MS:-60000}"
ATTEMPTS="${XSEC_BACKUP_ATTEMPTS:-3}"
RETRY_DELAY_SECONDS="${XSEC_BACKUP_RETRY_DELAY_SECONDS:-10}"

require_nonnegative_integer() {
  local name="$1"
  local value="$2"

  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "[backup_db] ${name} must be a non-negative integer: ${value}" >&2
    exit 1
  fi
}

require_nonnegative_integer XSEC_BACKUP_KEEP_DAYS "${KEEP_DAYS}"
require_nonnegative_integer XSEC_SECONDARY_BACKUP_KEEP_COUNT "${SECONDARY_KEEP_COUNT}"
require_nonnegative_integer XSEC_SECONDARY_BACKUP_RESERVE_KB "${SECONDARY_RESERVE_KB}"
require_nonnegative_integer XSEC_PRIMARY_BACKUP_RESERVE_KB "${PRIMARY_RESERVE_KB}"
require_nonnegative_integer XSEC_REQUIRE_SEPARATE_DEVICE "${REQUIRE_SEPARATE_DEVICE}"
require_nonnegative_integer XSEC_BACKUP_BUSY_TIMEOUT_MS "${BUSY_TIMEOUT_MS}"
require_nonnegative_integer XSEC_BACKUP_ATTEMPTS "${ATTEMPTS}"
require_nonnegative_integer XSEC_BACKUP_RETRY_DELAY_SECONDS "${RETRY_DELAY_SECONDS}"

if (( ATTEMPTS < 1 )); then
  echo "[backup_db] XSEC_BACKUP_ATTEMPTS must be at least 1" >&2
  exit 1
fi

if (( SECONDARY_KEEP_COUNT < 1 )); then
  echo "[backup_db] XSEC_SECONDARY_BACKUP_KEEP_COUNT must be at least 1" >&2
  exit 1
fi

if (( REQUIRE_SEPARATE_DEVICE != 0 && REQUIRE_SEPARATE_DEVICE != 1 )); then
  echo "[backup_db] XSEC_REQUIRE_SEPARATE_DEVICE must be 0 or 1" >&2
  exit 1
fi

if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "[backup_db] sqlite3 is not installed" >&2
  exit 1
fi

# Opening a missing path with sqlite3 creates a new empty database. Validate the
# source before invoking the CLI so a mount/path failure cannot look successful.
if [[ ! -f "${DB}" || ! -r "${DB}" || ! -s "${DB}" ]]; then
  echo "[backup_db] source DB is missing, unreadable, or empty: ${DB}" >&2
  exit 1
fi

mkdir -p "${BACKUP_DIR}"

SOURCE_DEVICE=$(stat -c %d "${DB}")
SOURCE_KB=$(( ($(stat -c %s "${DB}") + 1023) / 1024 ))
PRIMARY_AVAILABLE_KB=$(df -Pk -- "${BACKUP_DIR}" | awk 'NR == 2 {print $4}')
if (( PRIMARY_AVAILABLE_KB < SOURCE_KB + PRIMARY_RESERVE_KB )); then
  echo "[backup_db] insufficient primary space: need ${SOURCE_KB} KiB plus ${PRIMARY_RESERVE_KB} KiB reserve" >&2
  exit 1
fi
if [[ -n "${SECONDARY_BACKUP_DIR}" ]]; then
  mkdir -p "${SECONDARY_BACKUP_DIR}"
  if [[ "$(realpath "${BACKUP_DIR}")" == "$(realpath "${SECONDARY_BACKUP_DIR}")" ]]; then
    echo "[backup_db] primary and secondary directories must differ" >&2
    exit 1
  fi
  SECONDARY_DEVICE=$(stat -c %d "${SECONDARY_BACKUP_DIR}")
  if (( REQUIRE_SEPARATE_DEVICE == 1 )) && [[ "${SOURCE_DEVICE}" == "${SECONDARY_DEVICE}" ]]; then
    echo "[backup_db] secondary backup must be on a different physical filesystem: ${SECONDARY_BACKUP_DIR}" >&2
    exit 1
  fi
  SECONDARY_AVAILABLE_KB=$(df -Pk -- "${SECONDARY_BACKUP_DIR}" | awk 'NR == 2 {print $4}')
  if (( SECONDARY_AVAILABLE_KB < SOURCE_KB + SECONDARY_RESERVE_KB )); then
    echo "[backup_db] insufficient secondary space: need ${SOURCE_KB} KiB plus ${SECONDARY_RESERVE_KB} KiB reserve" >&2
    exit 1
  fi
fi

TS=$(date -u +%Y%m%dT%H%M%SZ)
TARGET="${BACKUP_DIR}/crypto_data_${TS}_$$.db"
PARTIAL="${TARGET}.partial"
SECONDARY_TARGET=""
SECONDARY_PARTIAL=""
if [[ -n "${SECONDARY_BACKUP_DIR}" ]]; then
  SECONDARY_TARGET="${SECONDARY_BACKUP_DIR}/${TARGET##*/}"
  SECONDARY_PARTIAL="${SECONDARY_TARGET}.partial"
fi

cleanup_partial() {
  rm -f -- "${PARTIAL}"
  if [[ -n "${SECONDARY_PARTIAL}" ]]; then
    rm -f -- "${SECONDARY_PARTIAL}"
  fi
}

trap cleanup_partial EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# A persistent timer may catch up at boot alongside measure/retrain jobs. Wait
# for transient SQLite locks, while staying inside the service timeout budget.
backup_ok=0
for (( attempt = 1; attempt <= ATTEMPTS; attempt++ )); do
  cleanup_partial
  if sqlite3 -readonly -cmd ".timeout ${BUSY_TIMEOUT_MS}" \
      "${DB}" ".backup '${PARTIAL}'"; then
    backup_ok=1
    break
  fi

  if (( attempt < ATTEMPTS )); then
    echo "[backup_db] attempt ${attempt}/${ATTEMPTS} failed; retrying in ${RETRY_DELAY_SECONDS}s" >&2
    sleep "${RETRY_DELAY_SECONDS}"
  fi
done

if (( backup_ok == 0 )); then
  echo "[backup_db] backup failed after ${ATTEMPTS} attempts" >&2
  exit 1
fi

if [[ ! -s "${PARTIAL}" ]]; then
  echo "[backup_db] backup output is empty: ${PARTIAL}" >&2
  exit 2
fi

if ! CHECK=$(sqlite3 -readonly "${PARTIAL}" "PRAGMA integrity_check;"); then
  echo "[backup_db] integrity_check could not read ${PARTIAL}" >&2
  exit 2
fi
if [[ "${CHECK}" != "ok" ]]; then
  echo "[backup_db] integrity_check failed on ${PARTIAL}" >&2
  exit 2
fi

HAS_CRYPTO_DATA=$(sqlite3 -readonly "${PARTIAL}" \
  "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='crypto_data';")
if [[ "${HAS_CRYPTO_DATA}" != "1" ]]; then
  echo "[backup_db] required crypto_data table is missing from ${PARTIAL}" >&2
  exit 2
fi

# An explicitly configured secondary is still verified before publication.
if [[ -n "${SECONDARY_BACKUP_DIR}" ]]; then
  cp --reflink=never -- "${PARTIAL}" "${SECONDARY_PARTIAL}"
  sync -f "${SECONDARY_PARTIAL}"
  if ! cmp -s -- "${PARTIAL}" "${SECONDARY_PARTIAL}"; then
    echo "[backup_db] secondary copy verification failed: ${SECONDARY_PARTIAL}" >&2
    exit 2
  fi
  mv -- "${SECONDARY_PARTIAL}" "${SECONDARY_TARGET}"
fi

# Publish only complete, verified snapshots.
sync -f "${PARTIAL}"
mv -- "${PARTIAL}" "${TARGET}"
sync -f "${BACKUP_DIR}"
trap - EXIT HUP INT TERM

# Prune older than KEEP_DAYS
find "${BACKUP_DIR}" -maxdepth 1 -type f -name "crypto_data_*.db" \
  -mtime "+${KEEP_DAYS}" -delete

prune_to_count() {
  local directory="$1"
  local keep_count="$2"
  local entry
  local remove_count
  local -a entries=()

  while IFS= read -r -d '' entry; do
    entries+=("${entry}")
  done < <(find "${directory}" -maxdepth 1 -type f -name "crypto_data_*.db" \
    -printf '%T@ %p\0' | sort -z -n)

  remove_count=$(( ${#entries[@]} - keep_count ))
  for (( index = 0; index < remove_count; index++ )); do
    rm -f -- "${entries[$index]#* }"
  done
}

if [[ -n "${SECONDARY_BACKUP_DIR}" ]]; then
  prune_to_count "${SECONDARY_BACKUP_DIR}" "${SECONDARY_KEEP_COUNT}"
fi

# Report
COUNT=$(find "${BACKUP_DIR}" -maxdepth 1 -type f -name "crypto_data_*.db" \
  -size +0c | wc -l)
SIZE=$(du -sh "${BACKUP_DIR}" | cut -f1)
echo "[backup_db] ok: ${TARGET} (kept ${COUNT} primary snapshots, total ${SIZE})"
if [[ -n "${SECONDARY_BACKUP_DIR}" ]]; then
  SECONDARY_COUNT=$(find "${SECONDARY_BACKUP_DIR}" -maxdepth 1 -type f \
    -name "crypto_data_*.db" -size +0c | wc -l)
  if [[ "${SOURCE_DEVICE}" != "${SECONDARY_DEVICE}" ]]; then
    echo "[backup_db] off-device: ${SECONDARY_TARGET} (kept ${SECONDARY_COUNT})"
  else
    echo "[backup_db] same-device secondary: ${SECONDARY_TARGET} (disk failure NOT protected)"
  fi
else
  echo "[backup_db] local recovery only; off-device backup disabled (disk failure NOT protected)"
fi
