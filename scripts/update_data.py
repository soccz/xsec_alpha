#!/usr/bin/env python3
"""
Incremental data updater for both Upbit (crypto_data) and Binance (binance_data).

Run before fetch_and_rank.py or as its --collect step.

Usage:
    python scripts/update_data.py                # update both
    python scripts/update_data.py --upbit-only   # Upbit only
    python scripts/update_data.py --binance-only  # Binance only
    python scripts/update_data.py --days 120     # initial backfill depth
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import sqlite3
import requests
from datetime import datetime, timedelta, timezone

from config import config
from utils.logger import logger
from utils.run_lock import data_access_lock, run_lock
from data.database import get_db_connection, init_db
from data.collector import run_all as upbit_run_all, get_all_krw_markets


# Leave enough room inside the installed xsec-alpha TimeoutStartSec=600 for
# track_ic + fetch_and_rank after a peer updater finishes.
_UPDATE_LOCK_WAIT_SEC = 480.0
_PEER_MAX_DATA_AGE_HOURS = 6.0
_PEER_MAX_SOURCE_LAG_HOURS = 2.0
_PEER_MIN_BINANCE_FRESH_RATIO = 0.90


# ── Binance helpers ──────────────────────────────────────────────────────────

def _get_binance_usdt_symbols() -> list[tuple[str, str]]:
    """Return [(upbit_market, binance_symbol), ...] for KRW coins that also trade on Binance USDT."""
    upbit_markets = get_all_krw_markets()
    upbit_coins = {m.replace("KRW-", "") for m in upbit_markets}

    resp = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=15)
    resp.raise_for_status()
    binance_usdt = {
        s["symbol"].replace("USDT", ""): s["symbol"]
        for s in resp.json()["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
    }

    pairs = []
    for coin in upbit_coins:
        if coin in binance_usdt:
            pairs.append((f"KRW-{coin}", binance_usdt[coin]))

    # Priority ordering: major coins first
    priority = {"BTC", "ETH", "XRP", "SOL", "DOGE"}
    pairs.sort(key=lambda x: (x[0].replace("KRW-", "") not in priority, x[0]))

    logger.info(f"Binance USDT ∩ Upbit KRW: {len(pairs)} markets")
    return pairs


def _get_binance_last_ts(conn: sqlite3.Connection, market: str) -> str | None:
    """Get latest timestamp string in binance_data for a market."""
    cur = conn.execute(
        "SELECT MAX(timestamp) FROM binance_data WHERE market = ?", (market,)
    )
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def _ensure_binance_table(conn: sqlite3.Connection):
    """Create binance_data table + index if not exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS binance_data (
            timestamp TEXT,
            market TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (timestamp, market)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_binance_ts_mkt ON binance_data(timestamp, market)"
    )
    conn.commit()


def _collect_binance_market(
    conn: sqlite3.Connection, binance_symbol: str, upbit_market: str, days: int
) -> int:
    """Incrementally fetch Binance 1h candles for one market. Returns rows inserted."""
    latest_ts_str = _get_binance_last_ts(conn, upbit_market)

    if latest_ts_str:
        latest_dt = datetime.strptime(latest_ts_str[:19], "%Y-%m-%dT%H:%M:%S")
        start_ms = int(latest_dt.timestamp() * 1000) + 3600000  # next hour
    else:
        start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    end_ms = int(datetime.utcnow().timestamp() * 1000)
    if start_ms >= end_ms:
        return 0

    all_rows = []
    cursor_ms = start_ms
    retries = 0
    total_inserted = 0

    while cursor_ms < end_ms:
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol": binance_symbol,
                    "interval": "1h",
                    "limit": 1000,
                    "startTime": cursor_ms,
                },
                timeout=15,
            )

            if resp.status_code == 429:
                logger.warning(f"  {binance_symbol}: rate limited, waiting 60s")
                time.sleep(60)
                continue

            resp.raise_for_status()
            data = resp.json()
            if not data:
                break

            for k in data:
                ts = datetime.utcfromtimestamp(k[0] / 1000).strftime("%Y-%m-%dT%H:%M:%S")
                all_rows.append((
                    ts, upbit_market,
                    float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]),
                ))

            cursor_ms = data[-1][0] + 3600000  # next candle
            retries = 0

            # Flush in batches
            if len(all_rows) >= 5000:
                conn.executemany(
                    "INSERT OR REPLACE INTO binance_data "
                    "(timestamp, market, open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    all_rows,
                )
                conn.commit()
                total_inserted += len(all_rows)
                all_rows = []

            time.sleep(0.3)

        except requests.exceptions.RequestException as e:
            retries += 1
            if retries > 3:
                logger.error(f"  {binance_symbol}: 3 consecutive failures, skipping — {e}")
                break
            time.sleep(2**retries)

    if all_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO binance_data "
            "(timestamp, market, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            all_rows,
        )
        conn.commit()
        total_inserted += len(all_rows)

    return total_inserted


def update_binance(days: int = 120):
    """Incrementally update Binance data for all KRW-matching USDT markets."""
    pairs = _get_binance_usdt_symbols()
    if not pairs:
        logger.error("No Binance markets found")
        return

    conn = get_db_connection()
    try:
        _ensure_binance_table(conn)

        total = 0
        for i, (upbit_mkt, binance_sym) in enumerate(pairs):
            rows = _collect_binance_market(conn, binance_sym, upbit_mkt, days)
            total += rows
            status = f"[{i + 1}/{len(pairs)}] {binance_sym}: +{rows} rows"
            if rows > 0:
                logger.info(status)
            else:
                logger.debug(status)

        logger.info(f"Binance update complete: +{total} rows across {len(pairs)} markets")
    finally:
        conn.close()


def _parse_utc_timestamp(value) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _peer_update_is_complete() -> tuple[bool, str]:
    """Validate a peer updater's DB before letting systemd continue.

    The global timestamps catch a skipped Binance phase.  The per-market ratio
    catches a peer that stopped after updating only the first few Binance
    symbols, even though the table-wide MAX(timestamp) already looks fresh.
    """
    conn = get_db_connection()
    try:
        upbit_row = conn.execute("SELECT MAX(timestamp) FROM crypto_data").fetchone()
        binance_row = conn.execute("SELECT MAX(timestamp) FROM binance_data").fetchone()
        upbit_latest = _parse_utc_timestamp(upbit_row[0] if upbit_row else None)
        binance_latest = _parse_utc_timestamp(binance_row[0] if binance_row else None)
        if upbit_latest is None or binance_latest is None:
            return False, "missing Upbit or Binance latest timestamp"

        age_h = (datetime.now(timezone.utc) - upbit_latest).total_seconds() / 3600
        source_lag_h = abs((upbit_latest - binance_latest).total_seconds()) / 3600
        if age_h > _PEER_MAX_DATA_AGE_HOURS:
            return False, f"Upbit data is stale ({age_h:.1f}h)"
        if source_lag_h > _PEER_MAX_SOURCE_LAG_HOURS:
            return False, f"Upbit/Binance latest timestamps differ by {source_lag_h:.1f}h"

        known_row = conn.execute(
            "SELECT COUNT(DISTINCT market) FROM binance_data"
        ).fetchone()
        known_markets = int(known_row[0] if known_row else 0)
        fresh_row = conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT market
                FROM binance_data
                GROUP BY market
                HAVING julianday(MAX(timestamp)) >= julianday(?) - (? / 24.0)
            )
            """,
            (upbit_latest.isoformat(timespec="seconds"), _PEER_MAX_SOURCE_LAG_HOURS),
        ).fetchone()
        fresh_markets = int(fresh_row[0] if fresh_row else 0)
        fresh_ratio = fresh_markets / known_markets if known_markets else 0.0
        if fresh_ratio < _PEER_MIN_BINANCE_FRESH_RATIO:
            return False, (
                f"only {fresh_markets}/{known_markets} Binance markets are fresh "
                f"({fresh_ratio:.1%})"
            )

        return True, (
            f"Upbit/Binance lag={source_lag_h:.1f}h, "
            f"fresh Binance markets={fresh_markets}/{known_markets} ({fresh_ratio:.1%})"
        )
    except (sqlite3.Error, OSError) as exc:
        return False, f"peer DB validation failed: {exc}"
    finally:
        conn.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Update Upbit + Binance data incrementally")
    parser.add_argument("--upbit-only", action="store_true", help="Update Upbit only")
    parser.add_argument("--binance-only", action="store_true", help="Update Binance only")
    parser.add_argument("--days", type=int, default=120, help="Backfill depth in days (default: 120)")
    args = parser.parse_args()

    do_upbit = not args.binance_only
    do_binance = not args.upbit_only

    # This command is used as systemd ExecStartPre for both alpha and retrain.
    # Wait for a peer updater instead of letting the caller consume its partial
    # market-by-market commits.  Once the peer releases the lock, validate its
    # completed DB while holding the lock ourselves; skip a duplicate 5+ minute
    # collection only when that validation passes.
    with run_lock(
        "update_data",
        timeout_sec=_UPDATE_LOCK_WAIT_SEC,
        exit_code=os.EX_TEMPFAIL,
    ) as waited_for_peer:
        if waited_for_peer:
            ready, reason = _peer_update_is_complete()
            if not ready:
                logger.error(f"[Lock] Peer updater finished but DB is incomplete: {reason}")
                raise SystemExit(os.EX_TEMPFAIL)
            logger.info(f"[Lock] Peer updater completed; reusing fresh DB ({reason})")
            return

        # Keep market-by-market commits away from multi-query IC readers.  This
        # lock is intentionally distinct from update_data's peer-dedup lock: a
        # reader must never be mistaken for an updater whose work can be reused.
        with data_access_lock(timeout_sec=_UPDATE_LOCK_WAIT_SEC):
            init_db()
            started = time.time()

            if do_binance:
                logger.info("=== Binance update ===")
                update_binance(days=args.days)

            # Keep the price source used for ranking as close as possible to
            # the recommendation run. In a full update Binance is collected
            # first and Upbit last; single-source modes retain their behavior.
            if do_upbit:
                logger.info("=== Upbit update ===")
                upbit_run_all(days=args.days)

            elapsed = time.time() - started
            logger.info(f"Data update finished in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
