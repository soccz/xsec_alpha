#!/usr/bin/env python3
"""Backfill Upbit KRW 1h candles into a SEPARATE research DB (never the live DB).

Purpose: the model-zoo historical map (docs/prereg/) needs a longer Upbit
cross-section than the live DB holds (broad coverage only since 2025-09).

- Public quotation API only; pauses around live jobs (scripts/research_net.py).
- Only completed hours are stored; the current hour is excluded.
- Idempotent and resumable: PRIMARY KEY(market, ts_utc) + INSERT OR IGNORE,
  a top-up pass to "now" and a deep pass back to --start or the listing date.
- Survivorship: delisted markets are not served by the API. Each run stores
  the market-list snapshot (incl. warning flags) for the report.

Usage:
  nice -n 19 ionice -c3 python scripts/backfill_upbit_history.py [--start 2021-01-01]
      [--markets KRW-BTC,KRW-ETH] [--db PATH] [--pace 0.2]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_net import get_json, stamp, utcnow, wait_outside_live_window  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "research" / "upbit_backfill"
API = "https://api.upbit.com/v1"
PAGE = 200

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    market TEXT NOT NULL, ts_utc TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, value_krw REAL, fetched_at TEXT NOT NULL,
    PRIMARY KEY (market, ts_utc)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS market_status (
    market TEXT PRIMARY KEY, reached_listing INTEGER NOT NULL DEFAULT 0,
    reached_start TEXT, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    started_at TEXT, finished_at TEXT, start_bound TEXT, markets INTEGER,
    rows_added INTEGER, requests INTEGER, status TEXT
);
"""


def log(msg: str) -> None:
    print(f"{stamp()} {msg}", flush=True)


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def krw_markets_by_turnover() -> list[str]:
    markets = get_json(f"{API}/market/all?isDetails=true", log=log) or []
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"markets_{utcnow().strftime('%Y%m%dT%H%M%SZ')}.json").write_text(
        json.dumps(markets, ensure_ascii=False))
    krw = [m["market"] for m in markets if m.get("market", "").startswith("KRW-")]
    turnover: dict[str, float] = {}
    for i in range(0, len(krw), 100):
        chunk = ",".join(krw[i:i + 100])
        for t in get_json(f"{API}/ticker?markets={urllib.parse.quote(chunk, safe=',-')}", log=log) or []:
            turnover[t["market"]] = float(t.get("acc_trade_price_24h") or 0.0)
        time.sleep(0.2)
    return sorted(krw, key=lambda m: -turnover.get(m, 0.0))


def page_back(con, market: str, to_ts: str, stop_ts: str, cutoff: str, pace: float,
              deep: bool, counters: dict) -> None:
    """Fetch pages backward from to_ts (exclusive) until a page reaches stop_ts."""
    cursor = to_ts
    while True:
        wait_outside_live_window(log)
        url = f"{API}/candles/minutes/60?market={market}&to={cursor}Z&count={PAGE}"
        rows = get_json(url, log=log)
        counters["requests"] += 1
        time.sleep(pace)
        if not rows:
            if deep:
                mark(con, market, reached_listing=True)
            return
        fetched = stamp()
        batch = [(market, r["candle_date_time_utc"], r["opening_price"], r["high_price"],
                  r["low_price"], r["trade_price"], r["candle_acc_trade_volume"],
                  r["candle_acc_trade_price"], fetched)
                 for r in rows if r["candle_date_time_utc"] < cutoff]
        before = con.total_changes
        con.executemany("INSERT OR IGNORE INTO candles VALUES (?,?,?,?,?,?,?,?,?)", batch)
        con.commit()
        counters["rows"] += con.total_changes - before
        oldest = min(r["candle_date_time_utc"] for r in rows)
        if oldest <= stop_ts:
            if deep:
                mark(con, market, reached_start=stop_ts)
            return
        if len(rows) < PAGE:
            if deep:
                mark(con, market, reached_listing=True)
            return
        cursor = oldest


def mark(con, market: str, reached_listing: bool = False, reached_start: str | None = None) -> None:
    con.execute(
        "INSERT INTO market_status(market, reached_listing, reached_start, updated_at) VALUES (?,?,?,?) "
        "ON CONFLICT(market) DO UPDATE SET reached_listing=max(reached_listing, excluded.reached_listing), "
        "reached_start=coalesce(excluded.reached_start, reached_start), updated_at=excluded.updated_at",
        (market, int(reached_listing), reached_start, stamp()))
    con.commit()


def backfill(con, market: str, start: str, pace: float, counters: dict) -> None:
    now_hour = utcnow().replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")
    newest, oldest = con.execute(
        "SELECT max(ts_utc), min(ts_utc) FROM candles WHERE market=?", (market,)).fetchone()
    status = con.execute(
        "SELECT reached_listing, reached_start FROM market_status WHERE market=?", (market,)).fetchone()
    done_deep = bool(status and (status[0] or (status[1] and status[1] <= start)))
    if newest is None:
        page_back(con, market, now_hour, start, now_hour, pace, True, counters)
        return
    page_back(con, market, now_hour, newest, now_hour, pace, False, counters)  # top-up
    if not done_deep and oldest > start:
        page_back(con, market, oldest, start, now_hour, pace, True, counters)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="2021-01-01", help="oldest hour to fetch (UTC date)")
    ap.add_argument("--markets", default="", help="comma list; default: all KRW markets by 24h turnover")
    ap.add_argument("--db", default=str(OUT / "upbit_1h.sqlite"))
    ap.add_argument("--pace", type=float, default=0.2, help="seconds between requests")
    args = ap.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    con = open_db(Path(args.db))
    markets = [m.strip() for m in args.markets.split(",") if m.strip()] or krw_markets_by_turnover()
    started = stamp()
    counters = {"rows": 0, "requests": 0}
    log(f"start: {len(markets)} markets, start bound {start}, db {args.db}")
    status = "complete"
    try:
        for i, market in enumerate(markets, 1):
            before = counters["rows"]
            backfill(con, market, start, args.pace, counters)
            log(f"[{i}/{len(markets)}] {market}: +{counters['rows'] - before} rows "
                f"(total {counters['rows']}, requests {counters['requests']})")
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?)",
                    (started, stamp(), start, len(markets), counters["rows"], counters["requests"], status))
        con.commit()
        con.close()
    log(f"done ({status}): {counters['rows']} rows, {counters['requests']} requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
