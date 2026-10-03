#!/usr/bin/env python3
"""Preserve Bitget USDT-perp funding-rate history in a research DB.

Bitget serves only ~90 days of history-fund-rate, so this is meant to be run
regularly (idempotent: PRIMARY KEY(symbol, funding_time) + INSERT OR IGNORE).
Research data only (model-zoo M5 funding-crowding track, docs/prereg/).
Public endpoints only; pauses around live jobs (scripts/research_net.py).

Usage:
  nice -n 19 ionice -c3 python scripts/dump_bitget_funding.py [--symbols BTCUSDT,ETHUSDT] [--pace 0.25]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_net import get_json, stamp, utcnow, wait_outside_live_window  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "research" / "bitget_funding"
BASE = "https://api.bitget.com/api/v2/mix/market/"
PAGE = 100
MAX_PAGES = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS funding (
    symbol TEXT NOT NULL, funding_time INTEGER NOT NULL, funding_rate REAL NOT NULL,
    fetched_at TEXT NOT NULL, PRIMARY KEY (symbol, funding_time)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS runs (
    started_at TEXT, finished_at TEXT, symbols INTEGER, rows_added INTEGER, requests INTEGER, status TEXT
);
"""


def log(msg: str) -> None:
    print(f"{stamp()} {msg}", flush=True)


def check(payload) -> list:
    if not isinstance(payload, dict) or payload.get("code") != "00000":
        raise RuntimeError(f"Bitget API rejected request: {str(payload)[:200]}")
    return payload.get("data") or []


def normal_symbols() -> list[str]:
    payload = get_json(BASE + "contracts?productType=USDT-FUTURES", log=log)
    contracts = check(payload)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"contracts_{utcnow().strftime('%Y%m%dT%H%M%SZ')}.json").write_text(json.dumps(payload))
    return sorted(c["symbol"] for c in contracts if c.get("symbolStatus") == "normal" and c.get("symbol"))


def dump_symbol(con, symbol: str, pace: float, counters: dict) -> None:
    for page in range(1, MAX_PAGES + 1):
        wait_outside_live_window(log)
        params = urllib.parse.urlencode({"symbol": symbol, "productType": "USDT-FUTURES",
                                         "pageSize": str(PAGE), "pageNo": str(page)})
        rows = check(get_json(BASE + "history-fund-rate?" + params, log=log))
        counters["requests"] += 1
        time.sleep(pace)
        if not rows:
            return
        fetched = stamp()
        before = con.total_changes
        con.executemany("INSERT OR IGNORE INTO funding VALUES (?,?,?,?)",
                        [(symbol, int(r["fundingTime"]), float(r["fundingRate"]), fetched) for r in rows])
        con.commit()
        added = con.total_changes - before
        counters["rows"] += added
        if len(rows) < PAGE or added == 0:  # end of history, or reached already-stored records
            return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--symbols", default="", help="comma list; default: all normal USDT-FUTURES symbols")
    ap.add_argument("--db", default=str(OUT / "funding.sqlite"))
    ap.add_argument("--pace", type=float, default=0.25)
    args = ap.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(args.db)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or normal_symbols()
    started, counters, status = stamp(), {"rows": 0, "requests": 0}, "complete"
    log(f"start: {len(symbols)} symbols, db {args.db}")
    try:
        for i, symbol in enumerate(symbols, 1):
            dump_symbol(con, symbol, args.pace, counters)
            if i % 50 == 0 or i == len(symbols):
                log(f"[{i}/{len(symbols)}] rows {counters['rows']}, requests {counters['requests']}")
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?)",
                    (started, stamp(), len(symbols), counters["rows"], counters["requests"], status))
        con.commit()
        con.close()
    log(f"done ({status}): {counters['rows']} rows, {counters['requests']} requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
