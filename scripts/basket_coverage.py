"""Daily read-only source audit. This is not a pilot evaluator or a signal job."""

import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import fcntl
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
BACKUPS = Path("/home/soccz/22tb/backups/xsec_db")
BACKUP_NAME = re.compile(r"crypto_data_(\d{8}T\d{6}Z)_\d+\.db")
SCHEMA = """
CREATE TABLE IF NOT EXISTS slots (
 slot TEXT PRIMARY KEY, regime TEXT, captured TEXT, entry_at TEXT, exit_at TEXT,
 official TEXT NOT NULL, evaluated TEXT, official_gap INTEGER, backup TEXT, state TEXT NOT NULL,
 missing_keys INTEGER, model_net_pct REAL, reversal_net_pct REAL, source_ic REAL);
CREATE TABLE IF NOT EXISTS prices (
 slot TEXT, market TEXT, leg TEXT, candle TEXT, model INTEGER, reversal INTEGER,
 status TEXT NOT NULL, price REAL, PRIMARY KEY(slot,market,leg));
CREATE INDEX IF NOT EXISTS price_gaps ON prices(status,candle,market);
CREATE TABLE IF NOT EXISTS probes (
 market TEXT, candle TEXT, checked_at TEXT, result TEXT, price REAL, tries INTEGER,
 PRIMARY KEY(market,candle));
CREATE TABLE IF NOT EXISTS probe_attempts (
 market TEXT, candle TEXT, checked_at TEXT, result TEXT, detail TEXT);
"""


def utc(value):
    stamp = datetime.fromisoformat(value)
    return (
        stamp.replace(tzinfo=timezone.utc)
        if stamp.tzinfo is None
        else stamp.astimezone(timezone.utc)
    )


def ro(path):
    return path.resolve().as_uri() + "?mode=ro"


def summary_file(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, allow_nan=False, ensure_ascii=False, indent=2) + "\n"
    )
    tmp.replace(path)


def backup_for(folder, now):
    for path in sorted(folder.glob("crypto_data_*.db"), reverse=True):
        match = BACKUP_NAME.fullmatch(path.name)
        if match and path.stat().st_size:
            stamp = datetime.strptime(match[1], "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
            if not timedelta(0) <= now - stamp <= timedelta(hours=40):
                raise ValueError("daily price backup timestamp outside expected range")
            return path, stamp
    raise FileNotFoundError("no completed daily price backup")


def inspect_slot(db, raw, slot, obs, outcome, backup):
    members = obs["rows"]
    if len({r["market"] for r in members}) != len(members) or any(
        sum(r[k] is True for r in members) != 5
        for k in ("model_selected", "baseline_selected")
    ):
        raise ValueError("invalid saved basket membership: " + slot)
    rows, values = [], {}
    for key in ("entry_at", "exit_at"):
        stamp = utc(obs[key]).strftime("%Y-%m-%dT%H:%M:%S")
        found = {}
        for market, price in raw.execute(
            "SELECT market,open FROM crypto_data WHERE timestamp IN (?,?)",
            (stamp, stamp.replace("T", " ")),
        ):
            found.setdefault(market, []).append(price)
        for member in members:
            market = member["market"]
            hits = found.get(market, [])
            valid = (
                len(hits) == 1
                and isinstance(hits[0], (float, int))
                and math.isfinite(hits[0])
                and hits[0] > 0
            )
            status = (
                "ok"
                if valid
                else "absent"
                if not hits
                else "duplicate"
                if len(hits) > 1
                else "invalid"
            )
            price = hits[0] if valid else None
            rows.append(
                (
                    slot,
                    market,
                    key,
                    stamp,
                    member["model_selected"],
                    member["baseline_selected"],
                    status,
                    price,
                )
            )
            values[market, key] = price
    db.executemany("INSERT INTO prices VALUES (?,?,?,?,?,?,?,?)", rows)
    bad = [r for r in rows if r[6] != "ok"]
    selected_bad = any(r[4] or r[5] for r in bad)
    costs = obs["evaluation"]["cost_bps"]
    cost_pct = (
        2 * (costs["one_way_fee"] + costs["one_way_slippage"]) + costs["short_extra"]
    ) / 100

    def net(flag):
        if selected_bad:
            return None
        returns = [
            values[r["market"], "exit_at"] / values[r["market"], "entry_at"] - 1
            for r in members
            if r[flag]
        ]
        return -100 * sum(returns) / len(returns) - cost_pct

    gap = (
        outcome["required_price_rows"] - outcome["observed_price_rows"]
        if "required_price_rows" in outcome
        else None
    )
    db.execute(
        "INSERT INTO slots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            slot,
            obs["regime"]["regime"],
            obs["recorded_at"],
            obs["entry_at"],
            obs["exit_at"],
            outcome["status"],
            outcome["evaluated_at"],
            gap,
            backup,
            "basket_incomplete" if selected_bad else "basket_only" if bad else "full",
            len(bad),
            net("model_selected"),
            net("baseline_selected"),
            outcome.get("model_ic"),
        ),
    )


def collect(db, raw, backup, backup_at):
    # The observer has no terminal date. The pilot is joined for attribution only.
    latest = raw.execute("SELECT MAX(timestamp) FROM crypto_data").fetchone()[0]
    if not latest:
        raise ValueError("empty price backup")
    pending = 0
    for slot, payload, result in db.execute("""
        SELECT o.slot,o.payload,r.payload FROM observer.observations o
        LEFT JOIN observer.outcomes r USING(slot) LEFT JOIN slots a USING(slot)
        WHERE a.slot IS NULL ORDER BY o.slot"""):
        obs = json.loads(payload)
        if obs["status"] == "missed":
            db.execute(
                "INSERT INTO slots(slot,official,state) VALUES (?,'missed','missed')",
                (slot,),
            )
            continue
        if not result or utc(obs["exit_at"]) + timedelta(hours=1) > min(
            utc(latest), backup_at
        ):
            pending += 1
            continue
        inspect_slot(db, raw, slot, obs, json.loads(result), backup.name)
    return pending


def probe(market, candle):
    end = (utc(candle) + timedelta(hours=1)).isoformat()
    url = "https://api.upbit.com/v1/candles/minutes/60?" + urlencode(
        {"market": market, "to": end, "count": 1}
    )
    with urlopen(url, timeout=4) as response:
        body = json.loads(response.read(65536))
    if not isinstance(body, list) or len(body) > 1:
        raise ValueError("unexpected candle list")
    if not body:
        return "not_returned_now", None
    row = body[0]
    found = utc(row["candle_date_time_utc"])
    if row["market"] != market or found > utc(candle):
        raise ValueError("unexpected exchange candle")
    if found < utc(candle):
        return "not_returned_now", None
    price = row["opening_price"]
    if not isinstance(price, (float, int)) or not math.isfinite(price) or price <= 0:
        raise ValueError("invalid exchange price")
    return "available_now", price


def probe_gaps(db, quota, now, fetch=probe):
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    used = db.execute(
        "SELECT COUNT(*) FROM probes WHERE checked_at>=?", (midnight,)
    ).fetchone()[0]
    quota = min(quota, 30 - used)
    if quota <= 0:
        return
    keys = db.execute(
        """
        SELECT p.market,p.candle FROM prices p LEFT JOIN probes a USING(market,candle)
        WHERE p.status='absent' AND (a.market IS NULL OR
          (a.result='unknown' AND a.tries<3 AND a.checked_at<? AND julianday(a.checked_at)<julianday(?)-0.9))
        GROUP BY p.market,p.candle ORDER BY a.tries IS NOT NULL,MAX(p.model OR p.reversal) DESC,p.candle,p.market
        LIMIT ?""",
        (midnight, now.isoformat(), quota),
    ).fetchall()
    for market, candle in keys:
        halt = False
        detail = None
        try:
            result, price = fetch(market, candle)
        except HTTPError as exc:
            result, price, halt = (
                "unknown",
                None,
                exc.code in (418, 429) or exc.code >= 500,
            )
            detail = f"http_{exc.code}"
        except (
            HTTPException,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            result, price, halt = "unknown", None, True
            detail = type(exc).__name__
        with db:
            db.execute(
                "INSERT INTO probe_attempts VALUES (?,?,?,?,?)",
                (market, candle, now.isoformat(), result, detail),
            )
            db.execute(
                """INSERT INTO probes VALUES (?,?,?,?,?,1) ON CONFLICT(market,candle) DO UPDATE SET
                       checked_at=excluded.checked_at,result=excluded.result,price=excluded.price,tries=tries+1""",
                (market, candle, now.isoformat(), result, price),
            )
        if halt:
            break
        time.sleep(0.25)


def pilot_comparison(db):
    pilot = """ FROM pilot.observations p LEFT JOIN slots s USING(slot)
       LEFT JOIN pilot.outcomes r USING(slot) """
    invalid = dict(
        db.execute(
            "SELECT s.state,count(*)"
            + pilot
            + "WHERE json_extract(r.payload,'$.status')='invalid_prices' AND s.state IS NOT NULL GROUP BY s.state"
        )
    )
    misses = [
        dict(
            slot=s,
            detected_at=t,
            reason=r,
            deadline=(utc(s) + timedelta(minutes=45)).isoformat(),
        )
        for s, t, r in db.execute("""SELECT slot,json_extract(payload,'$.recorded_at'),json_extract(payload,'$.reason')
                  FROM pilot.observations WHERE json_extract(payload,'$.status')='missed' ORDER BY slot""")
    ]
    # Only use weights that were actually saved by the registered pilot. The
    # perpetual observer has no mandate to invent a new policy after its end.
    paired, exercised, paired_pp = db.execute("""
        WITH pairs AS (
          SELECT s.reversal_net_pct-s.model_net_pct AS spread,
                 json_extract(p.payload,'$.weights.reversal') AS w,
                 json_extract(p.payload,'$.previous_reversal_weight') AS prev,
                 json_extract(contract.payload,'$.weight_change_bps') AS bps
          FROM pilot.observations p JOIN slots s USING(slot) CROSS JOIN pilot.protocol contract
          WHERE s.model_net_pct IS NOT NULL AND json_extract(p.payload,'$.status')='recorded')
        SELECT count(*),count(CASE WHEN w>0 THEN 1 END),avg(w*spread-abs(w-prev)*bps/100.0)
        FROM pairs""").fetchone()
    return {
        "pilot_invalid": invalid,
        "pilot_misses": misses,
        "pilot_backup_comparison": {
            "paired": paired,
            "exercised": exercised,
            "mean_difference_pp": paired_pp,
        },
    }


def aggregates(db, now, backup, pending):
    last = db.execute("""SELECT slot FROM observer.observations
                       WHERE json_extract(payload,'$.status')='recorded' ORDER BY slot DESC LIMIT 1""").fetchone()
    return {
        "schema": 1,
        "checked_at": now.isoformat(),
        "backup": backup.name,
        "pending": pending,
        "latest_capture": last[0] if last else None,
        "capture_stale": not last or now - utc(last[0]) > timedelta(hours=8),
        "observer": dict(db.execute("SELECT state,count(*) FROM slots GROUP BY state")),
        "by_regime": [
            dict(regime=r, state=s, n=n)
            for r, s, n in db.execute(
                "SELECT regime,state,count(*) FROM slots WHERE regime IS NOT NULL GROUP BY regime,state ORDER BY regime,state"
            )
        ],
        "probes": dict(
            db.execute("SELECT result,count(*) FROM probes GROUP BY result")
        ),
        "large_gap_count": db.execute(
            "SELECT count(*) FROM slots WHERE official_gap>=20"
        ).fetchone()[0],
        "large_gap_timelines": [
            dict(
                slot=s,
                captured_at=c,
                entry_at=e,
                exit_at=x,
                evaluated_at=t,
                official_missing=g,
                backup_missing=m,
            )
            for s, c, e, x, t, g, m in db.execute("""SELECT slot,captured,entry_at,exit_at,evaluated,
                                    official_gap,missing_keys FROM slots WHERE official_gap>=20 ORDER BY slot DESC LIMIT 12""")
        ],
    }


def run(root, backups, quota, now):
    folder = root / "output/basket_coverage"
    folder.mkdir(parents=True, exist_ok=True)
    backup, stamp = backup_for(backups, now)
    with (
        open(folder / "worker.lock", "a") as lock,
        closing(sqlite3.connect(folder / "audit.sqlite", uri=True)) as db,
        closing(sqlite3.connect(ro(backup), uri=True)) as raw,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db.execute("PRAGMA temp_store=MEMORY")
        db.executescript(SCHEMA)
        db.execute(
            "ATTACH DATABASE ? AS observer",
            (ro(root / "output/regime_observation/ledger.sqlite"),),
        )
        with db:
            pending = collect(db, raw, backup, stamp)
        probe_gaps(db, quota, now)
        report = aggregates(db, now, backup, pending)
        try:
            db.execute(
                "ATTACH DATABASE ? AS pilot",
                (ro(root / "output/rotation_pilot/ledger.sqlite"),),
            )
            report.update(pilot_comparison(db))
        except sqlite3.DatabaseError as exc:
            report.update(
                pilot_invalid=None,
                pilot_misses=None,
                pilot_backup_comparison=None,
                pilot_error=type(exc).__name__,
            )
        recovery = folder / "recovery.new.sqlite"
        with closing(sqlite3.connect(recovery)) as dest:
            db.backup(dest)
            if dest.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise sqlite3.DatabaseError("coverage recovery copy failed quick_check")
        recovery.replace(folder / "recovery.sqlite")
        summary_file(folder / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-limit", type=int, choices=range(0, 31), default=30, metavar="0..30"
    )
    args = parser.parse_args()
    os.umask(0o077)
    state = run(ROOT, BACKUPS, args.probe_limit, datetime.now(timezone.utc))
    print(
        json.dumps(
            {
                key: state[key]
                for key in (
                    "checked_at",
                    "observer",
                    "pending",
                    "pilot_invalid",
                    "probes",
                )
            }
        )
    )
