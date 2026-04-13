#!/usr/bin/env python3
"""
Operational health check for xsec_alpha.

Usage:
    cd /mnt/20t/main/gan_t/xsec_alpha
    python scripts/healthcheck.py

Checks:
  [DB]       DB accessible, latest timestamp within 6h
  [MODEL]    Model file exists and is < 7 days old
  [RECS]     Latest recommendations CSV exists and is < 13h old
  [UNIVERSE] Universe >= 100 coins
  [BINANCE]  Binance data within 12h
  [IC]       If IC history exists, latest IC > 0.05

Exit 0 if all OK, exit 1 if any FAIL.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import sqlite3
from datetime import datetime, timezone

from config import config
from data.database import get_latest_db_timestamp, get_all_krw_markets_in_db, get_db_connection


def _hours_ago(dt):
    """Return hours between now (UTC) and dt."""
    if dt is None:
        return float("inf")
    now = datetime.now(timezone.utc)
    return (now - dt).total_seconds() / 3600


def _file_age_days(path):
    """Return file age in days, or inf if missing."""
    if not os.path.exists(path):
        return float("inf")
    mtime = os.path.getmtime(path)
    return (time.time() - mtime) / 86400


def _file_age_hours(path):
    """Return file age in hours, or inf if missing."""
    if not os.path.exists(path):
        return float("inf")
    mtime = os.path.getmtime(path)
    return (time.time() - mtime) / 3600


def check_db():
    """[DB] DB accessible, latest timestamp within 6h."""
    try:
        latest = get_latest_db_timestamp()
    except Exception as e:
        return "FAIL", f"DB connection error: {e}"

    if latest is None:
        return "FAIL", "No data in DB"

    hours = _hours_ago(latest)
    detail = f"latest={latest.strftime('%Y-%m-%d %H:%M')} UTC ({hours:.1f}h ago)"
    if hours > 6:
        return "FAIL", f"Stale: {detail}"
    return "OK", detail


def check_model():
    """[MODEL] Model file exists and is < 7 days old."""
    root = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(root, config.Model.MODEL_PATH)

    if not os.path.exists(model_path):
        return "FAIL", f"Not found: {model_path}"

    age = _file_age_days(model_path)
    detail = f"{model_path} ({age:.1f}d old)"
    if age > 7:
        return "WARN", f"Stale model: {detail}"
    return "OK", detail


def check_recs():
    """[RECS] Latest recommendations CSV exists and is < 13h old."""
    root = os.path.dirname(os.path.dirname(__file__))
    recs_path = os.path.join(root, "output", "latest.csv")

    if not os.path.exists(recs_path):
        return "FAIL", f"Not found: {recs_path}"

    age = _file_age_hours(recs_path)
    detail = f"{recs_path} ({age:.1f}h old)"
    if age > 13:
        return "FAIL", f"Stale: {detail}"
    return "OK", detail


def check_universe():
    """[UNIVERSE] Universe >= 100 coins."""
    try:
        markets = get_all_krw_markets_in_db()
    except Exception as e:
        return "FAIL", f"Error querying universe: {e}"

    n = len(markets)
    detail = f"{n} coins"
    if n < 100:
        return "FAIL", f"Too few: {detail}"
    return "OK", detail


def check_binance():
    """[BINANCE] Binance data within 12h."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT MAX(timestamp) FROM binance_data")
        row = cur.fetchone()
        conn.close()
    except Exception as e:
        return "WARN", f"binance_data table not accessible: {e}"

    if row is None or row[0] is None:
        return "WARN", "No binance_data rows"

    ts_str = row[0]
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
    except Exception:
        return "WARN", f"Cannot parse timestamp: {ts_str}"

    hours = _hours_ago(dt)
    detail = f"latest={dt.strftime('%Y-%m-%d %H:%M')} UTC ({hours:.1f}h ago)"
    if hours > 12:
        return "FAIL", f"Stale: {detail}"
    return "OK", detail


def check_ic():
    """[IC] If IC history exists, latest IC > 0.05."""
    root = os.path.dirname(os.path.dirname(__file__))
    ic_path = os.path.join(root, "logs", "ic_history.csv")

    if not os.path.exists(ic_path):
        return "OK", "No IC history yet (skipped)"

    try:
        import pandas as pd
        df = pd.read_csv(ic_path)
        if df.empty:
            return "OK", "IC history empty (skipped)"

        # Expect column named 'ic' or 'mean_ic'
        ic_col = None
        for c in ["ic", "mean_ic", "IC"]:
            if c in df.columns:
                ic_col = c
                break
        if ic_col is None:
            return "WARN", f"IC history has no 'ic' column; cols={list(df.columns)}"

        latest_ic = df[ic_col].iloc[-1]
        detail = f"latest IC={latest_ic:.4f}"
        if latest_ic < 0.05:
            return "WARN", f"Low: {detail}"
        return "OK", detail
    except Exception as e:
        return "WARN", f"Error reading IC history: {e}"


def main():
    checks = [
        ("DB", check_db),
        ("MODEL", check_model),
        ("RECS", check_recs),
        ("UNIVERSE", check_universe),
        ("BINANCE", check_binance),
        ("IC", check_ic),
    ]

    has_fail = False
    for name, fn in checks:
        status, detail = fn()
        tag = {"OK": "\033[32mOK\033[0m", "WARN": "\033[33mWARN\033[0m", "FAIL": "\033[31mFAIL\033[0m"}
        print(f"  [{tag.get(status, status):>4s}] {name:10s} {detail}")
        if status == "FAIL":
            has_fail = True

    if has_fail:
        print("\nHealth check: FAIL")
        sys.exit(1)
    else:
        print("\nHealth check: OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
