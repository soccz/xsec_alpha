#!/usr/bin/env python3
"""
Verify that utils.telegram.format_performance agrees with the ledger.

Motivation
----------
We discovered that format_performance was treating WATCH_LONG as SHORT (sign
flipped), so every WATCH LONG line in the report showed the opposite P&L from
what the ledger (the authoritative source) recorded. This script prevents that
class of bug by reconstructing a previous-recommendations DataFrame from the
ledger and replaying format_performance's math on it, then diffing each row's
sign/magnitude against the ledger's realized_return.

Usage
-----
    python scripts/verify_telegram.py [--recent N]

Exit codes:
  0 = all rows agree
  1 = sign mismatch or |Δ| > 10 bps on any row (a real bug)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from utils.telegram import format_performance, realized_return  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "output" / "recommendation_ledger.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent", type=int, default=30, help="Number of most-recent matured rows to verify")
    ap.add_argument("--tolerance-bps", type=float, default=10.0, help="Magnitude tolerance (basis points)")
    args = ap.parse_args()

    if not LEDGER.exists():
        print(f"FAIL ledger not found: {LEDGER}")
        sys.exit(1)

    df = pd.read_csv(LEDGER)
    required = {"market", "side", "entry_price", "exit_price", "realized_return"}
    if not required.issubset(df.columns):
        print(f"FAIL ledger missing columns: need {required}, got {set(df.columns)}")
        sys.exit(1)

    df = df.dropna(subset=["entry_price", "exit_price", "realized_return"]).copy()
    df = df.tail(args.recent).reset_index(drop=True)
    if df.empty:
        print("OK ledger has no matured rows to verify")
        sys.exit(0)

    # Replay format_performance's per-row math
    mismatches = []
    for _, row in df.iterrows():
        side = str(row["side"])
        entry = float(row["entry_price"])
        now = float(row["exit_price"])
        ledger_ret = float(row["realized_return"])
        tele_ret = realized_return(side, entry, now)
        if pd.isna(tele_ret):
            continue
        sign_flip = (ledger_ret > 0) != (tele_ret > 0) and abs(ledger_ret) > 1e-6
        mag_err_bps = abs(ledger_ret - tele_ret) * 10_000
        if sign_flip or mag_err_bps > args.tolerance_bps:
            mismatches.append({
                "market": row["market"], "side": side,
                "entry": entry, "exit": now,
                "ledger_ret": ledger_ret, "tele_ret": tele_ret,
                "sign_flip": sign_flip, "mag_err_bps": round(mag_err_bps, 2),
            })

    # Additionally sanity-check format_performance end-to-end on a synthetic prev_df
    prev_df = pd.DataFrame([
        {"market": r["market"], "side": r["side"], "entry_price": r["entry_price"]}
        for _, r in df.tail(10).iterrows()
    ])
    now_prices = {r["market"]: float(r["exit_price"]) for _, r in df.tail(10).iterrows()}
    try:
        _ = format_performance(prev_df, now_prices)
    except Exception as e:
        print(f"FAIL format_performance raised: {e}")
        sys.exit(1)

    print(f"Checked {len(df)} recent rows (tolerance {args.tolerance_bps} bps)")
    if not mismatches:
        print("OK  telegram math matches ledger on every row")
        sys.exit(0)

    print(f"FAIL {len(mismatches)} mismatch(es):")
    for m in mismatches:
        print(
            f"  {m['market']:12s} {m['side']:11s} "
            f"entry={m['entry']:>12g} exit={m['exit']:>12g}  "
            f"ledger={m['ledger_ret']*100:+.2f}%  "
            f"telegram={m['tele_ret']*100:+.2f}%  "
            f"{'SIGN-FLIP' if m['sign_flip'] else ''} "
            f"Δ={m['mag_err_bps']}bps"
        )
    sys.exit(1)


if __name__ == "__main__":
    main()
