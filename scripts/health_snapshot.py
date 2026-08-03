#!/usr/bin/env python3
"""
Session kickoff health snapshot for xsec_alpha.

Run this at the start of every Claude/operator session before any edits:
    python scripts/health_snapshot.py

Aggregates existing signals into one structured view:
  A. Operational status (DB / model / recs / universe / binance freshness)
  B. IC health (ic_history.json short + long: last 1/7/30 reads, §7 gate status)
  C. Realized performance (recommendation_ledger.csv last 30d by side)
  D. Current picks (latest.csv: count, score range, watch/execute)
  E. Structural warnings (gate breached, ledger/telegram mismatch, known bugs)

Outputs:
  - stdout: human-readable report
  - output/health_snapshot.json: machine-readable (for agents)
  - logs/health_snapshot_YYYY-MM-DD.md: archived markdown

Exit codes:
  0 = healthy (all gates pass)
  1 = gate breached (§7 IC freeze/liquidate, or realized pnl < 0 for >14d)
  2 = operational failure (stale data / missing files)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
LOGS_DIR = ROOT / "logs"


# ---------------------------- helpers ---------------------------- #

def _status_tag(status: str) -> str:
    colors = {
        "OK":    "\033[32mOK  \033[0m",
        "WARN":  "\033[33mWARN\033[0m",
        "FAIL":  "\033[31mFAIL\033[0m",
        "FREEZE": "\033[31mFRZ \033[0m",
        "LIQ":   "\033[31mLIQ \033[0m",
        "BLOCKED": "\033[36mBLK \033[0m",
    }
    return colors.get(status, f"{status:>4s}")


def _load_json_records(path: Path):
    if not path.exists():
        return []
    try:
        with path.open() as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:
        return []


# --------------------- A. operational status --------------------- #

def section_operational():
    """Reuse healthcheck.py functions so we have one source of truth."""
    from scripts.healthcheck import (
        check_db, check_model, check_recs, check_universe, check_binance,
    )
    rows = []
    worst = "OK"
    for name, fn in [
        ("DB",       check_db),
        ("MODEL",    check_model),
        ("RECS",     check_recs),
        ("UNIVERSE", check_universe),
        ("BINANCE",  check_binance),
    ]:
        status, detail = fn()
        rows.append({"check": name, "status": status, "detail": detail})
        if status == "FAIL":
            worst = "FAIL"
        elif status == "WARN" and worst != "FAIL":
            worst = "WARN"
    return worst, rows


# ------------------------- B. IC health ------------------------- #

def _ic_gate_status(values: list[float]) -> tuple[str, str]:
    """CLAUDE.md §7 enforcement on a list of IC values (most-recent last)."""
    if not values:
        return "WARN", "no readings"
    last3 = values[-3:]
    # Count trailing consecutive < 0.03
    trailing_low = 0
    for v in reversed(values):
        if v < 0.03:
            trailing_low += 1
        else:
            break
    if trailing_low >= 3:
        return "LIQ", f"{trailing_low} consecutive IC<0.03 — §7: liquidate + 2-week halt"
    if trailing_low >= 2:
        return "FREEZE", f"{trailing_low} consecutive IC<0.03 — §7: freeze positions"
    if any(v < 0.03 for v in last3):
        return "WARN", f"last-3 has IC<0.03 — tightening warning"
    if any(v < 0.05 for v in last3):
        return "WARN", f"last-3 has IC<0.05 — signal softening"
    return "OK", f"last-3 all >= 0.05"


def section_ic():
    short = _load_json_records(OUTPUT_DIR / "ic_history.json")
    long_ = _load_json_records(OUTPUT_DIR / "ic_history_long.json")
    from utils.ic_gate import _policy_block_reason

    def summarise(records, side):
        if not records:
            return {"n": 0}
        latest_contract = next(
            (
                row.get("contract_version")
                for row in reversed(records)
                if row.get("contract_version")
            ),
            None,
        )
        if latest_contract:
            records = [
                row for row in records
                if row.get("contract_version") == latest_contract
            ]
        ics = [float(r["ic"]) for r in records if "ic" in r and r["ic"] is not None]
        if not ics:
            return {"n": 0}
        last1 = ics[-1]
        last7 = ics[-7:]
        last30 = ics[-30:]
        raw_status, note = _ic_gate_status(ics)
        policy_reason = _policy_block_reason(side)
        status = "BLOCKED" if policy_reason and raw_status in ("OK", "WARN") else raw_status
        return {
            "n":       len(ics),
            "last1":   last1,
            "last7":   {"mean": sum(last7)/len(last7),  "min": min(last7),  "n": len(last7)},
            "last30":  {"mean": sum(last30)/len(last30), "min": min(last30), "n": len(last30),
                        "neg_frac": sum(1 for v in last30 if v < 0) / len(last30)},
            "status":  status,
            "raw_status": raw_status,
            "note":    note,
            "execution_blocked": policy_reason is not None,
            "policy_reason": policy_reason,
            "latest_ts": records[-1].get("timestamp"),
            "horizon_h": records[-1].get("horizon_h"),
            "contract_version": latest_contract,
        }

    short_summary = summarise(short, "short")
    long_summary  = summarise(long_, "long")
    worst = "OK"
    for s in (short_summary.get("status"), long_summary.get("status")):
        if s in ("LIQ",):
            worst = "LIQ"
        elif s == "FREEZE" and worst not in ("LIQ",):
            worst = "FREEZE"
        elif s == "WARN" and worst == "OK":
            worst = "WARN"
    return worst, {"short": short_summary, "long": long_summary}


# ------------------ C. realized performance ------------------ #

def section_realized():
    import pandas as pd
    path = OUTPUT_DIR / "recommendation_ledger.csv"
    if not path.exists():
        return "WARN", {"note": "ledger not found"}
    df = pd.read_csv(path)
    if df.empty or "realized_return" not in df.columns:
        return "WARN", {"note": "ledger empty or missing realized_return"}

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    recent = df[df["entry_time"] >= cutoff].copy()
    recent["realized_return"] = pd.to_numeric(recent["realized_return"], errors="coerce")
    if "net_return" in recent.columns:
        recent["net_return"] = pd.to_numeric(recent["net_return"], errors="coerce")
        recent["performance_return"] = recent["net_return"].fillna(recent["realized_return"])
        return_metric = "net_return (legacy rows fall back to gross)"
    else:
        recent["performance_return"] = recent["realized_return"]
        return_metric = "gross realized_return (legacy ledger)"
    recent = recent.dropna(subset=["performance_return"])

    from utils.ic_gate import _policy_block_reason

    by_side = {}
    worst = "OK"
    for side, grp in recent.groupby("side"):
        n = len(grp)
        avg = float(grp["performance_return"].mean())
        std = float(grp["performance_return"].std()) if n > 1 else 0.0
        win_rate = float((grp["performance_return"] > 0).mean())
        raw_status = "OK"
        # 30d avg < 0 with at least 20 observations is concerning
        if n >= 20 and avg < 0:
            raw_status = "WARN"
        if n >= 20 and win_rate < 0.45:
            raw_status = "WARN"
        policy_reason = _policy_block_reason("long") if "LONG" in str(side).upper() else None
        status = "BLOCKED" if policy_reason else raw_status
        if raw_status == "WARN" and not policy_reason:
            worst = worst if worst in ("FAIL",) else "WARN"
        by_side[side] = {
            "n": n, "avg_return": avg, "std": std, "win_rate": win_rate,
            "status": status, "raw_status": raw_status,
            "execution_blocked": policy_reason is not None,
            "policy_reason": policy_reason,
        }
    return worst, {
        "window": "30d",
        "return_metric": return_metric,
        "by_side": by_side,
        "total": len(recent),
    }


# ------------------ D. current picks ------------------ #

def section_current_picks():
    import pandas as pd
    path = OUTPUT_DIR / "latest.csv"
    if not path.exists():
        return "WARN", {"note": "latest.csv missing"}
    try:
        df = pd.read_csv(os.path.realpath(path))
    except Exception as e:
        return "WARN", {"note": f"could not read latest.csv: {e}"}

    summary = {
        "path": str(path),
        "rows": len(df),
        "sides": {},
    }
    if df.empty or "side" not in df.columns:
        return "WARN", {"note": "latest.csv empty or missing 'side'"}
    for side, grp in df.groupby("side"):
        scores = pd.to_numeric(grp.get("score"), errors="coerce").dropna()
        summary["sides"][side] = {
            "n": len(grp),
            "score_min": float(scores.min()) if len(scores) else None,
            "score_max": float(scores.max()) if len(scores) else None,
            "score_abs_max": float(scores.abs().max()) if len(scores) else None,
            "actionable_n": int(grp.get("actionable", False).sum()) if "actionable" in grp.columns else None,
        }
    from utils.ic_gate import _policy_block_reason
    long_policy_reason = _policy_block_reason("long")
    if long_policy_reason:
        long_rows = int(df["side"].astype(str).str.contains("LONG", case=False, na=False).sum())
        summary["blocked_sides"] = {"LONG": long_policy_reason}
        if long_rows:
            summary["policy_violation"] = (
                f"LONG execution is blocked but latest.csv still contains {long_rows} LONG/WATCH_LONG row(s)"
            )
            return "FAIL", summary
    # timestamp drift: entry_time vs now
    if "entry_time" in df.columns:
        entry_ts = pd.to_datetime(df["entry_time"], utc=True, errors="coerce").max()
        if entry_ts is not None:
            age_h = (datetime.now(timezone.utc) - entry_ts).total_seconds() / 3600
            summary["latest_entry_age_h"] = round(age_h, 2)
    return "OK", summary


# ------------------ F. feature health ------------------ #

def section_feature_health():
    """Read output/feature_health.json (written by fetch_and_rank).

    Returns (status, info) where info includes per-factor NaN% & flags.
    """
    fh_path = OUTPUT_DIR / "feature_health.json"
    if not fh_path.exists():
        return "WARN", {"available": False, "factors": {}}
    try:
        data = json.loads(fh_path.read_text())
    except Exception as e:
        return "WARN", {"available": False, "error": str(e), "factors": {}}

    factors = data.get("factors", {}) or {}
    worst = "OK"
    for col, info in factors.items():
        st = info.get("status", "OK")
        if st == "WARN":
            worst = "WARN"
    return worst, {
        "available": True,
        "generated_at": data.get("generated_at"),
        "rows": data.get("rows"),
        "factors": factors,
    }


# ------------------ E. structural warnings ------------------ #

def section_warnings(operational_rows, ic_info, realized_info, feature_health=None):
    warns = []
    if feature_health and feature_health.get("available"):
        for col, info in feature_health.get("factors", {}).items():
            if info.get("status") == "WARN":
                warns.append(
                    f"Feature health: {col} NaN={info.get('nan_pct')}% — {info.get('note')}"
                )
    # Known bug: healthcheck.py points at wrong IC path
    hc_path = ROOT / "scripts" / "healthcheck.py"
    if hc_path.exists():
        try:
            body = hc_path.read_text()
            if "logs/ic_history.csv" in body or "logs\", \"ic_history.csv\"" in body:
                warns.append("healthcheck.py reads logs/ic_history.csv but IC is written to output/ic_history.json — IC check is effectively disabled")
        except Exception:
            pass

    # Gate breached?
    for side, s in ic_info.items():
        if s.get("status") in ("FREEZE", "LIQ"):
            warns.append(f"§7 gate: {side.upper()} IC = {s.get('status')} ({s.get('note')})")
        elif s.get("status") == "BLOCKED" and s.get("raw_status") == "WARN":
            warns.append(
                f"Contained: {side.upper()} raw IC=WARN ({s.get('note')}); "
                f"execution blocked by {s.get('policy_reason')}"
            )

    # Realized perf warning
    for side, r in realized_info.get("by_side", {}).items():
        if r.get("status") == "WARN":
            warns.append(
                f"Realized perf: {side} 30d avg={r['avg_return']*100:+.2f}%, "
                f"win_rate={r['win_rate']*100:.0f}% over n={r['n']}"
            )
        elif r.get("status") == "BLOCKED" and r.get("raw_status") == "WARN":
            warns.append(
                f"Contained: {side} realized 30d avg={r['avg_return']*100:+.2f}%, "
                f"win_rate={r['win_rate']*100:.0f}% over n={r['n']}; execution blocked"
            )
    return warns


# ------------------ composition ------------------ #

def _fmt_pct(v, digits=4):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def render_text(snap):
    lines = []
    lines.append("━" * 68)
    lines.append(f"  🩺 xsec_alpha health snapshot  |  {snap['generated_at']}")
    lines.append("━" * 68)

    # A
    lines.append("")
    lines.append("A. OPERATIONAL")
    for r in snap["operational"]["rows"]:
        lines.append(f"   [{_status_tag(r['status'])}] {r['check']:9s} {r['detail']}")

    # B
    lines.append("")
    lines.append("B. SIGNAL HEALTH (IC)")
    for side, s in snap["ic"].items():
        if s.get("n", 0) == 0:
            lines.append(f"   [{_status_tag('WARN')}] {side.upper():5s} no history")
            continue
        lines.append(
            f"   [{_status_tag(s['status'])}] {side.upper():5s} h{s['horizon_h']}  "
            f"last={_fmt_pct(s['last1'])}  "
            f"mean7={_fmt_pct(s['last7']['mean'])}  "
            f"mean30={_fmt_pct(s['last30']['mean'])}  "
            f"neg30={s['last30']['neg_frac']*100:.0f}%"
        )
        lines.append(f"            {s['note']}  ({s['n']} reads, latest {s['latest_ts']})")
        if s.get("execution_blocked"):
            lines.append(
                f"            raw={s.get('raw_status')} | execution BLOCKED: {s.get('policy_reason')}"
            )

    # C
    lines.append("")
    metric = snap["realized"].get("return_metric", "realized_return")
    lines.append(f"C. REALIZED PERFORMANCE (last 30d, {metric})")
    for side, r in snap["realized"].get("by_side", {}).items():
        lines.append(
            f"   [{_status_tag(r['status'])}] {side:11s} n={r['n']:3d}  "
            f"avg={r['avg_return']*100:+.2f}%  win={r['win_rate']*100:.0f}%  "
            f"σ={r['std']*100:.2f}%"
        )
        if r.get("execution_blocked"):
            lines.append(
                f"               raw={r.get('raw_status')} | execution BLOCKED: {r.get('policy_reason')}"
            )
    if not snap["realized"].get("by_side"):
        lines.append("   (no matured positions in last 30d)")

    # D
    lines.append("")
    lines.append("D. CURRENT PICKS (output/latest.csv)")
    cur = snap["current_picks"]
    if "note" in cur:
        lines.append(f"   {cur['note']}")
    else:
        if "latest_entry_age_h" in cur:
            lines.append(f"   latest entry age: {cur['latest_entry_age_h']}h")
        for side, info in cur.get("sides", {}).items():
            extra = f" actionable={info['actionable_n']}" if info.get("actionable_n") is not None else ""
            lines.append(
                f"   {side:11s} n={info['n']}  "
                f"score∈[{_fmt_pct(info['score_min'])}, {_fmt_pct(info['score_max'])}]"
                f"{extra}"
            )
        for side, reason in cur.get("blocked_sides", {}).items():
            lines.append(f"   {side:11s} execution BLOCKED: {reason}")
        if cur.get("policy_violation"):
            lines.append(f"   [{_status_tag('FAIL')}] {cur['policy_violation']}")

    # F (feature health)
    fh = snap.get("feature_health") or {}
    if fh.get("available"):
        lines.append("")
        lines.append("F. FEATURE HEALTH (NaN%)")
        for col, info in fh.get("factors", {}).items():
            tag = _status_tag(info["status"])
            note = f"  {info['note']}" if info.get("note") else ""
            lines.append(f"   [{tag}] {col:<22} {info['nan_pct']:>5.2f}%{note}")

    # E
    lines.append("")
    lines.append("E. WARNINGS")
    if snap["warnings"]:
        for w in snap["warnings"]:
            lines.append(f"   ⚠  {w}")
    else:
        lines.append("   (none)")

    lines.append("")
    lines.append(f"OVERALL: {_status_tag(snap['overall'])}")
    lines.append("━" * 68)
    return "\n".join(lines)


def render_markdown(snap):
    lines = [
        f"# xsec_alpha health snapshot — {snap['generated_at']}",
        "",
        f"**Overall: `{snap['overall']}`**",
        "",
        "## A. Operational",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for r in snap["operational"]["rows"]:
        lines.append(f"| {r['check']} | `{r['status']}` | {r['detail']} |")

    lines += ["", "## B. Signal health (IC)", ""]
    for side, s in snap["ic"].items():
        if s.get("n", 0) == 0:
            lines.append(f"- **{side.upper()}**: no history")
            continue
        lines.append(
            f"- **{side.upper()} (h{s['horizon_h']})** `{s['status']}` — "
            f"last={_fmt_pct(s['last1'])}, mean7={_fmt_pct(s['last7']['mean'])}, "
            f"mean30={_fmt_pct(s['last30']['mean'])}, "
            f"neg30={s['last30']['neg_frac']*100:.0f}% "
            f"({s['n']} reads)"
        )
        lines.append(f"  - {s['note']}")
        if s.get("execution_blocked"):
            lines.append(
                f"  - raw=`{s.get('raw_status')}`; execution `BLOCKED`: {s.get('policy_reason')}"
            )

    metric = snap["realized"].get("return_metric", "realized_return")
    lines += ["", f"## C. Realized performance (30d, {metric})", ""]
    for side, r in snap["realized"].get("by_side", {}).items():
        lines.append(
            f"- **{side}** `{r['status']}` — n={r['n']}, "
            f"avg={r['avg_return']*100:+.2f}%, win={r['win_rate']*100:.0f}%"
        )
        if r.get("execution_blocked"):
            lines.append(
                f"  - raw=`{r.get('raw_status')}`; execution `BLOCKED`: {r.get('policy_reason')}"
            )

    lines += ["", "## D. Current picks", ""]
    cur = snap["current_picks"]
    if "note" in cur:
        lines.append(f"- {cur['note']}")
    else:
        for side, info in cur.get("sides", {}).items():
            lines.append(
                f"- **{side}**: n={info['n']}, "
                f"score∈[{_fmt_pct(info['score_min'])}, {_fmt_pct(info['score_max'])}]"
            )
        for side, reason in cur.get("blocked_sides", {}).items():
            lines.append(f"- **{side}** execution `BLOCKED`: {reason}")
        if cur.get("policy_violation"):
            lines.append(f"- **FAIL**: {cur['policy_violation']}")

    fh = snap.get("feature_health") or {}
    if fh.get("available"):
        lines += ["", "## F. Feature health (NaN%)", "", "| factor | status | NaN% | note |", "|---|---|---|---|"]
        for col, info in fh.get("factors", {}).items():
            note = info.get("note", "") or ""
            lines.append(f"| `{col}` | `{info['status']}` | {info['nan_pct']}% | {note} |")

    lines += ["", "## E. Warnings", ""]
    if snap["warnings"]:
        for w in snap["warnings"]:
            lines.append(f"- ⚠ {w}")
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-only", action="store_true", help="Emit only the JSON to stdout")
    args = ap.parse_args()

    op_worst, op_rows = section_operational()
    ic_worst, ic_info = section_ic()
    rl_worst, rl_info = section_realized()
    cp_worst, cp_info = section_current_picks()
    fh_worst, fh_info = section_feature_health()

    # Resolve overall
    precedence = ["LIQ", "FREEZE", "FAIL", "WARN", "OK"]
    all_status = [op_worst, ic_worst, rl_worst, cp_worst, fh_worst]
    overall = min(all_status, key=lambda s: precedence.index(s) if s in precedence else 99)

    warnings = section_warnings(op_rows, ic_info, rl_info, fh_info)

    snap = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall": overall,
        "operational": {"status": op_worst, "rows": op_rows},
        "ic":            ic_info,
        "realized":      rl_info,
        "current_picks": cp_info,
        "feature_health": fh_info,
        "warnings":      warnings,
    }

    # Write JSON
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "health_snapshot.json").write_text(json.dumps(snap, indent=2, default=str))

    # Write markdown archive (one per day, overwritten within day)
    LOGS_DIR.mkdir(exist_ok=True)
    md_path = LOGS_DIR / f"health_snapshot_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.md"
    md_path.write_text(render_markdown(snap))

    if args.json_only:
        print(json.dumps(snap, indent=2, default=str))
    else:
        print(render_text(snap))
        print(f"\nWrote: {OUTPUT_DIR/'health_snapshot.json'}")
        print(f"Wrote: {md_path}")

    # Exit code
    if overall in ("LIQ", "FREEZE"):
        sys.exit(1)
    if overall == "FAIL":
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
