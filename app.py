"""xsec_alpha Flask dashboard — port 5555."""

import json
import math
import os
import sys
from datetime import datetime, timezone

from flask import Flask, Response, jsonify

sys.path.insert(0, os.path.dirname(__file__))
from config import config
from data.database import get_latest_db_timestamp, get_all_krw_markets_in_db

app = Flask(__name__)

_ROOT = os.path.dirname(os.path.abspath(__file__))
_OUTPUT = os.path.join(_ROOT, "output")
_LATEST_CSV = os.path.join(_OUTPUT, "latest.csv")
_IC_HISTORY = os.path.join(_OUTPUT, "ic_history.json")
_IC_HISTORY_LONG = os.path.join(_OUTPUT, "ic_history_long.json")
_LEDGER_CSV = os.path.join(_OUTPUT, "recommendation_ledger.csv")
_MODEL_PATH = os.path.join(_ROOT, config.Model.MODEL_PATH)


# ---------------------------------------------------------------------------
# NaN-safe JSON serialization
# ---------------------------------------------------------------------------

def _nan_safe(obj):
    """Replace NaN/Infinity with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_safe(v) for v in obj]
    return obj


def _json_response(data, status=200):
    body = json.dumps(_nan_safe(data), ensure_ascii=False, default=str)
    return Response(body, status=status, mimetype="application/json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_latest_csv():
    """Load output/latest.csv if it exists. Returns (df_or_None, mtime_utc_or_None)."""
    import pandas as pd

    if not os.path.exists(_LATEST_CSV):
        return None, None
    try:
        df = pd.read_csv(_LATEST_CSV)
        mtime = datetime.fromtimestamp(os.path.getmtime(_LATEST_CSV), tz=timezone.utc)
        return df, mtime
    except Exception:
        return None, None


def _load_ic_history():
    if not os.path.exists(_IC_HISTORY):
        return None
    try:
        with open(_IC_HISTORY) as f:
            return json.load(f)
    except Exception:
        return None


def _load_long_ic_history():
    if not os.path.exists(_IC_HISTORY_LONG):
        return None
    try:
        with open(_IC_HISTORY_LONG) as f:
            return json.load(f)
    except Exception:
        return None


def _load_ledger_csv():
    import pandas as pd

    if not os.path.exists(_LEDGER_CSV):
        return None
    try:
        return pd.read_csv(_LEDGER_CSV)
    except Exception:
        return None


def _model_age():
    """Return model file mtime as UTC datetime, or None."""
    if not os.path.exists(_MODEL_PATH):
        return None
    try:
        return datetime.fromtimestamp(os.path.getmtime(_MODEL_PATH), tz=timezone.utc)
    except Exception:
        return None


def _fmt_ts(dt):
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _safe_float(v):
    try:
        v = float(v)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _safe_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _latest_ic_point(history):
    if not history:
        return None
    try:
        last = history[-1]
    except Exception:
        return None
    return {
        "timestamp": last.get("timestamp"),
        "ic": _safe_float(last.get("ic")),
        "n_coins": last.get("n_coins"),
        "horizon_h": last.get("horizon_h"),
        "side": last.get("side"),
    }


def _live_meta(df):
    meta = {
        "long_count": 0,
        "short_count": 0,
        "long_next_rebalance": None,
        "short_next_rebalance": None,
        "long_refresh_reason": None,
        "short_refresh_reason": None,
        "long_mode": "WATCH",
        "short_mode": "EXEC",
    }
    if df is None or df.empty or "side" not in df.columns:
        return meta

    side_series = df["side"].astype(str)
    long_df = df[side_series.str.contains("long", case=False, na=False)].copy()
    short_df = df[side_series.str.contains("short", case=False, na=False)].copy()

    meta["long_count"] = int(len(long_df))
    meta["short_count"] = int(len(short_df))

    if not long_df.empty:
        meta["long_next_rebalance"] = long_df["next_rebalance_at"].iloc[0] if "next_rebalance_at" in long_df.columns else None
        meta["long_refresh_reason"] = long_df["refresh_reason"].iloc[0] if "refresh_reason" in long_df.columns else None
        if long_df["side"].astype(str).str.upper().eq("LONG").any():
            meta["long_mode"] = "EXEC"

    if not short_df.empty:
        meta["short_next_rebalance"] = short_df["next_rebalance_at"].iloc[0] if "next_rebalance_at" in short_df.columns else None
        meta["short_refresh_reason"] = short_df["refresh_reason"].iloc[0] if "refresh_reason" in short_df.columns else None

    return meta


def _ledger_summary(df):
    import pandas as pd
    from utils.dashboard_export import _pick_history, _realized_cohorts

    summary = {
        "total_closed": 0,
        "actionable_closed": 0,
        "watch_long_closed": 0,
        "short_closed": 0,
        "mean_return_all": None,
        "mean_return_actionable": None,
        "mean_return_watch_long": None,
        "mean_return_short": None,
        "mean_net_all": None,
        "mean_net_actionable": None,
        "mean_net_watch_long": None,
        "mean_net_short": None,
        "last_exit_time": None,
        "decision_cohorts_30d": {},
    }
    if df is None or df.empty:
        return summary

    summary["total_closed"] = int(len(df))
    summary["decision_cohorts_30d"] = _realized_cohorts(
        _pick_history(df.to_dict(orient="records"), days=30), [30],
    )
    if "actionable" in df.columns:
        actionable_mask = df["actionable"].apply(_safe_bool)
        summary["actionable_closed"] = int(actionable_mask.sum())
    else:
        actionable_mask = None

    if "side" in df.columns:
        side_series = df["side"].astype(str)
        long_mask = side_series.str.contains("long", case=False, na=False)
        short_mask = side_series.str.contains("short", case=False, na=False)
        watch_long_mask = side_series.str.upper().eq("WATCH_LONG")
        summary["watch_long_closed"] = int(watch_long_mask.sum())
        summary["short_closed"] = int(short_mask.sum())
    else:
        long_mask = short_mask = watch_long_mask = None

    if "realized_return" in df.columns:
        rr = df["realized_return"].apply(_safe_float)
        summary["mean_return_all"] = rr.dropna().mean() if rr.notna().any() else None
        if actionable_mask is not None and actionable_mask.any():
            ar = rr[actionable_mask].dropna()
            summary["mean_return_actionable"] = ar.mean() if not ar.empty else None
        if watch_long_mask is not None and watch_long_mask.any():
            wr = rr[watch_long_mask].dropna()
            summary["mean_return_watch_long"] = wr.mean() if not wr.empty else None
        if short_mask is not None and short_mask.any():
            sr = rr[short_mask].dropna()
            summary["mean_return_short"] = sr.mean() if not sr.empty else None

    if "net_return" in df.columns:
        nr = df["net_return"].apply(_safe_float)
        summary["mean_net_all"] = nr.dropna().mean() if nr.notna().any() else None
        if actionable_mask is not None and actionable_mask.any():
            an = nr[actionable_mask].dropna()
            summary["mean_net_actionable"] = an.mean() if not an.empty else None
        if watch_long_mask is not None and watch_long_mask.any():
            wn = nr[watch_long_mask].dropna()
            summary["mean_net_watch_long"] = wn.mean() if not wn.empty else None
        if short_mask is not None and short_mask.any():
            sn = nr[short_mask].dropna()
            summary["mean_net_short"] = sn.mean() if not sn.empty else None

    if "exit_time_actual" in df.columns:
        exits = pd.to_datetime(df["exit_time_actual"], utc=True, errors="coerce").dropna()
        if not exits.empty:
            summary["last_exit_time"] = exits.max().isoformat()

    return summary


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/api/recommendations")
def api_recommendations():
    df, mtime = _load_latest_csv()
    if df is None:
        return _json_response({"error": "No recommendations yet", "data": []}, 200)
    records = df.to_dict(orient="records")
    return _json_response({
        "generated_at": mtime.isoformat() if mtime else None,
        "count": len(records),
        "data": records,
    })


@app.route("/api/ic")
def api_ic():
    short_ic = _load_ic_history()
    long_ic = _load_long_ic_history()
    if short_ic is None and long_ic is None:
        return _json_response({"error": "No IC history yet", "data": {}}, 200)
    return _json_response({
        "short": short_ic or [],
        "long": long_ic or [],
        "latest_short": _latest_ic_point(short_ic),
        "latest_long": _latest_ic_point(long_ic),
    })


@app.route("/api/ledger")
def api_ledger():
    df = _load_ledger_csv()
    if df is None:
        return _json_response({"error": "No ledger yet", "summary": _ledger_summary(None), "data": []}, 200)
    records = df.to_dict(orient="records")
    return _json_response({
        "summary": _ledger_summary(df),
        "count": len(records),
        "data": records[-100:],
    })


@app.route("/api/experiment")
def api_experiment():
    from utils.experiment_supervisor import review_summary
    try:
        return _json_response(review_summary())
    except Exception:
        app.logger.exception("Prospective experiment unavailable")
        return _json_response({"status": "error"}, 503)


@app.route("/api/status")
def api_status():
    db_ts = get_latest_db_timestamp()
    model_ts = _model_age()
    try:
        markets = get_all_krw_markets_in_db()
    except Exception:
        markets = []

    latest_df, rec_mtime = _load_latest_csv()
    ledger_df = _load_ledger_csv()
    meta = _live_meta(latest_df)
    ledger = _ledger_summary(ledger_df)

    now = datetime.now(timezone.utc)
    db_age_h = round((now - db_ts).total_seconds() / 3600, 1) if db_ts else None
    model_age_h = round((now - model_ts).total_seconds() / 3600, 1) if model_ts else None

    return _json_response({
        "db_latest": db_ts.isoformat() if db_ts else None,
        "db_age_hours": db_age_h,
        "model_updated": model_ts.isoformat() if model_ts else None,
        "model_age_hours": model_age_h,
        "universe_size": len(markets),
        "last_recommendation": rec_mtime.isoformat() if rec_mtime else None,
        "long_mode": meta["long_mode"],
        "short_mode": meta["short_mode"],
        "long_count": meta["long_count"],
        "short_count": meta["short_count"],
        "long_next_rebalance": meta["long_next_rebalance"],
        "short_next_rebalance": meta["short_next_rebalance"],
        "long_refresh_reason": meta["long_refresh_reason"],
        "short_refresh_reason": meta["short_refresh_reason"],
        "ledger_summary": ledger,
    })


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>xsec_alpha Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
         background: #0d1117; color: #c9d1d9; padding: 20px; }
  h1 { color: #58a6ff; margin-bottom: 4px; font-size: 1.4em; }
  .subtitle { color: #8b949e; font-size: 0.85em; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; min-width: 0; overflow-x: auto; }
  .card h2 { color: #58a6ff; font-size: 1em; margin-bottom: 10px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
  .stat { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 4px 12px; padding: 4px 0; font-size: 0.9em; }
  .stat .label { color: #8b949e; }
  .stat .value { color: #f0f6fc; font-weight: 600; }
  .long { color: #3fb950; }
  .short { color: #f85149; }
  .watch { color: #d29922; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  th { text-align: left; color: #8b949e; padding: 6px 8px; border-bottom: 1px solid #30363d; }
  td { padding: 5px 8px; border-bottom: 1px solid #21262d; }
  .empty { color: #8b949e; font-style: italic; padding: 20px; text-align: center; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: 600; }
  .badge-long { background: #0f2d1a; color: #3fb950; }
  .badge-short { background: #2d0f0f; color: #f85149; }
  .badge-watch { background: #2d2200; color: #d29922; }
  .full-width { grid-column: 1 / -1; }
  .refresh-note { text-align: center; color: #484f58; font-size: 0.75em; margin-top: 12px; }
  @media (max-width: 700px) {
    body { padding: 12px; }
    .grid { grid-template-columns: minmax(0, 1fr); }
    .stat .value { overflow-wrap: anywhere; }
  }
</style>
</head>
<body>
<h1>xsec_alpha</h1>
<div class="subtitle">Cross-sectional crypto ranking | Auto-refresh 5 min</div>

<div class="grid">
  <!-- Status card -->
  <div class="card">
    <h2>System Status</h2>
    <div class="stat"><span class="label">DB Latest</span><span class="value">{{db_latest}}</span></div>
    <div class="stat"><span class="label">DB Age</span><span class="value">{{db_age}}</span></div>
    <div class="stat"><span class="label">Universe Size</span><span class="value">{{universe_size}}</span></div>
    <div class="stat"><span class="label">Model Updated</span><span class="value">{{model_updated}}</span></div>
    <div class="stat"><span class="label">Last Run</span><span class="value">{{last_run}}</span></div>
    <div class="stat"><span class="label">Long Mode</span><span class="value">{{long_mode}}</span></div>
    <div class="stat"><span class="label">Short Mode</span><span class="value">{{short_mode}}</span></div>
    <div class="stat"><span class="label">Long Next Rebalance</span><span class="value">{{long_next_rebalance}}</span></div>
    <div class="stat"><span class="label">Short Next Rebalance</span><span class="value">{{short_next_rebalance}}</span></div>
    <div class="stat"><span class="label">Long Refresh</span><span class="value">{{long_refresh_reason}}</span></div>
    <div class="stat"><span class="label">Short Refresh</span><span class="value">{{short_refresh_reason}}</span></div>
  </div>

  <!-- IC card -->
  <div class="card">
    <h2>IC Summary</h2>
    {{ic_block}}
  </div>

  <!-- Ledger card -->
  <div class="card">
    <h2>Realized Ledger</h2>
    {{ledger_block}}
  </div>

  <!-- Live meta card -->
  <div class="card">
    <h2>Live Mix</h2>
    <div class="stat"><span class="label">{{long_label}} Count</span><span class="value">{{long_count}}</span></div>
    <div class="stat"><span class="label">{{short_label}} Count</span><span class="value">{{short_count}}</span></div>
    <div class="stat"><span class="label">{{long_label}} Horizon</span><span class="value">{{long_horizon}}</span></div>
    <div class="stat"><span class="label">{{short_label}} Horizon</span><span class="value">{{short_horizon}}</span></div>
  </div>

  <!-- Long positions -->
  <div class="card">
    <h2><span class="long">{{long_label}}</span> Positions (Top {{long_n}})</h2>
    {{long_table}}
  </div>

  <!-- Short positions -->
  <div class="card">
    <h2><span class="short">{{short_label}}</span> Positions (Bottom {{short_n}})</h2>
    {{short_table}}
  </div>

  <!-- Full table -->
  <div class="card full-width">
    <h2>All Recommendations ({{total_count}})</h2>
    {{full_table}}
  </div>
</div>

<div class="refresh-note">Page auto-refreshes every 5 minutes | {{now_utc}}</div>
</body>
</html>"""


def _render_rec_table(df, max_rows=None):
    """Render a pandas DataFrame subset as an HTML table."""
    if df is None or df.empty:
        return '<div class="empty">No recommendations yet</div>'

    cols = list(df.columns)
    rows = df.head(max_rows) if max_rows else df

    html = ["<table><thead><tr>"]
    for c in cols:
        html.append(f"<th>{c}</th>")
    html.append("</tr></thead><tbody>")

    for _, row in rows.iterrows():
        html.append("<tr>")
        for c in cols:
            val = row[c]
            # Format floats
            if isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    cell = "N/A"
                else:
                    cell = f"{val:.4f}" if abs(val) < 10 else f"{val:.2f}"
            else:
                cell = str(val) if val is not None else "N/A"

            # Color signal column
            if c.lower() in ("signal", "side", "direction", "action"):
                lv = cell.lower()
                if "watch" in lv:
                    cell = f'<span class="badge badge-watch">{cell}</span>'
                elif "long" in lv:
                    cell = f'<span class="badge badge-long">{cell}</span>'
                elif "short" in lv:
                    cell = f'<span class="badge badge-short">{cell}</span>'

            html.append(f"<td>{cell}</td>")
        html.append("</tr>")

    html.append("</tbody></table>")
    return "".join(html)


def _render_ic_block(ic_data):
    if ic_data is None:
        return '<div class="empty">No IC data yet</div>'

    if isinstance(ic_data, dict) and ("latest_short" in ic_data or "latest_long" in ic_data):
        html = []
        for label, point in [("Short", ic_data.get("latest_short")), ("Long", ic_data.get("latest_long"))]:
            if not point:
                html.append(f'<div class="stat"><span class="label">{label} IC</span><span class="value">N/A</span></div>')
                continue
            ic_val = point.get("ic")
            ic_disp = f"{ic_val:+.4f}" if ic_val is not None else "N/A"
            ts_disp = point.get("timestamp", "N/A")
            n_disp = point.get("n_coins", "N/A")
            h_disp = point.get("horizon_h", "N/A")
            html.append(f'<div class="stat"><span class="label">{label} IC</span><span class="value">{ic_disp}</span></div>')
            html.append(f'<div class="stat"><span class="label">{label} Horizon</span><span class="value">{h_disp}h</span></div>')
            html.append(f'<div class="stat"><span class="label">{label} Coins</span><span class="value">{n_disp}</span></div>')
            html.append(f'<div class="stat"><span class="label">{label} Timestamp</span><span class="value">{ts_disp}</span></div>')
        return "".join(html)

    # Handle generic dict-of-stats formats
    if isinstance(ic_data, dict):
        stats = ic_data.get("summary", ic_data.get("stats", ic_data))
        if isinstance(stats, dict):
            html = []
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        disp = "N/A"
                    else:
                        disp = f"{v:.4f}" if isinstance(v, float) else str(v)
                else:
                    disp = str(v) if v is not None else "N/A"
                html.append(f'<div class="stat"><span class="label">{k}</span><span class="value">{disp}</span></div>')
            return "".join(html) if html else '<div class="empty">No IC stats</div>'

    return '<div class="empty">IC data format not recognized</div>'


def _render_ledger_block(summary):
    if not summary or summary.get("total_closed", 0) == 0:
        return '<div class="empty">No matured positions yet</div>'

    def _pct(v):
        if v is None:
            return "N/A"
        return f"{v:+.2%}"

    html = [
        '<div class="stat"><span class="label">Return Basis</span><span class="value">Paper / Upbit proxy</span></div>',
        f'<div class="stat"><span class="label">Closed Positions</span><span class="value">{summary["total_closed"]}</span></div>',
        f'<div class="stat"><span class="label">Actionable Closed</span><span class="value">{summary["actionable_closed"]}</span></div>',
        f'<div class="stat"><span class="label">Watch Long Closed</span><span class="value">{summary["watch_long_closed"]}</span></div>',
        f'<div class="stat"><span class="label">Short Closed</span><span class="value">{summary["short_closed"]}</span></div>',
        f'<div class="stat"><span class="label">Mean Return All</span><span class="value">{_pct(summary["mean_return_all"])}</span></div>',
        f'<div class="stat"><span class="label">Mean Return Actionable</span><span class="value">{_pct(summary["mean_return_actionable"])}</span></div>',
        f'<div class="stat"><span class="label">Mean Return Watch Long</span><span class="value">{_pct(summary["mean_return_watch_long"])}</span></div>',
        f'<div class="stat"><span class="label">Mean Return Short</span><span class="value">{_pct(summary["mean_return_short"])}</span></div>',
        f'<div class="stat"><span class="label">Mean Net All</span><span class="value">{_pct(summary["mean_net_all"])}</span></div>',
        f'<div class="stat"><span class="label">Mean Net Actionable</span><span class="value">{_pct(summary["mean_net_actionable"])}</span></div>',
        f'<div class="stat"><span class="label">Mean Net Short</span><span class="value">{_pct(summary["mean_net_short"])}</span></div>',
        f'<div class="stat"><span class="label">Last Exit</span><span class="value">{summary["last_exit_time"] or "N/A"}</span></div>',
    ]
    for cohort, label in (("actionable", "Recommendations"), ("watch", "Observations")):
        stats = summary.get("decision_cohorts_30d", {}).get(cohort, {}).get("SHORT", {}).get("d30", {})
        net = stats.get("avg_net_per_window")
        value = f"{net:+.2f}%" if net is not None else "N/A"
        html.append(
            f'<div class="stat"><span class="label">30d SHORT {label}</span>'
            f'<span class="value">{value} ({stats.get("n_windows", 0)} baskets)</span></div>'
        )
    return "".join(html)


def _live_recommendation_labels():
    mode = getattr(config.Portfolio, "LIVE_EXECUTION_MODE", "balanced")
    if mode == "short_only":
        return {
            "long_label": "WATCH LONG",
            "long_n": getattr(config.Portfolio, "LIVE_WATCH_LONG_N", config.Portfolio.LONG_N),
            "short_label": "SHORT EXEC",
            "short_n": getattr(config.Portfolio, "LIVE_EXEC_SHORT_N", config.Portfolio.SHORT_N),
        }
    return {
        "long_label": "LONG",
        "long_n": config.Portfolio.LONG_N,
        "short_label": "SHORT",
        "short_n": config.Portfolio.SHORT_N,
    }


@app.route("/")
def dashboard():
    import pandas as pd

    # Status
    db_ts = get_latest_db_timestamp()
    model_ts = _model_age()
    try:
        markets = get_all_krw_markets_in_db()
    except Exception:
        markets = []

    now = datetime.now(timezone.utc)

    db_age_str = "N/A"
    if db_ts:
        age_h = (now - db_ts).total_seconds() / 3600
        db_age_str = f"{age_h:.1f}h ago"

    # Recommendations
    df, rec_mtime = _load_latest_csv()
    meta = _live_meta(df)
    ledger_df = _load_ledger_csv()
    ledger = _ledger_summary(ledger_df)

    # Split long/short
    long_df = short_df = None
    if df is not None and not df.empty:
        sig_col = None
        for c in ("signal", "side", "direction", "action"):
            if c in df.columns:
                sig_col = c
                break
        if sig_col:
            long_df = df[df[sig_col].str.lower().str.contains("long", na=False)]
            short_df = df[df[sig_col].str.lower().str.contains("short", na=False)]

    # IC
    ic_data = {
        "latest_short": _latest_ic_point(_load_ic_history()),
        "latest_long": _latest_ic_point(_load_long_ic_history()),
    }
    live_labels = _live_recommendation_labels()

    html = _DASHBOARD_HTML
    html = html.replace("{{db_latest}}", _fmt_ts(db_ts))
    html = html.replace("{{db_age}}", db_age_str)
    html = html.replace("{{universe_size}}", str(len(markets)))
    html = html.replace("{{model_updated}}", _fmt_ts(model_ts))
    html = html.replace("{{last_run}}", _fmt_ts(rec_mtime))
    html = html.replace("{{long_mode}}", meta["long_mode"])
    html = html.replace("{{short_mode}}", meta["short_mode"])
    html = html.replace("{{long_next_rebalance}}", str(meta["long_next_rebalance"] or "N/A"))
    html = html.replace("{{short_next_rebalance}}", str(meta["short_next_rebalance"] or "N/A"))
    html = html.replace("{{long_refresh_reason}}", str(meta["long_refresh_reason"] or "N/A"))
    html = html.replace("{{short_refresh_reason}}", str(meta["short_refresh_reason"] or "N/A"))
    html = html.replace("{{long_label}}", live_labels["long_label"])
    html = html.replace("{{short_label}}", live_labels["short_label"])
    html = html.replace("{{long_n}}", str(live_labels["long_n"]))
    html = html.replace("{{short_n}}", str(live_labels["short_n"]))
    html = html.replace("{{long_count}}", str(meta["long_count"]))
    html = html.replace("{{short_count}}", str(meta["short_count"]))
    html = html.replace("{{long_horizon}}", "12h")
    html = html.replace("{{short_horizon}}", "6h")
    html = html.replace("{{long_table}}", _render_rec_table(long_df, max_rows=live_labels["long_n"]))
    html = html.replace("{{short_table}}", _render_rec_table(short_df, max_rows=live_labels["short_n"]))
    html = html.replace("{{total_count}}", str(len(df)) if df is not None else "0")
    html = html.replace("{{full_table}}", _render_rec_table(df, max_rows=100))
    html = html.replace("{{ic_block}}", _render_ic_block(ic_data))
    html = html.replace("{{ledger_block}}", _render_ledger_block(ledger))
    html = html.replace("{{now_utc}}", now.strftime("%Y-%m-%d %H:%M UTC"))

    return Response(html, mimetype="text/html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(_OUTPUT, exist_ok=True)
    app.run(host="0.0.0.0", port=5555, debug=False)
