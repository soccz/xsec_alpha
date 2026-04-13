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
    ic = _load_ic_history()
    if ic is None:
        return _json_response({"error": "No IC history yet", "data": {}}, 200)
    return _json_response(ic)


@app.route("/api/status")
def api_status():
    db_ts = get_latest_db_timestamp()
    model_ts = _model_age()
    try:
        markets = get_all_krw_markets_in_db()
    except Exception:
        markets = []

    _, rec_mtime = _load_latest_csv()

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
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .card h2 { color: #58a6ff; font-size: 1em; margin-bottom: 10px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
  .stat { display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.9em; }
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
  </div>

  <!-- IC card -->
  <div class="card">
    <h2>IC Summary</h2>
    {{ic_block}}
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

    # Handle both dict-of-stats and list-of-records formats
    if isinstance(ic_data, dict):
        # Try common keys
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
    ic_data = _load_ic_history()
    live_labels = _live_recommendation_labels()

    html = _DASHBOARD_HTML
    html = html.replace("{{db_latest}}", _fmt_ts(db_ts))
    html = html.replace("{{db_age}}", db_age_str)
    html = html.replace("{{universe_size}}", str(len(markets)))
    html = html.replace("{{model_updated}}", _fmt_ts(model_ts))
    html = html.replace("{{last_run}}", _fmt_ts(rec_mtime))
    html = html.replace("{{long_label}}", live_labels["long_label"])
    html = html.replace("{{short_label}}", live_labels["short_label"])
    html = html.replace("{{long_n}}", str(live_labels["long_n"]))
    html = html.replace("{{short_n}}", str(live_labels["short_n"]))
    html = html.replace("{{long_table}}", _render_rec_table(long_df, max_rows=live_labels["long_n"]))
    html = html.replace("{{short_table}}", _render_rec_table(short_df, max_rows=live_labels["short_n"]))
    html = html.replace("{{total_count}}", str(len(df)) if df is not None else "0")
    html = html.replace("{{full_table}}", _render_rec_table(df, max_rows=100))
    html = html.replace("{{ic_block}}", _render_ic_block(ic_data))
    html = html.replace("{{now_utc}}", now.strftime("%Y-%m-%d %H:%M UTC"))

    return Response(html, mimetype="text/html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(_OUTPUT, exist_ok=True)
    app.run(host="0.0.0.0", port=5555, debug=False)
