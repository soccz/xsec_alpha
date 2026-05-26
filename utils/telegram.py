"""Telegram notification for xsec_alpha recommendations."""
import os
import requests
from utils.logger import logger

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }, timeout=(3.05, 10))
        resp.raise_for_status()
        logger.info("Telegram message sent successfully")
        return True
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


def _fmt_krw(price):
    if price is None:
        return "N/A"
    if price >= 1_000_000:
        return f"{price:,.0f}"
    elif price >= 100:
        return f"{price:,.0f}"
    elif price >= 1:
        return f"{price:,.1f}"
    else:
        return f"{price:.4f}"


def format_regime_header(latest_ts, btc_regime_row=None, ic_state: dict | None = None,
                          preflight: dict | None = None) -> str:
    """Top-of-message regime + model-health summary."""
    ts_str = str(latest_ts)[:16]
    lines = [f"📅 <code>{ts_str} UTC</code>"]

    if btc_regime_row is not None:
        try:
            import math
            def _fmt(v):
                if v is None: return "—"
                if isinstance(v, float) and math.isnan(v): return "—"
                return f"{v:+.1%}"
            btc7 = btc_regime_row.get("btc_ret_7d")
            btc30 = btc_regime_row.get("btc_ret_30d")
            bull = btc_regime_row.get("regime_bull")
            if isinstance(bull, float) and math.isnan(bull): bull = None
            tag = "🟢 bull" if bull == 1.0 else ("🟡 neutral/bear" if bull is not None else "❓")
            lines.append(f"📈 Regime: {tag}  BTC 7d {_fmt(btc7)} / 30d {_fmt(btc30)}")
        except Exception:
            pass

    if ic_state:
        parts = []
        for side, s in ic_state.items():
            status = s.get("status", "?")
            last = s.get("last_ic")
            last_s = f"{last:+.3f}" if last is not None else "—"
            emoji = {"OK": "🟢", "WARN": "🟡", "FREEZE": "🔴", "LIQUIDATE": "⛔"}.get(status, "⚪")
            parts.append(f"{emoji} {side.upper()} {last_s}")
        lines.append("🩺 Model IC: " + " | ".join(parts))

    if preflight:
        if preflight.get("batch_suppression"):
            lines.append(f"⚠ Pre-flight: <b>{preflight['batch_suppression']}</b> → 전체 watch-only")
        elif preflight.get("warnings"):
            lines.append("⚠ " + "; ".join(preflight["warnings"])[:150])

    # Factor drift (if state file available)
    try:
        import json
        from pathlib import Path
        drift_path = Path(__file__).resolve().parent.parent / "output" / "drift_state.json"
        if drift_path.exists():
            drift = json.loads(drift_path.read_text())
            flagged = [(c, v) for c, v in drift.get("factors", {}).items()
                       if v.get("status") in ("DRIFT", "SIGN_FLIP")]
            if flagged:
                parts = [f"{c}({v['status']})" for c, v in flagged[:3]]
                lines.append("📉 Factor drift: " + ", ".join(parts))
    except Exception:
        pass

    return "\n".join(lines)


def format_compact(short_df, long_df, latest_ts, btc_regime_row=None,
                    ic_state: dict | None = None, max_per_horizon: int = 5,
                    min_sigma: float = 1.5, prices: dict | None = None) -> str:
    """Single compact telegram message — A+B hybrid, ~12-14 rows total.

    Header: timestamp + regime (single line, optionally IC flag if any side WARN+)
    Body: per-horizon top `max_per_horizon` coins by |σ|
          Row format: `{tier} {coin:6s} {arrow} {pct:+.1f}% @{price} {trust_tag}`
    Footer: legend only if any trust tag present (⭐/⚠)

    Filters:
      - show |σ| >= min_sigma (default 1.5 = 🔥 tier + top ✅)
      - skip suppressed rows (preflight batch-blocked)
      - LONG (12h) section is skipped entirely when LIVE_LONG_TELEGRAM_SILENT is set.
        Ledger continues to record paper PnL; we just stop alerting until 1-2 weeks
        of observation data is collected. The dashboard's "long_paper_observation"
        badge lets the user/reviewer see the silence is intentional.

    prices: optional {market: KRW close price at latest_ts} for display. Missing
            entries fall back to no price suffix.
    """
    prices = prices or {}
    import math
    def _nan(v): return v is None or (isinstance(v, float) and v != v)

    # Paper-observation gate for LONG side (config-driven, env-overridable).
    long_silent = False
    try:
        from config import config as _cfg
        long_silent = bool(getattr(_cfg.Portfolio, "LIVE_LONG_TELEGRAM_SILENT", False))
    except Exception:
        long_silent = False

    lines = []

    # --- Header (1-2 lines) ---
    ts_str = str(latest_ts)[:16]
    hdr = f"📊 <b>xsec</b> · <code>{ts_str} UTC</code>"

    if btc_regime_row is not None:
        try:
            bull = btc_regime_row.get("regime_bull")
            if isinstance(bull, float) and math.isnan(bull): bull = None
            if bull == 1.0:
                hdr += " · 🟢 bull"
            elif bull is not None:
                hdr += " · 🟡 neutral/bear"
        except Exception: pass

    # IC flag only if something is concerning
    ic_flag = ""
    if ic_state:
        statuses = [s.get("status", "OK") for s in ic_state.values()]
        if any(s in ("FREEZE", "LIQUIDATE") for s in statuses):
            ic_flag = " · 🔴 IC"
        elif any(s == "WARN" for s in statuses):
            ic_flag = " · 🟡 IC"
    lines.append(hdr + ic_flag)
    lines.append("")

    # --- Body: 6h and 12h sections ---
    has_trust_tag = False

    for pred_df, horizon_h, title in [(short_df, 6, "6h"), (long_df, 12, "12h")]:
        if pred_df is None or len(pred_df) == 0:
            continue
        # Paper-observation: skip the 12h LONG section entirely when silent.
        if horizon_h == 12 and long_silent:
            continue
        strong = pred_df[pred_df["sigma"] >= min_sigma].head(max_per_horizon)
        if strong.empty:
            continue

        # Horizon header with hit rate summary for 🔥 tier and ✅ tier
        from utils.magnitude import _load_calibration
        calib = _load_calibration().get(f"{'short' if horizon_h==6 else 'long'}_{horizon_h}h", [])
        hit_fire = hit_check = None
        for b in calib:
            if b.get("sigma_low") == 2.0:
                hit_fire = b.get("hit_rate")
            elif b.get("sigma_low") == 1.0 and b.get("sigma_high") == 1.5:
                hit_check = b.get("hit_rate")
        hit_str = []
        if hit_fire is not None: hit_str.append(f"🔥 {hit_fire*100:.0f}%")
        if hit_check is not None: hit_str.append(f"✅ {hit_check*100:.0f}%")
        hit_suffix = f"  <code>({' · '.join(hit_str)})</code>" if hit_str else ""

        lines.append(f"<b>{title}</b>{hit_suffix}")

        for mkt, r in strong.iterrows():
            coin = str(mkt).replace("KRW-", "")
            arrow = "↑" if r["direction"] > 0 else ("↓" if r["direction"] < 0 else "·")
            exp = r.get("expected_pct")
            exp_s = f"{exp:+.1f}%" if not _nan(exp) else "—"
            price = prices.get(mkt)
            price_s = _fmt_krw(price) if price and not _nan(price) else None
            trust_tag = r.get("trust_tag", "") or ""
            if trust_tag: has_trust_tag = True
            tier = r.get("tag", "·")

            lines.append(
                f"  {tier} <b>{coin:<6}</b> {arrow} <code>{exp_s}</code>"
                + (f" @<code>{price_s}</code>" if price_s else "")
                + (f"  {trust_tag}" if trust_tag else "")
            )
        lines.append("")

    # --- Footer: legend only if trust tags appeared ---
    if has_trust_tag:
        lines.append("<code>⭐ 신뢰 (과거 적중≥60%) · ⚠ 역신호 (≤40%)</code>")

    return "\n".join(lines).rstrip()


def format_actionable_signals(recommendations_df, latest_ts, prices: dict | None = None,
                              min_sigma: float = 1.5,
                              min_expected_abs_pct: float = 0.20,
                              hide_untrusted: bool = True,
                              realized_summary: dict | None = None) -> str:
    """Telegram surface — designed for an operator who trades manually.

    The user executes both sides by hand on Bitget / Upbit, so the message
    needs enough information to support a manual judgment. Every pick that
    might be tradable shows up, with quality tier + trust tag attached:

      - SHORT rows (actionable): the 5 picks fetch_and_rank emits as exec
      - LONG  rows (watch-only): top WATCH_LONG candidates by σ (since the
        user trades these manually too, they need to see them)
      - Each row carries σ, expected%, trust_tag, entry_price, and a tier
        icon (🔥 ≥2σ, ✅ ≥1.5σ, ▫ weaker). Untrusted (⚠) signals are kept
        but visibly tagged so the user can skip them.

    If absolutely nothing meets even the soft threshold (σ ≥ 1.0), we still
    emit a heartbeat with reason — confirms the loop ran.
    """
    prices = prices or {}
    ts_str = str(latest_ts)[:16]

    def _nan(v):
        return v is None or (isinstance(v, float) and v != v)

    def _truthy(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        return str(v).strip().lower() in {"1", "true", "yes", "y"}

    def _realized_tail() -> str:
        if not realized_summary:
            return ""
        bits = []
        for side, label in (("SHORT", "SHORT"), ("WATCH_LONG", "LONG")):
            d = (realized_summary.get(side) or {}).get("d30") or {}
            net = d.get("avg_net")
            n = d.get("n")
            if net is not None and n:
                sign = "+" if net > 0 else ""
                bits.append(f"{label} {sign}{net:.2f}% (n={n})")
        return "  ·  ".join(bits) if bits else ""

    def _clean_str(v) -> str:
        # NaN slips through `or ""` because float('nan') is truthy. Guard explicitly.
        if _nan(v):
            return ""
        s = str(v).strip()
        return "" if s.lower() in ("nan", "none", "<na>") else s

    def _row(row, side_key: str) -> str:
        market = str(row.get("market", ""))
        coin = market.replace("KRW-", "")
        exp = row.get("expected_pct")
        exp_s = f"{float(exp):+.1f}%" if not _nan(exp) else "—"
        sigma = row.get("sigma")
        sigma_s = f"{float(sigma):.1f}σ" if not _nan(sigma) else "—"
        tag = _clean_str(row.get("tag")) or "·"
        trust = _clean_str(row.get("trust_tag"))
        price = prices.get(market) or row.get("entry_price")
        price_s = _fmt_krw(float(price)) if price and not _nan(price) else None
        arrow = "↓" if side_key == "SHORT" else "↑"
        return (
            f"  {tag} <b>{coin}</b> {arrow} <code>{exp_s}</code> "
            f"<code>{sigma_s}</code>"
            + (f" @<code>{price_s}</code>" if price_s else "")
            + (f" {trust}" if trust else "")
        )

    def _heartbeat(reason: str, raw_df=None) -> str:
        lines = [
            f"📌 <b>xsec</b> · <code>{ts_str} UTC</code>",
            "",
            "⏸ <b>강한 신호 없음</b>",
            f"<code>{reason}</code>",
        ]
        if raw_df is not None and len(raw_df):
            total = len(raw_df)
            lines.append(f"<code>후보 {total}개 · 모두 σ &lt; 1.0</code>")
        tail = _realized_tail()
        if tail:
            lines.append("")
            lines.append(f"<code>📊 30d net  {tail}</code>")
        lines.append("<code>세부는 대시보드/ledger</code>")
        return "\n".join(lines).rstrip()

    if recommendations_df is None or len(recommendations_df) == 0:
        return _heartbeat("추천 데이터 없음")

    raw_df = recommendations_df.copy()
    if "sigma" in raw_df.columns:
        raw_df["sigma"] = raw_df["sigma"].apply(
            lambda v: float(v) if not _nan(v) and str(v) != "" else float("nan")
        )
    if "expected_pct" in raw_df.columns:
        raw_df["expected_pct"] = raw_df["expected_pct"].apply(
            lambda v: float(v) if not _nan(v) and str(v) != "" else float("nan")
        )

    # Split sides BEFORE filtering so we can show both regardless of actionable.
    side_u = raw_df["side"].astype(str).str.upper() if "side" in raw_df.columns else pd.Series([""] * len(raw_df))
    short_df = raw_df[side_u.str.contains("SHORT", na=False)].copy()
    long_df = raw_df[side_u.str.contains("LONG", na=False)].copy()

    # Direction sanity per side
    if "expected_pct" in short_df.columns:
        short_df = short_df[short_df["expected_pct"] <= 0]
    if "expected_pct" in long_df.columns:
        long_df = long_df[long_df["expected_pct"] >= 0]

    # Soft floor: σ >= 1.0 lets weaker signals through with their tier visible.
    # The min_sigma / min_expected_abs_pct args now drive the "🔥 strong" tier
    # split, not a hard cut.
    SOFT_SIGMA = 1.0
    short_df = short_df[short_df["sigma"].fillna(0) >= SOFT_SIGMA]
    long_df = long_df[long_df["sigma"].fillna(0) >= SOFT_SIGMA]

    # Sort by sigma desc per side, cap to 5 each
    if not short_df.empty:
        short_df = short_df.sort_values("sigma", ascending=False).head(5)
    if not long_df.empty:
        long_df = long_df.sort_values("sigma", ascending=False).head(5)

    if short_df.empty and long_df.empty:
        return _heartbeat(
            f"전 사이드 σ &lt; {SOFT_SIGMA:.1f} (강신호 0)",
            raw_df,
        )

    lines = [f"📌 <b>xsec</b> · <code>{ts_str} UTC</code>", ""]

    def _summary_chip(part) -> str:
        if part.empty:
            return ""
        strong = int((part["sigma"] >= min_sigma).sum()) if "sigma" in part.columns else 0
        trusted = int((part.get("trust_tag", "").astype(str) == "⭐").sum()) if "trust_tag" in part.columns else 0
        untrusted = int((part.get("trust_tag", "").astype(str) == "⚠").sum()) if "trust_tag" in part.columns else 0
        bits = []
        if strong: bits.append(f"🔥{strong}")
        if trusted: bits.append(f"⭐{trusted}")
        if untrusted: bits.append(f"⚠{untrusted}")
        return f" <code>({' · '.join(bits)})</code>" if bits else ""

    if not short_df.empty:
        horizon = int(short_df["horizon_h"].iloc[0]) if "horizon_h" in short_df.columns else 6
        lines.append(f"🔴 <b>SHORT</b> <code>{horizon}h</code>{_summary_chip(short_df)}")
        for _, row in short_df.iterrows():
            lines.append(_row(row, "SHORT"))
        lines.append("")

    if not long_df.empty:
        horizon = int(long_df["horizon_h"].iloc[0]) if "horizon_h" in long_df.columns else 12
        lines.append(f"🟢 <b>LONG</b> <code>{horizon}h</code>{_summary_chip(long_df)}")
        for _, row in long_df.iterrows():
            lines.append(_row(row, "LONG"))
        lines.append("")

    # Realized 30d net summary line for context (manual trader needs to know
    # how the signal has actually paid off recently).
    tail = _realized_tail()
    if tail:
        lines.append(f"<code>📊 30d net  {tail}</code>")

    lines.append("<code>⭐ 신뢰 (적중≥60%) · ⚠ 역신호 (≤40%) · 세부는 대시보드</code>")
    return "\n".join(lines).rstrip()


def format_per_coin_predictions(pred_df, horizon_h: int, title: str,
                                   min_sigma: float = 1.0, max_rows: int = 15) -> str:
    """Probabilistic per-coin view.

    Each row: tier | coin | arrow | P(hit) | expected% | [CI_low, CI_high] | size
    """
    if pred_df is None or len(pred_df) == 0:
        return ""
    strong = pred_df[pred_df["sigma"] >= min_sigma].head(max_rows)
    if strong.empty:
        return f"<b>{title}</b> <code>{horizon_h}h</code> — |σ|≥{min_sigma} 신호 없음"

    total = len(pred_df)
    strong_count = int((pred_df["sigma"] >= min_sigma).sum())
    suppressed = pred_df.get("actionable")
    suppressed_n = int((~suppressed.fillna(False)).sum()) if suppressed is not None else 0

    lines = [
        f"<b>{title}</b> <code>{horizon_h}h</code>",
        f"<code>총 {total} | |σ|≥{min_sigma}: {strong_count} | 표시 {len(strong)}개"
        + (f" | 억제 {suppressed_n}" if suppressed_n else "")
        + "</code>",
        "",
    ]
    def _nan(v):
        return v is None or (isinstance(v, float) and v != v)

    for mkt, r in strong.iterrows():
        coin = str(mkt).replace("KRW-", "")
        arrow = "↑" if r["direction"] > 0 else ("↓" if r["direction"] < 0 else "·")
        prob = r.get("direction_prob")
        exp  = r.get("expected_pct")
        lo   = r.get("ci_95_low")
        hi   = r.get("ci_95_high")
        size = r.get("position_size_pct")
        vol  = r.get("coin_vol_pct")

        prob_s = f"{prob*100:.0f}%" if not _nan(prob) else "—"
        exp_s  = f"{exp:+.2f}%"     if not _nan(exp)  else "N/A"
        vol_s  = f"±{vol:.1f}%"     if not _nan(vol)  else ""
        ci_s   = f"[{lo:+.1f}, {hi:+.1f}]" if not _nan(lo) and not _nan(hi) else ""
        size_s = f" 💰<code>{size:.1f}%</code>" if not _nan(size) and size > 0 else ""
        tail   = "  <i>(억제)</i>" if (suppressed is not None and not bool(r.get("actionable", True))) else ""

        # Enrichment tags (consensus + trust)
        cons_tag  = r.get("consensus_tag", "") or ""
        trust_tag = r.get("trust_tag", "") or ""
        enrich = (cons_tag + trust_tag).strip()
        enrich_s = f" {enrich}" if enrich else ""

        # Format: tag | coin | arrow | prob | expected | vol | CI | size | enrich | suppression
        lines.append(
            f"  {r.get('tag', '·')} <b>{coin:<6}</b> {arrow}  "
            f"<code>{prob_s}</code> 기대<code>{exp_s}</code> "
            f"변<code>{vol_s}</code> <code>{ci_s}</code>{size_s}{enrich_s}{tail}"
        )
    return "\n".join(lines)


def realized_return(side: str, entry_price: float, exit_price: float) -> float:
    """Single source of truth for realized-return sign convention.

    Shared by format_performance and scripts/verify_telegram.py so any regression
    in one is caught by the other.

    'SHORT' in side → short math, else long math. WATCH_LONG is long-direction.
    """
    if not (entry_price and exit_price and entry_price > 0):
        return float("nan")
    if "SHORT" in str(side).upper():
        return (entry_price - exit_price) / entry_price
    return (exit_price - entry_price) / entry_price


def _confidence(score, std):
    """Return (stars, label) based on sigma distance."""
    if std <= 0:
        return "▪", "low"
    sigma = abs(score) / std
    if sigma >= 2.0:
        return "🔥", "strong"
    elif sigma >= 1.0:
        return "✅", "ok"
    else:
        return "▫", "weak"


def format_report(longs, shorts, timestamp, prices: dict = None,
                  horizon_h: int = 6, all_scores=None, min_sigma: float = 1.0,
                  capital: float = None, stop_loss_pct: float = None,
                  long_header: str = "LONG", long_watch_only: bool = False,
                  long_horizon_h: int | None = None, short_horizon_h: int | None = None,
                  next_long_rebalance_at=None, next_short_rebalance_at=None) -> str:
    """
    Format a trading report. Only shows coins above min_sigma confidence.

    Parameters
    ----------
    longs : pd.Series — index=market, values=score (sorted descending)
    shorts : pd.Series — index=market, values=score (sorted ascending)
    timestamp : datetime-like
    prices : dict — {market: current_price}
    horizon_h : int
    all_scores : pd.Series — full score series (for std calculation)
    min_sigma : float — minimum sigma to include (1.0 = skip noise)
    """
    import numpy as np

    prices = prices or {}
    ts_str = str(timestamp)[:16]

    long_horizon_h = long_horizon_h or horizon_h
    short_horizon_h = short_horizon_h or horizon_h
    header_horizon = (
        f"L{long_horizon_h}h / S{short_horizon_h}h"
        if long_horizon_h != short_horizon_h
        else f"{horizon_h}h"
    )

    # Calculate std from all scores for confidence levels
    if all_scores is not None and len(all_scores) > 0:
        std = float(all_scores.std())
    else:
        import pandas as pd
        combined = pd.concat([longs, shorts])
        std = float(combined.std()) if len(combined) > 1 else 0.003

    # Filter by min_sigma
    filtered_longs = [(m, s) for m, s in longs.items() if abs(s) / std >= min_sigma] if std > 0 else list(longs.items())
    filtered_shorts = [(m, s) for m, s in shorts.items() if abs(s) / std >= min_sigma] if std > 0 else list(shorts.items())

    # Confidence-weighted position sizing (sized per side, not mixed)
    # 🔥 (2σ+) = weight 2, ✅ (1σ+) = weight 1
    weights = {}
    for mkt, score in filtered_longs + filtered_shorts:
        sigma = abs(score) / std if std > 0 else 1.0
        weights[mkt] = 2.0 if sigma >= 2.0 else 1.0

    long_total_weight = sum(weights[m] for m, _ in filtered_longs) if filtered_longs else 0
    short_total_weight = sum(weights[m] for m, _ in filtered_shorts) if filtered_shorts else 0

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"  📊 <b>xsec_alpha</b>",
        f"  <code>{ts_str} | {header_horizon}</code>",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "",
    ]

    if next_long_rebalance_at is not None or next_short_rebalance_at is not None:
        next_long_str = str(next_long_rebalance_at)[:16] if next_long_rebalance_at is not None else "N/A"
        next_short_str = str(next_short_rebalance_at)[:16] if next_short_rebalance_at is not None else "N/A"
        lines.append(f"⏱ LONG 다음 {next_long_str} | SHORT 다음 {next_short_str}")
        lines.append("")

    # Summary line
    if capital and (long_total_weight > 0 or short_total_weight > 0):
        lines.append(f"💰 총 자본 <b>{int(capital):,}원</b> | 🔥 2배 배분 | 손절 <b>{stop_loss_pct or 3.0}%</b>")
        lines.append("")

    sl_pct = (stop_loss_pct or 3.0) / 100

    # LONG section
    if filtered_longs:
        long_icon = "🟡" if long_watch_only else "🟢"
        lines.append(f"{long_icon} <b>{long_header} ({len(filtered_longs)})</b> <code>{long_horizon_h}h</code>")
        lines.append("")
        for i, (mkt, score) in enumerate(filtered_longs, 1):
            coin = mkt.replace("KRW-", "")
            icon, _ = _confidence(score, std)
            now = prices.get(mkt)
            exp_pct = score * 100

            if now and now > 0:
                stop = now * (1 - sl_pct)
                position_amt = int(capital * weights[mkt] / long_total_weight) if capital and long_total_weight > 0 else None
                qty = f"{position_amt / now:,.2f}" if position_amt else ""
                action_label = "관찰" if long_watch_only else "매수"
                lines.append(
                    f"  {icon} <b>{coin}</b>  잔차 {exp_pct:+.2f}%"
                    f"\n     {action_label} <code>{_fmt_krw(now)}</code>"
                    f"  손절 <code>{_fmt_krw(stop)}</code>"
                )
                if position_amt and not long_watch_only:
                    lines.append(f"     수량 <code>{qty}</code>개  금액 <code>{position_amt:,}</code>원")
            else:
                lines.append(f"  {icon} <b>{coin}</b>  잔차 <b>{exp_pct:+.2f}%</b>")
        lines.append("")
    else:
        empty_label = long_header if long_watch_only else "LONG"
        empty_icon = "🟡" if long_watch_only else "🟢"
        lines.append(f"{empty_icon} <b>{empty_label}</b> <code>{long_horizon_h}h</code> — 신뢰할 만한 신호 없음")
        lines.append("")

    # SHORT section
    if filtered_shorts:
        lines.append(f"🔴 <b>SHORT ({len(filtered_shorts)})</b> <code>{short_horizon_h}h</code>")
        lines.append("")
        for i, (mkt, score) in enumerate(filtered_shorts, 1):
            coin = mkt.replace("KRW-", "")
            icon, _ = _confidence(score, std)
            now = prices.get(mkt)
            exp_pct = score * 100

            if now and now > 0:
                stop = now * (1 + sl_pct)   # short stop = price goes UP
                lines.append(
                    f"  {icon} <b>{coin}</b>  잔차 {exp_pct:+.2f}%"
                    f"\n     매도 <code>{_fmt_krw(now)}</code>"
                    f"  손절 <code>{_fmt_krw(stop)}</code>"
                )
            else:
                lines.append(f"  {icon} <b>{coin}</b>  잔차 <b>{exp_pct:+.2f}%</b>")
        lines.append("")
    else:
        lines.append(f"🔴 <b>SHORT</b> <code>{short_horizon_h}h</code> — 신뢰할 만한 신호 없음")
        lines.append("")

    # Legend
    lines.append("<code>🔥 강한 신호 (2σ+)  ✅ 보통 (1σ+)</code>")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines)


def format_performance(prev_df, now_prices: dict) -> str:
    """
    Format previous recommendations performance report.

    prev_df: DataFrame with columns [market, score, side, entry_price]
    now_prices: {market: current_price}
    """
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "  📋 <b>이전 추천 성적표</b>",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "",
    ]

    long_pnl = []
    watch_long_pnl = []
    short_pnl = []

    for side_name, side_label, pnl_list in [
        ("LONG", "🟢 LONG", long_pnl),
        ("WATCH_LONG", "🟡 WATCH LONG", watch_long_pnl),
        ("SHORT", "🔴 SHORT", short_pnl),
    ]:
        rows = prev_df[prev_df["side"] == side_name]
        if rows.empty:
            continue

        lines.append(f"<b>{side_label}</b>")
        for _, row in rows.iterrows():
            mkt = row["market"]
            coin = mkt.replace("KRW-", "")
            entry = row.get("entry_price")
            now = now_prices.get(mkt)

            if entry and now and entry > 0:
                ret = realized_return(side_name, entry, now)
                pnl_list.append(ret)
                icon = "✅" if ret > 0 else "❌"
                lines.append(
                    f"  {icon} {coin:<8}"
                    f" {_fmt_krw(entry)} → {_fmt_krw(now)}"
                    f"  <b>{ret*100:+.2f}%</b>"
                )
            else:
                lines.append(f"  ▫ {coin:<8} 가격 없음")
        lines.append("")

    # Summary — report per side separately (horizons differ: long=12h, short=6h)
    def _summary(label, pnl):
        if not pnl:
            return None
        avg = sum(pnl) / len(pnl) * 100
        wins = sum(1 for p in pnl if p > 0)
        return f"  {label}: {wins}/{len(pnl)} 적중 · 평균 <b>{avg:+.2f}%</b>"

    any_data = bool(long_pnl or watch_long_pnl or short_pnl)
    if any_data:
        lines.append("<b>요약</b>")
        for row in [
            _summary("🟢 LONG (12h)", long_pnl),
            _summary("🟡 WATCH LONG (12h)", watch_long_pnl),
            _summary("🔴 SHORT (6h)", short_pnl),
        ]:
            if row:
                lines.append(row)
    else:
        lines.append("  이전 데이터 없음")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


# Backward compat
def format_recommendations(longs, shorts, timestamp, ic_value=None):
    return format_report(longs, shorts, timestamp)


def send_recommendations(longs, shorts, timestamp, ic_value=None):
    text = format_recommendations(longs, shorts, timestamp)
    return send_message(text)
