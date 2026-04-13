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
                  long_header: str = "LONG", long_watch_only: bool = False) -> str:
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

    # Confidence-weighted position sizing
    # 🔥 (2σ+) = weight 2, ✅ (1σ+) = weight 1
    weights = {}
    for mkt, score in filtered_longs + filtered_shorts:
        sigma = abs(score) / std if std > 0 else 1.0
        weights[mkt] = 2.0 if sigma >= 2.0 else 1.0

    total_weight = sum(weights.values()) if weights else 0

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"  📊 <b>xsec_alpha</b>",
        f"  <code>{ts_str} | {horizon_h}h</code>",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "",
    ]

    # Summary line
    if capital and total_weight > 0:
        lines.append(f"💰 총 자본 <b>{int(capital):,}원</b> | 🔥 2배 배분 | 손절 <b>{stop_loss_pct or 3.0}%</b>")
        lines.append("")

    sl_pct = (stop_loss_pct or 3.0) / 100

    # LONG section
    if filtered_longs:
        long_icon = "🟡" if long_watch_only else "🟢"
        lines.append(f"{long_icon} <b>{long_header} ({len(filtered_longs)})</b>")
        lines.append("")
        for i, (mkt, score) in enumerate(filtered_longs, 1):
            coin = mkt.replace("KRW-", "")
            icon, _ = _confidence(score, std)
            now = prices.get(mkt)
            exp_pct = score * 100

            if now and now > 0:
                target = now * (1 + score)
                stop = now * (1 - sl_pct)
                position_amt = int(capital * weights[mkt] / total_weight) if capital and total_weight > 0 else None
                qty = f"{position_amt / now:,.2f}" if position_amt else ""
                action_label = "관찰" if long_watch_only else "매수"
                lines.append(
                    f"  {icon} <b>{coin}</b>  {exp_pct:+.2f}%"
                    f"\n     {action_label} <code>{_fmt_krw(now)}</code>"
                    f"  목표 <code>{_fmt_krw(target)}</code>"
                    f"  손절 <code>{_fmt_krw(stop)}</code>"
                )
                if position_amt and not long_watch_only:
                    lines.append(f"     수량 <code>{qty}</code>개  금액 <code>{position_amt:,}</code>원")
            else:
                lines.append(f"  {icon} <b>{coin}</b>  (<b>{exp_pct:+.2f}%</b>)")
        lines.append("")
    else:
        empty_label = long_header if long_watch_only else "LONG"
        empty_icon = "🟡" if long_watch_only else "🟢"
        lines.append(f"{empty_icon} <b>{empty_label}</b> — 신뢰할 만한 신호 없음")
        lines.append("")

    # SHORT section
    if filtered_shorts:
        lines.append(f"🔴 <b>SHORT ({len(filtered_shorts)})</b>")
        lines.append("")
        for i, (mkt, score) in enumerate(filtered_shorts, 1):
            coin = mkt.replace("KRW-", "")
            icon, _ = _confidence(score, std)
            now = prices.get(mkt)
            exp_pct = score * 100

            if now and now > 0:
                target = now * (1 + score)  # score is negative
                stop = now * (1 + sl_pct)   # short stop = price goes UP
                lines.append(
                    f"  {icon} <b>{coin}</b>  {exp_pct:+.2f}%"
                    f"\n     매도 <code>{_fmt_krw(now)}</code>"
                    f"  목표 <code>{_fmt_krw(target)}</code>"
                    f"  손절 <code>{_fmt_krw(stop)}</code>"
                )
            else:
                lines.append(f"  {icon} <b>{coin}</b>  (<b>{exp_pct:+.2f}%</b>)")
        lines.append("")
    else:
        lines.append("🔴 <b>SHORT</b> — 신뢰할 만한 신호 없음")
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
    short_pnl = []

    for side_name, side_label, pnl_list in [
        ("LONG", "🟢 LONG", long_pnl),
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
                if side_name == "LONG":
                    ret = (now - entry) / entry  # bought → price went up = profit
                else:
                    ret = (entry - now) / entry  # shorted → price went down = profit

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

    # Summary
    all_pnl = long_pnl + short_pnl
    if all_pnl:
        avg = sum(all_pnl) / len(all_pnl) * 100
        wins = sum(1 for p in all_pnl if p > 0)
        total = len(all_pnl)
        long_avg = sum(long_pnl) / len(long_pnl) * 100 if long_pnl else 0
        short_avg = sum(short_pnl) / len(short_pnl) * 100 if short_pnl else 0

        lines.append(f"<b>요약</b>")
        lines.append(f"  적중 {wins}/{total} ({wins/total*100:.0f}%)")
        lines.append(f"  평균 수익: <b>{avg:+.2f}%</b>")
        lines.append(f"  LONG {long_avg:+.2f}% | SHORT {short_avg:+.2f}%")
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
