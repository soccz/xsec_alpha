"""Private operator reports, observation fallbacks and latest-report delivery."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import secrets

import pandas as pd

from utils.logger import logger
from utils.run_lock import run_lock

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_URL = "https://soccz.github.io/projects/xsec-alpha/dashboard/"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _read(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(document, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _records(frame):
    if frame is None or frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def build_report(signals=None, candidates=None, *, asof=None, reason="", error=False,
                 root=ROOT):
    """Observation names never enter the recommendation/performance ledger."""
    records = _records(signals)
    ideas = []
    if not any(str(r.get("side", "")).upper() == "SHORT" for r in records):
        source_rows = _records(candidates)
        source = "current_ranking"
        if not source_rows:
            prior = _read(Path(root) / "output" / "latest_operator_report.json")
            source_rows = prior.get("ideas", []) or [
                r for r in prior.get("signals", []) if str(r.get("side", "")).upper() == "SHORT"
            ]
            source = "previous_report"
        if not source_rows:
            try:
                latest = pd.read_csv(Path(root) / "output" / "latest.csv")
                source_rows = _records(latest[latest["side"].astype(str).str.upper().eq("SHORT")])
            except (OSError, ValueError, KeyError):
                source_rows = []
            source = "previous_recommendations"
        if not source_rows:
            source_rows = [{"market": "KRW-BTC"}, {"market": "KRW-ETH"}]
            source = "market_reference"
        seen = set()
        for row in source_rows:
            market = row.get("market")
            if not isinstance(market, str) or not market.startswith("KRW-") or market in seen:
                continue
            seen.add(market)
            ideas.append({
                "market": market, "actionable": False,
                "source": source if source != "previous_report" else row.get("source", "previous_report"),
                "asof": (str(asof) if asof is not None else None) if source == "current_ranking"
                        else row.get("asof") or row.get("entry_time") or row.get("timestamp"),
                "reference_only": source != "current_ranking" or error,
                "reason": reason or "no_actionable_basket",
                "model_sha256": row.get("model_sha256"),
            })
            if len(ideas) == 5:
                break
        if not ideas:
            # Malformed historical rows are not evidence of a fresh ranking.
            ideas = [{"market": "KRW-BTC", "actionable": False, "source": "market_reference",
                      "asof": None, "reference_only": True, "reason": reason or "data_unavailable"}]
    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + secrets.token_hex(3),
        "generated_at": _now(), "data_asof": str(asof) if asof is not None else None,
        "status": "error" if error else (
            "ok" if not reason and any(r.get("actionable") is True for r in records) else "watch"
        ),
        "reason": reason, "signals": records, "ideas": ideas,
        "telegram": {"state": "pending", "attempts": 0},
    }


def format_report(report, realized_summary=None):
    from html import escape
    from utils.telegram import format_actionable_signals
    from config import config

    timestamp = report.get("data_asof") or report["generated_at"]
    if report["signals"]:
        text = format_actionable_signals(
            pd.DataFrame(report["signals"]), timestamp,
            realized_summary=realized_summary, status_note=report.get("reason", ""),
            min_sigma=config.Notification.TELEGRAM_MIN_SIGMA,
        )
    else:
        text = f"<b>xsec: 코인 관찰 제안</b>\n<code>{escape(report['generated_at'][:16])} UTC</code>"
    if report["ideas"]:
        text += "\n\n<b>관찰 후보 · 매매 추천 아님</b>"
        if report.get("reason"):
            text += f"\n<code>{escape(report['reason'])}</code>"
        for row in report["ideas"]:
            label = "이전 자료 참고" if row["reference_only"] else "현재 상대순위 후보"
            if row["source"] == "market_reference":
                label = "시장 기준 종목 · 모델 선정 아님"
            stamp = str(row.get("asof") or "시각 미확인")[:25]
            text += f"\n  <b>{escape(row['market'].replace('KRW-', ''))}</b> · {label}"
            text += f"\n  <code>자료: {escape(stamp)}</code>"
        text += "\n<code>actionable=false · 과거 수치로 신규 진입 판단 금지</code>"
    return text + f'\n\n<a href="{DASHBOARD_URL}">대시보드</a>'


def _save(report, root):
    output = Path(root) / "output"
    _write(output / "latest_operator_report.json", report)
    _write(output / "operator_reports" / f"{report['run_id']}.json", report)


def _deliver(report, root):
    from utils.telegram import send_message

    delivery = report["telegram"]
    delivery["attempts"] += 1
    delivery["last_attempt_at"] = _now()
    # Persist intent first. A crash after API acknowledgement can cause a retry
    # duplicate, but cannot silently discard an unacknowledged report.
    _save(report, root)
    sent = send_message(report["message"])
    if sent:
        delivery.update(state="sent", acknowledged_at=_now())
    _save(report, root)
    return sent


def publish_report(report, *, send=True, realized_summary=None, root=ROOT):
    root = Path(root)
    with run_lock("operator_report", lock_dir=str(root / "logs" / "locks"), timeout_sec=45, exit_code=75):
        previous = _read(root / "output" / "latest_operator_report.json")
        if previous and previous.get("telegram", {}).get("state") == "pending":
            previous["telegram"].update(state="superseded", superseded_by=report["run_id"])
            _write(root / "output" / "operator_reports" / f"{previous['run_id']}.json", previous)
        report["message"] = format_report(report, realized_summary)
        report["telegram"]["state"] = "pending" if send else "disabled"
        _save(report, root)
        return _deliver(report, root) if send else False


def retry_latest(root=ROOT, now=None):
    root = Path(root)
    with run_lock("operator_report", lock_dir=str(root / "logs" / "locks"), timeout_sec=45, exit_code=75):
        report = _read(root / "output" / "latest_operator_report.json")
        if not report or report.get("telegram", {}).get("state") != "pending":
            return True
        now = now or datetime.now(timezone.utc)
        created = datetime.fromisoformat(report["generated_at"])
        if report.get("signals") and now - created >= timedelta(hours=6):
            replacement = build_report(reason="전송 지연으로 추천 기간 만료 · 이전 후보 참고용", root=root)
            replacement["replaces_run_id"] = report["run_id"]
            replacement["message"] = format_report(replacement)
            report["telegram"].update(state="expired", superseded_by=replacement["run_id"])
            _write(root / "output" / "operator_reports" / f"{report['run_id']}.json", report)
            report = replacement
        logger.info("Retrying operator report %s", report["run_id"])
        return _deliver(report, root)
