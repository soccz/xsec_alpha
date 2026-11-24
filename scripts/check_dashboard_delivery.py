"""Check the served public and encrypted summaries against the last local push."""

import hashlib
from http.client import HTTPException
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if __package__ in ("", None):
    sys.path.insert(0, str(ROOT))
from utils.report_identity import report_identity  # noqa: E402

PREFIX = "projects/xsec-alpha"
FILES = ("public_summary.json", "dashboard/data/summary.json")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(name, version):
    request = Request(
        f"https://soccz.github.io/{PREFIX}/{name}?coverage_check={version}",
        headers={"Cache-Control": "no-cache", "User-Agent": "xsec-delivery-check/1"},
    )
    with urlopen(request, timeout=6) as response:
        data = response.read(4 * 1024**2 + 1)
    if len(data) > 4 * 1024**2:
        raise ValueError("served summary too large")
    return data


def _optional_dict(path):
    try:
        value = json.loads(path.read_bytes())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def round_check(folder, summary_hash, now):
    now = now.astimezone(timezone.utc)
    # Allow the scheduled collection 45 minutes; never use export time as market time.
    cutoff = (now - timedelta(minutes=45)).replace(minute=0, second=0, microsecond=0)
    due = cutoff - timedelta(hours=(cutoff.hour - 5) % 6)
    latest = _optional_dict(folder.parent / "latest_operator_report.json")
    witness = _optional_dict(folder / "report_witness.json")
    served = (
        (witness.get("report") or {})
        if summary_hash and witness.get("summary_sha256") == summary_hash
        else {}
    )
    if not isinstance(served, dict):
        served = {}
    run_id = latest.get("run_id")
    try:
        asof = datetime.fromisoformat(latest["data_asof"])
        current = "future" if asof > now else "stale" if asof < due else "current"
    except (TypeError, ValueError, KeyError):
        current = "unknown"
    matched = (
        bool(run_id)
        and served.get("run_id") == run_id
        and served.get("data_asof") == latest.get("data_asof")
    )
    tg = latest.get("telegram") or {}
    return {
        "run_id": run_id,
        "data_asof": latest.get("data_asof"),
        "expected_slot": due.isoformat(),
        "freshness": current,
        "telegram_state": tg.get("state"),
        "served_run_id": served.get("run_id"),
        "match": "same"
        if matched
        else "different"
        if run_id and served.get("run_id")
        else "unknown",
        "same_picks": bool(
            matched
            and served.get("picks_sha256") == report_identity(latest)["picks_sha256"]
        ),
        "same_ack": bool(
            matched
            and tg.get("state") == "sent"
            and tg.get("acknowledged_at")
            and tg == served.get("telegram")
        ),
    }


def verify(folder, loader=fetch, now=None):
    now = now or datetime.now(timezone.utc)
    output = folder / "delivery.json"
    prior = _optional_dict(output)
    state = {"checked_at": now.isoformat(), "last_verified": prior.get("last_verified")}
    status = folder / "status.json"
    confirmed = None
    try:
        before = status.read_bytes()
        pushed = json.loads(before)
        if pushed.get("status") != "published":
            state["status"] = "no_confirmed_push"
        else:
            paths = [folder / "payloads" / PREFIX / name for name in FILES]
            data = [path.read_bytes() for path in paths]
            hashes = list(map(digest, data))
            public = json.loads(data[0])
            # An export runs before push and may also run without pushing.
            if (
                any(
                    path.stat().st_mtime_ns > status.stat().st_mtime_ns
                    for path in paths
                )
                or public.get("operating_receipt", {}).get("summary_sha256")
                != hashes[1]
            ):
                state["status"] = "local_generation_unconfirmed"
            else:
                state["asof"] = public["asof"]
                match = all(
                    digest(loader(name, hashes[0])) == expected
                    for name, expected in zip(FILES, hashes)
                )
                state["status"] = "verified" if match else "served_other_generation"
                if (
                    before != status.read_bytes()
                    or [digest(path.read_bytes()) for path in paths] != hashes
                ):
                    state["status"] = "local_generation_changed"
                if state["status"] == "verified":
                    confirmed = hashes[1]
                    state["last_verified"] = {
                        "asof": public["asof"],
                        "at": state["checked_at"],
                        "public_sha256": hashes[0],
                        "summary_sha256": hashes[1],
                    }
    except (OSError, HTTPException, ValueError, KeyError, TypeError) as err:
        state.update(status="unverified", error_type=type(err).__name__)
    state["round"] = round_check(folder, confirmed, now)
    if confirmed:
        state["last_verified"]["round"] = state["round"]
    current_round = state["round"]
    if any(
        current_round.get(k) != (prior.get("round") or {}).get(k)
        for k in (
            "run_id",
            "served_run_id",
            "match",
            "freshness",
            "telegram_state",
            "same_picks",
            "same_ack",
        )
    ):
        with (folder / "report-round-transitions.jsonl").open("a") as events:
            events.write(
                json.dumps({"checked_at": state["checked_at"], **current_round}) + "\n"
            )
    if state["status"] != prior.get("status"):
        with (folder / "delivery-transitions.jsonl").open("a") as events:
            events.write(json.dumps(state) + "\n")
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(output)
    return state


if __name__ == "__main__":
    print(json.dumps(verify(ROOT / "output/dashboard_publication")))
