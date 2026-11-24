"""Check the served public and encrypted summaries against the last local push."""

import hashlib
from http.client import HTTPException
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
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


def verify(folder, loader=fetch, now=None):
    now = now or datetime.now(timezone.utc)
    output = folder / "delivery.json"
    prior = json.loads(output.read_text()) if output.exists() else {}
    state = {"checked_at": now.isoformat(), "last_verified": prior.get("last_verified")}
    status = folder / "status.json"
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
                    state["last_verified"] = {
                        "asof": public["asof"],
                        "at": state["checked_at"],
                        "public_sha256": hashes[0],
                        "summary_sha256": hashes[1],
                    }
    except (OSError, HTTPException, ValueError, KeyError, TypeError) as err:
        state.update(status="unverified", error_type=type(err).__name__)
    if state["status"] != prior.get("status"):
        with (folder / "delivery-transitions.jsonl").open("a") as events:
            events.write(json.dumps(state) + "\n")
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(output)
    return state


if __name__ == "__main__":
    print(json.dumps(verify(ROOT / "output/dashboard_publication")))
