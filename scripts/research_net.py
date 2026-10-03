"""Shared helpers for research-only data collectors (no live-runtime imports).

Research collectors must never compete with the live jobs for API budget or disk:
they pause around the live alpha runs (23/05/11/17 UTC) and the +1h/+7h venue
observations (00/06/12/18 UTC), and retry politely on rate limits.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

USER_AGENT = "xsec-alpha-research/1"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def in_live_window(now: datetime) -> bool:
    """True while live jobs may be using the Upbit/Bitget public APIs."""
    hour, minute = now.hour, now.minute
    if hour % 6 == 5 and (minute < 20 or minute >= 55):  # alpha run + pre-observation
        return True
    if hour % 6 == 0 and minute < 10:  # +1h/+7h depth, ticker and funding observations
        return True
    return False


def wait_outside_live_window(log=print) -> None:
    announced = False
    while in_live_window(utcnow()):
        if not announced:
            log("live window: pausing")
            announced = True
        time.sleep(20)
    if announced:
        log("live window over: resuming")


def get_json(url: str, *, tries: int = 7, timeout: float = 15.0, log=print):
    """GET JSON with exponential backoff on throttling and transient errors; 404 -> None."""
    delay = 1.0
    for _ in range(tries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                remaining = response.headers.get("Remaining-Req", "")
                payload = json.load(response)
            if _remaining_per_second(remaining) is not None and _remaining_per_second(remaining) <= 2:
                time.sleep(1.0)
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code in (418, 429) or exc.code >= 500:
                log(f"HTTP {exc.code}; backing off {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            log(f"network error {type(exc).__name__}; backing off {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise RuntimeError(f"giving up after {tries} tries: {url}")


def _remaining_per_second(header: str) -> int | None:
    # Upbit: "group=candles; min=1799; sec=29"
    for part in header.split(";"):
        key, _, value = part.strip().partition("=")
        if key == "sec" and value.isdigit():
            return int(value)
    return None


def stamp() -> str:
    return utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
