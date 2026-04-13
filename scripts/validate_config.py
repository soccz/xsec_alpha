"""
Pre-flight config and data sanity check.

Usage:
    cd /mnt/20t/main/gan_t/xsec_alpha
    python scripts/validate_config.py
"""

import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from data.database import get_all_krw_markets_in_db, get_latest_db_timestamp

_STALE_HOURS = 48
_MIN_UNIVERSE = 50
_BTC_MARKET = "KRW-BTC"
_EXPECTED_HORIZON = 12
_EXPECTED_BETA_WINDOW = 168


def _ok(label: str, detail: str = "") -> bool:
    msg = f"[OK]   [{label}]"
    if detail:
        msg += f"  {detail}"
    print(msg)
    return True


def _fail(label: str, detail: str = "") -> bool:
    msg = f"[FAIL] [{label}]"
    if detail:
        msg += f"  {detail}"
    print(msg)
    return False


def check_db() -> bool:
    db_path = os.path.abspath(config.General.DB_PATH)
    if not os.path.exists(db_path):
        return _fail("DB", f"file not found: {db_path}")
    if not os.access(db_path, os.R_OK):
        return _fail("DB", f"not readable: {db_path}")
    size_mb = os.path.getsize(db_path) / 1024 / 1024
    return _ok("DB", f"{db_path}  ({size_mb:.1f} MB)")


def check_data_freshness() -> bool:
    latest = get_latest_db_timestamp()
    if latest is None:
        return _fail("DATA", "no timestamp found in DB (empty or inaccessible)")
    now = datetime.now(timezone.utc)
    # ensure latest is tz-aware
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    age_hours = (now - latest).total_seconds() / 3600
    if age_hours > _STALE_HOURS:
        return _fail("DATA", f"latest timestamp {latest.isoformat()} is {age_hours:.1f}h old (limit {_STALE_HOURS}h)")
    return _ok("DATA", f"latest {latest.isoformat()}  ({age_hours:.1f}h ago)")


def check_universe() -> bool:
    markets = get_all_krw_markets_in_db()
    n = len(markets)
    if n < _MIN_UNIVERSE:
        return _fail("UNIV", f"only {n} KRW markets in DB (need >= {_MIN_UNIVERSE})")
    return _ok("UNIV", f"{n} KRW markets in DB")


def check_btc() -> bool:
    markets = get_all_krw_markets_in_db()
    if _BTC_MARKET not in markets:
        return _fail("BTC", f"{_BTC_MARKET} not found in DB universe")
    return _ok("BTC", f"{_BTC_MARKET} present")


def check_horizon() -> bool:
    h = config.Data.PREDICT_HORIZON
    if h is None:
        return _fail("HORIZ", "PREDICT_HORIZON is not set")
    detail = f"PREDICT_HORIZON={h}h"
    if h != _EXPECTED_HORIZON:
        detail += f"  (expected {_EXPECTED_HORIZON}h)"
    return _ok("HORIZ", detail)


def check_beta_window() -> bool:
    w = config.Data.BETA_ROLLING_WINDOW
    if w is None:
        return _fail("BETA", "BETA_ROLLING_WINDOW is not set")
    detail = f"BETA_ROLLING_WINDOW={w}h"
    if w != _EXPECTED_BETA_WINDOW:
        detail += f"  (expected {_EXPECTED_BETA_WINDOW}h)"
    return _ok("BETA", detail)


def main():
    print("=== xsec_alpha pre-flight check ===\n")
    results = [
        check_db(),
        check_data_freshness(),
        check_universe(),
        check_btc(),
        check_horizon(),
        check_beta_window(),
    ]
    print()
    if all(results):
        print("All checks passed.")
        sys.exit(0)
    else:
        n_fail = results.count(False)
        print(f"{n_fail} check(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
