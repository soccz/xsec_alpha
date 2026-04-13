from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone

from utils.logger import logger

_ROOT = os.path.dirname(os.path.dirname(__file__))
_OUTPUT = os.path.join(_ROOT, "output")
_CACHE_PATH = os.path.join(_OUTPUT, "bitget_usdt_contracts.json")
_CONTRACTS_URL = "https://api.bitget.com/api/v2/mix/market/contracts?productType=USDT-FUTURES"


def _normalize_contracts(payload: dict) -> dict[str, dict]:
    contracts = payload.get("data", []) if isinstance(payload, dict) else []
    tradable = {}
    for row in contracts:
        if row.get("symbolStatus") != "normal":
            continue
        base = row.get("baseCoin")
        symbol = row.get("symbol")
        if not base or not symbol:
            continue
        tradable[base] = row
    return tradable


def _write_cache(payload: dict) -> None:
    os.makedirs(_OUTPUT, exist_ok=True)
    cache_payload = {
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    with open(_CACHE_PATH, "w") as f:
        json.dump(cache_payload, f)


def _read_cache() -> dict[str, dict]:
    if not os.path.exists(_CACHE_PATH):
        return {}
    try:
        with open(_CACHE_PATH) as f:
            cached = json.load(f)
        contracts = _normalize_contracts(cached.get("payload", {}))
        if contracts:
            logger.warning("Using cached Bitget contract list: %s symbols", len(contracts))
        return contracts
    except Exception as exc:
        logger.warning("Failed to read Bitget cache: %s", exc)
        return {}


def load_bitget_usdt_perp_map(timeout_sec: float = 10.0) -> dict[str, dict]:
    """Return tradable Bitget USDT perpetual contracts keyed by base coin."""
    try:
        with urllib.request.urlopen(_CONTRACTS_URL, timeout=timeout_sec) as resp:
            payload = json.load(resp)
        contracts = _normalize_contracts(payload)
        if contracts:
            _write_cache(payload)
            logger.info("Bitget tradable universe loaded: %s USDT perpetual symbols", len(contracts))
            return contracts
        logger.warning("Bitget contract response returned no tradable symbols")
    except Exception as exc:
        logger.warning("Bitget contract fetch failed: %s", exc)

    return _read_cache()


def market_to_bitget_symbol(market: str, contract_map: dict[str, dict]) -> str | None:
    base = market.replace("KRW-", "")
    row = contract_map.get(base)
    return row.get("symbol") if row else None


def filter_markets_to_bitget(markets: list[str], contract_map: dict[str, dict]) -> list[str]:
    return [market for market in markets if market_to_bitget_symbol(market, contract_map)]
