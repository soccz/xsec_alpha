"""Upbit data collector — adapted from gan_t/data/collector.py."""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Optional

from config import config
from utils.logger import logger
from data.database import save_data, get_db_connection, init_db


def _request_upbit_json(url: str, params: dict, timeout: int = 10):
    retries = int(config.Data.COLLECTOR_REQUEST_MAX_RETRIES)
    backoff_sec = float(config.Data.COLLECTOR_REQUEST_BACKOFF_SEC)
    last_err = None

    for attempt in range(1, retries + 1):
        for fmt in ("T", " "):
            req_params = dict(params)
            if "to" in req_params and isinstance(req_params["to"], str):
                req_params["to"] = req_params["to"].replace("T", fmt)
            try:
                res = requests.get(url, params=req_params, timeout=timeout)
                res.raise_for_status()
                data = res.json()
                if isinstance(data, dict) and "error" in data:
                    last_err = RuntimeError(str(data.get("error")))
                    continue
                return data, attempt, None
            except requests.exceptions.RequestException as e:
                last_err = e
                continue

        if attempt < retries:
            time.sleep(backoff_sec * attempt)

    return None, retries, last_err


def _get_last_timestamp(market: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(timestamp) FROM crypto_data WHERE market = ?", (market,))
    result = cursor.fetchone()
    conn.close()
    if result and result[0]:
        ts = result[0]
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S") if "T" in ts else datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    return None


def _save_candles(data: list, market: str):
    df = pd.DataFrame(data)
    df.rename(columns={
        "candle_date_time_utc": "timestamp",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume",
    }, inplace=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["market"] = market
    df.drop_duplicates(subset=["timestamp", "market"], inplace=True)
    save_data(df, "crypto_data")


def collect_market_data(market: str, days: int = 120):
    logger.info(f"Collecting {market} ({days} days)")
    url = "https://api.upbit.com/v1/candles/minutes/60"
    last_ts = _get_last_timestamp(market)

    if last_ts:
        to_datetime = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    else:
        to_datetime = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")

    all_data = []
    while True:
        params = {"market": market, "count": 200, "to": to_datetime}
        data, _, last_err = _request_upbit_json(url, params=params)
        if data is None:
            logger.error(f"Request failed for {market}: {last_err}")
            break
        if not data:
            break

        if last_ts:
            new_data = []
            stop = False
            for candle in data:
                ts = datetime.strptime(candle["candle_date_time_utc"], "%Y-%m-%dT%H:%M:%S")
                if ts > last_ts:
                    new_data.append(candle)
                else:
                    stop = True
                    break
            all_data.extend(new_data)
            if stop:
                break
        else:
            all_data.extend(data)

        oldest_str = data[-1]["candle_date_time_utc"]
        oldest_ts = datetime.strptime(oldest_str, "%Y-%m-%dT%H:%M:%S")

        if not last_ts and (datetime.utcnow() - oldest_ts).days >= days:
            break

        to_datetime = oldest_ts.strftime("%Y-%m-%dT%H:%M:%S")
        time.sleep(float(config.Data.COLLECTOR_PAGE_SLEEP_SEC))

        if len(all_data) >= 5000:
            _save_candles(all_data, market)
            all_data = []

    if all_data:
        _save_candles(all_data, market)
    logger.info(f"Done {market}: saved {len(all_data)} new rows")


def get_all_krw_markets() -> list:
    try:
        res = requests.get("https://api.upbit.com/v1/market/all", timeout=10)
        res.raise_for_status()
        return [item["market"] for item in res.json() if item["market"].startswith("KRW-")]
    except Exception as e:
        logger.error(f"Failed to fetch markets: {e}")
        return []


def run_all(days: int = 120):
    init_db()
    markets = get_all_krw_markets()
    logger.info(f"Collecting {len(markets)} KRW markets")
    for i, market in enumerate(markets):
        logger.info(f"[{i+1}/{len(markets)}] {market}")
        collect_market_data(market, days)
        time.sleep(1.1)
    logger.info("Collection complete")


def get_current_price(market: str) -> Optional[float]:
    try:
        res = requests.get("https://api.upbit.com/v1/ticker", params={"markets": market}, timeout=10)
        res.raise_for_status()
        data = res.json()
        return data[0].get("trade_price") if data else None
    except Exception:
        return None
