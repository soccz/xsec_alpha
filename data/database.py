import sqlite3
import pandas as pd
from datetime import datetime, timedelta, timezone

from config import config
from utils.logger import logger


def _parse_db_ts(ts: str):
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except Exception:
        return None


def get_db_connection():
    conn = sqlite3.connect(config.General.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA busy_timeout=5000")
    except Exception:
        pass
    return conn


def init_db():
    logger.info("Initializing database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS crypto_data (
        timestamp DATETIME,
        market TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        PRIMARY KEY (timestamp, market)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)
    conn.commit()
    conn.close()
    logger.info("DB initialized.")


def save_data(df: pd.DataFrame, table_name: str):
    if df.empty:
        logger.warning(f"Empty DataFrame for '{table_name}'. Nothing to save.")
        return
    conn = get_db_connection()
    try:
        df.to_sql(table_name, conn, if_exists="append", index=False)
    except Exception as e:
        logger.error(f"Error saving to '{table_name}': {e}")
    finally:
        conn.close()


def load_data(query: str, params=None) -> pd.DataFrame:
    conn = get_db_connection()
    try:
        return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        logger.error(f"DB load error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_latest_db_timestamp(markets: list = None):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        if markets:
            placeholders = ", ".join("?" for _ in markets)
            cur.execute(f"SELECT MAX(timestamp) FROM crypto_data WHERE market IN ({placeholders})", markets)
        else:
            cur.execute("SELECT MAX(timestamp) FROM crypto_data")
        row = cur.fetchone()
        return _parse_db_ts(row[0] if row else None)
    except Exception:
        return None
    finally:
        conn.close()


def get_all_krw_markets_in_db(exclude_patterns=None, freshness_days: int = 7) -> list:
    """Return KRW markets present in DB whose latest data is within freshness_days of the global max.

    freshness_days=7 means: exclude any coin whose latest timestamp is more than
    7 days behind the most recent timestamp in the DB. This filters out delisted or
    stale coins that would pollute the holdout period with forward-filled flat prices.
    Set freshness_days=0 to disable the filter (return all markets).
    """
    exclude_patterns = exclude_patterns or config.Data.DYNAMIC_UNIVERSE_EXCLUDE
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Global DB maximum timestamp
        cur.execute("SELECT MAX(timestamp) FROM crypto_data WHERE market LIKE 'KRW-%'")
        row = cur.fetchone()
        global_max_str = row[0] if row else None
        global_max = _parse_db_ts(global_max_str)

        # Per-market latest timestamps
        cur.execute(
            "SELECT market, MAX(timestamp) as latest FROM crypto_data "
            "WHERE market LIKE 'KRW-%' GROUP BY market"
        )
        rows = cur.fetchall() or []

        markets = []
        for r in rows:
            market = r[0]
            latest = _parse_db_ts(r[1])
            if freshness_days > 0 and global_max and latest:
                if (global_max - latest).total_seconds() > freshness_days * 86400:
                    continue  # stale: skip
            markets.append(market)

        filtered = []
        for m in markets:
            coin = m.replace("KRW-", "")
            if not any(ex in coin for ex in exclude_patterns):
                filtered.append(m)

        logger.info(
            f"Universe after freshness filter (≤{freshness_days}d stale): "
            f"{len(filtered)} markets"
        )
        return sorted(filtered)
    except Exception:
        return []
    finally:
        conn.close()


def load_binance_ohlcv(markets: list, days: int = 120) -> pd.DataFrame:
    """Load Binance OHLCV from binance_data table for given markets."""
    latest = get_latest_db_timestamp()  # use global latest as reference
    if latest is None:
        return pd.DataFrame()
    start = latest - timedelta(days=days)
    placeholders = ", ".join("?" for _ in markets)
    query = f"""
        SELECT timestamp, market, close, volume
        FROM binance_data
        WHERE market IN ({placeholders})
          AND timestamp >= ?
        ORDER BY timestamp
    """
    params = markets + [start.strftime("%Y-%m-%dT%H:%M:%S")]
    df = load_data(query, params=params)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_ohlcv(markets: list, days: int = 120) -> pd.DataFrame:
    """
    Load OHLCV for given markets over last `days` days.
    Returns long-format DataFrame: (timestamp, market, open, high, low, close, volume).
    """
    latest = get_latest_db_timestamp(markets)
    if latest is None:
        logger.error("No data in DB.")
        return pd.DataFrame()

    start = latest - timedelta(days=days)
    placeholders = ", ".join("?" for _ in markets)
    query = f"""
        SELECT timestamp, market, open, high, low, close, volume
        FROM crypto_data
        WHERE market IN ({placeholders})
          AND timestamp >= ?
        ORDER BY timestamp
    """
    params = markets + [start.strftime("%Y-%m-%dT%H:%M:%S")]
    df = load_data(query, params=params)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
