"""Cross-sectional feature computation for xsec_alpha."""
import numpy as np
import pandas as pd
from math import sqrt

from config import config
from utils.logger import logger


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_binance_pivot(markets: list, days: int = 120) -> pd.DataFrame:
    """
    Load Binance close prices and pivot to (timestamp × market).
    Returns closes only (USDT-denominated). NaN where binance has no data.
    """
    from data.database import load_binance_ohlcv
    bn_markets = [m for m in markets if m.startswith("KRW-")]
    df = load_binance_ohlcv(bn_markets, days=days)
    if df.empty:
        return pd.DataFrame()
    df = df.drop_duplicates(subset=["timestamp", "market"])
    closes = df.pivot(index="timestamp", columns="market", values="close")
    closes.ffill(inplace=True)
    return closes


def load_and_pivot(days: int = 120) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load OHLCV from DB and pivot to (timestamp × market) DataFrames.
    Returns: closes, opens, highs, lows, volumes
    """
    from data.database import get_all_krw_markets_in_db, load_ohlcv

    markets = get_all_krw_markets_in_db()
    logger.info(f"Universe: {len(markets)} KRW markets")

    df = load_ohlcv(markets, days=days)
    if df.empty:
        raise RuntimeError("No OHLCV data loaded from DB")

    # Pivot each OHLCV column
    def pivot(col):
        return df.pivot(index="timestamp", columns="market", values=col)

    closes  = pivot("close")
    opens   = pivot("open")
    highs   = pivot("high")
    lows    = pivot("low")
    volumes = pivot("volume")

    # Forward-fill only; NO bfill (bfill leaks future prices into past)
    for p in (closes, opens, highs, lows, volumes):
        p.ffill(inplace=True)

    # Drop coins that still have NaN (never had data in this window)
    valid_coins = closes.columns[closes.notna().all()].tolist()
    logger.info(f"Coins with complete data: {len(valid_coins)}/{len(markets)}")

    return (
        closes[valid_coins],
        opens[valid_coins],
        highs[valid_coins],
        lows[valid_coins],
        volumes[valid_coins],
    )


def build_top_liquidity_universe_index(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    top_n: int | None = None,
    lookback_hours: int | None = None,
):
    """Build a per-timestamp active universe from traded value.

    Returns
    -------
    selected_index:
        MultiIndex(timestamp, market) of rows that belong to the active universe.
    coverage:
        Simple diagnostics for logging.
    """
    if top_n is None:
        top_n = getattr(config.Data, "LIQUIDITY_TOP_N", 0)
    if lookback_hours is None:
        lookback_hours = getattr(config.Data, "LIQUIDITY_LOOKBACK_HOURS", 24)

    if top_n <= 0:
        counts = closes.notna().sum(axis=1)
        coverage = {
            "avg_selected": float(counts.mean()) if not counts.empty else 0.0,
            "min_selected": int(counts.min()) if not counts.empty else 0,
            "max_selected": int(counts.max()) if not counts.empty else 0,
        }
        selected_index = closes.stack(future_stack=True).dropna().index
        return selected_index, coverage

    traded_value = (closes * volumes).rolling(lookback_hours).sum()
    liquidity_long = traded_value.stack(future_stack=True).rename("traded_value").dropna()
    rank = liquidity_long.groupby(level="timestamp").rank(method="first", ascending=False)
    selected = liquidity_long[rank <= top_n]
    counts = selected.groupby(level="timestamp").size()
    coverage = {
        "avg_selected": float(counts.mean()) if not counts.empty else 0.0,
        "min_selected": int(counts.min()) if not counts.empty else 0,
        "max_selected": int(counts.max()) if not counts.empty else 0,
    }
    return selected.index, coverage


def filter_long_frame_by_universe(df: pd.DataFrame, selected_index: pd.MultiIndex | None) -> pd.DataFrame:
    if selected_index is None:
        return df
    return df.loc[df.index.intersection(selected_index)]


def filter_long_series_by_universe(series: pd.Series, selected_index: pd.MultiIndex | None) -> pd.Series:
    if selected_index is None:
        return series
    return series.loc[series.index.intersection(selected_index)]


# ---------------------------------------------------------------------------
# Beta and residual returns
# ---------------------------------------------------------------------------

def compute_beta_series(coin_returns: pd.Series, btc_returns: pd.Series, window: int = 168) -> pd.Series:
    """Rolling OLS beta via cov/var."""
    cov = coin_returns.rolling(window).cov(btc_returns)
    var = btc_returns.rolling(window).var()
    return cov / var.replace(0, np.nan)


def compute_all_betas(returns: pd.DataFrame, btc_col: str = "KRW-BTC", window: int = 168) -> pd.DataFrame:
    if btc_col not in returns.columns:
        raise ValueError(f"{btc_col} not in returns DataFrame")
    btc = returns[btc_col]
    betas = {}
    for col in returns.columns:
        betas[col] = compute_beta_series(returns[col], btc, window)
    return pd.DataFrame(betas)


def compute_forward_returns(
    prices: pd.DataFrame,
    horizon: int = 12,
    execution_lag: int = 0,
) -> pd.DataFrame:
    """
    Forward return aligned to the signal timestamp.

    execution_lag=0:
        return[t] = price[t+h] / price[t] - 1

    execution_lag=1:
        return[t] = price[t+1+h] / price[t+1] - 1
    """
    entry = prices.shift(-execution_lag) if execution_lag > 0 else prices
    exit_ = prices.shift(-(execution_lag + horizon))
    return exit_ / entry - 1


def compute_residual_returns(
    closes: pd.DataFrame,
    horizon: int = 12,
    beta_window: int = 168,
    execution_lag: int = 0,
) -> pd.DataFrame:
    """
    Forward residual return at each timestamp:
        residual[t] = coin_fwd_return[t] - beta[t] * btc_fwd_return[t]

    coin_fwd_return[t] = price[t+lag+horizon] / price[t+lag] - 1
    beta[t] = rolling beta computed on past `beta_window` hourly returns

    Returns DataFrame same shape as closes, with NaN for the last
    `horizon + execution_lag` rows.
    """
    returns_1h = closes.pct_change(1)

    btc_col = "KRW-BTC"
    if btc_col not in closes.columns:
        raise ValueError("KRW-BTC not in universe — cannot compute residual returns")

    betas = compute_all_betas(returns_1h, btc_col=btc_col, window=beta_window)

    # Forward returns over `horizon` hours, optionally delayed by `execution_lag`
    fwd_returns = compute_forward_returns(closes, horizon=horizon, execution_lag=execution_lag)
    btc_fwd = fwd_returns[btc_col]

    # residual = coin_fwd - beta * btc_fwd
    residuals = fwd_returns.subtract(betas.multiply(btc_fwd, axis=0))
    return residuals


# ---------------------------------------------------------------------------
# 6 cross-sectional factors
# ---------------------------------------------------------------------------

CALENDAR_COLS = ["dow_bull", "hour_vol"]  # broadcast per-timestamp, skip cross-sectional z-score


def compute_factors(
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    volumes: pd.DataFrame,
    binance_closes: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Compute cross-sectional factors. All inputs: (timestamp × market).

    Cross-sectional factors (z-scored per timestamp):
        reversal_1h, reversal_4h       — short-term mean reversion
        volatility_inv_24h             — low-vol coins outperform (proven IC=+0.081)
        order_flow_bear                — net selling pressure last 6h

    Calendar factors (same value all coins at a timestamp — NOT z-scored):
        dow_bull     — Mon/Tue/Wed KST=+1, Thu=0, Fri/Sat/Sun=-1
        hour_vol     — 1 if KST hour in [7,8,9,22,23,0] (8am/11pm volatility windows)
    """
    returns_1h = closes.pct_change(1)
    btc_col = "KRW-BTC"

    # --- Cross-sectional factors ---

    # 1. Reversal 1h: recent losers mean-revert over the next few hours.
    reversal_1h = -closes.pct_change(1)

    # 2. Reversal 4h: same idea on a slightly wider window.
    reversal_4h = -closes.pct_change(4)

    # NOTE:
    # btc_lead_{1h,4h} was removed from the active factor set because after
    # per-timestamp cross-sectional z-scoring it collapses to reversal_{1h,4h}.
    # Keeping both wastes model capacity on a duplicated signal.

    # 3. Volatility inverse: lower vol → outperforms (IC=+0.081 proven)
    volatility_inv_24h = -returns_1h.rolling(24).std()

    # 4. Bearish order flow: heavier recent selling tends to mean-revert.
    direction = np.sign(closes - opens)
    direction[direction == 0] = 1
    signed_vol = volumes * direction
    vol_sum_6h = volumes.rolling(6).sum().replace(0, np.nan)
    order_flow_bear = -(signed_vol.rolling(6).sum() / vol_sum_6h)

    factor_dict = {
        "reversal_1h":        reversal_1h,
        "reversal_4h":        reversal_4h,
        "volatility_inv_24h": volatility_inv_24h,
        "order_flow_bear":    order_flow_bear,
    }

    # --- Binance cross-exchange factors (optional) ---
    if binance_closes is not None and not binance_closes.empty:
        # Align timestamps: inner join on common timestamps
        common_ts = closes.index.intersection(binance_closes.index)
        common_coins = [c for c in closes.columns if c in binance_closes.columns]

        if len(common_ts) > 0 and len(common_coins) > 0:
            upbit_r1 = closes[common_coins].loc[common_ts].pct_change(1)
            bn_r1    = binance_closes[common_coins].loc[common_ts].pct_change(1)

            # binance_lead_1h: binance 1h return > upbit 1h return → upbit hasn't caught up → BUY
            bl1h = (bn_r1 - upbit_r1).reindex(index=closes.index, columns=closes.columns)
            factor_dict["binance_lead_1h"] = bl1h

            # kimchi_premium: log(upbit/binance) — high = Korea premium = overheated → reversion
            # We INVERT (negate) so high kimchi = sell signal → factor direction is: low premium = buy
            upbit_p  = closes[common_coins].reindex(index=closes.index)
            bn_p     = binance_closes[common_coins].reindex(index=closes.index)
            raw_premium = np.log(upbit_p / bn_p.replace(0, np.nan))
            # negate: low premium (under-valued vs global) → buy signal
            factor_dict["kimchi_inv"] = (-raw_premium).reindex(columns=closes.columns)

            logger.info(f"Binance factors added: {len(common_coins)} coins, {len(common_ts)} timestamps")

    stacked = pd.concat(
        {name: df.stack(future_stack=True) for name, df in factor_dict.items()},
        axis=1,
    )
    stacked.index.names = ["timestamp", "market"]

    # --- Calendar factors (broadcast same value to all coins per timestamp) ---
    timestamps = closes.index
    kst = timestamps + pd.Timedelta(hours=9)

    dow = kst.dayofweek  # 0=Mon … 6=Sun
    dow_bull_vals = np.where(dow < 3, 1.0, np.where(dow == 3, 0.0, -1.0))

    hour = kst.hour
    hour_vol_vals = (((hour >= 7) & (hour <= 9)) | (hour >= 22) | (hour <= 1)).astype(float)

    # Repeat calendar values for every (timestamp, market) row in stacked
    n_coins = len(closes.columns)
    stacked["dow_bull"] = np.repeat(dow_bull_vals, n_coins)
    stacked["hour_vol"] = np.repeat(hour_vol_vals, n_coins)

    return stacked


# ---------------------------------------------------------------------------
# Long-specialist factors (trend / breakout / momentum)
# ---------------------------------------------------------------------------

LONG_CALENDAR_COLS = ["dow_bull", "hour_vol"]  # same calendar cols


def compute_long_factors(
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    volumes: pd.DataFrame,
    binance_closes: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Compute long-specialist cross-sectional factors (v3).

    v3 design principles (from 10-agent Loop 1 review):
      - All momentum/trend factors removed (IC negative at every horizon)
      - range_contraction replaces vol_inv (IC +0.179 vs +0.168, lower collinearity with others)
      - reversal factors restored (IC positive in bull regime: buy the dip works)
      - Only factors with empirically verified positive IC in bull regime retained

    Cross-sectional factors (z-scored per timestamp):
        range_contraction_12h — avg intrabar range shrinking = compression before expansion
                                IC: +0.109 (6h), +0.137 (12h), +0.179 (24h). Stable across Q1-Q4.
        reversal_1h           — 1h losers bounce. IC: +0.102 (6h), +0.078 (12h)
        reversal_4h           — 4h losers bounce. IC: +0.094 (6h), +0.073 (12h)

    Calendar factors (NOT z-scored):
        dow_bull     — Mon/Tue/Wed KST=+1, Thu=0, Fri/Sat/Sun=-1
        hour_vol     — 1 if KST hour in [7,8,9,22,23,0]
    """
    returns_1h = closes.pct_change(1)

    # --- Cross-sectional factors ---

    # 1. Range contraction 12h: shrinking intrabar range = compression before breakout
    #    IC=+0.179 at 24h (strongest single factor), stable across all quarters
    intrabar_range = (highs - lows) / closes.replace(0, np.nan)
    range_contraction_12h = -intrabar_range.rolling(12).mean()

    # 2. Reversal 1h: recent 1h losers bounce (IC=+0.102 at 6h in bull regime)
    reversal_1h = -closes.pct_change(1)

    # 3. Reversal 4h: recent 4h losers bounce (IC=+0.094 at 6h in bull regime)
    reversal_4h = -closes.pct_change(4)

    factor_dict = {
        "range_contraction_12h": range_contraction_12h,
        "reversal_1h":           reversal_1h,
        "reversal_4h":           reversal_4h,
    }

    # --- Binance cross-exchange factors (optional) ---
    if binance_closes is not None and not binance_closes.empty:
        common_ts = closes.index.intersection(binance_closes.index)
        common_coins = [c for c in closes.columns if c in binance_closes.columns]

        if len(common_ts) > 0 and len(common_coins) > 0:
            upbit_r1 = closes[common_coins].loc[common_ts].pct_change(1)
            bn_r1 = binance_closes[common_coins].loc[common_ts].pct_change(1)

            # binance_lead_1h: same as short model — global leads local
            bl1h = (bn_r1 - upbit_r1).reindex(index=closes.index, columns=closes.columns)
            factor_dict["binance_lead_1h"] = bl1h

            logger.info(f"Binance factors added (long): {len(common_coins)} coins, {len(common_ts)} timestamps")

    stacked = pd.concat(
        {name: df.stack(future_stack=True) for name, df in factor_dict.items()},
        axis=1,
    )
    stacked.index.names = ["timestamp", "market"]

    # --- Calendar factors (same as short model) ---
    timestamps = closes.index
    kst = timestamps + pd.Timedelta(hours=9)

    dow = kst.dayofweek
    dow_bull_vals = np.where(dow < 3, 1.0, np.where(dow == 3, 0.0, -1.0))

    hour = kst.hour
    hour_vol_vals = (((hour >= 7) & (hour <= 9)) | (hour >= 22) | (hour <= 1)).astype(float)

    n_coins = len(closes.columns)
    stacked["dow_bull"] = np.repeat(dow_bull_vals, n_coins)
    stacked["hour_vol"] = np.repeat(hour_vol_vals, n_coins)

    return stacked


# ---------------------------------------------------------------------------
# BTC regime gate for long model
# ---------------------------------------------------------------------------

def compute_btc_regime(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute BTC regime indicators for long model gating.

    Returns DataFrame indexed by timestamp with columns:
        btc_ret_7d:  BTC 7-day return
        btc_ret_30d: BTC 30-day return
        btc_above_sma20: 1 if BTC close > 20-day SMA, else 0
        regime_bull: 1 if btc_ret_7d > gate AND btc_ret_30d > floor
    """
    btc_col = "KRW-BTC"
    if btc_col not in closes.columns:
        raise ValueError("KRW-BTC not in universe — cannot compute BTC regime")

    btc = closes[btc_col]
    btc_ret_7d = btc.pct_change(7 * 24)   # 7 days in hourly bars
    btc_ret_30d = btc.pct_change(30 * 24)  # 30 days
    btc_sma20 = btc.rolling(20 * 24).mean()
    btc_above_sma20 = (btc > btc_sma20).astype(float)

    regime = pd.DataFrame({
        "btc_ret_7d": btc_ret_7d,
        "btc_ret_30d": btc_ret_30d,
        "btc_above_sma20": btc_above_sma20,
    }, index=closes.index)

    from config import config
    gate_7d = getattr(config.LongModel, "BTC_7D_RETURN_GATE", 0.0)
    floor_30d = getattr(config.LongModel, "BTC_30D_RETURN_FLOOR", -0.10)
    # NaN-safe: if 30d return is NaN (insufficient data), only check 7d gate
    cond_7d = btc_ret_7d > gate_7d
    cond_30d = (btc_ret_30d > floor_30d) | btc_ret_30d.isna()
    regime["regime_bull"] = (cond_7d & cond_30d).astype(float)

    return regime


# ---------------------------------------------------------------------------
# Cross-sectional normalization
# ---------------------------------------------------------------------------

def crosssection_zscore(factor_df: pd.DataFrame, cols: list = None, clip: float = None) -> pd.DataFrame:
    """
    Per-timestamp z-score normalization with optional |z| clipping.
    factor_df: MultiIndex(timestamp, market).
    CALENDAR_COLS are automatically excluded (they're constant across coins per timestamp).
    """
    if clip is None:
        clip = config.Data.ZSCORE_CLIP
    if cols is None:
        cols = [c for c in factor_df.columns if c not in CALENDAR_COLS]

    result = factor_df.copy()
    for col in cols:
        grp = result[col].groupby(level="timestamp")
        result[col] = (grp.transform("mean")
                        .rsub(result[col])
                        .div(grp.transform("std").replace(0, np.nan)))
        result[col] = result[col].clip(-clip, clip)
    return result


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

def compute_ic_series(factor_df: pd.DataFrame, residuals_long: pd.Series, factor_col: str) -> pd.Series:
    """
    Compute Spearman IC per timestamp between one factor and residual returns.
    Coins with NaN in either factor or residual are excluded at each timestamp.

    Returns pd.Series of IC values indexed by timestamp.
    """
    from scipy.stats import spearmanr

    merged = pd.concat([factor_df[factor_col], residuals_long.rename("residual")], axis=1).dropna()

    ic_values = {}
    for ts, group in merged.groupby(level="timestamp"):
        if len(group) < 5:
            continue
        ic, _ = spearmanr(group[factor_col], group["residual"])
        ic_values[ts] = ic

    return pd.Series(ic_values, name=factor_col)


def summarize_ic(ic_series: pd.Series) -> dict:
    """Return mean IC, IC std, t-stat, and IC_IR."""
    n = len(ic_series.dropna())
    mean = ic_series.mean()
    std  = ic_series.std()
    t_stat = mean / (std / sqrt(n)) if n > 1 and std > 0 else 0.0
    return {
        "factor":   ic_series.name,
        "mean_ic":  round(float(mean), 4),
        "ic_std":   round(float(std), 4),
        "t_stat":   round(float(t_stat), 3),
        "n_periods": n,
    }
