import os

_ROOT = os.path.dirname(__file__)


class Config:
    class General:
        APP_NAME = "xsec_alpha"
        # Read from gan_t's existing DB by default; override with env var
        DB_PATH = os.getenv("XSEC_DB_PATH", os.path.join(_ROOT, "..", "data", "crypto_data.db"))
        LOG_DIR = os.path.join(_ROOT, "logs")

    class Data:
        BETA_ROLLING_WINDOW = 168        # hours for rolling beta
        PREDICT_HORIZON = 6              # hours forward (6h rebalance)
        ZSCORE_CLIP = 3.0
        MIN_ROWS_PER_COIN = 480          # warmup rows to drop per coin
        LIQUIDITY_TOP_N = 100            # active universe = top 100 by 24h traded value
        LIQUIDITY_LOOKBACK_HOURS = 24
        DYNAMIC_UNIVERSE_EXCLUDE = ["USDT", "BUSD", "DAI", "USDC", "UP", "DOWN", "BEAR", "BULL"]

        COLLECTOR_REQUEST_MAX_RETRIES = 3
        COLLECTOR_REQUEST_BACKOFF_SEC = 1.0
        COLLECTOR_PAGE_SLEEP_SEC = 0.5

    class Portfolio:
        LONG_N = 20    # top 10% of ~200 coins
        SHORT_N = 20   # bottom 10%
        LIVE_EXECUTION_MODE = os.getenv("XSEC_LIVE_EXECUTION_MODE", "short_only")
        LIVE_WATCH_LONG_N = int(os.getenv("XSEC_LIVE_WATCH_LONG_N", "5"))
        LIVE_EXEC_SHORT_N = int(os.getenv("XSEC_LIVE_EXEC_SHORT_N", "5"))
        LIVE_REQUIRE_BITGET_TRADABLE = os.getenv("XSEC_LIVE_REQUIRE_BITGET_TRADABLE", "1") != "0"
        TOTAL_CAPITAL_KRW = 10_000_000  # 총 운용 자본 (KRW). 환경변수로 오버라이드 가능
        MAX_POSITION_PCT = 10.0         # 단일 포지션 최대 비중 (%)
        STOP_LOSS_PCT = 3.0             # 손절 기준 (%)
        REBAL_BUFFER = 10               # 기존 포지션 유지 버퍼 (rank N+buffer까지 유지)

    class Model:
        # F1 unified (2026-04-25): 10-feature unified factor library, all-regime,
        # absolute-return target. Paired with LongModel below — both models share
        # identical features; only target horizon differs (6h vs 12h).
        MODEL_PATH = "models/xsec_6h.pkl"
        HOLDOUT_RATIO = 0.2

        # XGBoost hyperparameters
        XGB_N_ESTIMATORS = 500
        XGB_MAX_DEPTH = 4
        XGB_LEARNING_RATE = 0.05
        XGB_SUBSAMPLE = 0.8
        XGB_COLSAMPLE_BYTREE = 0.8
        XGB_MIN_CHILD_WEIGHT = 20
        XGB_RANDOM_STATE = 42

    class LongModel:
        # F1 unified (2026-04-25): same factor library as Model above.
        MODEL_PATH = "models/xsec_12h.pkl"
        PREDICT_HORIZON = 12            # long model uses 12h (short uses 6h)
        REBALANCE_ANCHOR_HOUR_UTC = 11  # 11/23 UTC under the current 6h timer cadence

        # Regime gate: long model only active when BTC is bullish
        # BTC 7d return > threshold AND BTC not in severe drawdown (30d > floor)
        BTC_7D_RETURN_GATE = 0.0       # BTC 7d return must be > 0%
        BTC_30D_RETURN_FLOOR = -0.10   # BTC 30d return must be > -10%

        # Long-specific portfolio
        LONG_N = 5                      # top 5 for execution
        # "long_only" = actionable LONG, "watch" = score but WATCH only, "disabled" = skip entirely
        # Production approval requires: holdout >= 60 timestamps, pred_std >= 0.01, long-only IC > 0
        EXECUTION_MODE = "watch"


config = Config()
