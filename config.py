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
        # Upbit candle endpoint limit is 10 req/s. 0.15s caps sequential
        # per-market collection at ~6.7 req/s before request latency.
        COLLECTOR_MARKET_SLEEP_SEC = 0.15

    class Portfolio:
        LONG_N = 20    # top 10% of ~200 coins
        SHORT_N = 20   # bottom 10%
        LIVE_EXECUTION_MODE = os.getenv("XSEC_LIVE_EXECUTION_MODE", "short_only")
        # 2026-07-11 WATCH_LONG KILL (README §14-bis 사전등록 재계약): 30d 실현 net −0.22%/trade,
        # 승률 45.3%로 §14 승격 전제와 모순 → N=0 (실행 픽·recommendation ledger 중지).
        # 단, Notification의 고정 강신호 게이트를 통과한 shadow 후보는 최대 1개까지
        # WATCH 알림으로만 보낼 수 있다. 이 경로는 actionable=False이며 실행과 분리된다.
        # 부활은 새 사전등록 + 코드 변경으로만 가능하다. 환경변수 우회 불가.
        LIVE_WATCH_LONG_N = 0
        LIVE_EXEC_SHORT_N = int(os.getenv("XSEC_LIVE_EXEC_SHORT_N", "5"))
        LIVE_REQUIRE_BITGET_TRADABLE = os.getenv("XSEC_LIVE_REQUIRE_BITGET_TRADABLE", "1") != "0"
        # Paper-observation mode for the LONG side. When True:
        #   - Telegram alerts SUPPRESS the LONG/WATCH_LONG section (no spam during observation).
        #   - Ledger continues to record realized PnL for WATCH_LONG picks (paper trading).
        #   - Dashboard surfaces a "long-paper-observation" status so the gap is auditable.
        # Set XSEC_LIVE_LONG_TELEGRAM_SILENT=1 to enter observation period; flip back when ready
        # to wire LONG to live execution (then also flip LIVE_EXECUTION_MODE).
        LIVE_LONG_TELEGRAM_SILENT = os.getenv("XSEC_LIVE_LONG_TELEGRAM_SILENT", "0") == "1"
        LIVE_LONG_OBSERVATION_START = os.getenv("XSEC_LIVE_LONG_OBSERVATION_START", "")
        TOTAL_CAPITAL_KRW = 10_000_000  # 총 운용 자본 (KRW). 환경변수로 오버라이드 가능
        MAX_POSITION_PCT = 10.0         # 단일 포지션 최대 비중 (%)
        STOP_LOSS_PCT = 3.0             # 손절 기준 (%)
        REBAL_BUFFER = 10               # 기존 포지션 유지 버퍼 (rank N+buffer까지 유지)

    class Costs:
        # Background accounting only. Telegram stays recommendation-focused;
        # dashboard/ledger disclose the cost assumptions and net return.
        ONE_WAY_FEE_BPS = float(os.getenv("XSEC_ONE_WAY_FEE_BPS", "6"))
        SLIPPAGE_BPS = float(os.getenv("XSEC_SLIPPAGE_BPS", "4"))
        SHORT_EXTRA_COST_BPS = float(os.getenv("XSEC_SHORT_EXTRA_COST_BPS", "10"))

    class Notification:
        # Only these quality-screened rows reach Telegram. Full details are still
        # written to latest.csv / latest_predictions.csv / recommendation_ledger.csv.
        TELEGRAM_MIN_SIGMA = float(os.getenv("XSEC_TELEGRAM_MIN_SIGMA", "1.5"))
        TELEGRAM_MIN_EXPECTED_ABS_PCT = float(os.getenv("XSEC_TELEGRAM_MIN_EXPECTED_ABS_PCT", "0.20"))
        TELEGRAM_HIDE_UNTRUSTED = os.getenv("XSEC_TELEGRAM_HIDE_UNTRUSTED", "1") != "0"

        # WATCH_LONG KILL을 우회하지 않는 관찰전용 강신호 알림 계약.
        # 안전 임계치는 환경변수로 낮출 수 없다. OOS calibration의 2σ+ LONG
        # bucket(n=59, hit=66.1%, mean=+1.97%)과 §5의 실용 IC 기준을 사용한다.
        LONG_WATCH_ALERT_MAX_N = 1
        LONG_WATCH_ALERT_REQUIRE_BITGET_TRADABLE = True
        LONG_WATCH_ALERT_MIN_SIGMA = 2.0
        LONG_WATCH_ALERT_MIN_EXPECTED_PCT = 0.20
        LONG_WATCH_ALERT_MIN_DIRECTION_PROB = 0.65
        LONG_WATCH_ALERT_MIN_IC = 0.10
        LONG_WATCH_ALERT_MIN_IC_READS = 3
        LONG_WATCH_ALERT_IC_FLOOR = 0.05
        LONG_WATCH_ALERT_REQUIRE_CONSENSUS = True
        LONG_WATCH_ALERT_HIDE_UNTRUSTED = True

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
        # Weekly model promotion is only a shadow-model refresh: exact-anchor
        # OOS IC plus >=10 regime-active periods and mean LONG net > 0.
        # It never lifts Portfolio.LIVE_WATCH_LONG_N=0; execution reactivation
        # still requires the prospective preregistration in README §14-bis.
        EXECUTION_MODE = "watch"


config = Config()
