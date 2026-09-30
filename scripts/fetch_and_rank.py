#!/usr/bin/env python3
"""
Main entry point for xsec_alpha (systemd / cron).

Steps:
  1. Collect fresh OHLCV data for all KRW markets
  2. Compute cross-sectional factors
  3. Rank coins → Long top 20 / Short bottom 20
  4. Save recommendations CSV

Usage:
    cd /mnt/20t/main/gan_t/xsec_alpha
    python scripts/fetch_and_rank.py [--collect] [--dry-run]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from config import config
from utils.bitget import filter_markets_to_bitget, load_bitget_usdt_perp_map, market_to_bitget_symbol
from utils.eval_metrics import select_positions_with_buffer
from utils.logger import logger
from utils.run_lock import run_lock
from utils.model_release import artifact_sha256, model_release_guard


def _effective_long_pick_limit(model_limit: int, portfolio_limit: int, blocked: bool) -> int:
    """Honor the strictest LONG limit; an IC/policy block always wins."""
    if blocked:
        return 0
    return max(0, min(int(model_limit), int(portfolio_limit)))


def _short_pick_limit(requested: int, blocked: bool = False) -> int:
    return 0 if blocked else max(0, min(int(requested), 5))


def _short_suppression_reason(score, prediction, basket_size, batch_reason="") -> str:
    if batch_reason:
        return str(batch_reason)
    if basket_size < 5:
        return "insufficient_candidates"
    if prediction.get("suppression"):
        return str(prediction["suppression"])
    values = pd.to_numeric(
        pd.Series([score, prediction.get("sigma"), prediction.get("expected_pct")]),
        errors="coerce",
    )
    if not np.isfinite(values).all():
        return "prediction_unavailable"
    raw_score, sigma, expected = values
    if raw_score >= 0 or expected >= 0:
        return "direction_mismatch"
    if sigma < 1.0:
        return "weak_signal"
    return ""


def _long_regime_allows(regime_row, btc_7d_gate: float, btc_30d_floor: float) -> bool:
    """Return True only when both preregistered BTC LONG gates pass."""
    if regime_row is None:
        return False
    try:
        btc_7d = float(regime_row.get("btc_ret_7d"))
        btc_30d = float(regime_row.get("btc_ret_30d"))
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(
        np.isfinite(btc_7d)
        and np.isfinite(btc_30d)
        and btc_7d > float(btc_7d_gate)
        and btc_30d > float(btc_30d_floor)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="Run data collection before ranking")
    parser.add_argument("--dry-run", action="store_true", help="Print recommendations without saving")
    parser.add_argument("--no-model", action="store_true", help="Force equal-weight fallback (ignore XGBoost model)")
    parser.add_argument("--no-telegram", action="store_true", help="Suppress Telegram notification")
    parser.add_argument("--no-dashboard-export", action="store_true",
                        help="Skip dashboard payload refresh after run")
    parser.add_argument("--no-dashboard-push", action="store_true",
                        help="Skip auto-pushing dashboard payloads to soccz.github.io")
    args = parser.parse_args()

    with run_lock("fetch_and_rank"):
        try:
            _run(args)
        except Exception:
            logger.exception("Recommendation run failed")
            if not args.dry_run:
                from utils.operator_report import build_report, publish_report
                try:
                    publish_report(
                        build_report(error=True, reason="분석 실패 · 이전 후보는 참고용입니다"),
                        send=not args.no_telegram,
                    )
                except Exception:
                    logger.exception("Could not persist/deliver analysis failure report")
            raise
        finally:
            if not args.dry_run:
                from utils.prospective import maintain_experiment
                maintain_experiment()
            if not args.dry_run and not args.no_dashboard_export:
                publish_dashboard(push=not args.no_dashboard_push)


def publish_dashboard(push=True):
    from utils.dashboard_publish import publish_dashboard as isolated_publish
    return isolated_publish(push=push)


_KIMCHI_NAN_EXPECTED_PCT = 33.0  # ~25% Binance-unmatched coins + 24h warmup
_NAN_FLOOR_GENERIC_PCT = 30.0     # CLAUDE.md §8 rule for non-kimchi factors


def _write_feature_health(factor_df) -> None:
    """Write per-factor NaN% + flag for health_snapshot.py.

    Output: output/feature_health.json
    Fields:
      - generated_at: ISO timestamp
      - rows: total rows in factor_df
      - factors: {col: {nan_pct, status, note}}
        status = OK / EXPECTED (kimchi structural) / WARN (>30% non-kimchi)
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    rows = len(factor_df)
    if rows == 0:
        return
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
        "factors": {},
    }
    for col in factor_df.columns:
        pct = float(factor_df[col].isna().sum() / rows * 100)
        if "kimchi" in col and pct <= _KIMCHI_NAN_EXPECTED_PCT + 5:
            status = "EXPECTED"
            note = f"structural: ~25% coins lack Binance + 24h warmup"
        elif pct > _NAN_FLOOR_GENERIC_PCT:
            status = "WARN"
            note = f"NaN% > {_NAN_FLOOR_GENERIC_PCT}% (CLAUDE.md §8)"
        else:
            status = "OK"
            note = ""
        payload["factors"][col] = {
            "nan_pct": round(pct, 2),
            "status": status,
            "note": note,
        }
    out = Path(__file__).resolve().parent.parent / "output" / "feature_health.json"
    out.write_text(json.dumps(payload, indent=2))


def _compute_score(latest_factors: "pd.DataFrame", args) -> "pd.Series":
    """Return composite score Series (index=market).

    Priority:
      1. XGBoost model (XSecRanker) if model file exists and --no-model not set
      2. Equal-weight mean of all factors (fallback)
    """
    fallback = latest_factors.mean(axis=1)
    fallback.attrs["is_fallback"] = True
    if args.no_model:
        logger.info("--no-model flag set: using equal-weight fallback")
        return fallback

    try:
        model_path = getattr(getattr(config, "Model", None), "MODEL_PATH", "models/xsec_xgb.pkl")
        abs_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)

        if not os.path.exists(abs_model_path):
            logger.warning(f"Model not found at {abs_model_path}: using equal-weight fallback")
            return fallback

        from models.xgb_ranker import XSecRanker
        model = XSecRanker.load(abs_model_path)
        scores_array = model.predict(latest_factors)
        score = pd.Series(scores_array, index=latest_factors.index)
        score.attrs["model_sha256"] = artifact_sha256(Path(abs_model_path))
        logger.info(f"XGBoost model loaded from {abs_model_path}: {len(score)} coins scored")
        return score

    except Exception as e:
        logger.warning(f"Model load/predict failed ({e}): using equal-weight fallback")
        return fallback


def _load_previous_recommendations(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            return pd.read_csv(os.path.realpath(path))
    except Exception as e:
        logger.warning(f"Could not load previous recommendations ({e})")
    return pd.DataFrame()


def _is_rebalance_slot(ts, horizon_h: int, anchor_hour_utc: int) -> bool:
    if horizon_h <= 0:
        return True
    ts_hour = pd.Timestamp(ts).tz_convert("UTC").hour if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).hour
    return ((ts_hour - anchor_hour_utc) % horizon_h) == 0


def _next_rebalance_ts(ts, horizon_h: int, anchor_hour_utc: int | None = None):
    base = pd.Timestamp(ts)
    if horizon_h <= 0:
        return base
    if anchor_hour_utc is None:
        return base + pd.Timedelta(hours=horizon_h)
    probe = base
    for _ in range(1, horizon_h + 2):
        probe = probe + pd.Timedelta(hours=1)
        if _is_rebalance_slot(probe, horizon_h, anchor_hour_utc):
            return probe
    return base + pd.Timedelta(hours=horizon_h)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _entry_observed_timestamp(row: dict, fallback) -> pd.Timestamp:
    """Use the actual price-observation time, falling back for legacy rows."""
    observed = pd.to_datetime(row.get("entry_observed_at"), utc=True, errors="coerce")
    if pd.isna(observed):
        observed = pd.Timestamp(fallback)
        if observed.tzinfo is None:
            observed = observed.tz_localize("UTC")
        else:
            observed = observed.tz_convert("UTC")
    return observed


def _long_recommendation_is_actionable(
    side: str,
    model_active: bool,
    prediction_actionable: bool,
) -> bool:
    """Fail closed unless execution mode and the per-row quality gate both pass."""
    return str(side).upper() == "LONG" and bool(model_active) and bool(prediction_actionable)


def _select_strong_long_watch_alerts(
    predictions_df: pd.DataFrame | None,
    shadow_markets,
    long_gate,
    *,
    regime_eligible: bool,
    rebalance_due: bool,
    bitget_tradable_enforced: bool,
) -> pd.DataFrame:
    """Return strict shadow-only LONG alerts while execution remains killed.

    This path is deliberately narrower than recommendation selection.  It only
    consumes the point-in-time Bitget-filtered shadow basket and never returns
    an actionable row.  Missing columns or gate evidence fail closed.
    """
    empty = pd.DataFrame()
    if predictions_df is None or predictions_df.empty:
        return empty
    if not regime_eligible or not rebalance_due or not bitget_tradable_enforced:
        return empty
    if not bool(getattr(long_gate, "policy_blocked", False)):
        # Once the KILL is explicitly lifted, the normal recommendation path is
        # the single source of Telegram rows; never duplicate it here.
        return empty
    if str(getattr(long_gate, "status", "")) != "OK":
        return empty

    min_reads = int(getattr(config.Notification, "LONG_WATCH_ALERT_MIN_IC_READS", 3))
    ic_floor = float(getattr(config.Notification, "LONG_WATCH_ALERT_IC_FLOOR", 0.05))
    min_ic = float(getattr(config.Notification, "LONG_WATCH_ALERT_MIN_IC", 0.10))
    try:
        ic_tail = [float(value) for value in getattr(long_gate, "ic_tail", [])]
        last_ic = float(getattr(long_gate, "last_ic", np.nan))
    except (TypeError, ValueError):
        return empty
    if (
        len(ic_tail) < min_reads
        or not np.isfinite(last_ic)
        or last_ic < min_ic
        or any(not np.isfinite(value) or value < ic_floor for value in ic_tail[-min_reads:])
    ):
        return empty

    required = {
        "actionable",
        "consensus",
        "direction",
        "direction_prob",
        "expected_pct",
        "sigma",
        "suppression",
        "trust_tag",
    }
    if not required.issubset(predictions_df.columns):
        return empty

    ordered_markets = [market for market in (shadow_markets or []) if market in predictions_df.index]
    if not ordered_markets:
        return empty
    candidates = predictions_df.loc[ordered_markets].copy()
    candidates["sigma"] = pd.to_numeric(candidates["sigma"], errors="coerce")
    candidates["expected_pct"] = pd.to_numeric(candidates["expected_pct"], errors="coerce")
    candidates["direction"] = pd.to_numeric(candidates["direction"], errors="coerce")
    candidates["direction_prob"] = pd.to_numeric(
        candidates["direction_prob"], errors="coerce"
    )

    min_sigma = float(getattr(config.Notification, "LONG_WATCH_ALERT_MIN_SIGMA", 2.0))
    min_expected = float(
        getattr(config.Notification, "LONG_WATCH_ALERT_MIN_EXPECTED_PCT", 0.20)
    )
    min_direction_prob = float(
        getattr(config.Notification, "LONG_WATCH_ALERT_MIN_DIRECTION_PROB", 0.65)
    )
    eligible = (
        candidates["actionable"].apply(_as_bool)
        & candidates["direction"].gt(0)
        & candidates["direction_prob"].ge(min_direction_prob)
        & candidates["sigma"].ge(min_sigma)
        & candidates["expected_pct"].ge(min_expected)
        & candidates["suppression"].fillna("").astype(str).str.strip().eq("")
    )
    if bool(getattr(config.Notification, "LONG_WATCH_ALERT_REQUIRE_CONSENSUS", True)):
        eligible &= candidates["consensus"].apply(_as_bool)
    if bool(getattr(config.Notification, "LONG_WATCH_ALERT_HIDE_UNTRUSTED", True)):
        eligible &= candidates["trust_tag"].fillna("").astype(str).ne("⚠")

    alerts = candidates.loc[eligible].copy()
    if alerts.empty:
        return empty
    alerts.insert(0, "market", alerts.index.astype(str))
    alerts["side"] = "WATCH_LONG"
    alerts["horizon_h"] = int(getattr(config.LongModel, "PREDICT_HORIZON", 12))
    alerts["actionable"] = False
    alerts["position_size_pct"] = 0.0
    alerts["alert_contract"] = "strong_shadow_watch_v1"
    alerts["execution_policy"] = "KILL"
    max_n = min(
        int(getattr(config.Notification, "LONG_WATCH_ALERT_MAX_N", 1)),
        int(getattr(config.LongModel, "LONG_N", 5)),
    )
    return alerts.head(max(0, max_n)).reset_index(drop=True)


def _cost_assumptions_for_side(side: str) -> dict:
    fee_bps = float(getattr(config.Costs, "ONE_WAY_FEE_BPS", 6.0))
    slippage_bps = float(getattr(config.Costs, "SLIPPAGE_BPS", 4.0))
    short_extra_bps = (
        float(getattr(config.Costs, "SHORT_EXTRA_COST_BPS", 10.0))
        if "short" in str(side).lower()
        else 0.0
    )
    estimated_cost_return = (2.0 * (fee_bps + slippage_bps) + short_extra_bps) / 10000.0
    return {
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "short_extra_cost_bps": short_extra_bps,
        "estimated_cost_return": estimated_cost_return,
    }


def _prediction_meta_for(market: str, side: str, short_df=None, long_df=None) -> dict:
    pred_df = short_df if "short" in str(side).lower() else long_df
    fields = {
        "sigma": None,
        "tag": "",
        "expected_pct": None,
        "hit_rate": None,
        "typical_move_pct": None,
        "coin_vol_pct": None,
        "consensus": None,
        "consensus_tag": "",
        "trust_tag": "",
        "trust_note": "",
        "suppression": "",
        "model_sha256": None,
        "calibration_generated_at": None,
    }
    if pred_df is None or market not in pred_df.index:
        return fields
    row = pred_df.loc[market]
    for key in fields:
        if key in row:
            val = row.get(key)
            if not isinstance(val, (list, tuple, dict)) and pd.isna(val):
                val = None
            fields[key] = val
    return fields


def _backfill_cost_columns(ledger_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if ledger_df is None or ledger_df.empty or "realized_return" not in ledger_df.columns:
        return ledger_df, False

    df = ledger_df.copy()
    changed = False
    for col in (
        "gross_return",
        "fee_bps",
        "slippage_bps",
        "short_extra_cost_bps",
        "estimated_cost_return",
        "net_return",
    ):
        if col not in df.columns:
            df[col] = np.nan
            changed = True

    for idx, row in df.iterrows():
        gross = pd.to_numeric(row.get("gross_return"), errors="coerce")
        if pd.isna(gross):
            gross = pd.to_numeric(row.get("realized_return"), errors="coerce")
            if pd.isna(gross):
                continue
            df.at[idx, "gross_return"] = float(gross)
            changed = True

        cost = _cost_assumptions_for_side(row.get("side", ""))
        for col in ("fee_bps", "slippage_bps", "short_extra_cost_bps", "estimated_cost_return"):
            if pd.isna(pd.to_numeric(row.get(col), errors="coerce")):
                df.at[idx, col] = cost[col]
                changed = True

        expected_net = float(gross) - float(df.at[idx, "estimated_cost_return"])
        net = pd.to_numeric(row.get("net_return"), errors="coerce")
        if pd.isna(net):
            df.at[idx, "net_return"] = expected_net
            changed = True

    return df, changed


def _update_long_shadow_ledger(now=None, price_fetcher=None) -> None:
    """Mature exact-entry LONG shadow picks without emitting executable rows."""
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    ledger_path = os.path.join(out_dir, "long_shadow_ledger.csv")
    os.makedirs(out_dir, exist_ok=True)
    columns = [
        "market",
        "signal_time",
        "regime_eligible",
        "horizon_h",
        "entry_price",
        "entry_observed_at",
        "exit_time_target",
        "exit_price",
        "exit_observed_at",
        "gross_return",
        "estimated_cost_return",
        "net_return",
        "score",
        "sigma",
        "price_source",
        "source_file",
        "model_sha256",
        "calibration_generated_at",
    ]
    if os.path.exists(ledger_path):
        try:
            ledger = pd.read_csv(ledger_path)
        except Exception as exc:
            logger.warning("Could not read LONG shadow ledger (%s)", exc)
            ledger = pd.DataFrame(columns=columns)
    else:
        ledger = pd.DataFrame(columns=columns)

    existing = set()
    key_cols = {"market", "signal_time", "horizon_h"}
    if not ledger.empty and key_cols.issubset(ledger.columns):
        for row in ledger[["market", "signal_time", "horizon_h"]].itertuples(index=False):
            signal = pd.to_datetime(row.signal_time, utc=True, errors="coerce")
            if pd.notna(signal):
                existing.add((str(row.market), signal.isoformat(), int(row.horizon_h)))

    fixed_now = pd.Timestamp(now) if now is not None else None
    if fixed_now is not None:
        fixed_now = fixed_now.tz_localize("UTC") if fixed_now.tzinfo is None else fixed_now.tz_convert("UTC")
    if price_fetcher is None:
        from data.collector import get_current_price

        price_fetcher = get_current_price

    matured = []
    for fname in sorted(os.listdir(out_dir)):
        if not (fname.startswith("predictions_") and fname.endswith(".csv")):
            continue
        try:
            predictions = pd.read_csv(os.path.join(out_dir, fname))
        except Exception:
            continue
        required = {
            "timestamp",
            "market",
            "horizon_h",
            "shadow_selected",
            "shadow_entry_price",
            "entry_observed_at",
        }
        if not required.issubset(predictions.columns):
            continue

        for row in predictions.to_dict(orient="records"):
            if not _as_bool(row.get("shadow_selected")):
                continue
            market = str(row.get("market", ""))
            horizon_h = pd.to_numeric(row.get("horizon_h"), errors="coerce")
            entry_price = pd.to_numeric(row.get("shadow_entry_price"), errors="coerce")
            signal_time = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
            entry_observed_at = pd.to_datetime(
                row.get("entry_observed_at"), utc=True, errors="coerce"
            )
            if (
                not market
                or pd.isna(horizon_h)
                or pd.isna(entry_price)
                or entry_price <= 0
                or pd.isna(signal_time)
                or pd.isna(entry_observed_at)
            ):
                continue
            horizon_h = int(horizon_h)
            key = (market, signal_time.isoformat(), horizon_h)
            if key in existing:
                continue

            target_exit = entry_observed_at + pd.Timedelta(hours=horizon_h)
            maturity_now = fixed_now or pd.Timestamp.now(tz="UTC")
            if target_exit > maturity_now:
                continue
            exit_price = pd.to_numeric(price_fetcher(market), errors="coerce")
            if pd.isna(exit_price) or exit_price <= 0:
                continue
            exit_observed_at = fixed_now or pd.Timestamp.now(tz="UTC")
            gross = (float(exit_price) - float(entry_price)) / float(entry_price)
            cost = _cost_assumptions_for_side("LONG")
            matured.append({
                "market": market,
                "signal_time": signal_time.isoformat(),
                "regime_eligible": _as_bool(row.get("regime_eligible")),
                "horizon_h": horizon_h,
                "entry_price": float(entry_price),
                "entry_observed_at": entry_observed_at.isoformat(),
                "exit_time_target": target_exit.isoformat(),
                "exit_price": float(exit_price),
                "exit_observed_at": exit_observed_at.isoformat(),
                "gross_return": gross,
                "estimated_cost_return": cost["estimated_cost_return"],
                "net_return": gross - cost["estimated_cost_return"],
                "score": pd.to_numeric(row.get("score"), errors="coerce"),
                "sigma": pd.to_numeric(row.get("sigma"), errors="coerce"),
                "price_source": "upbit_ticker_proxy",
                "source_file": fname,
                "model_sha256": row.get("model_sha256"),
                "calibration_generated_at": row.get("calibration_generated_at"),
            })
            existing.add(key)

    if not matured:
        if not os.path.exists(ledger_path):
            pd.DataFrame(columns=columns).to_csv(ledger_path, index=False)
        return

    matured_df = pd.DataFrame(matured)
    ledger = matured_df if ledger.empty else pd.concat([ledger, matured_df], ignore_index=True)
    ledger = ledger.sort_values(["signal_time", "market"]).reset_index(drop=True)
    ledger.to_csv(ledger_path, index=False)
    logger.info("LONG shadow ledger updated: +%s -> %s", len(matured), ledger_path)


def _update_realized_ledger(closes: "pd.DataFrame", latest_ts) -> None:
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    ledger_path = os.path.join(out_dir, "recommendation_ledger.csv")
    os.makedirs(out_dir, exist_ok=True)
    ledger_cols = [
        "market",
        "side",
        "actionable",
        "entry_time",
        "entry_observed_at",
        "exit_time_target",
        "exit_time_actual",
        "horizon_h",
        "entry_price",
        "exit_price",
        "gross_return",
        "fee_bps",
        "slippage_bps",
        "short_extra_cost_bps",
        "estimated_cost_return",
        "net_return",
        "realized_return",
        "refresh_reason",
        "source_file",
        "model_sha256",
        "calibration_generated_at",
    ]

    key_cols = ["market", "side", "entry_time", "horizon_h"]
    if os.path.exists(ledger_path):
        try:
            ledger_df = pd.read_csv(ledger_path)
        except Exception as e:
            logger.warning(f"Could not read ledger ({e}) — rebuilding from scratch")
            ledger_df = pd.DataFrame()
    else:
        ledger_df = pd.DataFrame()

    ledger_df, ledger_backfilled = _backfill_cost_columns(ledger_df)
    ledger_schema_changed = False
    if "entry_observed_at" not in ledger_df.columns:
        ledger_df["entry_observed_at"] = ledger_df.get(
            "entry_time",
            pd.Series(index=ledger_df.index, dtype=object),
        )
        ledger_schema_changed = True
    else:
        missing_observed = ledger_df["entry_observed_at"].isna()
        if missing_observed.any() and "entry_time" in ledger_df.columns:
            ledger_df.loc[missing_observed, "entry_observed_at"] = ledger_df.loc[
                missing_observed, "entry_time"
            ]
            ledger_schema_changed = True

    existing_keys = set()
    if not ledger_df.empty and set(key_cols).issubset(ledger_df.columns):
        for row in ledger_df[key_cols].itertuples(index=False):
            existing_keys.add((str(row.market), str(row.side), str(row.entry_time), int(row.horizon_h)))

    latest_ts = pd.Timestamp(latest_ts)
    matured_rows = []

    for fname in sorted(os.listdir(out_dir)):
        if not (fname.startswith("recommendations_") and fname.endswith(".csv")):
            continue
        fpath = os.path.join(out_dir, fname)
        try:
            rec_df = pd.read_csv(fpath)
        except Exception:
            continue
        required = {"market", "side", "entry_price", "entry_time", "horizon_h"}
        if not required.issubset(rec_df.columns):
            continue

        for row in rec_df.to_dict(orient="records"):
            market = str(row.get("market", ""))
            side = str(row.get("side", ""))
            entry_price = pd.to_numeric(row.get("entry_price"), errors="coerce")
            horizon_h = pd.to_numeric(row.get("horizon_h"), errors="coerce")
            entry_time = pd.to_datetime(row.get("entry_time"), utc=True, errors="coerce")
            if not market or pd.isna(entry_price) or entry_price <= 0 or pd.isna(horizon_h) or pd.isna(entry_time):
                continue

            horizon_h = int(horizon_h)
            key = (market, side, entry_time.isoformat(), horizon_h)
            if key in existing_keys:
                continue

            entry_observed_at = _entry_observed_timestamp(row, entry_time)
            target_exit_ts = entry_observed_at + pd.Timedelta(hours=horizon_h)
            if target_exit_ts > latest_ts:
                continue
            if market not in closes.columns:
                continue

            eligible = closes.index[closes.index <= target_exit_ts]
            if len(eligible) == 0:
                continue
            exit_ts = eligible[-1]
            exit_price = closes.at[exit_ts, market]
            if pd.isna(exit_price):
                continue

            side_lower = side.lower()
            if "short" in side_lower:
                gross_return = (entry_price - exit_price) / entry_price
            else:
                gross_return = (exit_price - entry_price) / entry_price
            cost = _cost_assumptions_for_side(side)
            net_return = gross_return - cost["estimated_cost_return"]

            matured_rows.append({
                "market": market,
                "side": side,
                "actionable": _as_bool(row.get("actionable", False)),
                "entry_time": entry_time.isoformat(),
                "entry_observed_at": entry_observed_at.isoformat(),
                "exit_time_target": target_exit_ts.isoformat(),
                "exit_time_actual": pd.Timestamp(exit_ts).isoformat(),
                "horizon_h": horizon_h,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "gross_return": float(gross_return),
                "fee_bps": cost["fee_bps"],
                "slippage_bps": cost["slippage_bps"],
                "short_extra_cost_bps": cost["short_extra_cost_bps"],
                "estimated_cost_return": cost["estimated_cost_return"],
                "net_return": float(net_return),
                "realized_return": float(gross_return),
                "refresh_reason": row.get("refresh_reason", ""),
                "source_file": fname,
                "model_sha256": row.get("model_sha256"),
                "calibration_generated_at": row.get("calibration_generated_at"),
            })
            existing_keys.add(key)

    if not matured_rows:
        if not os.path.exists(ledger_path):
            pd.DataFrame(columns=ledger_cols).to_csv(ledger_path, index=False)
            logger.info("Ledger initialized with no matured positions yet -> %s", ledger_path)
            return
        if ledger_backfilled or ledger_schema_changed:
            ledger_df.to_csv(ledger_path, index=False)
            logger.info("Ledger schema/backfill updated -> %s", ledger_path)
            return
        logger.info("Ledger unchanged: no newly matured positions")
        return

    append_df = pd.DataFrame(matured_rows)
    ledger_df = pd.concat([ledger_df, append_df], ignore_index=True) if not ledger_df.empty else append_df
    ledger_df = ledger_df.sort_values(["entry_time", "market", "side"]).reset_index(drop=True)
    ledger_df.to_csv(ledger_path, index=False)

    long_mask = ~append_df["side"].astype(str).str.lower().str.contains("short", na=False)
    short_mask = append_df["side"].astype(str).str.lower().str.contains("short", na=False)
    logger.info(
        "Ledger updated: +%s matured positions (long/watch=%s, short=%s) -> %s",
        len(append_df),
        int(long_mask.sum()),
        int(short_mask.sum()),
        ledger_path,
    )


@model_release_guard()
def _run(args):
    logger.info("=== fetch_and_rank START ===")

    # 1. Collect data if requested
    if args.collect:
        logger.info("Collecting fresh data...")
        from data.collector import run_all
        run_all(days=120)

    # 2. Load and compute
    from data.features import (
        load_and_pivot,
        load_binance_pivot,
        compute_factors,
        compute_long_factors,
        compute_unified_factors,
        compute_btc_regime,
        crosssection_zscore,
        build_top_liquidity_universe_index,
        filter_long_frame_by_universe,
        LONG_CALENDAR_COLS,
        UNIFIED_CALENDAR_COLS,
    )

    # Keep enough pre-warmup history for the 30d BTC regime gate. Computing
    # the gate after dropping the 480h feature warmup made btc_ret_30d NaN.
    live_history_days = 45
    closes, opens, highs, lows, volumes = load_and_pivot(days=live_history_days)
    regime_closes = closes
    warmup = config.Data.MIN_ROWS_PER_COIN
    closes  = closes.iloc[warmup:]
    opens   = opens.iloc[warmup:]
    highs   = highs.iloc[warmup:]
    lows    = lows.iloc[warmup:]
    volumes = volumes.iloc[warmup:]

    binance_closes = load_binance_pivot(closes.columns.tolist(), days=live_history_days)
    if not binance_closes.empty:
        binance_closes = binance_closes.iloc[warmup:]
    # F1 unified (2026-04-25): single factor pipeline for both 6h and 12h models
    factor_df = compute_unified_factors(closes, opens, highs, lows, volumes, binance_closes=binance_closes)
    zcols = [c for c in factor_df.columns if c not in UNIFIED_CALENDAR_COLS]
    factor_df = crosssection_zscore(factor_df, cols=zcols)
    top_n = getattr(config.Data, "LIQUIDITY_TOP_N", 0)
    selected_index, coverage = build_top_liquidity_universe_index(closes, volumes, top_n=top_n)
    factor_df = filter_long_frame_by_universe(factor_df, selected_index)
    logger.info(
        "Active universe: top %s by %sh traded value (avg selected=%.1f, min=%s, max=%s)",
        top_n,
        getattr(config.Data, "LIQUIDITY_LOOKBACK_HOURS", 24),
        coverage["avg_selected"],
        coverage["min_selected"],
        coverage["max_selected"],
    )

    # 3. Persist feature-health snapshot (NaN% per column at latest timestamp).
    # Used by health_snapshot.py section F. Cheap (≤10ms) and gives operators
    # a record of structural NaN drift (e.g. Binance match rate dropping).
    try:
        _write_feature_health(factor_df)
    except Exception as e:
        logger.warning(f"Feature-health snapshot failed (non-fatal): {e}")

    # 3. Use latest timestamp
    latest_ts = factor_df.index.get_level_values("timestamp").max()
    _update_realized_ledger(closes, latest_ts)
    _update_long_shadow_ledger()
    short_horizon_h = config.Data.PREDICT_HORIZON
    long_horizon_h = getattr(config.LongModel, "PREDICT_HORIZON", short_horizon_h)
    long_anchor_hour = getattr(config.LongModel, "REBALANCE_ANCHOR_HOUR_UTC", 11)
    long_rebalance_due = _is_rebalance_slot(latest_ts, long_horizon_h, long_anchor_hour)
    next_short_rebalance_ts = _next_rebalance_ts(latest_ts, short_horizon_h)
    next_long_rebalance_ts = _next_rebalance_ts(latest_ts, long_horizon_h, long_anchor_hour)
    latest_factors_raw = factor_df.xs(latest_ts, level="timestamp")

    # Run pre-flight gates BEFORE imputation so NaN rates are honest
    from utils.preflight import run_preflight
    _prev_df_early = _load_previous_recommendations(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "latest.csv")
    )
    _prev_universe = (
        set(_prev_df_early["market"])
        if not _prev_df_early.empty and "market" in _prev_df_early.columns
        else None
    )
    preflight = run_preflight(
        latest_ts=latest_ts,
        latest_factors=latest_factors_raw,
        current_universe=set(latest_factors_raw.index),
        previous_universe=_prev_universe,
    )
    if preflight["batch_suppression"]:
        logger.warning(f"Pre-flight FAIL: {preflight['batch_suppression']} — all picks → watch-only")
    if preflight["bad_features"]:
        logger.warning(f"Pre-flight bad features (NaN>20%): {preflight['bad_features']}")
    for w in preflight["warnings"]:
        logger.info(f"Pre-flight warn: {w}")

    # Binance factors may be NaN if binance data lags — fill with 0 (neutral signal)
    latest_factors = latest_factors_raw.fillna(0)
    # Drop coins with all zeros (truly no data)
    latest_factors = latest_factors[latest_factors.any(axis=1)]

    # Composite score (short model): XGBoost model if available, else equal-weight fallback
    score = _compute_score(latest_factors, args)
    if score.attrs.get("is_fallback"):
        preflight["batch_suppression"] = "model_unavailable"
    score_sorted = score.sort_values(ascending=False)

    score_witness = None
    if not args.dry_run and score.attrs.get("model_sha256"):
        try:
            from utils.score_evidence import record_score_evidence
            score_witness = record_score_evidence("signal", latest_ts, latest_factors_raw.loc[score.index],
                                                  score, score.attrs["model_sha256"])
        except (Exception, SystemExit) as exc:
            logger.warning("Signal diagnostic witness failed; scoring unchanged: %s", exc)

    # --- Per-coin full-universe prediction (direction + expected% + confidence) ---
    # Uses sigma-bucket calibration (output/calibration_sigma.json). Saved alongside
    # the basket CSV so the user can see predictions for all 100 coins, not just picks.
    try:
        from utils.magnitude import predict_batch as _predict_batch
        if score.attrs.get("is_fallback"):
            raise ValueError("Equal-weight fallback has no production calibration")
        short_predictions_df = _predict_batch("short_6h", score, closes=closes, horizon_h=short_horizon_h)
        short_predictions_df["model_sha256"] = score.attrs.get("model_sha256")
        short_predictions_df["timestamp"] = pd.Timestamp(latest_ts).isoformat()
        # Apply pre-flight suppression
        short_predictions_df["suppression"] = preflight["batch_suppression"] or ""
        short_predictions_df["actionable"] = (
            (short_predictions_df["sigma"] >= 1.0) & (preflight["batch_suppression"] is None)
        )
        logger.info(
            "Full-universe SHORT predictions: 🔥%s ✅%s ▫%s (n=%s)",
            int((short_predictions_df["sigma"] >= 2.0).sum()),
            int(((short_predictions_df["sigma"] >= 1.0) & (short_predictions_df["sigma"] < 2.0)).sum()),
            int(((short_predictions_df["sigma"] >= 0.5) & (short_predictions_df["sigma"] < 1.0)).sum()),
            len(short_predictions_df),
        )
    except Exception as e:
        logger.warning(f"Per-coin prediction (short) failed: {e}")
        short_predictions_df = None

    # --- Long model scoring (independent, regime-gated, execution-mode-gated) ---
    long_model_score_sorted = None
    long_model_scored = False
    long_regime_eligible = False
    long_model_active = False
    long_predictions_df = None
    long_watch_source_df = None
    long_execution_mode = getattr(config.LongModel, "EXECUTION_MODE", "disabled")
    try:
        long_model_path = getattr(config.LongModel, "MODEL_PATH", "models/xsec_long.pkl")
        abs_long_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), long_model_path)

        if long_execution_mode == "disabled":
            logger.info("Long model DISABLED (LongModel.EXECUTION_MODE='disabled')")
        elif os.path.exists(abs_long_model_path) and not args.no_model:
            # F1 unified (2026-04-25): 12h model uses the SAME unified factor pipeline
            # as the 6h model — only target horizon differs. Just reuse latest_factors
            # (already z-scored above). Rank correlation between 6h/12h predictions is 0.91.
            btc_regime = compute_btc_regime(regime_closes)
            latest_regime = btc_regime.loc[latest_ts] if latest_ts in btc_regime.index else None
            latest_long_factors = latest_factors  # same features as short model

            from models.xgb_ranker import XSecRanker
            long_model = XSecRanker.load(abs_long_model_path)
            long_scores = long_model.predict(latest_long_factors)
            long_model_score_sorted = pd.Series(long_scores, index=latest_long_factors.index).sort_values(ascending=False)
            long_model_scored = True
            long_regime_eligible = _long_regime_allows(
                latest_regime,
                getattr(config.LongModel, "BTC_7D_RETURN_GATE", 0.0),
                getattr(config.LongModel, "BTC_30D_RETURN_FLOOR", -0.10),
            )
            long_model_active = long_regime_eligible
            regime_str = "bull" if latest_regime is not None and latest_regime.get("regime_bull") == 1.0 else "neutral/bear"
            btc7 = (latest_regime['btc_ret_7d'] if latest_regime is not None else None)
            btc30 = (latest_regime['btc_ret_30d'] if latest_regime is not None else None)
            logger.info(
                f"Long model {'ACTIVE' if long_model_active else 'SCORED_ONLY'} "
                f"(regime={regime_str}, eligible={long_regime_eligible}, "
                f"btc_7d={btc7}, btc_30d={btc30}): "
                f"{len(long_model_score_sorted)} coins scored"
            )
            # Per-coin full-universe predictions (long model, 12h)
            try:
                from utils.magnitude import predict_batch as _predict_batch_long
                long_predictions_df = _predict_batch_long(
                    "long_12h",
                    long_model_score_sorted.sort_index(),
                    closes=closes,
                    horizon_h=long_horizon_h,
                )
                long_predictions_df["timestamp"] = pd.Timestamp(latest_ts).isoformat()
                long_predictions_df["model_sha256"] = artifact_sha256(Path(abs_long_model_path))
                long_predictions_df["suppression"] = preflight["batch_suppression"] or ""
                long_predictions_df["actionable"] = (
                    (long_predictions_df["sigma"] >= 1.0)
                    & (preflight["batch_suppression"] is None)
                    & long_model_active
                )
            except Exception as _pe:
                logger.warning(f"Per-coin prediction (long) failed: {_pe}")
                long_predictions_df = None
    except Exception as e:
        logger.warning(f"Long model scoring failed: {e}")

    # --- Enrich predictions with consensus + per-coin reliability ---
    try:
        from utils.enrich import enrich_predictions
        short_predictions_df, long_predictions_df = enrich_predictions(
            short_predictions_df, long_predictions_df
        )
        if short_predictions_df is not None:
            cons_n = int(short_predictions_df.get("consensus", pd.Series([False]*len(short_predictions_df))).sum())
            trust_n = int((short_predictions_df.get("trust_tag", pd.Series([""]*len(short_predictions_df))) == "⭐").sum())
            warn_n  = int((short_predictions_df.get("trust_tag", pd.Series([""]*len(short_predictions_df))) == "⚠").sum())
            logger.info(f"Enrichment SHORT: consensus={cons_n}  ⭐trusted={trust_n}  ⚠untrusted={warn_n}")
    except Exception as _ee:
        logger.warning(f"Enrichment failed: {_ee}")

    # Preserve the intrinsic data/regime quality decision before the execution
    # KILL demotes every LONG prediction to non-actionable.  This copy is used
    # only by the strict shadow WATCH alert selector below.
    if long_predictions_df is not None:
        long_watch_source_df = long_predictions_df.copy(deep=True)

    execution_mode = getattr(config.Portfolio, "LIVE_EXECUTION_MODE", "short_only")
    watch_long_n = getattr(config.Portfolio, "LIVE_WATCH_LONG_N", config.Portfolio.LONG_N)
    exec_short_n = getattr(config.Portfolio, "LIVE_EXEC_SHORT_N", config.Portfolio.SHORT_N)
    require_bitget = getattr(config.Portfolio, "LIVE_REQUIRE_BITGET_TRADABLE", False)

    # --- IC gate (CLAUDE.md §7 auto-enforcement) ---
    # Evaluate gate state per-side from ic_history*.json. If a side is in
    # FREEZE/LIQUIDATE, force watch-only / block. Always persist state for
    # health_snapshot to read.
    from utils.ic_gate import evaluate_all as _evaluate_gates
    gates = _evaluate_gates(persist=True)
    short_gate = gates["short"]
    long_gate = gates["long"]
    logger.info(
        f"IC gate: short={short_gate.status} (last={short_gate.last_ic}); "
        f"long={long_gate.status} (last={long_gate.last_ic})"
    )
    if short_gate.watch_only and execution_mode == "short_only":
        logger.warning(f"IC gate SHORT={short_gate.status}: forcing short to watch-only ({short_gate.reason})")
    if long_gate.watch_only:
        logger.warning(f"IC gate LONG={long_gate.status}: long forced to watch-only ({long_gate.reason})")
    short_blocked = short_gate.block
    long_blocked = long_gate.block
    prev_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "latest.csv")
    prev_df = _load_previous_recommendations(prev_path)

    contract_map = {}
    tradable_score_sorted = score_sorted
    if require_bitget:
        contract_map = load_bitget_usdt_perp_map()
        tradable_markets = filter_markets_to_bitget(score_sorted.index.tolist(), contract_map)
        tradable_score_sorted = score_sorted.loc[tradable_markets]
        logger.info(
            "Tradable Bitget universe: %s/%s scored coins",
            len(tradable_score_sorted),
            len(score_sorted),
        )

    if execution_mode == "short_only":
        long_n = watch_long_n
        short_n = _short_pick_limit(exec_short_n)
        long_side_label = "WATCH_LONG"
    else:
        long_n = config.Portfolio.LONG_N
        short_n = _short_pick_limit(exec_short_n)
        long_side_label = "LONG"

        # Calendar filter: suppress LONG on Fri/Sat/Sun KST
        now_kst = datetime.now(timezone.utc) + timedelta(hours=9)
        dow_kst = now_kst.weekday()  # 0=Mon … 6=Sun
        is_bullish_day = dow_kst < 4  # Mon/Tue/Wed/Thu OK, Fri/Sat/Sun suppress longs
        if not is_bullish_day:
            logger.warning(f"Calendar filter: {now_kst.strftime('%A')} KST — suppressing LONG signals (Fri/weekend)")
            long_n = 0

    if not long_model_scored:
        logger.warning("LONG model was not scored: failing closed with 0 long picks")
        long_n = 0
    elif not long_regime_eligible:
        logger.warning("LONG regime gate is OFF: emitting 0 long picks while retaining shadow scores")
        long_n = 0

    # --- Apply IC gate overrides (§7 auto-enforcement) ---
    # Block takes precedence over watch-only. Watch-only demotes 'actionable'
    # and re-labels the long side to WATCH_LONG.
    if long_blocked:
        block_reason = long_gate.policy_reason or long_gate.reason
        logger.warning(f"LONG execution blocked: emitting 0 long picks ({block_reason})")
        long_n = 0
    elif long_gate.watch_only and long_side_label == "LONG":
        logger.warning("IC gate LONG=FREEZE: demoting LONG → WATCH_LONG")
        long_side_label = "WATCH_LONG"
    if short_blocked:
        logger.warning("IC gate SHORT=LIQUIDATE: emitting 0 short picks")
        short_n = 0
    short_watch_only = short_gate.watch_only and not short_blocked
    short_watch_reason = f"ic_{short_gate.status.lower()}" if short_watch_only else ""
    if preflight["batch_suppression"]:
        short_watch_only = True
        short_watch_reason = preflight["batch_suppression"]
    if long_predictions_df is not None and (
        long_blocked
        or long_gate.watch_only
        or long_execution_mode != "long_only"
        or not long_model_active
    ):
        long_predictions_df["actionable"] = False

    # --- Rebalancing buffer: reduce turnover by keeping existing positions ---
    rebal_buffer = config.Portfolio.REBAL_BUFFER
    prev_longs = set()
    prev_shorts = set()
    prev_long_rows = pd.DataFrame()
    if not prev_df.empty and "side" in prev_df.columns and "market" in prev_df.columns:
        side_col = prev_df["side"].astype(str).str.lower()
        prev_long_rows = prev_df[side_col.str.contains("long", na=False)].copy()
        prev_longs = set(prev_long_rows["market"])
        prev_shorts = set(prev_df[side_col.str.contains("short", na=False)]["market"])

    selection_scores = tradable_score_sorted if require_bitget else score_sorted

    if prev_longs or prev_shorts:
        longs, shorts = select_positions_with_buffer(
            selection_scores,
            long_n=long_n,
            short_n=short_n,
            prev_longs=prev_longs,
            prev_shorts=prev_shorts,
            rebal_buffer=rebal_buffer,
        )
        turnover_l = len(set(longs.index) - prev_longs) / max(len(longs), 1)
        turnover_s = len(set(shorts.index) - prev_shorts) / max(len(shorts), 1)
        logger.info(f"Rebal buffer: turnover L={turnover_l:.0%} S={turnover_s:.0%} (buffer={rebal_buffer})")
    else:
        longs = selection_scores.head(long_n) if long_n > 0 else pd.Series(dtype=float)
        shorts = selection_scores.tail(short_n)

    # --- Override long picks with long-specialist model if active ---
    carry_longs = False
    long_refresh_reason = "watch_seed"
    if long_model_active and long_model_score_sorted is not None:
        long_model_n = _effective_long_pick_limit(
            getattr(config.LongModel, "LONG_N", 5), long_n, long_blocked,
        )
        if require_bitget:
            tradable_long = filter_markets_to_bitget(long_model_score_sorted.index.tolist(), contract_map)
            long_model_tradable = long_model_score_sorted.loc[tradable_long]
        else:
            long_model_tradable = long_model_score_sorted

        if long_model_n == 0:
            longs = pd.Series(dtype=float)
            carry_longs = False
            long_refresh_reason = "policy_blocked"
            logger.warning("Long model scored for monitoring, but execution policy emitted 0 picks")
        elif long_rebalance_due or prev_long_rows.empty:
            longs = long_model_tradable.head(long_model_n)
            carry_longs = False
            long_refresh_reason = "refresh_12h"
            logger.info(
                "Long model rebalance due: refreshing %s picks (h=%sh, anchor=%s UTC)",
                len(longs), long_horizon_h, long_anchor_hour,
            )
        else:
            carry_longs = True
            long_refresh_reason = "carry_12h"
            prev_long_rows = prev_long_rows.drop_duplicates(subset=["market"], keep="last")
            carry_markets = [m for m in prev_long_rows["market"].tolist() if m in long_model_tradable.index]
            if carry_markets:
                longs = prev_long_rows.set_index("market").loc[carry_markets, "score"]
                logger.info(
                    "Long model HOLD: carrying %s previous picks until next %sh rebalance",
                    len(longs), long_horizon_h,
                )
            else:
                longs = long_model_tradable.head(long_model_n)
                carry_longs = False
                long_refresh_reason = "refresh_12h"
                logger.info("Long model carry unavailable: seeding fresh picks")
    elif long_execution_mode == "disabled":
        long_refresh_reason = "disabled"
    elif not long_model_active:
        long_refresh_reason = "regime_off"

    # Only upgrade to actionable LONG if execution mode permits
    if (
        long_model_active
        and long_execution_mode == "long_only"
        and not long_gate.watch_only
        and not long_blocked
    ):
        long_side_label = "LONG"
        if carry_longs:
            logger.info("Long model EXECUTE HOLD: carrying existing LONG basket")
        else:
            logger.info(f"Long model EXECUTE: {len(longs)} positions (EXECUTION_MODE=long_only)")
    elif long_model_active:
        long_side_label = "WATCH_LONG"
        mode_note = "HOLD" if carry_longs else "WATCH"
        logger.info(f"Long model {mode_note}: {len(longs)} picks (EXECUTION_MODE={long_execution_mode}, not yet approved)")

    # Guard: need minimum coins
    if len(longs) < 5 and long_n > 0:
        logger.warning(f"Too few long candidates: {len(longs)} → watch-only for longs")
    if len(shorts) < 5:
        logger.warning(f"Too few coins for ranking: shorts={len(shorts)} → watch-only")

    logger.info(f"As of {latest_ts}")
    logger.info(f"{long_side_label} ({len(longs)}): {longs.index.tolist()}")
    short_label = "WATCH" if short_watch_only or len(shorts) < 5 else "CANDIDATES"
    logger.info(f"SHORT {short_label} ({len(shorts)}): {shorts.index.tolist()}")

    saved_recommendations_df = pd.DataFrame()
    long_watch_alerts_df = pd.DataFrame()
    if not args.dry_run:
        saved_recommendations_df = _save_recommendations(
            latest_ts,
            longs,
            shorts,
            short_predictions_df=short_predictions_df,
            long_predictions_df=long_predictions_df,
            long_side_label=long_side_label,
            short_side_label="SHORT",
            contract_map=contract_map,
            long_model_active=long_model_active,
            long_horizon_h=long_horizon_h if long_model_active else short_horizon_h,
            short_horizon_h=short_horizon_h,
            carry_longs=carry_longs,
            prev_long_rows=prev_long_rows,
            long_refresh_reason=long_refresh_reason,
            next_long_rebalance_ts=next_long_rebalance_ts,
            next_short_rebalance_ts=next_short_rebalance_ts,
            short_watch_only=short_watch_only,
            short_watch_reason=short_watch_reason,
        )
        if short_predictions_df is not None:
            short_predictions_df["actionable"] = False
            short_predictions_df["selected"] = False
            for row in saved_recommendations_df.to_dict(orient="records"):
                market = row["market"]
                if row["side"] == "SHORT" and market in short_predictions_df.index:
                    short_predictions_df.loc[market, "selected"] = True
                    short_predictions_df.loc[market, "actionable"] = bool(row["actionable"])
                    short_predictions_df.loc[market, "suppression"] = row["suppression"]

        # Preserve exact paper-entry observations for the preregistered LONG
        # challenger even while the execution KILL emits no positions.
        long_shadow_markets = []
        if long_model_score_sorted is not None and long_rebalance_due:
            shadow_scores = long_model_score_sorted
            if require_bitget:
                shadow_tradable = filter_markets_to_bitget(shadow_scores.index.tolist(), contract_map)
                shadow_scores = shadow_scores.loc[shadow_tradable]
            long_shadow_markets = shadow_scores.head(
                int(getattr(config.LongModel, "LONG_N", 5))
            ).index.tolist()

        # Save full-universe per-coin predictions (both horizons) for user to browse
        _save_predictions_full(
            latest_ts,
            short_predictions_df,
            long_predictions_df,
            long_shadow_markets=long_shadow_markets,
            long_regime_eligible=long_regime_eligible,
        )

        long_watch_alerts_df = _select_strong_long_watch_alerts(
            long_watch_source_df,
            long_shadow_markets,
            long_gate,
            regime_eligible=long_regime_eligible,
            rebalance_due=long_rebalance_due,
            bitget_tradable_enforced=bool(require_bitget and contract_map),
        )
        logger.info(
            "LONG strong shadow WATCH alerts: %s (execution rows remain 0)",
            len(long_watch_alerts_df),
        )

    # Report-only candidates survive execution blocks without creating trades.
    if not args.dry_run:
        try:
            from utils.operator_report import build_report, publish_report

            # Realized 30d summary so the message can show recent paper performance
            # alongside the new signals — manual trader needs that context.
            realized_summary = None
            try:
                from utils.dashboard_export import build_summary_payload
                realized_summary = (build_summary_payload() or {}).get("realized_summary")
            except Exception as e:
                logger.debug(f"Could not load realized summary for telegram tail: {e}")

            telegram_df = saved_recommendations_df
            if not long_watch_alerts_df.empty:
                telegram_df = pd.concat(
                    [saved_recommendations_df, long_watch_alerts_df],
                    ignore_index=True,
                    sort=False,
                )
            observed_scores = selection_scores.dropna().sort_values().head(5)
            candidates = pd.DataFrame({"market": observed_scores.index, "score": observed_scores.values})
            report = build_report(
                telegram_df, candidates, asof=latest_ts,
                reason=(f"SHORT IC {short_gate.status}: {short_gate.reason}"
                        if short_blocked else short_watch_reason),
            )
            report["score_evidence_sha256"] = score_witness.stem if score_witness else None
            report["audit_context"] = {
                "short_gate": short_gate.status,
                "tradable_symbols": {m: symbol for m in score.index
                                     if (symbol := market_to_bitget_symbol(m, contract_map or {}))},
            }
            sent = publish_report(report, send=not args.no_telegram, realized_summary=realized_summary)
            if sent:
                logger.info("Telegram coin proposal report acknowledged")
        except Exception as e:
            logger.warning(f"Operator report failed: {e}")

        # Shadow recording follows Telegram; it cannot delay or change live picks.
        try:
            from utils.regime_observer import load_btc_history, record_observation, refresh_observation
            observed_markets = tradable_score_sorted.index
            result = record_observation(
                latest_ts, load_btc_history(latest_ts), score.reindex(observed_markets),
                latest_factors_raw["reversal_4h"].reindex(observed_markets),
                score.attrs.get("model_sha256"),
            )
            refresh_observation()
            logger.info("Regime observation: %s", result)
        except Exception:
            logger.exception("Regime observation failed; live recommendations unchanged")

        try:
            from utils.prospective import record_snapshot

            live_columns = [c for c in ("market", "actionable", "suppression", "model_sha256")
                            if c in saved_recommendations_df.columns]
            live_rows = json.loads(saved_recommendations_df[live_columns].to_json(orient="records"))
            result = record_snapshot(
                latest_ts, latest_factors, latest_factors_raw,
                tradable_score_sorted.index.tolist(), live_rows=live_rows,
                context={"short_gate": short_gate.status,
                         "preflight_suppression": preflight["batch_suppression"],
                         "live_model_sha256": score.attrs.get("model_sha256")},
            )
            logger.info("Prospective snapshot: %s", result)
        except Exception:
            logger.exception("Prospective snapshot failed; live recommendations unchanged")

    logger.info("=== fetch_and_rank END ===")


def _save_predictions_full(
    ts,
    short_df,
    long_df,
    long_shadow_markets=None,
    long_regime_eligible=False,
):
    """Save full-universe per-coin predictions (both horizons) to output/predictions_*.csv.

    Columns: timestamp, market, horizon_h, score, sigma, tag, label, direction,
             expected_pct, hit_rate, typical_move_pct, coin_vol_pct
    """
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    ts_str = ts.strftime("%Y%m%dT%H%M") if hasattr(ts, "strftime") else str(ts)[:16].replace(" ", "T")

    frames = []
    if short_df is not None and len(short_df):
        d = short_df.reset_index().copy()
        d["horizon_h"] = 6
        d["side_hint"] = d["direction"].map({1: "UP", -1: "DOWN", 0: "FLAT"})
        frames.append(d)
    if long_df is not None and len(long_df):
        d = long_df.reset_index().copy()
        d["horizon_h"] = 12
        d["side_hint"] = d["direction"].map({1: "UP", -1: "DOWN", 0: "FLAT"})
        d["regime_eligible"] = bool(long_regime_eligible)
        d["shadow_selected"] = d["market"].isin(set(long_shadow_markets or []))
        d["shadow_entry_price"] = np.nan
        d["entry_observed_at"] = None
        if d["shadow_selected"].any():
            from data.collector import get_current_price
            import time as _time

            for market in d.loc[d["shadow_selected"], "market"]:
                observed_price = get_current_price(market)
                observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                mask = d["market"] == market
                d.loc[mask, "shadow_entry_price"] = observed_price
                d.loc[mask, "entry_observed_at"] = observed_at
                _time.sleep(float(getattr(config.Data, "COLLECTOR_MARKET_SLEEP_SEC", 0.15)))
        frames.append(d)
    if not frames:
        logger.warning("No predictions to save")
        return

    df = pd.concat(frames, ignore_index=True)
    cols = ["timestamp", "market", "horizon_h", "side_hint", "direction",
            "model_sha256", "calibration_generated_at", "selected",
            "direction_prob", "sigma", "tag", "label",
            "expected_pct", "ci_95_low", "ci_95_high",
            "hit_rate", "typical_move_pct", "coin_vol_pct",
            "position_size_pct",
            "consensus", "consensus_tag", "consensus_note",
            "trust_tag", "trust_note", "trust_hit_rate",
            "suppression", "actionable", "score",
            "regime_eligible", "shadow_selected", "shadow_entry_price",
            "entry_observed_at"]
    df = df[[c for c in cols if c in df.columns]]

    path = os.path.join(out_dir, f"predictions_{ts_str}.csv")
    df.to_csv(path, index=False)
    # latest-predictions symlink
    latest = os.path.join(out_dir, "latest_predictions.csv")
    try:
        if os.path.lexists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(path), latest)
    except Exception:
        pass
    logger.info(
        "Saved full-universe predictions: %s (n=%s, 🔥%s ✅%s)",
        path, len(df),
        int((df["sigma"] >= 2.0).sum()),
        int(((df["sigma"] >= 1.0) & (df["sigma"] < 2.0)).sum()),
    )


def _save_recommendations(
    ts,
    longs,
    shorts,
    short_predictions_df=None,
    long_predictions_df=None,
    long_side_label="LONG",
    short_side_label="SHORT",
    contract_map=None,
    long_model_active=False,
    long_horizon_h=6,
    short_horizon_h=6,
    carry_longs=False,
    prev_long_rows=None,
    long_refresh_reason="watch_seed",
    next_long_rebalance_ts=None,
    next_short_rebalance_ts=None,
    short_watch_only=False,
    short_watch_reason="",
):
    shorts = shorts.head(_short_pick_limit(len(shorts)))
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    ts_str = ts.strftime("%Y%m%dT%H%M") if hasattr(ts, "strftime") else str(ts)[:16].replace(" ", "T")
    csv_path = os.path.join(out_dir, f"recommendations_{ts_str}.csv")

    # Fetch entry prices for LONG/SHORT positions
    from data.collector import get_current_price
    import time as _time

    rows = []
    prev_long_meta = {}
    if prev_long_rows is not None and not prev_long_rows.empty and "market" in prev_long_rows.columns:
        prev_long_meta = prev_long_rows.drop_duplicates(subset=["market"], keep="last").set_index("market").to_dict("index")

    for market, score in longs.items():
        side = long_side_label
        cost = _cost_assumptions_for_side(side)
        pred_meta = _prediction_meta_for(market, side, short_predictions_df, long_predictions_df)
        prev_meta = prev_long_meta.get(market, {})
        if carry_longs and prev_meta:
            entry_price = prev_meta.get("entry_price")
            entry_time = prev_meta.get("entry_time", pd.Timestamp(ts).isoformat())
            entry_observed_at = prev_meta.get("entry_observed_at") or entry_time
            bitget_symbol = prev_meta.get("bitget_symbol") or market_to_bitget_symbol(market, contract_map or {})
        else:
            entry_price = get_current_price(market)
            entry_observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            entry_time = pd.Timestamp(ts).isoformat()
            bitget_symbol = market_to_bitget_symbol(market, contract_map or {})
            _time.sleep(0.1)
        prediction_actionable = False
        if long_predictions_df is not None and market in long_predictions_df.index:
            prediction_actionable = _as_bool(long_predictions_df.loc[market].get("actionable", False))
        rows.append({
            "market": market,
            "score": round(float(score), 4),
            "side": side,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "entry_observed_at": entry_observed_at,
            "horizon_h": long_horizon_h,
            "refresh_reason": long_refresh_reason,
            "next_rebalance_at": pd.Timestamp(next_long_rebalance_ts).isoformat() if next_long_rebalance_ts is not None else None,
            "bitget_symbol": bitget_symbol,
            "actionable": _long_recommendation_is_actionable(
                side,
                long_model_active,
                prediction_actionable,
            ),
            **cost,
            **pred_meta,
        })

    for market, score in shorts.items():
        cost = _cost_assumptions_for_side(short_side_label)
        pred_meta = _prediction_meta_for(market, short_side_label, short_predictions_df, long_predictions_df)
        reason = _short_suppression_reason(
            score, pred_meta, len(shorts),
            short_watch_reason or ("watch_only" if short_watch_only else ""),
        )
        pred_meta["suppression"] = reason
        entry_price = get_current_price(market)
        entry_observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _time.sleep(0.1)
        rows.append({
            "market": market,
            "score": round(float(score), 4),
            "side": short_side_label,
            "entry_price": entry_price,
            "entry_time": pd.Timestamp(ts).isoformat(),
            "entry_observed_at": entry_observed_at,
            "horizon_h": short_horizon_h,
            "refresh_reason": "refresh_6h",
            "next_rebalance_at": pd.Timestamp(next_short_rebalance_ts).isoformat() if next_short_rebalance_ts is not None else None,
            "bitget_symbol": market_to_bitget_symbol(market, contract_map or {}),
            "actionable": not bool(reason),
            **cost,
            **pred_meta,
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    # Latest symlink
    latest_path = os.path.join(out_dir, "latest.csv")
    try:
        if os.path.lexists(latest_path):
            os.remove(latest_path)
        os.symlink(os.path.basename(csv_path), latest_path)
    except Exception as e:
        logger.warning(f"Could not update latest symlink: {e}")

    return df


if __name__ == "__main__":
    main()
