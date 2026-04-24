"""
XGBoost cross-sectional ranking model training pipeline.

Usage:
    cd /mnt/20t/main/gan_t/xsec_alpha
    python scripts/train.py [--days 90] [--holdout-ratio 0.2]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from scipy.stats import spearmanr

from config import Config
from data.dataset import build_dataset
from models.xgb_ranker import XSecRanker, RidgeRanker, LGBMRanker

# --------------------------------------------------------------------------- #
# Config defaults with fallback (Config.Model may not exist yet)
# --------------------------------------------------------------------------- #
_MODEL_PATH = getattr(
    getattr(Config, "Model", None), "MODEL_PATH", "models/xsec_xgb.pkl"
)
_PREDICT_HORIZON = Config.Data.PREDICT_HORIZON  # 12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost cross-sectional ranker.")
    p.add_argument("--days", type=int, default=90, help="Training window in days (default: 90)")
    p.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Fraction of data reserved as temporal holdout (default: 0.2)",
    )
    p.add_argument(
        "--model-type",
        type=str,
        choices=["ridge", "lgbm", "xgb", "ensemble"],
        default="ridge",
        help="Model type: ridge (default), lgbm (LightGBM), xgb (XGBoost)",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override prediction horizon in hours for training/evaluation experiments",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Output path for trained model pickle (default: auto from --side)",
    )
    p.add_argument(
        "--side",
        type=str,
        choices=["short", "long", "unified"],
        default="short",
        help="short / long (legacy, different factor sets) or unified (F1: same 10-feature set for both horizons)",
    )
    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=None,
        help="Ridge regularization alpha (default: 1.0 for short, 0.1 for long)",
    )
    p.add_argument(
        "--target",
        type=str,
        choices=["residual", "absolute"],
        default="residual",
        help="Training target: residual (coin−β·BTC, legacy) or absolute (coin_fwd, user-visible %)",
    )
    p.add_argument(
        "--regime-filter",
        type=str,
        choices=["default", "all"],
        default="default",
        help="Regime filter for long side: default (bull-only, legacy) or all (no filter — unifies baseline with short model)",
    )
    p.add_argument(
        "--no-macro-globals",
        action="store_true",
        help="LONG side: exclude global macro features (btc_mom_7d, btc_vol_7d, alt_season_index) — force pure cross-sectional",
    )
    return p.parse_args()


def print_feature_importance(importance: dict, top_n: int = 20) -> None:
    """Print feature importance sorted by gain (descending)."""
    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    header = f"{'Rank':<6}{'Feature':<35}{'Gain':>12}"
    print("\n" + "=" * len(header))
    print("Feature Importance (sorted by gain)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for rank, (feat, gain) in enumerate(sorted_feats[:top_n], start=1):
        print(f"{rank:<6}{feat:<35}{gain:>12.6f}")
    if len(sorted_feats) > top_n:
        print(f"  ... ({len(sorted_feats) - top_n} more features omitted)")
    print("=" * len(header))


def compute_insample_ic(model: XSecRanker, X_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Quick in-sample IC check (informational only — not a gate).
    Computes mean Spearman IC across all training samples treated as a single slice.
    Returns NaN on failure.
    """
    try:
        preds = model.predict(X_train)
        ic, _ = spearmanr(preds, y_train)
        return float(ic)
    except Exception as exc:
        print(f"[WARN] In-sample IC computation failed: {exc}")
        return float("nan")


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Build dataset
    # ------------------------------------------------------------------ #
    print(f"[train] Building dataset: days={args.days}, holdout_ratio={args.holdout_ratio}, side={args.side}")
    ds = build_dataset(days=args.days, holdout_ratio=args.holdout_ratio, horizon=args.horizon, side=args.side, target=args.target, regime_filter=args.regime_filter, include_macro_globals=not args.no_macro_globals)

    X_train: np.ndarray = ds["X_train"]
    y_train: np.ndarray = ds["y_train"]
    X_holdout: np.ndarray = ds["X_holdout"]
    y_holdout: np.ndarray = ds["y_holdout"]
    factor_cols: list = ds["factor_cols"]
    split_ts = ds["split_ts"]

    # ------------------------------------------------------------------ #
    # 2. Log shapes and split timestamp
    # ------------------------------------------------------------------ #
    print(f"[train] Train  shape: X={X_train.shape}, y={y_train.shape}")
    print(f"[train] Holdout shape: X={X_holdout.shape}, y={y_holdout.shape}")
    print(f"[train] Temporal split at: {split_ts}")
    print(f"[train] Predict horizon: {ds['horizon']}h")
    print(f"[train] Features ({len(factor_cols)}): {factor_cols}")

    # ------------------------------------------------------------------ #
    # 3. Sanity checks — no NaN allowed in training data
    # ------------------------------------------------------------------ #
    nan_X = int(np.isnan(np.array(X_train, dtype=float)).sum())
    nan_y = int(np.isnan(np.array(y_train, dtype=float)).sum())
    assert nan_X == 0, f"X_train contains {nan_X} NaN values — fix data pipeline first."
    assert nan_y == 0, f"y_train contains {nan_y} NaN values — fix data pipeline first."
    print("[train] Sanity check passed: no NaN in X_train or y_train.")

    # ------------------------------------------------------------------ #
    # 4. Train model (ridge / lgbm / xgb)
    # ------------------------------------------------------------------ #
    if args.model_type == "ensemble":
        from models.xgb_ranker import EnsembleRanker
        print("[train] Training EnsembleRanker (Ridge + LightGBM) ...")
        model = EnsembleRanker()
    elif args.model_type == "lgbm":
        print("[train] Training LGBMRanker (min_child_samples=50) ...")
        model = LGBMRanker(min_child_samples=50)
    elif args.model_type == "xgb":
        print("[train] Training XSecRanker (XGBoost) ...")
        model = XSecRanker()
    else:
        ridge_alpha = args.ridge_alpha if args.ridge_alpha is not None else (0.1 if args.side == "long" else 1.0)
        print(f"[train] Training RidgeRanker (alpha={ridge_alpha}) ...")
        model = RidgeRanker(alpha=ridge_alpha)
    model.fit(X_train, y_train)
    print(f"[train] Training complete ({type(model).__name__}).")

    # ------------------------------------------------------------------ #
    # 5. Save model
    # ------------------------------------------------------------------ #
    if args.model_path is not None:
        model_path = args.model_path
    elif args.side == "long":
        model_path = getattr(
            getattr(Config, "LongModel", None), "MODEL_PATH", "models/xsec_long.pkl"
        )
    else:
        model_path = _MODEL_PATH
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    model.save(model_path)

    # ------------------------------------------------------------------ #
    # 6. Feature importance / coefficient table
    # ------------------------------------------------------------------ #
    if hasattr(model, "feature_importance_") and model.feature_importance_:
        imp = model.feature_importance_
        if isinstance(model, RidgeRanker):
            label = "Coefficient (Ridge)"
        elif isinstance(model, LGBMRanker):
            label = "Importance (LightGBM)"
        else:
            label = "Gain (XGBoost)"
        print(f"\n{'='*53}")
        print(f"Feature Importance — {label}")
        print(f"{'='*53}")
        print(f"{'Rank':<6}{'Feature':<40}{'Value':>10}")
        print(f"{'-'*53}")
        for i, (feat, val) in enumerate(imp.items(), 1):
            print(f"{i:<6}{feat:<40}{val:>10.6f}")
        print(f"{'='*53}")
    else:
        print("[WARN] feature_importance_ not available on trained model.")

    # ------------------------------------------------------------------ #
    # 7. Quick in-sample IC (informational)
    # ------------------------------------------------------------------ #
    ic_insample = compute_insample_ic(model, X_train, y_train)
    print(f"\n[train] In-sample Spearman IC (informational): {ic_insample:.4f}")
    if not np.isnan(ic_insample) and abs(ic_insample) < 0.005:
        print("[WARN] In-sample IC near zero — model may not be learning. Check features/target.")

    # ------------------------------------------------------------------ #
    # 8. Done
    # ------------------------------------------------------------------ #
    print(f"\n[train] Model saved to {model_path}")


if __name__ == "__main__":
    main()
