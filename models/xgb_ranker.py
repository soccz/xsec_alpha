import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# xgboost is required at runtime. If not installed, a clear error is raised in __init__.
# Install with: pip install xgboost

logger = logging.getLogger(__name__)


def _derive_legacy_feature(X: pd.DataFrame, feature_name: str) -> pd.Series:
    """Reconstruct deprecated or renamed factors for pickled models."""
    if feature_name == "btc_lead_1h":
        if "momentum_1h" in X.columns:
            return -X["momentum_1h"]
        if "reversal_1h" in X.columns:
            return X["reversal_1h"]
    if feature_name == "btc_lead_4h":
        if "momentum_4h" in X.columns:
            return -X["momentum_4h"]
        if "reversal_4h" in X.columns:
            return X["reversal_4h"]
    if feature_name == "momentum_1h" and "reversal_1h" in X.columns:
        return -X["reversal_1h"]
    if feature_name == "momentum_4h" and "reversal_4h" in X.columns:
        return -X["reversal_4h"]
    if feature_name == "order_flow_bull" and "order_flow_bear" in X.columns:
        return -X["order_flow_bear"]
    if feature_name == "reversal_1h" and "momentum_1h" in X.columns:
        return -X["momentum_1h"]
    if feature_name == "reversal_4h" and "momentum_4h" in X.columns:
        return -X["momentum_4h"]
    if feature_name == "order_flow_bear" and "order_flow_bull" in X.columns:
        return -X["order_flow_bull"]
    raise KeyError(feature_name)


def _align_feature_frame(X, feature_names: Optional[list]):
    """Align incoming features to the model's training schema.

    New training code can drop deprecated duplicates while older pickles still
    ask for them. Reconstruct those columns lazily at prediction time.
    """
    if feature_names is None:
        return X if isinstance(X, pd.DataFrame) else np.asarray(X)

    if isinstance(X, pd.DataFrame):
        aligned = X.copy()
        missing = [name for name in feature_names if name not in aligned.columns]
        for name in list(missing):
            try:
                aligned[name] = _derive_legacy_feature(aligned, name)
                missing.remove(name)
            except KeyError:
                pass
        if missing:
            raise KeyError(f"Missing feature columns for prediction: {missing}")
        return aligned[feature_names]

    arr = np.asarray(X)
    if arr.ndim != 2 or arr.shape[1] != len(feature_names):
        raise ValueError(
            f"Expected {len(feature_names)} features, got shape {arr.shape}"
        )
    return arr


class XSecRanker:
    """XGBoost wrapper for cross-sectional return prediction.

    Predicts a score per coin per timestamp. Higher score = predicted higher
    residual return. Callers are responsible for cross-sectional ranking of
    the returned scores.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 20,
        random_state: int = 42,
    ):
        try:
            import xgboost  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "xgboost is required for XSecRanker. "
                "Install it with: pip install xgboost"
            ) from e

        from xgboost import XGBRegressor

        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            random_state=random_state,
            objective="reg:squarederror",
            tree_method="hist",
            verbosity=0,
        )
        self._model: Optional[XGBRegressor] = XGBRegressor(**self._params)
        self._feature_names: Optional[list] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XSecRanker":
        """Train on (X, y). Logs feature importances after fit. Returns self."""
        n_samples, n_features = X.shape
        logger.info(
            "XSecRanker.fit: %d samples, %d features", n_samples, n_features
        )

        self._feature_names = list(X.columns)
        self._model.fit(X, y)
        self._is_fitted = True

        # Log top-10 feature importances by gain
        importances = self.feature_importance_
        top10 = list(importances.items())[:10]
        logger.info(
            "Feature importances (gain, top 10): %s",
            ", ".join(f"{k}={v:.4f}" for k, v in top10),
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns raw scores (higher = predicted higher residual return)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(_align_feature_frame(X, self._feature_names))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model to path using pickle. Creates parent directories if needed."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("XSecRanker saved to %s", path)

    @classmethod
    def load(cls, path: str):
        """Load a previously saved model (XSecRanker or RidgeRanker) from path."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not hasattr(obj, "predict"):
            raise TypeError(f"Loaded object {type(obj).__name__} has no predict method.")
        logger.info("Model loaded from %s (%s)", path, type(obj).__name__)
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_importance_(self) -> dict:
        """Returns {feature_name: importance_score} sorted descending by gain."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        scores = self._model.get_booster().get_score(importance_type="gain")

        if self._feature_names:
            # get_score returns keys like 'f0', 'f1', ... when fit with arrays.
            # Map back to original feature names.
            named: dict = {}
            for key, val in scores.items():
                if key.startswith("f") and key[1:].isdigit():
                    idx = int(key[1:])
                    if idx < len(self._feature_names):
                        named[self._feature_names[idx]] = val
                    else:
                        named[key] = val
                else:
                    named[key] = val
            scores = named

        return dict(
            sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        )

    def __repr__(self) -> str:
        fitted = "fitted" if self._is_fitted else "not fitted"
        return (
            f"XSecRanker({fitted}, "
            f"n_estimators={self._params['n_estimators']}, "
            f"max_depth={self._params['max_depth']}, "
            f"lr={self._params['learning_rate']})"
        )


class RidgeRanker:
    """Ridge regression wrapper for cross-sectional return prediction.

    Same interface as XSecRanker: fit(X, y), predict(X), save/load.
    Preferred when universe is small (< 100 coins) where XGBoost tends to overfit.
    """

    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge

        self._alpha = alpha
        self._model = Ridge(alpha=alpha)
        self._feature_names: Optional[list] = None
        self._is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeRanker":
        self._feature_names = list(X.columns)
        self._model.fit(X, y)
        self._is_fitted = True
        # Log coefficients as "importances"
        coefs = dict(zip(self._feature_names, self._model.coef_))
        logger.info(
            "RidgeRanker.fit: %d samples, %d features. Coefs: %s",
            len(X),
            len(self._feature_names),
            ", ".join(f"{k}={v:.5f}" for k, v in coefs.items()),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(_align_feature_frame(X, self._feature_names))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("RidgeRanker saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "RidgeRanker":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, (cls, XSecRanker)):
            raise TypeError(
                f"Loaded object is {type(obj).__name__}, expected RidgeRanker or XSecRanker."
            )
        logger.info("Model loaded from %s (type=%s)", path, type(obj).__name__)
        return obj

    @property
    def feature_importance_(self) -> dict:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        coefs = dict(zip(self._feature_names, self._model.coef_))
        # Sort by absolute coefficient value descending
        return dict(sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True))

    def __repr__(self) -> str:
        fitted = "fitted" if self._is_fitted else "not fitted"
        return f"RidgeRanker({fitted}, alpha={self._alpha})"


class LGBMRanker:
    """LightGBM for cross-sectional ranking — captures factor interactions.

    Same interface as XSecRanker / RidgeRanker: fit(X, y), predict(X), save/load.
    Generally faster than XGBoost with comparable accuracy. Uses higher
    min_child_samples (50) by default to reduce overfitting on noisy crypto data.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 50,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        try:
            import lightgbm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "lightgbm is required for LGBMRanker. "
                "Install it with: pip install lightgbm"
            ) from e

        import lightgbm as lgb

        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            verbose=-1,
        )
        self._model = lgb.LGBMRegressor(**self._params)
        self._feature_names: Optional[list] = None
        self._is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMRanker":
        n_samples, n_features = X.shape
        logger.info(
            "LGBMRanker.fit: %d samples, %d features", n_samples, n_features
        )
        self._feature_names = list(X.columns)
        self._model.fit(X, y)
        self._is_fitted = True

        # Log top-10 feature importances
        importances = self.feature_importance_
        top10 = list(importances.items())[:10]
        logger.info(
            "Feature importances (top 10): %s",
            ", ".join(f"{k}={v:.4f}" for k, v in top10),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(_align_feature_frame(X, self._feature_names))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("LGBMRanker saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "LGBMRanker":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not hasattr(obj, "predict"):
            raise TypeError(
                f"Loaded object {type(obj).__name__} has no predict method."
            )
        logger.info("Model loaded from %s (type=%s)", path, type(obj).__name__)
        return obj

    @property
    def feature_importance_(self) -> dict:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        imp = self._model.feature_importances_
        names = self._feature_names or [f"f{i}" for i in range(len(imp))]
        return dict(sorted(zip(names, imp), key=lambda x: -x[1]))

    def __repr__(self) -> str:
        fitted = "fitted" if self._is_fitted else "not fitted"
        return (
            f"LGBMRanker({fitted}, "
            f"n_estimators={self._params['n_estimators']}, "
            f"max_depth={self._params['max_depth']}, "
            f"lr={self._params['learning_rate']})"
        )


class EnsembleRanker:
    """50/50 blend of Ridge + LightGBM. Best IC in model comparison."""

    def __init__(self, ridge_alpha=1.0, lgbm_kwargs=None):
        self._ridge = RidgeRanker(alpha=ridge_alpha)
        self._lgbm = LGBMRanker(**(lgbm_kwargs or {"min_child_samples": 50}))
        self._is_fitted = False
        self._feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleRanker":
        logger.info("EnsembleRanker.fit: training Ridge + LightGBM")
        self._ridge.fit(X, y)
        self._lgbm.fit(X, y)
        self._feature_names = list(X.columns)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pr = self._ridge.predict(X)
        pl = self._lgbm.predict(X)
        return 0.5 * pr + 0.5 * pl

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("EnsembleRanker saved to %s", path)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not hasattr(obj, "predict"):
            raise TypeError(f"Loaded object {type(obj).__name__} has no predict method.")
        logger.info("Model loaded from %s (%s)", path, type(obj).__name__)
        return obj

    @property
    def feature_importance_(self) -> dict:
        ri = self._ridge.feature_importance_
        li = self._lgbm.feature_importance_
        # Normalize LGBM importances to same scale as Ridge coefficients
        li_sum = sum(abs(v) for v in li.values()) or 1
        ri_sum = sum(abs(v) for v in ri.values()) or 1
        combined = {}
        for k in ri:
            r_norm = abs(ri.get(k, 0)) / ri_sum
            l_norm = abs(li.get(k, 0)) / li_sum
            combined[k] = round(0.5 * r_norm + 0.5 * l_norm, 4)
        return dict(sorted(combined.items(), key=lambda x: -x[1]))

    def __repr__(self) -> str:
        fitted = "fitted" if self._is_fitted else "not fitted"
        return f"EnsembleRanker({fitted}, Ridge+LightGBM)"
