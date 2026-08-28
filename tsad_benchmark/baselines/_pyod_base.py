# -*- coding: utf-8 -*-


from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability


def prepare_array(data) -> np.ndarray:

    if isinstance(data, pd.DataFrame):
        X = data.values
    else:
        X = np.asarray(data)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.size and not np.all(np.isfinite(X)):
        mask = ~np.isfinite(X)
        col_mean = np.nanmean(np.where(mask, np.nan, X), axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        X = np.where(mask, col_mean, X)
    return X


class PyODBaseModel(AnomalyModelBase):

    capability = ModelCapability.score_and_label()
    _scale_inputs: bool = False

    def __init__(self, contamination: float = 0.05) -> None:
        self.contamination = float(contamination)
        self.model_hyper_params = {"contamination": self.contamination}
        self._model = None
        self._n_features: int = 0
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _build_pyod_model(self):

        raise NotImplementedError

    def _preprocess_X(self, X: np.ndarray, fitting: bool) -> np.ndarray:

        return X

    # ------------------------------------------------------------------
    # AnomalyModelBase interface
    # ------------------------------------------------------------------

    def _apply_scaler(self, X: np.ndarray, fitting: bool) -> np.ndarray:
        if not self._scale_inputs or X.size == 0:
            return X
        if fitting:
            self._scaler = StandardScaler().fit(X)
            return self._scaler.transform(X)
        if self._scaler is None:
            return X
        return self._scaler.transform(X)

    def fit(
        self,
        train_data: pd.DataFrame,
        train_label: Optional[pd.DataFrame] = None,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> None:
        X = prepare_array(train_data)
        self._n_features = X.shape[1] if X.size else 0
        self._scaler = None
        if X.size == 0:
            self._model = None
            return
        X = self._apply_scaler(X, fitting=True)
        X = self._preprocess_X(X, fitting=True)
        if X.size == 0:
            self._model = None
            return
        self._model = self._build_pyod_model()
        self._model.fit(X)

    def detect_score(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        X = prepare_array(test_data)
        if self._model is None or X.size == 0:
            return np.zeros(len(X), dtype=float)
        X = self._apply_scaler(X, fitting=False)
        Xp = self._preprocess_X(X, fitting=False)
        if Xp.size == 0:
            return np.zeros(len(X), dtype=float)
        return np.asarray(self._model.decision_function(Xp), dtype=float).ravel()

    def detect_label(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        X = prepare_array(test_data)
        if self._model is None or X.size == 0:
            return np.zeros(len(X), dtype=np.int32)
        X = self._apply_scaler(X, fitting=False)
        Xp = self._preprocess_X(X, fitting=False)
        if Xp.size == 0:
            return np.zeros(len(X), dtype=np.int32)
        return np.asarray(self._model.predict(Xp), dtype=np.int32).ravel()

    # ------------------------------------------------------------------
    # Cost estimation (default: no estimate)
    # ------------------------------------------------------------------

    def estimate_flops(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        **kwargs,
    ) -> float:
        return float("nan")

    def estimate_model_size_mb(self) -> float:
        if self._model is None:
            return float("nan")
        try:
            import pickle
            return float(len(pickle.dumps(self._model)) / (1024.0 * 1024.0))
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # Helpers exposed for subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _nrows(df) -> int:
        return int(len(df)) if df is not None else 0

    @staticmethod
    def _ncols(df, fallback: int = 1) -> int:
        if df is None:
            return max(fallback, 1)
        if hasattr(df, "shape") and len(getattr(df, "shape", ())) > 1:
            return max(int(df.shape[1]), 1)
        return max(fallback, 1)