# -*- coding: utf-8 -*-


from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from pyod.models.pca import PCA as _PyodPCA

from tsad_benchmark.baselines._pyod_base import PyODBaseModel, prepare_array
from tsad_benchmark.common.random_utils import DEFAULT_SEED


class PCAModel(PyODBaseModel):

    model_name = "PCA"

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        n_selected_components: Optional[int] = None,
        whiten: bool = False,
        weighted: bool = True,
        standardization: bool = True,
        contamination: float = 0.05,
        random_state: int = DEFAULT_SEED,
        min_variance: float = 1e-8,
    ) -> None:
        super().__init__(contamination=contamination)
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.whiten = bool(whiten)
        self.weighted = bool(weighted)
        self.standardization = bool(standardization)
        self.random_state = int(random_state)
        self.min_variance = float(min_variance)
        self.model_hyper_params.update(
            {
                "n_components": self.n_components,
                "n_selected_components": self.n_selected_components,
                "weighted": self.weighted,
                "standardization": self.standardization,
                "random_state": self.random_state,
                "min_variance": self.min_variance,
            }
        )
        self._kept_cols: Optional[np.ndarray] = None

    def _build_pyod_model(self):
        return _PyodPCA(
            n_components=self.n_components,
            n_selected_components=self.n_selected_components,
            contamination=self.contamination,
            whiten=self.whiten,
            standardization=self.standardization,
            weighted=self.weighted,
            random_state=self.random_state,
        )

    def fit(self, train_data, train_label=None, covariates=None, **kwargs) -> None:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            super().fit(
                train_data,
                train_label=train_label,
                covariates=covariates,
                **kwargs,
            )
        if self._model is None or not hasattr(self._model, "selected_w_components_"):
            return

        w = np.asarray(self._model.selected_w_components_, dtype=float)
        self._model.selected_w_components_ = np.maximum(w, self.min_variance)

        X = prepare_array(train_data)
        X = self._preprocess_X(X, fitting=False)
        if X.size == 0:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scores = np.asarray(
                self._model.decision_function(X), dtype=float
            ).ravel()
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        self._model.decision_scores_ = scores
        if 0.0 < self.contamination < 1.0:
            pct = 100.0 * (1.0 - self.contamination)
            self._model.threshold_ = float(np.percentile(scores, pct))

    def detect_score(self, test_data, covariates=None, **kwargs) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw = super().detect_score(test_data, covariates=covariates, **kwargs)
        return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    def detect_label(self, test_data, covariates=None, **kwargs) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return super().detect_label(test_data, covariates=covariates, **kwargs)

    def _preprocess_X(self, X: np.ndarray, fitting: bool) -> np.ndarray:
        if X.size == 0:
            return X
        if fitting:
            var = np.nanvar(X, axis=0)
            kept = var > self.min_variance
            self._kept_cols = kept
            if not kept.any():
                return X[:, :0]
            return X[:, kept]

        if self._kept_cols is None:
            return X
        if not self._kept_cols.any():
            return X[:, :0]
        if self._kept_cols.shape[0] != X.shape[1]:
            return X
        return X[:, self._kept_cols]

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        n_test = self._nrows(test_data)
        d = self._ncols(test_data, fallback=self._n_features or 1)
        c = d
        if isinstance(self.n_components, int) and self.n_components > 0:
            c = max(min(self.n_components, d), 1)
        flops = n_test * d * c if n_test > 0 else 0.0
        return float(flops) if flops > 0 else float("nan")
