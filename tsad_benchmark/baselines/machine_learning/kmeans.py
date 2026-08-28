# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pickle
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from tsad_benchmark.baselines._pyod_base import prepare_array
from tsad_benchmark.baselines._thresholding import (
    normalize_anomaly_ratios,
    percentile_label_maps,
)
from tsad_benchmark.common.random_utils import DEFAULT_SEED
from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability


class KMeansModel(AnomalyModelBase):

    model_name = "KMeans"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        n_clusters: int = 20,
        window_size: int = 50,
        stride: int = 1,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
        random_state: int = DEFAULT_SEED,
        n_init: int = 10,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.random_state = int(random_state)
        self.n_init = int(n_init)

        self.model_hyper_params = {
            "n_clusters": self.n_clusters,
            "window_size": self.window_size,
            "stride": self.stride,
            "anomaly_ratio": self.anomaly_ratio,
            "random_state": self.random_state,
        }

        self._scaler: Optional[StandardScaler] = None
        self._kmeans: Optional[KMeans] = None
        self._n_features: int = 0
        self._train_scores: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # AnomalyModelBase interface
    # ------------------------------------------------------------------

    def fit(
        self,
        train_data: pd.DataFrame,
        train_label: Optional[pd.DataFrame] = None,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> None:
        X = prepare_array(train_data)
        if X.size == 0:
            self._scaler = None
            self._kmeans = None
            self._train_scores = None
            return
        self._n_features = X.shape[1]
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        windows, _ = self._make_windows(Xs)
        if windows.shape[0] == 0:
            self._kmeans = None
            self._train_scores = None
            return
        k = max(min(self.n_clusters, windows.shape[0]), 1)
        self._kmeans = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        self._kmeans.fit(windows)
        self._train_scores = self._score_scaled(Xs)

    def detect_score(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        X = prepare_array(test_data)
        if X.size == 0 or self._kmeans is None or self._scaler is None:
            return np.zeros(len(X), dtype=float)
        Xs = self._scaler.transform(X)
        return self._score_scaled(Xs)

    def detect_label(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ):
        scores = self.detect_score(test_data)
        return percentile_label_maps(scores, self._train_scores, self.anomaly_ratio), scores

    def _score_scaled(self, Xs: np.ndarray) -> np.ndarray:
        if self._kmeans is None or Xs.size == 0:
            return np.zeros(Xs.shape[0], dtype=float)
        windows, padding = self._make_windows(Xs)
        if windows.shape[0] == 0:
            return np.zeros(Xs.shape[0], dtype=float)
        clusters = self._kmeans.predict(windows)
        diffs = np.linalg.norm(
            windows - self._kmeans.cluster_centers_[clusters], axis=1
        )
        return self._reverse_window(diffs, total=Xs.shape[0], padding=padding)

    # ------------------------------------------------------------------
    # Cost estimation (rough order-of-magnitude FLOPs)
    # ------------------------------------------------------------------

    def estimate_flops(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        **kwargs,
    ) -> float:
        n_test = int(len(test_data)) if test_data is not None else 0
        d = self._n_features
        if d <= 0 and hasattr(test_data, "shape") and len(getattr(test_data, "shape", ())) > 1:
            d = int(test_data.shape[1])
        d = max(d, 1)
        k = max(self.n_clusters, 1)
        win_d = self.window_size * d
        n_test_w = max((n_test - self.window_size) // max(self.stride, 1) + 1, 0)
        flops = n_test_w * k * win_d
        return float(flops) if flops > 0 else float("nan")

    def estimate_model_size_mb(self) -> float:
        if self._kmeans is None:
            return float("nan")
        try:
            state = {
                "kmeans": self._kmeans,
                "scaler": self._scaler,
                "n_features": self._n_features,
                "model_hyper_params": self.model_hyper_params,
            }
            payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
            return float(len(payload) / (1024.0 * 1024.0))
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # Sliding-window helpers
    # ------------------------------------------------------------------

    def _make_windows(self, X: np.ndarray) -> tuple[np.ndarray, int]:

        n, d = X.shape
        if n < self.window_size or self.window_size <= 0:
            return np.empty((0, max(self.window_size * d, 1))), n
        flat_shape = (n - (self.window_size - 1), -1)
        slides = sliding_window_view(X, window_shape=self.window_size, axis=0)
        slides = slides.reshape(flat_shape)[:: max(self.stride, 1), :]
        padding = n - (slides.shape[0] * max(self.stride, 1) + self.window_size - max(self.stride, 1))
        return slides, max(padding, 0)

    def _reverse_window(
        self, scores: np.ndarray, total: int, padding: int
    ) -> np.ndarray:

        if scores.size == 0 or total <= 0:
            return np.zeros(total, dtype=float)
        stride = max(self.stride, 1)
        begins = np.arange(scores.shape[0]) * stride
        ends = begins + self.window_size
        unwindowed_length = stride * (scores.shape[0] - 1) + self.window_size + padding
        out = np.full(unwindowed_length, fill_value=np.nan, dtype=float)
        indices = np.unique(np.r_[begins, ends])
        for i, j in zip(indices[:-1], indices[1:]):
            mask = (begins <= i) & (j - 1 < ends)
            sel = np.flatnonzero(mask)
            if sel.size:
                out[i:j] = float(np.nanmean(scores[sel]))
        np.nan_to_num(out, copy=False)
        if out.size >= total:
            return out[:total]
        # If for whatever reason out is shorter, right-pad with zeros.
        return np.concatenate([out, np.zeros(total - out.size, dtype=float)])
