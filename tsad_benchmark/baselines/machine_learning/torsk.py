# -*- coding: utf-8 -*-

from __future__ import annotations

import pickle
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from tsad_benchmark.baselines._pyod_base import prepare_array
from tsad_benchmark.baselines._thresholding import (
    normalize_anomaly_ratios,
    percentile_label_maps,
)
from tsad_benchmark.common.random_utils import DEFAULT_SEED
from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability


class TorskModel(AnomalyModelBase):
    """Echo State Network one-step predictor as anomaly detector."""

    model_name = "Torsk"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        reservoir_dim: int = 256,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.5,
        leak_rate: float = 0.3,
        sparsity: float = 0.1,
        ridge_lambda: float = 1e-4,
        transient: int = 200,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.reservoir_dim = int(reservoir_dim)
        self.spectral_radius = float(spectral_radius)
        self.input_scaling = float(input_scaling)
        self.leak_rate = float(leak_rate)
        # Fraction of W_res entries kept non-zero; 0.1 => 10% density.
        self.sparsity = float(sparsity)
        self.ridge_lambda = float(ridge_lambda)
        self.transient = int(transient)
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.seed = int(seed)

        self.model_hyper_params = {
            "reservoir_dim": self.reservoir_dim,
            "spectral_radius": self.spectral_radius,
            "input_scaling": self.input_scaling,
            "leak_rate": self.leak_rate,
            "sparsity": self.sparsity,
            "ridge_lambda": self.ridge_lambda,
            "transient": self.transient,
            "anomaly_ratio": self.anomaly_ratio,
            "seed": self.seed,
        }

        self._w_in: Optional[np.ndarray] = None
        self._w_res = None  # csr_matrix
        self._w_out: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._n_features: int = 0
        self._train_scores: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Reservoir construction & propagation
    # ------------------------------------------------------------------

    def _build_reservoir(self, d_in: int) -> None:
        rng = np.random.default_rng(self.seed)
        N = self.reservoir_dim

        W = rng.uniform(-0.5, 0.5, size=(N, N))
        # Sparsify
        keep_mask = rng.random(size=(N, N)) < self.sparsity
        W[~keep_mask] = 0.0
        # Rescale to target spectral radius
        try:
            eig = float(np.max(np.abs(np.linalg.eigvals(W))))
        except np.linalg.LinAlgError:
            eig = 0.0
        if eig > 1e-12:
            W *= self.spectral_radius / eig
        self._w_res = csr_matrix(W)

        self._w_in = rng.uniform(
            -self.input_scaling, self.input_scaling, size=(N, d_in)
        )

    def _run_reservoir(self, U: np.ndarray) -> np.ndarray:
        T, _ = U.shape
        N = self.reservoir_dim
        states = np.empty((T, N), dtype=float)
        x = np.zeros(N, dtype=float)
        alpha = self.leak_rate
        W_in = self._w_in
        W_res = self._w_res
        for t in range(T):
            pre = W_in @ U[t] + W_res @ x
            x = (1.0 - alpha) * x + alpha * np.tanh(pre)
            states[t] = x
        return states

    # ------------------------------------------------------------------
    # AnomalyModelBase interface
    # ------------------------------------------------------------------

    def fit(self, train_data, train_label=None, covariates=None, **kwargs) -> None:
        X = prepare_array(train_data)
        if X.size == 0 or X.shape[0] < 2:
            self._w_out = None
            self._train_scores = None
            return
        T, d = X.shape
        self._n_features = d
        # Standardise so input scaling is interpretable
        self._mean = X.mean(axis=0, keepdims=True)
        self._std = np.maximum(X.std(axis=0, keepdims=True), 1e-8)
        Xn = (X - self._mean) / self._std

        self._build_reservoir(d)
        states = self._run_reservoir(Xn)

        # Build (states[t], u[t]) → predict u[t+1] for t in [t0, T-1)
        T_pairs = T - 1
        if T_pairs <= 0:
            self._w_out = None
            return
        t0 = max(0, min(self.transient, T_pairs - 1))
        n_eff = T_pairs - t0
        if n_eff <= 0:
            t0 = 0
            n_eff = T_pairs
        Z = np.concatenate(
            [states[t0:t0 + n_eff], Xn[t0:t0 + n_eff], np.ones((n_eff, 1))],
            axis=1,
        )
        Y = Xn[t0 + 1:t0 + 1 + n_eff]
        # Ridge regression: W_out = (Z^T Z + λI)^{-1} Z^T Y
        ZTZ = Z.T @ Z
        ZTY = Z.T @ Y
        reg = np.eye(ZTZ.shape[0]) * self.ridge_lambda
        try:
            self._w_out = np.linalg.solve(ZTZ + reg, ZTY)
        except np.linalg.LinAlgError:
            self._w_out = np.linalg.pinv(ZTZ + reg) @ ZTY
        self._train_scores = self._score_array(X)

    def detect_score(self, test_data, covariates=None, **kwargs) -> np.ndarray:
        X = prepare_array(test_data)
        return self._score_array(X)

    def detect_label(self, test_data, covariates=None, **kwargs):
        scores = self.detect_score(test_data)
        return percentile_label_maps(scores, self._train_scores, self.anomaly_ratio), scores

    def _score_array(self, X: np.ndarray) -> np.ndarray:
        T = int(X.shape[0]) if X.ndim == 2 else 0
        if T == 0 or self._w_out is None or self._w_in is None:
            return np.zeros(T, dtype=float)
        # Project onto train statistics; reservoir was built for d cols
        d_train = self._n_features
        if X.shape[1] != d_train:
            # Best effort: pad / truncate
            if X.shape[1] < d_train:
                pad = np.zeros((T, d_train - X.shape[1]))
                X = np.concatenate([X, pad], axis=1)
            else:
                X = X[:, :d_train]
        Xn = (X - self._mean) / self._std
        states = self._run_reservoir(Xn)
        if T < 2:
            return np.zeros(T, dtype=float)
        Z = np.concatenate(
            [states[:-1], Xn[:-1], np.ones((T - 1, 1))], axis=1
        )
        Y_pred = Z @ self._w_out
        err = np.sum((Y_pred - Xn[1:]) ** 2, axis=1)
        # Pad first time-step with the median error so output length == T.
        scores = np.empty(T, dtype=float)
        scores[0] = float(np.median(err)) if err.size else 0.0
        scores[1:] = err
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        n_test = int(len(test_data)) if test_data is not None else 0
        d = self._n_features or self._ncols(test_data, fallback=1)
        N = self.reservoir_dim
        per_step = N * N * self.sparsity + N * d
        m = N + d + 1
        flops = n_test * per_step + n_test * m * d
        return float(flops) if flops > 0 else float("nan")

    def estimate_model_size_mb(self) -> float:
        if self._w_out is None or self._w_in is None or self._w_res is None:
            return float("nan")
        try:
            state = {
                "w_in": self._w_in,
                "w_res": self._w_res,
                "w_out": self._w_out,
                "mean": self._mean,
                "std": self._std,
                "n_features": self._n_features,
                "model_hyper_params": self.model_hyper_params,
            }
            payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
            return float(len(payload) / (1024.0 * 1024.0))
        except Exception:
            return float("nan")

    @staticmethod
    def _ncols(df, fallback: int = 1) -> int:
        if df is None:
            return max(fallback, 1)
        if hasattr(df, "shape") and len(getattr(df, "shape", ())) > 1:
            return max(int(df.shape[1]), 1)
        return max(fallback, 1)
