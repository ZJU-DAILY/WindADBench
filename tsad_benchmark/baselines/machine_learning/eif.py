# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import eif as iso

from tsad_benchmark.baselines._pyod_base import prepare_array
from tsad_benchmark.baselines._thresholding import (
    normalize_anomaly_ratios,
    percentile_label_maps,
)
from tsad_benchmark.common.random_utils import DEFAULT_SEED
from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability


class EIFModel(AnomalyModelBase):
    """Extended Isolation Forest detector."""

    model_name = "EIF"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        n_estimators: int = 200,
        sample_size: int = 256,
        extension_level: Optional[int] = None,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.sample_size = int(sample_size)
        # ``None`` => full EIF (extension_level = d - 1).  ``0`` falls
        # back to vanilla axis-aligned IF.
        self.extension_level = extension_level
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.seed = int(seed)

        self.model_hyper_params = {
            "n_estimators": self.n_estimators,
            "sample_size": self.sample_size,
            "extension_level": self.extension_level,
            "anomaly_ratio": self.anomaly_ratio,
            "seed": self.seed,
        }
        self._forest = None
        self._n_features = 0
        self._fit_sample_size = 0
        self._train_scores: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # AnomalyModelBase interface
    # ------------------------------------------------------------------

    def fit(self, train_data, train_label=None, covariates=None, **kwargs) -> None:
        X = prepare_array(train_data)
        if X.size == 0:
            self._forest = None
            self._fit_sample_size = 0
            self._train_scores = None
            return
        self._n_features = int(X.shape[1])
        # sahandha/eif requires C-contiguous float64 input
        X = np.ascontiguousarray(X, dtype=np.float64)
        sample_size = max(1, min(self.sample_size, X.shape[0]))
        self._fit_sample_size = int(sample_size)
        if self.extension_level is None:
            ext = max(self._n_features - 1, 0)
        else:
            ext = max(0, min(int(self.extension_level), self._n_features - 1))
        self._forest = iso.iForest(
            X,
            ntrees=self.n_estimators,
            sample_size=sample_size,
            ExtensionLevel=ext,
            seed=self.seed,
        )
        self._train_scores = self._score_array(X)

    def detect_score(self, test_data, covariates=None, **kwargs) -> np.ndarray:
        X = prepare_array(test_data)
        if self._forest is None or X.size == 0:
            return np.zeros(len(X), dtype=float)
        X = np.ascontiguousarray(X, dtype=np.float64)
        return self._score_array(X)

    def detect_label(self, test_data, covariates=None, **kwargs):
        scores = self.detect_score(test_data)
        return percentile_label_maps(scores, self._train_scores, self.anomaly_ratio), scores

    def _score_array(self, X: np.ndarray) -> np.ndarray:
        if self._forest is None or X.size == 0:
            return np.zeros(X.shape[0], dtype=float)
        X = np.ascontiguousarray(X, dtype=np.float64)
        scores = np.asarray(self._forest.compute_paths(X_in=X), dtype=float).ravel()
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Cost estimation (rough order-of-magnitude FLOPs)
    # ------------------------------------------------------------------

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        n_train = int(len(train_data)) if train_data is not None else 0
        n_test = int(len(test_data)) if test_data is not None else 0
        d = self._n_features or self._ncols(test_data, fallback=1)
        psi = float(min(self.sample_size, max(n_train, 1)))
        log_psi = math.log2(max(psi, 2.0))
        flops = self.n_estimators * n_test * log_psi * d
        return float(flops) if flops > 0 else float("nan")

    def estimate_model_size_mb(self) -> float:
        if self._forest is None:
            return math.nan
        nbytes = self._estimate_exposed_forest_bytes(self._forest)
        if nbytes <= 1024:
            nbytes = self._estimate_structural_forest_bytes()
        return float(nbytes / (1024.0 * 1024.0)) if nbytes > 0 else math.nan

    def _estimate_structural_forest_bytes(self) -> float:
        if self._n_features <= 0 or self._fit_sample_size <= 0:
            return math.nan

        d = int(self._n_features)
        psi = int(self._fit_sample_size)
        internal_nodes = max(psi - 1, 0)
        leaf_nodes = max(psi, 1)
        float64 = 8
        int64 = 8
        bytes_per_internal = 2 * d * float64 + 4 * int64
        bytes_per_leaf = 2 * int64
        per_tree = internal_nodes * bytes_per_internal + leaf_nodes * bytes_per_leaf
        return float(self.n_estimators * per_tree)

    @classmethod
    def _estimate_exposed_forest_bytes(cls, obj, *, max_depth: int = 24) -> int:
        seen: set[int] = set()
        total = 0

        def visit(value, depth: int) -> None:
            nonlocal total
            if value is None or depth > max_depth:
                return

            if isinstance(value, np.ndarray):
                total += int(value.nbytes)
                return
            if isinstance(value, np.generic):
                total += int(value.nbytes)
                return
            if isinstance(value, (bool, np.bool_)):
                total += 1
                return
            if isinstance(value, (int, np.integer, float, np.floating)):
                total += 8
                return
            if isinstance(value, (str, bytes)):
                total += len(value)
                return

            oid = id(value)
            if oid in seen:
                return
            seen.add(oid)

            if isinstance(value, dict):
                total += 8 * len(value)
                for key, item in value.items():
                    visit(key, depth + 1)
                    visit(item, depth + 1)
                return
            if isinstance(value, (list, tuple, set)):
                total += 8 * len(value)
                for item in value:
                    visit(item, depth + 1)
                return
            if hasattr(value, "__dict__"):
                visit(vars(value), depth + 1)
                return

            for name in dir(value):
                if name.startswith("_"):
                    continue
                try:
                    attr = getattr(value, name)
                except Exception:
                    continue
                if callable(attr):
                    continue
                visit(attr, depth + 1)

        try:
            visit(obj, 0)
        except Exception:
            return 0
        return int(total)

    @staticmethod
    def _ncols(df, fallback: int = 1) -> int:
        if df is None:
            return max(fallback, 1)
        if hasattr(df, "shape") and len(getattr(df, "shape", ())) > 1:
            return max(int(df.shape[1]), 1)
        return max(fallback, 1)
