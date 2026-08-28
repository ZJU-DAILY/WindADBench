# -*- coding: utf-8 -*-


from __future__ import annotations

from tsad_benchmark.baselines._merlion_base import MerlionBaseModel


class DeepPointModel(MerlionBaseModel):
    """Wrapper around Merlion's ``DeepPointAnomalyDetector``."""

    model_name = "DeepPoint"

    def __init__(
        self,
        enable_threshold: bool = True,
    ) -> None:
        super().__init__()
        self._n_features = None
        # Enable Merlion's internal thresholding by default so
        # detect_label follows the model post-rule instead of a benchmark
        # fallback.
        self.enable_threshold = bool(enable_threshold)
        self.model_hyper_params.update(
            {"enable_threshold": self.enable_threshold}
        )

    def fit(self, train_data, train_label=None, covariates=None, **kwargs) -> None:
        self._n_features = self._infer_n_features(train_data)
        return super().fit(
            train_data,
            train_label=train_label,
            covariates=covariates,
            **kwargs,
        )

    def _build_merlion_model(self):
        from merlion.models.anomaly.deep_point_anomaly_detector import (
            DeepPointAnomalyDetector,
            DeepPointAnomalyDetectorConfig,
        )


        config = DeepPointAnomalyDetectorConfig(
            enable_threshold=self.enable_threshold,
        )
        return DeepPointAnomalyDetector(config)

    def estimate_n_params(self) -> float:
        if self._model is None or self._n_features is None:
            return float("nan")

        hidden_dims = (400, 400, 400)
        total = 0
        prev_dim = 1
        for hidden_dim in hidden_dims:
            total += prev_dim * hidden_dim  # Linear(..., bias=False)
            total += 2 * hidden_dim  # BatchNorm1d weight + bias
            prev_dim = hidden_dim
        total += prev_dim * self._n_features + self._n_features  # final Linear
        return float(total)

    @staticmethod
    def _infer_n_features(train_data):
        try:
            shape = train_data.shape
        except Exception:
            return None
        try:
            if len(shape) == 1:
                return 1
            if len(shape) >= 2:
                return int(shape[1])
        except Exception:
            return None
        return None
