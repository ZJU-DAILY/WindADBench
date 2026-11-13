from __future__ import annotations

import numpy as np
import pandas as pd

from tsad_benchmark.baselines._merlion_base import MerlionBaseModel
from tsad_benchmark.baselines._thresholding import normalize_anomaly_ratios


class DeepPointModel(MerlionBaseModel):
    model_name = "DeepPoint"

    def __init__(
        self,
        enable_threshold: bool = True,
        score_protocol: str = "source_frozen_v1",
        anomaly_ratio=None,
    ) -> None:
        super().__init__()
        self._n_features = None
        if score_protocol != "source_frozen_v1":
            raise ValueError("DeepPoint supports score_protocol='source_frozen_v1'.")
        self.score_protocol = score_protocol
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self._train_scores = None
        self._feature_columns = None
        self._source_segment_lengths = None

        self.enable_threshold = bool(enable_threshold)
        self.model_hyper_params.update(
            {"enable_threshold": self.enable_threshold,
             "score_protocol": self.score_protocol,
             "anomaly_ratio": list(self.anomaly_ratio),
             "label_protocol": "source_train_percentile",
             "difference_warmup": 1,
             "source_difference_policy": "within_segment",
             "adapter_mlp_hidden": [400, 400, 400], "adapter_num_epochs": 20}
        )

    def fit(self, train_data, train_label=None, covariates=None, **kwargs) -> None:
        self._fit_source(train_data, [train_data], train_label, covariates, **kwargs)

    def fit_segments(self, segments, **kwargs) -> None:
        segments = list(segments)
        if not segments or any(segment is None or len(segment) == 0 for segment in segments):
            raise ValueError("DeepPoint requires nonempty source segments.")
        columns = list(segments[0].columns)
        for segment in segments:
            self._require_finite_input(segment)
            if list(segment.columns) != columns:
                raise ValueError("DeepPoint source segments must share ordered feature columns.")
        if sum(len(segment)-1 for segment in segments) < 2:
            raise ValueError("DeepPoint needs at least two within-segment source differences.")
        train_data = pd.concat(segments, ignore_index=True)
        self._fit_source(train_data, segments, **kwargs)

    def _fit_source(self, train_data, segments, train_label=None, covariates=None, **kwargs) -> None:
        self._require_finite_input(train_data)
        if train_data is None or len(train_data) < 3:
            raise ValueError("DeepPoint requires at least three source observations.")
        self._feature_columns = list(train_data.columns)
        self._source_segment_lengths = [len(segment) for segment in segments]
        self._n_features = self._infer_n_features(train_data)
        self._train_scores = None
        super().fit(
            train_data,
            train_label=train_label,
            covariates=covariates,
            **kwargs,
        )
        calibration = []
        for segment in segments:
            scores = self.detect_score(segment)
            calibration.append(scores[self.score_validity_mask(len(scores))])
        self._train_scores = np.concatenate(calibration)

    def _build_merlion_model(self):
        from merlion.models.anomaly.deep_point_anomaly_detector import DeepPointAnomalyDetectorConfig
        from ._causal_deeppoint import SourceFrozenDeepPointDetector

        config = DeepPointAnomalyDetectorConfig(
            enable_threshold=False,
        )
        return SourceFrozenDeepPointDetector(config, self._source_segment_lengths)

    @staticmethod
    def _require_finite_input(data):
        if data is not None and not np.isfinite(np.asarray(data, dtype=float)).all():
            raise ValueError("DeepPoint requires finite input from source-fitted imputation.")

    @staticmethod
    def score_validity_mask(length: int) -> np.ndarray:
        mask = np.ones(int(length), dtype=bool)
        mask[:1] = False
        return mask

    @staticmethod
    def _align_length(scores: np.ndarray, target_len: int) -> np.ndarray:
        if scores.size == target_len:
            return scores
        if scores.size != max(0, target_len-1):
            raise ValueError("Unexpected DeepPoint difference-score alignment.")
        out = np.zeros(target_len, dtype=float)
        if scores.size:
            out[1:] = scores
        return out

    def detect_score(self, test_data, covariates=None, **kwargs):
        self._require_finite_input(test_data)
        n = len(test_data) if test_data is not None else 0
        if getattr(self._model, "_score_network", None) is None:
            raise RuntimeError("DeepPoint needs a source_frozen_v1 fit; legacy checkpoints cannot be reused.")
        if n and list(test_data.columns) != self._feature_columns:
            raise ValueError("DeepPoint scoring columns differ from source training columns.")
        if n < 2:
            return np.zeros(n, dtype=float)
        return super().detect_score(test_data, covariates=covariates, **kwargs)

    def detect_label(self, test_data, covariates=None, **kwargs):
        scores = self.detect_score(test_data, covariates=covariates, **kwargs)
        if self._train_scores is None or not len(self._train_scores):
            raise RuntimeError("DeepPoint label calibration requires source training scores.")
        valid = self.score_validity_mask(len(scores))
        predictions = {
            str(r): ((scores > np.percentile(self._train_scores, 100.0-r)) & valid).astype(np.int32)
            for r in self.anomaly_ratio
        }
        return predictions, scores

    def estimate_n_params(self) -> float:
        if self._model is None or self._n_features is None:
            return float("nan")

        hidden_dims = (400, 400, 400)
        total = 0
        prev_dim = 1
        for hidden_dim in hidden_dims:
            total += prev_dim * hidden_dim
            total += 2 * hidden_dim
            prev_dim = hidden_dim
        total += prev_dim * self._n_features + self._n_features
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
