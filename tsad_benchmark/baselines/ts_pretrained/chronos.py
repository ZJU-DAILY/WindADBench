# -*- coding: utf-8 -*-
"""Chronos anomaly-detection adapter."""

from __future__ import annotations

import io
import logging
import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from tsad_benchmark.baselines._official_ad import percentile_labels, to_2d_array
from tsad_benchmark.baselines._thresholding import normalize_anomaly_ratios
from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability

logger = logging.getLogger(__name__)


class ChronosModel(AnomalyModelBase):

    model_name = "Chronos"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-base",
        context_length: int = 96,
        prediction_length: int = 1,
        batch_size: int = 64,
        forecast_batch_size: Optional[int] = None,
        num_samples: int = 20,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
        local_files_only: bool = False,
        torch_dtype: str = "auto",
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.batch_size = int(batch_size)
        self.forecast_batch_size = int(forecast_batch_size or batch_size)
        self.num_samples = int(num_samples)
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.local_files_only = bool(local_files_only)
        self.torch_dtype = torch_dtype
        self.device = str(device)
        self.model_hyper_params = {
            "model_id": self.model_id,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "batch_size": self.batch_size,
            "forecast_batch_size": self.forecast_batch_size,
            "num_samples": self.num_samples,
            "anomaly_ratio": self.anomaly_ratio,
            "local_files_only": self.local_files_only,
            "torch_dtype": self.torch_dtype,
            "device": self.device,
        }
        self._pipeline = None
        self._scaler = None
        self._device = None
        self._n_features = 0
        self._train_scores = None
        self._train_arr = None

    def _resolve_device(self):
        import torch

        if self._device is None:
            if self.device.lower() == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(self.device)
                if self._device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("ChronosModel was configured with device='cuda', but CUDA is not available.")
        return self._device

    def _resolve_dtype(self):
        import torch

        if self.torch_dtype == "auto":
            return torch.bfloat16 if self._resolve_device().type == "cuda" else torch.float32
        return getattr(torch, self.torch_dtype)

    def _load_pipeline(self):
        try:
            from chronos import BaseChronosPipeline
        except Exception as exc:
            raise ImportError(
                "ChronosModel requires the official Chronos package. Install it with "
                "`pip install git+https://github.com/amazon-science/chronos-forecasting.git`."
            ) from exc
        return BaseChronosPipeline.from_pretrained(
            self.model_id,
            device_map=str(self._resolve_device()),
            torch_dtype=self._resolve_dtype(),
            local_files_only=self.local_files_only,
        )

    def _scale_fit(self, arr: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        return self._scaler.fit_transform(arr).astype(np.float32)

    def _scale_apply(self, arr: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            return arr.astype(np.float32)
        return self._scaler.transform(arr).astype(np.float32)

    def _prepare_inference_array(self, data) -> np.ndarray:
        arr = to_2d_array(data)
        if arr.shape[1] < self._n_features:
            pad = np.zeros((arr.shape[0], self._n_features - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > self._n_features:
            arr = arr[:, : self._n_features]
        return self._scale_apply(arr)

    def _forecast_batch(self, contexts: np.ndarray, prediction_length: int) -> np.ndarray:
        import torch

        context = torch.from_numpy(contexts.astype(np.float32)).to(self._resolve_device())
        if hasattr(self._pipeline, "predict_quantiles"):
            _quantiles, pred = self._pipeline.predict_quantiles(
                context,
                prediction_length=int(prediction_length),
                quantile_levels=[0.5],
            )
        else:
            samples = self._pipeline.predict(
                context,
                prediction_length=int(prediction_length),
                num_samples=self.num_samples,
            )
            pred = samples.mean(dim=1) if samples.ndim == 3 else samples
        return pred.detach().cpu().numpy().astype(np.float32)

    def _timeline_scores(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        n, n_features = arr.shape
        scores = np.zeros(n, dtype=float)
        counts = np.zeros(n, dtype=float)
        if n <= self.context_length or n_features == 0:
            return scores
        horizon = max(1, self.prediction_length)
        starts = np.arange(self.context_length, n, horizon, dtype=int)
        forecast_batch_size = max(1, self.forecast_batch_size)
        call_count = 0
        for feature_start in range(0, n_features, self.batch_size):
            feature_end = min(feature_start + self.batch_size, n_features)
            n_chunk_features = feature_end - feature_start
            starts_per_call = max(1, forecast_batch_size // max(n_chunk_features, 1))
            start_idx = 0
            while start_idx < len(starts):
                pred_len = min(horizon, n - int(starts[start_idx]))
                end_idx = start_idx
                while (
                    end_idx < len(starts)
                    and end_idx - start_idx < starts_per_call
                    and min(horizon, n - int(starts[end_idx])) == pred_len
                ):
                    end_idx += 1
                batch_starts = starts[start_idx:end_idx]
                context_offsets = np.arange(-self.context_length, 0, dtype=int)
                contexts = arr[
                    batch_starts[:, None] + context_offsets[None, :],
                    feature_start:feature_end,
                ]
                contexts = contexts.transpose(0, 2, 1).reshape(-1, self.context_length)
                pred = self._forecast_batch(contexts, pred_len)
                pred = pred[:, :pred_len].reshape(len(batch_starts), n_chunk_features, pred_len)
                target_offsets = np.arange(pred_len, dtype=int)
                truth = arr[
                    batch_starts[:, None] + target_offsets[None, :],
                    feature_start:feature_end,
                ].transpose(0, 2, 1)
                err = (truth - pred) ** 2
                chunk_scores = np.mean(err, axis=1)
                for row_idx, target_start in enumerate(batch_starts):
                    target_start = int(target_start)
                    target_end = target_start + pred_len
                    scores[target_start:target_end] += chunk_scores[row_idx, : target_end - target_start]
                    counts[target_start:target_end] += 1.0
                call_count += 1
                start_idx = end_idx
        logger.info(
            "[Chronos] scored len=%d features=%d context=%d horizon=%d forecast_calls=%d forecast_batch=%d",
            n,
            n_features,
            self.context_length,
            horizon,
            call_count,
            forecast_batch_size,
        )
        seen = counts > 0
        scores[seen] /= counts[seen]
        if np.any(seen):
            first = int(np.argmax(seen))
            scores[:first] = scores[first]
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, train_data: pd.DataFrame, train_label=None, covariates=None, **kwargs) -> None:
        arr = to_2d_array(train_data)
        self._n_features = arr.shape[1] if arr.ndim == 2 else 0
        if arr.shape[0] <= self.context_length or self._n_features == 0:
            self._pipeline = None
            self._train_scores = None
            self._train_arr = None
            return
        arr = self._scale_fit(arr)
        self._pipeline = self._load_pipeline()
        self._train_arr = arr
        self._train_scores = None

    def detect_score(self, test_data: pd.DataFrame, covariates=None, **kwargs) -> np.ndarray:
        n = len(test_data) if test_data is not None else 0
        if self._pipeline is None or n == 0:
            return np.zeros(n, dtype=float)
        return self._timeline_scores(self._prepare_inference_array(test_data))

    def detect_label(self, test_data: pd.DataFrame, covariates=None, test_label=None, **kwargs):
        n = len(test_data) if test_data is not None else 0
        if self._pipeline is None or n == 0:
            empty = np.zeros(n, dtype=np.int32)
            return {str(r): empty.copy() for r in self.anomaly_ratio}, np.zeros(n, dtype=float)
        scores = self._timeline_scores(self._prepare_inference_array(test_data))
        if self._train_scores is None and self._train_arr is not None:
            self._train_scores = self._timeline_scores(self._train_arr)
        preds = percentile_labels(
            scores,
            self._train_scores,
            self.anomaly_ratio,
            test_label=test_label,
            apply_adjustment=False,
            threshold_test_scores=scores,
        )
        return preds, scores

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        return math.nan

    def estimate_n_params(self) -> float:
        if self._pipeline is None:
            return math.nan
        model = getattr(self._pipeline, "model", None)
        if model is None:
            return math.nan
        return float(sum(p.numel() for p in model.parameters()))

    def estimate_model_size_mb(self) -> float:
        if self._pipeline is None:
            return math.nan
        import torch

        model = getattr(self._pipeline, "model", None)
        if model is None:
            return math.nan
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return float(buf.tell() / (1024.0 * 1024.0))


def make_chronos(**kwargs) -> ChronosModel:
    return ChronosModel(**kwargs)


__all__ = ["ChronosModel", "make_chronos"]
