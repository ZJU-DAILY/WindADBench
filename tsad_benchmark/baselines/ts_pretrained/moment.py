# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import gc
import logging
import math
import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from tsad_benchmark.baselines._official_ad import (
    non_overlapping_timeline_scores,
    percentile_labels,
    to_2d_array,
)
from tsad_benchmark.baselines._thresholding import normalize_anomaly_ratios
from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability

logger = logging.getLogger(__name__)


class MOMENTModel(AnomalyModelBase):
    """MOMENT in official ``task_name='reconstruction'`` mode."""

    model_name = "MOMENT"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        model_id: str = "AutonLab/MOMENT-1-large",
        win_size: int = 512,
        batch_size: int = 16,
        fine_tune_epochs: int = 0,
        fine_tune_batch_size: int = 2,
        fine_tune_lr: float = 1e-4,
        fine_tune_step: int = 1,
        fine_tune_sample_rate: float = 1.0,
        fine_tune_val_ratio: float = 0.2,
        fine_tune_val_sample_rate: float = 1.0,
        fine_tune_patience: int = 3,
        train_score_step: int = 1,
        train_score_sample_rate: float = 1.0,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
        local_files_only: bool = False,
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.win_size = int(win_size)
        self.batch_size = int(batch_size)
        self.fine_tune_epochs = max(0, int(fine_tune_epochs))
        self.fine_tune_batch_size = max(1, int(fine_tune_batch_size))
        self.fine_tune_lr = float(fine_tune_lr)
        self.fine_tune_step = max(1, int(fine_tune_step))
        self.fine_tune_sample_rate = float(np.clip(fine_tune_sample_rate, 0.0, 1.0))
        self.fine_tune_val_ratio = float(np.clip(fine_tune_val_ratio, 0.0, 0.5))
        self.fine_tune_val_sample_rate = float(np.clip(fine_tune_val_sample_rate, 0.0, 1.0))
        self.fine_tune_patience = max(1, int(fine_tune_patience))
        self.train_score_step = max(1, int(train_score_step))
        self.train_score_sample_rate = float(np.clip(train_score_sample_rate, 0.0, 1.0))
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.local_files_only = bool(local_files_only)
        self.device = str(device)
        self.model_hyper_params = {
            "model_id": self.model_id,
            "win_size": self.win_size,
            "batch_size": self.batch_size,
            "fine_tune_epochs": self.fine_tune_epochs,
            "fine_tune_batch_size": self.fine_tune_batch_size,
            "fine_tune_lr": self.fine_tune_lr,
            "fine_tune_step": self.fine_tune_step,
            "fine_tune_sample_rate": self.fine_tune_sample_rate,
            "fine_tune_val_ratio": self.fine_tune_val_ratio,
            "fine_tune_val_sample_rate": self.fine_tune_val_sample_rate,
            "fine_tune_patience": self.fine_tune_patience,
            "train_score_step": self.train_score_step,
            "train_score_sample_rate": self.train_score_sample_rate,
            "anomaly_ratio": self.anomaly_ratio,
            "local_files_only": self.local_files_only,
            "device": self.device,
        }
        self._model = None
        self._scaler = None
        self._device = None
        self._n_features = 0
        self._train_arr = None
        self._train_scores = None

    def _clear_model(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if self._device is not None and self._device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _resolve_device(self):
        import torch

        if self._device is None:
            if self.device.lower() == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(self.device)
                if self._device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("MOMENTModel was configured with device='cuda', but CUDA is not available.")
        return self._device

    def _load_model(self):
        self._clear_model()

        try:
            from momentfm import MOMENTPipeline
        except Exception as exc:
            raise ImportError(
                "MOMENTModel requires the official MOMENT package. Install it with "
                "`pip install git+https://github.com/moment-timeseries-foundation-model/moment.git`."
            ) from exc

        model = MOMENTPipeline.from_pretrained(
            self.model_id,
            model_kwargs={"task_name": "reconstruction", "seq_len": self.win_size},
            local_files_only=self.local_files_only,
        )
        model.init()
        model.to(self._resolve_device())
        model.eval()
        return model

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

    def _window_starts(self, n: int, step: int, sample_rate: float = 1.0) -> np.ndarray:
        if n < self.win_size:
            return np.zeros(0, dtype=np.int64)
        starts = np.arange(0, int(n) - self.win_size + 1, int(step), dtype=np.int64)
        sample_rate = float(np.clip(sample_rate, 0.0, 1.0))
        if 0.0 < sample_rate < 1.0 and starts.size:
            keep = max(1, int(math.ceil(starts.size * sample_rate)))
            idx = np.linspace(0, starts.size - 1, keep, dtype=np.int64)
            starts = starts[np.unique(idx)]
        return starts

    def _window_scores(self, arr: np.ndarray, step: int = 1, sample_rate: float = 1.0) -> np.ndarray:
        import torch

        arr = np.asarray(arr, dtype=np.float32)
        starts = self._window_starts(arr.shape[0], step=step, sample_rate=sample_rate)
        if starts.size == 0:
            return np.zeros((0, self.win_size), dtype=float)
        offsets = np.arange(self.win_size, dtype=np.int64)
        chunks = []
        device = self._resolve_device()
        self._model.eval()
        with torch.no_grad():
            for start in range(0, starts.size, self.batch_size):
                batch_starts = starts[start : start + self.batch_size]
                windows = arr[batch_starts[:, None] + offsets[None, :]]
                batch = torch.from_numpy(windows.transpose(0, 2, 1).astype(np.float32)).to(device)
                output = self._model(x_enc=batch)
                recon = output.reconstruction
                score = torch.mean((batch - recon) ** 2, dim=1)
                chunks.append(score.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, self.win_size))

    def _reconstruction_loss(self, batch):
        import torch

        output = self._model(x_enc=batch)
        recon = output.reconstruction
        return torch.mean((batch - recon) ** 2)

    def _validation_loss(self, arr: np.ndarray) -> float:
        import torch

        starts = self._window_starts(
            arr.shape[0],
            step=self.fine_tune_step,
            sample_rate=self.fine_tune_val_sample_rate,
        )
        if starts.size == 0:
            return math.inf

        device = self._resolve_device()
        offsets = np.arange(self.win_size, dtype=np.int64)
        losses = []
        self._model.eval()
        with torch.no_grad():
            for start in range(0, starts.size, self.fine_tune_batch_size):
                batch_starts = starts[start : start + self.fine_tune_batch_size]
                windows = arr[batch_starts[:, None] + offsets[None, :]]
                batch = torch.from_numpy(windows.transpose(0, 2, 1).astype(np.float32)).to(device)
                losses.append(float(self._reconstruction_loss(batch).detach().cpu()))
        return float(np.mean(losses)) if losses else math.inf

    def _state_dict_cpu(self):
        return {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}

    def _fine_tune(self, train_arr: np.ndarray, valid_arr: Optional[np.ndarray] = None) -> None:
        if self.fine_tune_epochs <= 0:
            return

        import torch

        starts = self._window_starts(
            train_arr.shape[0],
            step=self.fine_tune_step,
            sample_rate=self.fine_tune_sample_rate,
        )
        if starts.size == 0:
            return

        device = self._resolve_device()
        offsets = np.arange(self.win_size, dtype=np.int64)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.fine_tune_lr)
        rng = np.random.default_rng(2026)
        best_loss = math.inf
        best_state = None
        bad_epochs = 0

        self._model.train()
        for _epoch in range(self.fine_tune_epochs):
            epoch_start = time.time()
            epoch_starts = rng.permutation(starts)
            train_loss_sum = 0.0
            train_loss_count = 0
            for start in range(0, epoch_starts.size, self.fine_tune_batch_size):
                batch_starts = epoch_starts[start : start + self.fine_tune_batch_size]
                windows = train_arr[batch_starts[:, None] + offsets[None, :]]
                batch = torch.from_numpy(windows.transpose(0, 2, 1).astype(np.float32)).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = self._reconstruction_loss(batch)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.detach().cpu())
                train_loss_count += 1

            if valid_arr is not None and valid_arr.shape[0] >= self.win_size:
                val_start = time.time()
                valid_loss = self._validation_loss(valid_arr)
                logger.info(
                    "[MOMENT] epoch=%d/%d train_loss=%.6f val_loss=%.6f train_time=%.1fs val_time=%.1fs",
                    _epoch + 1,
                    self.fine_tune_epochs,
                    train_loss_sum / max(train_loss_count, 1),
                    valid_loss,
                    time.time() - epoch_start,
                    time.time() - val_start,
                )
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_state = self._state_dict_cpu()
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.fine_tune_patience:
                        break
                self._model.train()
            else:
                logger.info(
                    "[MOMENT] epoch=%d/%d train_loss=%.6f train_time=%.1fs",
                    _epoch + 1,
                    self.fine_tune_epochs,
                    train_loss_sum / max(train_loss_count, 1),
                    time.time() - epoch_start,
                )

        self._model.eval()
        if best_state is not None:
            self._model.load_state_dict(best_state)
        del optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def fit(self, train_data: pd.DataFrame, train_label=None, covariates=None, **kwargs) -> None:
        arr = to_2d_array(train_data)
        self._n_features = arr.shape[1] if arr.ndim == 2 else 0
        if arr.shape[0] < self.win_size or self._n_features == 0:
            self._model = None
            self._train_arr = None
            self._train_scores = None
            return
        if self.fine_tune_epochs > 0 and self.fine_tune_val_ratio > 0.0:
            split = int(arr.shape[0] * (1.0 - self.fine_tune_val_ratio))
            split = min(max(split, self.win_size), arr.shape[0])
            scaler_fit_arr = arr[:split]
            valid_raw = arr[split:]
        else:
            scaler_fit_arr = arr
            valid_raw = None
        self._scale_fit(scaler_fit_arr)
        arr = self._scale_apply(arr)
        valid_arr = self._scale_apply(valid_raw) if valid_raw is not None and len(valid_raw) else None
        self._model = self._load_model()
        train_arr = arr[: len(scaler_fit_arr)]
        train_windows = self._window_starts(
            train_arr.shape[0],
            step=self.fine_tune_step,
            sample_rate=self.fine_tune_sample_rate,
        ).size
        val_windows = (
            self._window_starts(
                valid_arr.shape[0],
                step=self.fine_tune_step,
                sample_rate=self.fine_tune_val_sample_rate,
            ).size
            if valid_arr is not None
            else 0
        )
        train_score_windows = self._window_starts(
            arr.shape[0],
            step=self.train_score_step,
            sample_rate=self.train_score_sample_rate,
        ).size
        logger.info(
            "[MOMENT] fit windows train=%d val=%d train_score=%d batch=%d ft_batch=%d epochs=%d win=%d features=%d",
            train_windows,
            val_windows,
            train_score_windows,
            self.batch_size,
            self.fine_tune_batch_size,
            self.fine_tune_epochs,
            self.win_size,
            self._n_features,
        )
        self._fine_tune(train_arr, valid_arr=valid_arr)
        self._train_arr = arr
        self._train_scores = None

    def _get_train_scores(self) -> Optional[np.ndarray]:
        if self._train_scores is None and self._train_arr is not None:
            self._train_scores = self._window_scores(
                self._train_arr,
                step=self.train_score_step,
                sample_rate=self.train_score_sample_rate,
            ).reshape(-1)
        return self._train_scores

    def detect_score(self, test_data: pd.DataFrame, covariates=None, **kwargs) -> np.ndarray:
        n = len(test_data) if test_data is not None else 0
        if self._model is None or n == 0:
            return np.zeros(n, dtype=float)
        arr = self._prepare_inference_array(test_data)
        window_scores = self._window_scores(arr, step=self.win_size)
        tail_scores = self._window_scores(arr[-self.win_size :], step=1) if n >= self.win_size else None
        return non_overlapping_timeline_scores(window_scores, n, self.win_size, tail_scores)

    def detect_label(self, test_data: pd.DataFrame, covariates=None, test_label=None, **kwargs):
        n = len(test_data) if test_data is not None else 0
        if self._model is None or n == 0:
            empty = np.zeros(n, dtype=np.int32)
            return {str(r): empty.copy() for r in self.anomaly_ratio}, np.zeros(n)
        arr = self._prepare_inference_array(test_data)
        raw_window_scores = self._window_scores(arr, step=self.win_size)
        tail_scores = self._window_scores(arr[-self.win_size :], step=1) if n >= self.win_size else None
        scores = non_overlapping_timeline_scores(raw_window_scores, n, self.win_size, tail_scores)
        threshold_window_scores = self._window_scores(arr, step=1)
        preds = percentile_labels(
            scores,
            self._get_train_scores(),
            self.anomaly_ratio,
            test_label=test_label,
            apply_adjustment=False,
            threshold_test_scores=threshold_window_scores.reshape(-1),
        )
        return preds, scores

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        return math.nan

    def estimate_n_params(self) -> float:
        if self._model is None:
            return math.nan
        return float(sum(p.numel() for p in self._model.parameters()))

    def estimate_model_size_mb(self) -> float:
        if self._model is None:
            return math.nan
        import torch

        buf = io.BytesIO()
        torch.save(self._model.state_dict(), buf)
        return float(buf.tell() / (1024.0 * 1024.0))


def make_moment(**kwargs) -> MOMENTModel:
    return MOMENTModel(**kwargs)


__all__ = ["MOMENTModel", "make_moment"]
