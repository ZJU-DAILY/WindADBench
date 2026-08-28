# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import math
import random
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from tsad_benchmark.baselines._thresholding import (
    normalize_anomaly_ratios,
    percentile_label_maps,
)
from tsad_benchmark.baselines.deep_learning._models.usad import USADNetwork, evaluate
from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability


class USADModel(AnomalyModelBase):
    """USAD (Audibert et al., KDD 2020)."""

    model_name = "USAD"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        win_size: int = 12,
        batch_size: int = 128,
        num_epochs: int = 10,
        lr: float = 1e-3,
        val_ratio: float = 0.2,
        hidden_dim: int = 100,
        score_alpha: float = 0.5,
        score_beta: float = 0.5,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
        seed: int = 0,
    ) -> None:
        self.win_size = int(win_size)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.lr = float(lr)
        self.val_ratio = float(val_ratio)
        self.hidden_dim = int(hidden_dim)
        self.score_alpha = float(score_alpha)
        self.score_beta = float(score_beta)
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.seed = int(seed)
        self.model_hyper_params = {
            "win_size": self.win_size,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "val_ratio": self.val_ratio,
            "hidden_dim": self.hidden_dim,
            "score_alpha": self.score_alpha,
            "score_beta": self.score_beta,
            "anomaly_ratio": self.anomaly_ratio,
            "seed": self.seed,
        }
        self._network = None
        self._scaler = None
        self._n_features = 0
        self._device = None
        self._train_scores = None

    @staticmethod
    def _to_2d(df) -> np.ndarray:
        if isinstance(df, pd.DataFrame):
            arr = df.values
        else:
            arr = np.asarray(df)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.size and not np.all(np.isfinite(arr)):
            mask = ~np.isfinite(arr)
            col_mean = np.nanmean(np.where(mask, np.nan, arr), axis=0)
            col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
            arr = np.where(mask, col_mean, arr).astype(np.float32)
        return arr

    def _resolve_device(self):
        import torch

        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def _scale_fit(self, arr: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import MinMaxScaler

        self._scaler = MinMaxScaler()
        return self._scaler.fit_transform(arr).astype(np.float32)

    def _scale_apply(self, arr: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            return arr.astype(np.float32)
        return self._scaler.transform(arr).astype(np.float32)

    def _prepare_inference_array(self, data: pd.DataFrame) -> np.ndarray:
        arr = self._to_2d(data)
        if arr.shape[1] < self._n_features:
            pad = np.zeros(
                (arr.shape[0], self._n_features - arr.shape[1]),
                dtype=np.float32,
            )
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > self._n_features:
            arr = arr[:, : self._n_features]
        return self._scale_apply(arr)

    def _make_windows(self, arr: np.ndarray) -> np.ndarray:
        if arr is None or arr.shape[0] < self.win_size or arr.shape[1] == 0:
            return np.zeros((0, self.win_size, max(self._n_features, 1)), dtype=np.float32)
        starts = np.arange(arr.shape[0] - self.win_size + 1)
        offsets = np.arange(self.win_size)
        return arr[starts[:, None] + offsets[None, :]].astype(np.float32)

    def _make_loader(self, windows: np.ndarray, shuffle: bool):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if windows is None or len(windows) == 0:
            return None
        flat = windows.reshape(windows.shape[0], -1).astype(np.float32)
        dataset = TensorDataset(torch.from_numpy(flat))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def _align_window_scores(self, scores: np.ndarray, target_len: int) -> np.ndarray:
        out = np.zeros(target_len, dtype=float)
        if scores.size == 0:
            return out
        start = min(max(self.win_size - 1, 0), target_len)
        usable = min(scores.size, max(target_len - start, 0))
        if usable > 0:
            out[start : start + usable] = scores[:usable]
        return out

    def _align_window_labels(self, labels: np.ndarray, target_len: int) -> np.ndarray:
        out = np.zeros(target_len, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        if labels.size == 0:
            return out
        start = min(max(self.win_size - 1, 0), target_len)
        usable = min(labels.size, max(target_len - start, 0))
        if usable > 0:
            out[start : start + usable] = labels[:usable]
        return out

    def _raw_scores_from_scaled_array(self, arr: np.ndarray) -> np.ndarray:
        import torch

        windows = self._make_windows(arr)
        loader = self._make_loader(windows, shuffle=False)
        if loader is None:
            return np.zeros(0, dtype=float)

        device = self._resolve_device()
        chunks = []
        self._network.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.float().to(device)
                scores = self._network.window_scores(
                    batch,
                    alpha=self.score_alpha,
                    beta=self.score_beta,
                )
                chunks.append(scores.detach().cpu().numpy())
        raw_scores = np.concatenate(chunks, axis=0) if chunks else np.zeros(0)
        return np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(
        self,
        train_data: pd.DataFrame,
        train_label: Optional[pd.DataFrame] = None,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> None:
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        arr = self._to_2d(train_data)
        if arr.shape[0] < self.win_size or arr.shape[1] == 0:
            self._network = None
            self._train_scores = None
            return
        self._n_features = arr.shape[1]
        arr = self._scale_fit(arr)
        windows = self._make_windows(arr)
        if len(windows) == 0:
            self._network = None
            self._train_scores = None
            return

        val_n = int(len(windows) * self.val_ratio)
        if val_n > 0 and len(windows) - val_n > 0:
            train_windows = windows[: len(windows) - val_n]
            val_windows = windows[len(windows) - val_n :]
        else:
            train_windows = windows
            val_windows = windows

        train_loader = self._make_loader(train_windows, shuffle=True)
        val_loader = self._make_loader(val_windows, shuffle=False)
        if train_loader is None:
            self._network = None
            self._train_scores = None
            return

        device = self._resolve_device()
        w_size = self.win_size * self._n_features
        z_size = self.win_size * max(self.hidden_dim, 1)
        self._network = USADNetwork(w_size=w_size, z_size=z_size).to(device)
        optimizer1 = torch.optim.Adam(
            list(self._network.encoder.parameters())
            + list(self._network.decoder1.parameters()),
            lr=self.lr,
        )
        optimizer2 = torch.optim.Adam(
            list(self._network.encoder.parameters())
            + list(self._network.decoder2.parameters()),
            lr=self.lr,
        )

        for epoch in range(self.num_epochs):
            self._network.train()
            n = epoch + 1
            for (batch,) in train_loader:
                batch = batch.float().to(device)
                loss1, _loss2 = self._network.training_step(batch, n)
                loss1.backward()
                optimizer1.step()
                optimizer1.zero_grad()

                _loss1, loss2 = self._network.training_step(batch, n)
                loss2.backward()
                optimizer2.step()
                optimizer2.zero_grad()

            if val_loader is not None:
                self._network.eval()
                evaluate(self._network, val_loader, n, device)

        self._train_scores = self._raw_scores_from_scaled_array(arr)

    def detect_score(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        n = len(test_data) if test_data is not None else 0
        if self._network is None or n == 0 or self._n_features == 0:
            return np.zeros(n, dtype=float)
        arr = self._prepare_inference_array(test_data)
        raw_scores = self._raw_scores_from_scaled_array(arr)
        return self._align_window_scores(raw_scores, n)

    def detect_label(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ):
        n = len(test_data) if test_data is not None else 0
        if self._network is None or n == 0:
            empty = np.zeros(n, dtype=np.int32)
            aux = np.zeros(n, dtype=float)
            return {str(r): empty.copy() for r in self.anomaly_ratio}, aux

        arr = self._prepare_inference_array(test_data)
        raw_scores = self._raw_scores_from_scaled_array(arr)
        test_scores = self._align_window_scores(raw_scores, n)
        raw_preds = percentile_label_maps(raw_scores, self._train_scores, self.anomaly_ratio)
        preds = {
            ratio: self._align_window_labels(labels, n)
            for ratio, labels in raw_preds.items()
        }
        return preds, test_scores

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        return math.nan

    def estimate_n_params(self) -> float:
        if self._network is None:
            return math.nan
        try:
            return float(sum(p.numel() for p in self._network.parameters()))
        except Exception:
            return math.nan

    def estimate_model_size_mb(self) -> float:
        if self._network is None:
            return math.nan
        try:
            import torch

            buf = io.BytesIO()
            torch.save(self._network.state_dict(), buf)
            return float(buf.tell() / (1024.0 * 1024.0))
        except Exception:
            return math.nan


def make_usad(**kwargs) -> USADModel:
    return USADModel(**kwargs)


__all__ = ["USADModel", "make_usad"]
