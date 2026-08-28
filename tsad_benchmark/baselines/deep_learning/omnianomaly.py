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
from tsad_benchmark.baselines.deep_learning._models.omnianomaly import (
    OmniAnomalyNetwork,
)
from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability


class OmniAnomalyModel(AnomalyModelBase):
    

    model_name = "OmniAnomaly"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        win_size: int = 100,
        batch_size: int = 128,
        num_epochs: int = 10,
        lr: float = 1e-3,
        val_ratio: float = 0.2,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        n_layers: int = 2,
        beta: float = 0.01,
        patience: int = 3,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
        seed: int = 0,
    ) -> None:
        self.win_size = int(win_size)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.lr = float(lr)
        self.val_ratio = float(val_ratio)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.n_layers = int(n_layers)
        self.beta = float(beta)
        self.patience = int(patience)
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.seed = int(seed)
        self.model_hyper_params = {
            "win_size": self.win_size,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "val_ratio": self.val_ratio,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "n_layers": self.n_layers,
            "beta": self.beta,
            "patience": self.patience,
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
        dataset = TensorDataset(torch.from_numpy(windows))
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
        left = (self.win_size - 1) // 2
        right_start = left + scores.size
        usable = min(scores.size, max(target_len - left, 0))
        if usable > 0:
            out[left : left + usable] = scores[:usable]
            if left > 0:
                out[:left] = scores[0]
            if right_start < target_len:
                out[right_start:] = scores[min(scores.size - 1, usable - 1)]
        return out

    def _align_window_labels(self, labels: np.ndarray, target_len: int) -> np.ndarray:
        out = np.zeros(target_len, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        if labels.size == 0:
            return out
        left = (self.win_size - 1) // 2
        right_start = left + labels.size
        usable = min(labels.size, max(target_len - left, 0))
        if usable > 0:
            out[left : left + usable] = labels[:usable]
            if left > 0:
                out[:left] = labels[0]
            if right_start < target_len:
                out[right_start:] = labels[min(labels.size - 1, usable - 1)]
        return out

    def _epoch_loss(self, loader, criterion):
        import torch

        if loader is None:
            return math.inf
        losses = []
        self._network.eval()
        hidden = None
        with torch.no_grad():
            for idx, (batch,) in enumerate(loader):
                batch = batch.float().to(self._resolve_device())
                loss, hidden = self._network.loss(
                    batch,
                    criterion,
                    hidden if idx else None,
                )
                losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else math.inf

    def _raw_scores_from_scaled_array(self, arr: np.ndarray) -> np.ndarray:
        import torch

        windows = self._make_windows(arr)
        loader = self._make_loader(windows, shuffle=False)
        if loader is None:
            return np.zeros(0, dtype=float)

        criterion = torch.nn.MSELoss(reduction="none")
        chunks = []
        hidden = None
        self._network.eval()
        with torch.no_grad():
            for idx, (batch,) in enumerate(loader):
                batch = batch.float().to(self._resolve_device())
                scores, hidden = self._network.window_scores(
                    batch,
                    criterion,
                    hidden if idx else None,
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
        self._network = OmniAnomalyNetwork(
            n_features=self._n_features,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            n_layers=self.n_layers,
            beta=self.beta,
        ).to(device)
        criterion = torch.nn.MSELoss(reduction="none")
        optimizer = torch.optim.AdamW(self._network.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        best_state = None
        best_loss = math.inf
        stale = 0
        for _epoch in range(self.num_epochs):
            self._network.train()
            hidden = None
            for idx, (batch,) in enumerate(train_loader):
                batch = batch.float().to(device)
                optimizer.zero_grad()
                loss, hidden = self._network.loss(batch, criterion, hidden if idx else None)
                loss.backward()
                optimizer.step()
            scheduler.step()

            val_loss = self._epoch_loss(val_loader, criterion)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self._network.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self._network.load_state_dict(best_state)

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


def make_omnianomaly(**kwargs) -> OmniAnomalyModel:
    return OmniAnomalyModel(**kwargs)


__all__ = ["OmniAnomalyModel", "make_omnianomaly"]
