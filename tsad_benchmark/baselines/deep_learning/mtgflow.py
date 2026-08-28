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
from tsad_benchmark.baselines.deep_learning._models.mtgflow import MTGFlowNetwork
from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability


class MTGFlowModel(AnomalyModelBase):
    """MTGFlow (Zhou et al., AAAI 2023)."""

    model_name = "MTGFlow"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        win_size: int = 60,
        batch_size: int = 64,
        num_epochs: int = 40,
        lr: float = 2e-3,
        val_ratio: float = 0.0,
        hidden_size: int = 32,
        n_blocks: int = 1,
        n_hidden: int = 1,
        input_size: int = 1,
        train_stride: int = 10,
        weight_decay: float = 5e-4,
        grad_clip_value: float = 1.0,
        dropout: float = 0.0,
        batch_norm: bool = False,
        patience: int = 0,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
        seed: int = 0,
    ) -> None:
        self.win_size = int(win_size)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.lr = float(lr)
        self.val_ratio = float(val_ratio)
        self.hidden_size = int(hidden_size)
        self.n_blocks = int(n_blocks)
        self.n_hidden = int(n_hidden)
        self.input_size = int(input_size)
        self.train_stride = max(1, int(train_stride))
        self.weight_decay = float(weight_decay)
        self.grad_clip_value = float(grad_clip_value)
        self.dropout = float(dropout)
        self.batch_norm = bool(batch_norm)
        self.patience = int(patience)
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.seed = int(seed)
        self.model_hyper_params = {
            "win_size": self.win_size,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "val_ratio": self.val_ratio,
            "hidden_size": self.hidden_size,
            "n_blocks": self.n_blocks,
            "n_hidden": self.n_hidden,
            "input_size": self.input_size,
            "train_stride": self.train_stride,
            "weight_decay": self.weight_decay,
            "grad_clip_value": self.grad_clip_value,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
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
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
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

    def _make_loader(self, arr: np.ndarray, shuffle: bool, stride: int = 1):
        import torch
        from torch.utils.data import DataLoader, Dataset

        if arr is None or arr.shape[0] < self.win_size or arr.shape[1] == 0:
            return None

        win_size = self.win_size
        stride = max(1, int(stride))

        class _WindowDataset(Dataset):
            def __init__(self, values: np.ndarray):
                self.values = np.asarray(values, dtype=np.float32)

            def __len__(self):
                n_windows = max(0, int(self.values.shape[0]) - win_size + 1)
                return (n_windows + stride - 1) // stride

            def __getitem__(self, idx):
                start = int(idx) * stride
                window = self.values[start : start + win_size].T[:, :, None]
                return (torch.from_numpy(window.astype(np.float32, copy=False)),)

        return DataLoader(
            _WindowDataset(arr),
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

        loader = self._make_loader(arr, shuffle=False, stride=1)
        if loader is None:
            return np.zeros(0, dtype=float)

        device = self._resolve_device()
        chunks = []
        self._network.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.float().to(device)
                score = self._network.anomaly_scores(batch)
                chunks.append(score.detach().cpu().numpy())
        raw_scores = np.concatenate(chunks, axis=0) if chunks else np.zeros(0)
        return np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_network(self):
        return MTGFlowNetwork(
            n_blocks=self.n_blocks,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_hidden=self.n_hidden,
            window_size=self.win_size,
            n_sensor=self._n_features,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
        )

    def _validate(self, loader) -> float:
        import torch

        if loader is None or self._network is None:
            return math.inf
        device = self._resolve_device()
        losses = []
        self._network.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.float().to(device)
                loss = -self._network(batch)
                losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else math.inf

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

        val_n = int(len(arr) * self.val_ratio)
        if val_n >= self.win_size and len(arr) - val_n >= self.win_size:
            train_arr = arr[: len(arr) - val_n]
            val_arr = arr[len(arr) - val_n :]
        else:
            train_arr = arr
            val_arr = None

        train_loader = self._make_loader(train_arr, shuffle=True, stride=self.train_stride)
        val_loader = (
            self._make_loader(val_arr, shuffle=False, stride=self.train_stride)
            if val_arr is not None
            else None
        )
        if train_loader is None:
            self._network = None
            self._train_scores = None
            return

        device = self._resolve_device()
        self._network = self._build_network().to(device)
        optimizer = torch.optim.Adam(
            [{"params": self._network.parameters(), "weight_decay": self.weight_decay}],
            lr=self.lr,
            weight_decay=0.0,
        )

        best_state = None
        best_loss = math.inf
        stale = 0
        for _epoch in range(self.num_epochs):
            self._network.train()
            for (batch,) in train_loader:
                batch = batch.float().to(device)
                loss = -self._network(batch)
                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_value > 0:
                    torch.nn.utils.clip_grad_value_(
                        self._network.parameters(),
                        self.grad_clip_value,
                    )
                optimizer.step()

            if val_loader is not None:
                val_loss = self._validate(val_loader)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self._network.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
                    if self.patience > 0 and stale >= self.patience:
                        break

        if best_state is not None:
            self._network.load_state_dict(best_state)
            self._network.to(device)

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


def make_mtgflow(**kwargs) -> MTGFlowModel:
    return MTGFlowModel(**kwargs)


__all__ = ["MTGFlowModel", "make_mtgflow"]
