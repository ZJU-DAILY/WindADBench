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
from tsad_benchmark.baselines.deep_learning._models.mtad_gat import MTADGATNetwork
from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability


class MTADGATModel(AnomalyModelBase):
   

    model_name = "MTAD-GAT"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        win_size: int = 100,
        batch_size: int = 64,
        num_epochs: int = 5,
        lr: float = 1e-3,
        val_ratio: float = 0.2,
        kernel_size: int = 7,
        use_gatv2: bool = True,
        gru_hid_dim: int = 150,
        gru_n_layers: int = 1,
        forecast_hid_dim: int = 150,
        forecast_n_layers: int = 3,
        recon_hid_dim: int = 150,
        recon_n_layers: int = 1,
        dropout: float = 0.3,
        alpha: float = 0.2,
        gamma: float = 1.0,
        max_features: Optional[int] = 256,
        max_gat_pair_gib: float = 8.0,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
        seed: int = 0,
    ) -> None:
        self.win_size = int(win_size)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.lr = float(lr)
        self.val_ratio = float(val_ratio)
        self.kernel_size = int(kernel_size)
        self.use_gatv2 = bool(use_gatv2)
        self.gru_hid_dim = int(gru_hid_dim)
        self.gru_n_layers = int(gru_n_layers)
        self.forecast_hid_dim = int(forecast_hid_dim)
        self.forecast_n_layers = int(forecast_n_layers)
        self.recon_hid_dim = int(recon_hid_dim)
        self.recon_n_layers = int(recon_n_layers)
        self.dropout = float(dropout)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.max_features = None if max_features is None else int(max_features)
        if self.max_features is not None and self.max_features <= 0:
            self.max_features = None
        self.max_gat_pair_gib = float(max_gat_pair_gib)
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.seed = int(seed)
        self.model_hyper_params = {
            "win_size": self.win_size,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "val_ratio": self.val_ratio,
            "kernel_size": self.kernel_size,
            "use_gatv2": self.use_gatv2,
            "gru_hid_dim": self.gru_hid_dim,
            "gru_n_layers": self.gru_n_layers,
            "forecast_hid_dim": self.forecast_hid_dim,
            "forecast_n_layers": self.forecast_n_layers,
            "recon_hid_dim": self.recon_hid_dim,
            "recon_n_layers": self.recon_n_layers,
            "dropout": self.dropout,
            "gamma": self.gamma,
            "max_features": self.max_features,
            "max_gat_pair_gib": self.max_gat_pair_gib,
            "anomaly_ratio": self.anomaly_ratio,
            "seed": self.seed,
        }
        self._network = None
        self._scaler = None
        self._n_features = 0
        self._device = None
        self._train_scores = None
        self._selected_columns = None
        self._selected_indices = None

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

    def _select_fit_array(self, data) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            df = data
            n_cols = df.shape[1]
            if self.max_features is None or n_cols <= self.max_features:
                self._selected_columns = list(df.columns)
                self._selected_indices = None
                return self._to_2d(df)

            avg_cols = [c for c in df.columns if str(c).endswith("_avg")]
            candidate_cols = avg_cols if avg_cols else list(df.columns)
            if len(candidate_cols) > self.max_features:
                variances = df[candidate_cols].var(axis=0).fillna(0.0)
                candidate_cols = (
                    variances.sort_values(ascending=False)
                    .head(self.max_features)
                    .index
                    .tolist()
                )
            self._selected_columns = list(candidate_cols)
            self._selected_indices = None
            return self._to_2d(df.loc[:, self._selected_columns])

        arr = self._to_2d(data)
        n_cols = arr.shape[1]
        if self.max_features is None or n_cols <= self.max_features:
            self._selected_columns = None
            self._selected_indices = None
            return arr

        variances = np.nan_to_num(np.var(arr, axis=0), nan=0.0)
        self._selected_indices = np.argsort(variances)[-self.max_features :][::-1]
        self._selected_columns = None
        return arr[:, self._selected_indices]

    def _prepare_inference_array(self, data: pd.DataFrame) -> np.ndarray:
        if isinstance(data, pd.DataFrame) and self._selected_columns is not None:
            arr = np.zeros((len(data), len(self._selected_columns)), dtype=np.float32)
            for i, col in enumerate(self._selected_columns):
                if col in data.columns:
                    arr[:, i] = data[col].to_numpy(dtype=np.float32, copy=False)
        else:
            arr = self._to_2d(data)
            if self._selected_indices is not None:
                usable = self._selected_indices[self._selected_indices < arr.shape[1]]
                arr = arr[:, usable]
        if arr.shape[1] < self._n_features:
            pad = np.zeros(
                (arr.shape[0], self._n_features - arr.shape[1]),
                dtype=np.float32,
            )
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > self._n_features:
            arr = arr[:, : self._n_features]
        return self._scale_apply(arr)

    def _effective_batch_size(self) -> int:
        batch = max(1, int(self.batch_size))
        if self.max_gat_pair_gib <= 0 or self._n_features <= 0:
            return batch
        max_bytes = self.max_gat_pair_gib * (1024.0 ** 3)
        pair_bytes_per_item = self._n_features * self._n_features * (2 * self.win_size) * 4
        if pair_bytes_per_item <= 0:
            return batch
        capped = max(1, int(max_bytes // pair_bytes_per_item))
        return min(batch, capped)

    def _make_xy(self, arr: np.ndarray):
        if arr is None or arr.shape[0] <= self.win_size or arr.shape[1] == 0:
            empty_x = np.zeros((0, self.win_size, max(self._n_features, 1)), dtype=np.float32)
            empty_y = np.zeros((0, max(self._n_features, 1)), dtype=np.float32)
            return empty_x, empty_y
        starts = np.arange(arr.shape[0] - self.win_size)
        offsets = np.arange(self.win_size)
        x = arr[starts[:, None] + offsets[None, :]]
        y = arr[starts + self.win_size]
        return x.astype(np.float32), y.astype(np.float32)

    def _make_loader(self, x: np.ndarray, y: np.ndarray, shuffle: bool):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if x is None or y is None or len(x) == 0:
            return None
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        return DataLoader(
            dataset,
            batch_size=self._effective_batch_size(),
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def _build_network(self):
        return MTADGATNetwork(
            n_features=self._n_features,
            window_size=self.win_size,
            out_dim=self._n_features,
            kernel_size=self.kernel_size,
            use_gatv2=self.use_gatv2,
            gru_n_layers=self.gru_n_layers,
            gru_hid_dim=self.gru_hid_dim,
            forecast_n_layers=self.forecast_n_layers,
            forecast_hid_dim=self.forecast_hid_dim,
            recon_n_layers=self.recon_n_layers,
            recon_hid_dim=self.recon_hid_dim,
            dropout=self.dropout,
            alpha=self.alpha,
        )

    def _align_scores(self, scores: np.ndarray, target_len: int) -> np.ndarray:
        out = np.zeros(target_len, dtype=float)
        if scores.size == 0:
            return out
        start = min(self.win_size, target_len)
        usable = min(scores.size, max(target_len - start, 0))
        if usable > 0:
            out[start : start + usable] = scores[:usable]
        return out

    def _align_labels(self, labels: np.ndarray, target_len: int) -> np.ndarray:
        out = np.zeros(target_len, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        if labels.size == 0:
            return out
        start = min(self.win_size, target_len)
        usable = min(labels.size, max(target_len - start, 0))
        if usable > 0:
            out[start : start + usable] = labels[:usable]
        return out

    def _raw_scores_from_scaled_array(self, arr: np.ndarray) -> np.ndarray:
        import torch

        x, y = self._make_xy(arr)
        loader = self._make_loader(x, y, shuffle=False)
        if loader is None:
            return np.zeros(0, dtype=float)

        device = self._resolve_device()
        chunks = []
        self._network.eval()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                preds, recons = self._network(batch_x)
                recon_last = recons[:, -1, :]
                score = (batch_y - preds) ** 2 + self.gamma * (batch_y - recon_last) ** 2
                chunks.append(score.mean(dim=1).detach().cpu().numpy())
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

        arr = self._select_fit_array(train_data)
        if arr.shape[0] <= self.win_size or arr.shape[1] == 0:
            self._network = None
            self._train_scores = None
            return
        self._n_features = arr.shape[1]
        arr = self._scale_fit(arr)
        x, y = self._make_xy(arr)
        if len(x) == 0:
            self._network = None
            self._train_scores = None
            return

        val_n = int(len(x) * self.val_ratio)
        if val_n > 0 and len(x) - val_n > 0:
            train_x, val_x = x[: len(x) - val_n], x[len(x) - val_n :]
            train_y, val_y = y[: len(y) - val_n], y[len(y) - val_n :]
        else:
            train_x, val_x = x, x
            train_y, val_y = y, y

        train_loader = self._make_loader(train_x, train_y, shuffle=True)
        val_loader = self._make_loader(val_x, val_y, shuffle=False)
        if train_loader is None:
            self._network = None
            self._train_scores = None
            return

        device = self._resolve_device()
        self._network = self._build_network().to(device)
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.lr)

        for _epoch in range(self.num_epochs):
            self._network.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                preds, recons = self._network(batch_x)
                forecast_loss = torch.sqrt(torch.mean((batch_y - preds) ** 2))
                recon_loss = torch.sqrt(torch.mean((batch_x - recons) ** 2))
                loss = forecast_loss + recon_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                self._network.eval()
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.float().to(device)
                        batch_y = batch_y.float().to(device)
                        preds, recons = self._network(batch_x)
                        _ = torch.sqrt(torch.mean((batch_y - preds) ** 2)) + torch.sqrt(
                            torch.mean((batch_x - recons) ** 2)
                        )

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
        return self._align_scores(raw_scores, n)

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
        test_scores = self._align_scores(raw_scores, n)
        raw_preds = percentile_label_maps(raw_scores, self._train_scores, self.anomaly_ratio)
        preds = {ratio: self._align_labels(labels, n) for ratio, labels in raw_preds.items()}
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


def make_mtad_gat(**kwargs) -> MTADGATModel:
    return MTADGATModel(**kwargs)


__all__ = ["MTADGATModel", "make_mtad_gat"]
