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
from tsad_benchmark.baselines.deep_learning._models.sarad import SAR76
from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability


class SARADModel(AnomalyModelBase):
    """SARAD/SAR76 (Dai et al., NeurIPS 2024)."""

    model_name = "SARAD"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        win_size: int = 100,
        batch_size: int = 16,
        train_stride: int = 8,
        num_epochs: int = 3,
        lr: float = 7e-4,
        val_ratio: float = 0.0,
        model_size: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        num_patches: int = 2,
        detector_size: int = 64,
        dropout: float = 0.1,
        detec_weight: float = 100.0,
        scheduler_step_size: int = 1,
        scheduler_gamma: float = 0.5,
        is_diagonal_masked: bool = False,
        patience: int = 0,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
        seed: int = 0,
    ) -> None:
        self.win_size = int(win_size)
        self.batch_size = int(batch_size)
        self.train_stride = max(1, int(train_stride))
        self.num_epochs = int(num_epochs)
        self.lr = float(lr)
        self.val_ratio = float(val_ratio)
        self.model_size = int(model_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.num_patches = int(num_patches)
        self.detector_size = int(detector_size)
        self.dropout = float(dropout)
        self.detec_weight = float(detec_weight)
        self.scheduler_step_size = int(scheduler_step_size)
        self.scheduler_gamma = float(scheduler_gamma)
        self.is_diagonal_masked = bool(is_diagonal_masked)
        self.patience = int(patience)
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.seed = int(seed)
        if self.win_size % self.num_patches != 0:
            raise ValueError("win_size must be divisible by num_patches for SARAD patching")
        self.model_hyper_params = {
            "win_size": self.win_size,
            "batch_size": self.batch_size,
            "train_stride": self.train_stride,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "val_ratio": self.val_ratio,
            "model_size": self.model_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_patches": self.num_patches,
            "detector_size": self.detector_size,
            "dropout": self.dropout,
            "detec_weight": self.detec_weight,
            "scheduler_step_size": self.scheduler_step_size,
            "scheduler_gamma": self.scheduler_gamma,
            "is_diagonal_masked": self.is_diagonal_masked,
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

        if arr is None or arr.shape[0] == 0 or arr.shape[1] == 0:
            return None

        win_size = self.win_size
        stride = max(1, int(stride))

        class _WindowDataset(Dataset):
            def __init__(self, values: np.ndarray):
                self.values = np.asarray(values, dtype=np.float32)

            def __len__(self):
                n_windows = int(self.values.shape[0])
                return (n_windows + stride - 1) // stride

            def __getitem__(self, idx):
                idx = int(idx) * stride
                if idx < win_size:
                    pad_len = win_size - idx - 1
                    if pad_len > 0:
                        prefix = np.repeat(self.values[[0]], pad_len, axis=0)
                        window = np.concatenate((prefix, self.values[: idx + 1]), axis=0)
                    else:
                        window = self.values[: idx + 1]
                else:
                    window = self.values[idx - win_size + 1 : idx + 1]
                return torch.from_numpy(window.astype(np.float32, copy=False)), torch.tensor(0.0)

        return DataLoader(
            _WindowDataset(arr),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def _build_network(self):
        return SAR76(
            input_size=self._n_features,
            window_size=self.win_size,
            model_size=self.model_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_patches=self.num_patches,
            detector_size=self.detector_size,
            dropout=self.dropout,
            is_diagnoal_masked=self.is_diagonal_masked,
        )

    def _model_step(self, batch):
        import torch

        if self._network is None:
            raise RuntimeError("SARAD network is not initialized")
        x_hat, _s, q, q_bar = self._network(batch)
        recon_diagno = torch.mean((x_hat - batch) ** 2, dim=1)
        recon_losses = recon_diagno.mean(dim=-1)
        detec_diagno = torch.mean((q - q_bar) ** 2, dim=tuple(range(1, q.dim() - 1)))
        detec_losses = detec_diagno.mean(dim=-1)
        scores = (
            (recon_losses - self._network.recon_avg) / self._network.recon_std
            + (detec_losses - self._network.detec_avg) / self._network.detec_std
        )
        loss = recon_losses.mean() + self.detec_weight * detec_losses.mean()
        return {
            "loss": loss,
            "recon_losses": recon_losses,
            "detec_losses": detec_losses,
            "recon_diagno": recon_diagno,
            "detec_diagno": detec_diagno,
            "scores": scores,
        }

    def _update_validation_stats(self, loader) -> float:
        import torch

        if loader is None or self._network is None:
            return math.inf
        device = self._resolve_device()
        losses = []
        recon_scores = []
        detec_scores = []
        recon_diagno = []
        detec_diagno = []
        self._network.eval()
        with torch.no_grad():
            for batch, _labels in loader:
                batch = batch.float().to(device)
                out = self._model_step(batch)
                losses.append(float(out["loss"].detach().cpu().item()))
                recon_scores.append(out["recon_losses"].detach().cpu().numpy())
                detec_scores.append(out["detec_losses"].detach().cpu().numpy())
                recon_diagno.append(out["recon_diagno"].detach().cpu().numpy())
                detec_diagno.append(out["detec_diagno"].detach().cpu().numpy())

        if recon_scores:
            recon_scores_np = np.concatenate(recon_scores)
            detec_scores_np = np.concatenate(detec_scores)
            recon_diagno_np = np.concatenate(recon_diagno)
            detec_diagno_np = np.concatenate(detec_diagno)
            eps = 1e-6
            self._network.recon_avg.copy_(
                torch.tensor(float(recon_scores_np.mean()), dtype=torch.float32, device=device)
            )
            self._network.recon_std.copy_(
                torch.tensor(float(max(recon_scores_np.std(), eps)), dtype=torch.float32, device=device)
            )
            self._network.detec_avg.copy_(
                torch.tensor(float(detec_scores_np.mean()), dtype=torch.float32, device=device)
            )
            self._network.detec_std.copy_(
                torch.tensor(float(max(detec_scores_np.std(), eps)), dtype=torch.float32, device=device)
            )
            self._network.rdiag_avg.copy_(
                torch.tensor(recon_diagno_np.mean(0), dtype=torch.float32, device=device)
            )
            self._network.rdiag_std.copy_(
                torch.tensor(
                    np.maximum(recon_diagno_np.std(0), eps),
                    dtype=torch.float32,
                    device=device,
                )
            )
            self._network.ddiag_avg.copy_(
                torch.tensor(detec_diagno_np.mean(0), dtype=torch.float32, device=device)
            )
            self._network.ddiag_std.copy_(
                torch.tensor(
                    np.maximum(detec_diagno_np.std(0), eps),
                    dtype=torch.float32,
                    device=device,
                )
            )
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
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            self._network = None
            self._train_scores = None
            return
        self._n_features = arr.shape[1]
        arr = self._scale_fit(arr)

        val_n = int(len(arr) * self.val_ratio)
        if val_n > 0 and len(arr) - val_n > 0:
            train_arr = arr[: len(arr) - val_n]
            val_arr = arr[len(arr) - val_n :]
        else:
            train_arr = arr
            val_arr = arr

        train_loader = self._make_loader(train_arr, shuffle=True, stride=self.train_stride)
        val_loader = self._make_loader(val_arr, shuffle=False, stride=self.train_stride)
        if train_loader is None:
            self._network = None
            self._train_scores = None
            return

        device = self._resolve_device()
        self._network = self._build_network().to(device)
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(self.scheduler_step_size, 1),
            gamma=self.scheduler_gamma,
        )

        best_state = None
        best_loss = math.inf
        stale = 0
        for _epoch in range(self.num_epochs):
            self._network.train()
            for batch, _labels in train_loader:
                batch = batch.float().to(device)
                out = self._model_step(batch)
                optimizer.zero_grad()
                out["loss"].backward()
                optimizer.step()

            val_loss = self._update_validation_stats(val_loader)
            scheduler.step()
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

        self._train_scores = self.detect_score(train_data)

    def detect_score(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        import torch

        n = len(test_data) if test_data is not None else 0
        if self._network is None or n == 0 or self._n_features == 0:
            return np.zeros(n, dtype=float)
        arr = self._prepare_inference_array(test_data)
        loader = self._make_loader(arr, shuffle=False, stride=1)
        if loader is None:
            return np.zeros(n, dtype=float)

        device = self._resolve_device()
        chunks = []
        self._network.eval()
        with torch.no_grad():
            for batch, _labels in loader:
                batch = batch.float().to(device)
                out = self._model_step(batch)
                chunks.append(out["scores"].detach().cpu().numpy())
        raw_scores = np.concatenate(chunks, axis=0) if chunks else np.zeros(0)
        raw_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
        if raw_scores.shape[0] != n:
            out = np.zeros(n, dtype=float)
            usable = min(n, raw_scores.shape[0])
            out[:usable] = raw_scores[:usable]
            return out
        return raw_scores.astype(float)

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

        test_scores = self.detect_score(test_data, covariates=covariates, **kwargs)
        preds = percentile_label_maps(test_scores, self._train_scores, self.anomaly_ratio)
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


def make_sarad(**kwargs) -> SARADModel:
    return SARADModel(**kwargs)


__all__ = ["SARADModel", "make_sarad"]
