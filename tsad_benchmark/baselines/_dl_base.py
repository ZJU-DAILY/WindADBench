# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability
from tsad_benchmark.baselines._thresholding import (
    DEFAULT_ANOMALY_RATIOS,
    normalize_anomaly_ratios,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sliding-window dataset
# (train/val/test use step=1; threshold mode uses step=win_size so that
# scores can be flat-concatenated back to the original timeline.)
# ---------------------------------------------------------------------------


def _build_torch_dataset_cls():
    import torch
    from torch.utils.data import Dataset

    class _SegmentDataset(Dataset):
        def __init__(self, data: np.ndarray, win_size: int, mode: str = "train"):
            self.data = np.asarray(data, dtype=np.float32)
            self.win_size = int(win_size)
            self.mode = mode
            n = self.data.shape[0]
            if mode in ("train", "val", "test"):
                # step = 1
                self.length = max(0, n - self.win_size + 1)
            else:  # 'thre' / 'test' — non-overlapping windows
                self.length = max(0, (n - self.win_size) // self.win_size + 1)

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, idx: int) -> "torch.Tensor":
            if self.mode in ("train", "val", "test"):
                start = idx
            else:
                start = idx * self.win_size
                start = min(start, self.data.shape[0] - self.win_size)
            window = self.data[start : start + self.win_size]
            return torch.from_numpy(window)

    return _SegmentDataset


# ---------------------- -----------------------------------------------------
# Early stopping (kept tiny on purpose)
# ---------------------------------------------------------------------------


class _EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: Optional[float] = None
        self.counter = 0
        self.early_stop = False

    def step(self, value: float) -> None:
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class DLBaseModel(AnomalyModelBase):
    """Shared base for deep, self-implemented anomaly detectors."""

    capability = ModelCapability.score_and_label()
    model_name: str = "DLBaseModel"

    def __init__(
        self,
        win_size: int = 100,
        batch_size: int = 128,
        num_epochs: int = 10,
        lr: float = 1e-4,
        val_ratio: float = 0.2,
        patience: int = 3,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
    ) -> None:
        self.win_size = int(win_size)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.lr = float(lr)
        self.val_ratio = float(val_ratio)
        self.patience = int(patience)
        self.anomaly_ratio = self._normalize_anomaly_ratios(anomaly_ratio)
        self.model_hyper_params = {
            "win_size": self.win_size,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "anomaly_ratio": self.anomaly_ratio,
        }
        self._network = None
        self._scaler = None
        self._n_features: int = 0
        self._train_scores: Optional[np.ndarray] = None
        self._device = None  # resolved on fit/inference

    @staticmethod
    def _normalize_anomaly_ratios(
        anomaly_ratio: Optional[float | Sequence[float]],
    ) -> list[float]:
        return normalize_anomaly_ratios(anomaly_ratio)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _build_network(self, n_features: int):
        raise NotImplementedError

    def _train_one_step(self, model, batch_x, criterion, epoch, optimizer) -> float:
        raise NotImplementedError

    def _compute_window_scores(self, model, batch_x, criterion) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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

        if self._device is not None:
            return self._device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def _make_loader(self, arr: np.ndarray, mode: str, shuffle: bool):
        import torch
        from torch.utils.data import DataLoader

        ds_cls = _build_torch_dataset_cls()
        ds = ds_cls(arr, self.win_size, mode=mode)
        if len(ds) == 0:
            return None
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def _scale_fit(self, arr: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        return self._scaler.fit_transform(arr).astype(np.float32)

    def _scale_apply(self, arr: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            return arr.astype(np.float32)
        return self._scaler.transform(arr).astype(np.float32)

    # ------------------------------------------------------------------
    # AnomalyModelBase interface
    # ------------------------------------------------------------------

    def fit(
        self,
        train_data: pd.DataFrame,
        train_label: Optional[pd.DataFrame] = None,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> None:
        import torch

        arr = self._to_2d(train_data)
        n = arr.shape[0]
        if n < self.win_size or arr.shape[1] == 0:
            self._network = None
            self._train_scores = None
            return
        self._n_features = arr.shape[1]
        arr = self._scale_fit(arr)

        # train/val split; validation is cut from the tail to preserve order.
        val_n = int(n * self.val_ratio)
        if val_n >= self.win_size + 1 and (n - val_n) >= self.win_size:
            train_arr, val_arr = arr[: n - val_n], arr[n - val_n :]
        else:
            train_arr, val_arr = arr, None

        train_loader = self._make_loader(train_arr, mode="train", shuffle=True)
        if train_loader is None:
            self._network = None
            self._train_scores = None
            return
        val_loader = (
            self._make_loader(val_arr, mode="val", shuffle=False)
            if val_arr is not None
            else None
        )

        device = self._resolve_device()
        self._network = self._build_network(self._n_features).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.lr)
        early = _EarlyStopping(patience=self.patience)

        for epoch in range(self.num_epochs):
            self._network.train()
            losses = []
            for batch in train_loader:
                batch = batch.to(device)
                loss = self._train_one_step(
                    self._network, batch, criterion, epoch, optimizer
                )
                losses.append(float(loss))
            if val_loader is not None:
                val_loss = self._validate(val_loader, criterion, epoch)
                early.step(val_loss)
                if early.early_stop:
                    logger.debug(
                        "[%s] EarlyStopping at epoch %d (val=%.6f).",
                        self.model_name, epoch + 1, val_loss,
                    )
                    break

        # Cache per-timestep scores on the *training* set — needed at
        # detect_label time for the train+test concat percentile rule.
        self._train_scores = self._inference_scores(train_arr, criterion, mode="train")

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
        criterion = torch.nn.MSELoss()
        scores = self._inference_scores(arr, criterion, mode="thre")
        return self._align_length(scores, n)

    def detect_label(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        n = len(test_data) if test_data is not None else 0
        if self._network is None or n == 0:
            empty = np.zeros(n, dtype=np.int32)
            aux = np.zeros(n, dtype=float)
            return {str(r): empty.copy() for r in self.anomaly_ratio}, aux

        import torch

        arr = self._prepare_inference_array(test_data)
        criterion = torch.nn.MSELoss()
        raw_test_scores = self._inference_scores(arr, criterion, mode="thre")
        test_scores = self._align_length(raw_test_scores, n)
        if raw_test_scores.size == 0:
            empty = np.zeros(n, dtype=np.int32)
            return {str(r): empty.copy() for r in self.anomaly_ratio}, test_scores

        test_threshold_scores = self._inference_scores(arr, criterion, mode="test")
        if test_threshold_scores.size == 0:
            test_threshold_scores = raw_test_scores
        train_scores = self._train_scores
        if train_scores is None or train_scores.size == 0:
            combined = test_threshold_scores
        else:
            combined = np.concatenate([train_scores, test_threshold_scores], axis=0)

        preds = {}
        for ratio in self.anomaly_ratio:
            ratio = float(np.clip(ratio, 0.0, 100.0))
            if ratio <= 0.0:
                labels = np.zeros(raw_test_scores.shape[0], dtype=np.int32)
            elif ratio >= 100.0:
                labels = np.ones(raw_test_scores.shape[0], dtype=np.int32)
            else:
                thresh = float(np.percentile(combined, 100.0 - ratio))
                labels = (raw_test_scores > thresh).astype(np.int32)
            preds[str(ratio)] = self._align_label_length(labels, n)
        return preds, test_scores

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _validate(self, loader, criterion, epoch: int) -> float:
        import torch

        self._network.eval()
        device = self._resolve_device()
        losses = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                # We reuse the same train-step formula but without the
                # backward pass — feasible because subclasses compute
                # per-window scores via _compute_window_scores; the mean
                # of those scores is a sound validation proxy.
                w_scores = self._compute_window_scores(self._network, batch, criterion)
                losses.append(float(np.mean(w_scores)))
        self._network.train()
        return float(np.mean(losses)) if losses else 0.0

    def _inference_scores(self, arr: np.ndarray, criterion, mode: str = "thre") -> np.ndarray:
        import torch

        loader = self._make_loader(arr, mode=mode, shuffle=False)
        if loader is None:
            return np.zeros(arr.shape[0], dtype=float)
        device = self._resolve_device()
        self._network.eval()
        flat: list = []
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for batch in loader:
                batch = batch.to(device)
                w = self._compute_window_scores(self._network, batch, criterion)
                # w: (batch, win_size) — flatten in temporal order.
                w = np.asarray(w, dtype=float).reshape(-1)
                flat.append(w)
        scores = np.concatenate(flat, axis=0) if flat else np.zeros(0)
        if mode == "thre" and self.win_size < arr.shape[0] and 0 < scores.size < arr.shape[0]:
            missing = arr.shape[0] - scores.size
            tail_batch = torch.from_numpy(arr[-self.win_size:]).unsqueeze(0).to(device)
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tail = self._compute_window_scores(self._network, tail_batch, criterion)
            tail = np.asarray(tail, dtype=float).reshape(-1)
            if tail.size:
                scores = np.concatenate([scores, tail[-missing:]], axis=0)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores

    @staticmethod
    def _align_length(scores: np.ndarray, target_len: int) -> np.ndarray:
        if scores.size == target_len:
            return scores
        out = np.empty(target_len, dtype=float)
        if scores.size == 0:
            out.fill(0.0)
            return out
        if scores.size < target_len:
            out[:scores.size] = scores
            out[scores.size:] = 0.0
        else:
            out[:] = scores[:target_len]
        return out

    @staticmethod
    def _align_label_length(labels: np.ndarray, target_len: int) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        if labels.size == target_len:
            return labels
        if labels.size > target_len:
            return labels[:target_len]
        out = np.zeros(target_len, dtype=np.int32)
        out[:labels.size] = labels
        return out

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        if self._network is None:
            return math.nan
        try:
            import torch
            from fvcore.nn import FlopCountAnalysis
        except Exception:
            return math.nan
        n_features = max(int(self._n_features or 0), 1)
        n_test = int(len(test_data)) if test_data is not None else 0
        n_windows = max((n_test - self.win_size) // self.win_size + 1, 0)
        if n_test > self.win_size and n_test % self.win_size:
            n_windows += 1
        if n_windows <= 0:
            return math.nan
        device = self._resolve_device()
        dummy = torch.zeros(1, self.win_size, n_features, device=device)
        was_training = self._network.training
        self._network.eval()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with torch.no_grad():
                    fa = FlopCountAnalysis(self._network, dummy)
                    fa.unsupported_ops_warnings(False)
                    fa.uncalled_modules_warnings(False)
                    fa.tracer_warnings("none")
                    return float(fa.total()) * float(n_windows)
        except Exception:
            return math.nan
        finally:
            if was_training:
                self._network.train()

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
            import io
            import torch
            buf = io.BytesIO()
            torch.save(self._network.state_dict(), buf)
            return float(buf.tell() / (1024.0 * 1024.0))
        except Exception:
            return math.nan
