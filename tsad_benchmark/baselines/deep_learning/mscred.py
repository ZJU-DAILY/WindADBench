# -*- coding: utf-8 -*-


from __future__ import annotations

from collections import OrderedDict
import io
import logging
import math
import random
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from tsad_benchmark.baselines._thresholding import (
    normalize_anomaly_ratios,
    percentile_label_maps,
)
from tsad_benchmark.baselines.deep_learning._models.mscred import MSCREDNetwork
from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability

logger = logging.getLogger(__name__)

_STAT_SUFFIXES = {
    "avg": "_avg",
    "max": "_max",
    "min": "_min",
    "std": "_std",
}


def _normalise_int_list(value, fallback: Sequence[int]) -> list[int]:
    if value is None:
        return [int(v) for v in fallback]
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [int(value)]
    if isinstance(value, str):
        pieces = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
        return [int(float(p)) for p in pieces] if pieces else [int(v) for v in fallback]
    return [int(v) for v in value]


def _normalise_stat_list(value) -> list[str]:
    if value is None:
        return ["avg"]
    if isinstance(value, str):
        pieces = [p.strip().lower() for p in value.replace(";", ",").split(",") if p.strip()]
    else:
        pieces = [str(p).strip().lower() for p in value]
    valid = [p for p in pieces if p in _STAT_SUFFIXES]
    return valid or ["avg"]


class MSCREDModel(AnomalyModelBase):
    """MSCRED PyTorch port of the mainstream TensorFlow implementation."""

    model_name = "MSCRED"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        win_size: Optional[int | Sequence[int]] = None,
        signature_scales: Optional[Sequence[int]] = None,
        step_max: int = 5,
        time_steps: Optional[int] = None,
        gap_time: int = 10,
        batch_size: int = 16,
        num_epochs: int = 5,
        lr: float = 2e-4,
        learning_rate: Optional[float] = None,
        val_ratio: float = 0.2,
        patience: int = 3,
        feature_mode: str = "sensor_avg",
        sensor_stats: Optional[str | Sequence[str]] = None,
        max_sensors: Optional[int] = None,
        max_features: Optional[int] = None,
        signature_cache_max_mb: float = 768.0,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
        seed: int = 0,
    ) -> None:
        # Upstream defaults: win_size=[10, 30, 60], step_max=5,
        # gap_time=10, learning_rate=0.0002.
        scale_source = signature_scales if signature_scales is not None else win_size
        self.signature_scales = _normalise_int_list(scale_source, fallback=[10, 30, 60])
        self.win_size = list(self.signature_scales)
        self.step_max = int(step_max if time_steps is None else time_steps)
        self.time_steps = self.step_max
        self.gap_time = max(1, int(gap_time))
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.lr = float(lr if learning_rate is None else learning_rate)
        self.val_ratio = float(val_ratio)
        self.patience = int(patience)
        self.feature_mode = str(feature_mode or "sensor_avg").lower()
        self.sensor_stats = _normalise_stat_list(sensor_stats)
        sensor_cap = max_sensors if max_sensors is not None else max_features
        self.max_sensors = None if sensor_cap is None else int(sensor_cap)
        if self.max_sensors is not None and self.max_sensors <= 0:
            self.max_sensors = None
        self.signature_cache_max_mb = float(signature_cache_max_mb)
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.seed = int(seed)

        self.model_hyper_params = {
            "source": "7fantasysz/MSCRED@4bdfcacf (PyTorch port)",
            "win_size": self.win_size,
            "signature_scales": self.signature_scales,
            "step_max": self.step_max,
            "gap_time": self.gap_time,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "val_ratio": self.val_ratio,
            "patience": self.patience,
            "feature_mode": self.feature_mode,
            "sensor_stats": self.sensor_stats,
            "max_sensors": self.max_sensors,
            "signature_cache_max_mb": self.signature_cache_max_mb,
            "anomaly_ratio": self.anomaly_ratio,
            "seed": self.seed,
        }

        self._network = None
        self._device = None
        self._train_scores = None
        self._scale_min = None
        self._scale_den = None
        self._n_features = 0
        self._n_stat_channels = 0
        self._sensor_groups = None
        self._selected_indices = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_stat_suffix(col: str) -> tuple[str, Optional[str]]:
        name = str(col)
        for stat, suffix in _STAT_SUFFIXES.items():
            if name.endswith(suffix):
                return name[: -len(suffix)], stat
        return name, None

    @staticmethod
    def _to_2d(data) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            arr = data.values
        else:
            arr = np.asarray(data)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.size and not np.all(np.isfinite(arr)):
            mask = ~np.isfinite(arr)
            col_mean = np.nanmean(np.where(mask, np.nan, arr), axis=0)
            col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
            arr = np.where(mask, col_mean, arr).astype(np.float32)
        return arr

    @staticmethod
    def _finite_3d(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size and not np.all(np.isfinite(arr)):
            mask = ~np.isfinite(arr)
            mean = np.nanmean(np.where(mask, np.nan, arr), axis=0)
            mean = np.where(np.isnan(mean), 0.0, mean)
            arr = np.where(mask, mean, arr).astype(np.float32)
        return arr

    def _build_sensor_groups(self, df: pd.DataFrame) -> list[dict[str, list[str]]]:
        by_base: OrderedDict[str, dict[str, object]] = OrderedDict()
        for col in df.columns:
            base, stat = self._strip_stat_suffix(str(col))
            entry = by_base.setdefault(base, {"all": [], "raw": None})
            entry["all"].append(str(col))
            if stat is None:
                entry["raw"] = str(col)
            else:
                entry[stat] = str(col)

        bases = list(by_base.keys())
        if self.max_sensors is not None and len(bases) > self.max_sensors:
            scores = []
            for base in bases:
                columns = self._columns_for_stat(by_base[base], "avg")
                present = [c for c in columns if c in df.columns]
                if present:
                    values = df[present].astype(float)
                    score = float(values.mean(axis=1).var())
                else:
                    score = 0.0
                scores.append((score, base))
            bases = [b for _score, b in sorted(scores, reverse=True)[: self.max_sensors]]

        groups: list[dict[str, list[str]]] = []
        for base in bases:
            entry = by_base[base]
            group = {
                stat: self._columns_for_stat(entry, stat)
                for stat in self.sensor_stats
            }
            groups.append(group)
        return groups

    @staticmethod
    def _columns_for_stat(entry: dict[str, object], stat: str) -> list[str]:
        if stat in entry:
            return [str(entry[stat])]
        if stat == "avg" and entry.get("raw") is not None:
            return [str(entry["raw"])]
        if "avg" in entry:
            return [str(entry["avg"])]
        if entry.get("raw") is not None:
            return [str(entry["raw"])]
        return [str(c) for c in entry.get("all", [])]

    def _dataframe_to_sensor_array(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        if fit or self._sensor_groups is None:
            self._sensor_groups = self._build_sensor_groups(df)
        groups = self._sensor_groups or []
        n = len(df)
        arr = np.zeros((n, len(self.sensor_stats), len(groups)), dtype=np.float32)
        for sensor_idx, group in enumerate(groups):
            for stat_idx, stat in enumerate(self.sensor_stats):
                cols = [c for c in group.get(stat, []) if c in df.columns]
                if not cols:
                    continue
                if len(cols) == 1:
                    values = df[cols[0]].to_numpy(dtype=np.float32, copy=False)
                else:
                    values = df[cols].astype(np.float32).mean(axis=1).to_numpy()
                arr[:, stat_idx, sensor_idx] = values
        return self._finite_3d(arr)

    def _fit_array(self, data) -> np.ndarray:
        if isinstance(data, pd.DataFrame) and self.feature_mode != "raw":
            arr = self._dataframe_to_sensor_array(data, fit=True)
            self._selected_indices = None
            return arr

        arr2 = self._to_2d(data)
        if self.max_sensors is not None and arr2.shape[1] > self.max_sensors:
            variances = np.nan_to_num(np.var(arr2, axis=0), nan=0.0)
            self._selected_indices = np.argsort(variances)[-self.max_sensors :][::-1]
            arr2 = arr2[:, self._selected_indices]
        else:
            self._selected_indices = None
        return arr2[:, None, :].astype(np.float32)

    def _inference_array(self, data) -> np.ndarray:
        if isinstance(data, pd.DataFrame) and self._sensor_groups is not None:
            arr = self._dataframe_to_sensor_array(data, fit=False)
        else:
            arr2 = self._to_2d(data)
            if self._selected_indices is not None:
                usable = self._selected_indices[self._selected_indices < arr2.shape[1]]
                arr2 = arr2[:, usable]
            arr = arr2[:, None, :].astype(np.float32)

        if arr.shape[2] < self._n_features:
            pad = np.zeros(
                (arr.shape[0], arr.shape[1], self._n_features - arr.shape[2]),
                dtype=np.float32,
            )
            arr = np.concatenate([arr, pad], axis=2)
        elif arr.shape[2] > self._n_features:
            arr = arr[:, :, : self._n_features]
        return self._scale_apply(arr)

    def _scale_fit(self, arr: np.ndarray) -> np.ndarray:
        self._scale_min = np.nanmin(arr, axis=0, keepdims=True)
        max_value = np.nanmax(arr, axis=0, keepdims=True)
        self._scale_den = np.where(max_value > self._scale_min, max_value - self._scale_min, 1.0)
        return self._scale_apply(arr)

    def _scale_apply(self, arr: np.ndarray) -> np.ndarray:
        arr = self._finite_3d(arr)
        if self._scale_min is None or self._scale_den is None:
            return arr.astype(np.float32)
        arr = (arr - self._scale_min) / (self._scale_den + 1e-6)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Signature matrix generation
    # ------------------------------------------------------------------

    def _resolve_device(self):
        import torch

        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    @property
    def _n_signature_channels(self) -> int:
        return int(self._n_stat_channels * len(self.signature_scales))

    def _first_target_end(self) -> int:
        return int(max(self.signature_scales) - 1 + (self.step_max - 1) * self.gap_time)

    def _make_target_ends(self, arr: np.ndarray, target_stride: int = 1) -> np.ndarray:
        if arr is None or arr.shape[0] <= self._first_target_end() or arr.shape[2] == 0:
            return np.zeros(0, dtype=np.int64)
        step = self.gap_time * max(1, int(target_stride))
        return np.arange(self._first_target_end(), arr.shape[0], step, dtype=np.int64)

    def _signature_at(self, arr: np.ndarray, end_idx: int) -> np.ndarray:
        mats = []
        for stat_idx in range(arr.shape[1]):
            for scale in self.signature_scales:
                start = max(0, int(end_idx) - int(scale) + 1)
                seg = arr[start : int(end_idx) + 1, stat_idx, :]
                if seg.shape[0] == 0:
                    mat = np.zeros((self._n_features, self._n_features), dtype=np.float32)
                else:
                    mat = np.matmul(seg.T, seg) / float(max(seg.shape[0], 1))
                mats.append(mat.astype(np.float32))
        return np.stack(mats, axis=0)

    def _signature_cache_estimate_mb(self, n_signatures: int) -> float:
        bytes_per = self._n_signature_channels * self._n_features * self._n_features * 4
        return float(n_signatures * bytes_per / (1024.0 * 1024.0))

    def _make_signatures_for_ends(self, arr: np.ndarray, ends: np.ndarray) -> np.ndarray:
        ends = np.asarray(ends, dtype=np.int64)
        if ends.size == 0:
            return np.zeros(
                (0, self._n_signature_channels, self._n_features, self._n_features),
                dtype=np.float32,
            )

        max_end = int(ends[-1])
        buffers = np.zeros(
            (
                self._n_stat_channels,
                len(self.signature_scales),
                self._n_features,
                self._n_features,
            ),
            dtype=np.float32,
        )
        signatures = np.empty(
            (ends.size, self._n_signature_channels, self._n_features, self._n_features),
            dtype=np.float32,
        )
        out_idx = 0
        for t in range(max_end + 1):
            rows = arr[t]
            for stat_idx in range(self._n_stat_channels):
                current_outer = np.outer(rows[stat_idx], rows[stat_idx]).astype(np.float32, copy=False)
                for scale_idx, scale in enumerate(self.signature_scales):
                    buffers[stat_idx, scale_idx] += current_outer
                    old_idx = t - int(scale)
                    if old_idx >= 0:
                        old = arr[old_idx, stat_idx]
                        buffers[stat_idx, scale_idx] -= np.outer(old, old).astype(np.float32, copy=False)

            while out_idx < ends.size and int(ends[out_idx]) == t:
                channel = 0
                denom_base = t + 1
                for stat_idx in range(self._n_stat_channels):
                    for scale_idx, scale in enumerate(self.signature_scales):
                        denom = float(min(int(scale), denom_base))
                        signatures[out_idx, channel] = buffers[stat_idx, scale_idx] / denom
                        channel += 1
                out_idx += 1

        return signatures

    def _sequence_offsets(self) -> np.ndarray:
        return -np.arange(self.step_max - 1, -1, -1, dtype=np.int64) * self.gap_time

    def _make_loader_from_signature_cache(self, signatures, sequence_indices, shuffle: bool):
        import torch
        from torch.utils.data import DataLoader, Dataset

        if signatures is None or len(signatures) == 0 or len(sequence_indices) == 0:
            return None

        class _SignatureCacheDataset(Dataset):
            def __init__(self, sig: np.ndarray, seq_indices: np.ndarray):
                self.signatures = sig
                self.sequence_indices = seq_indices

            def __len__(self) -> int:
                return int(self.sequence_indices.shape[0])

            def __getitem__(self, idx: int):
                indices = self.sequence_indices[int(idx)]
                x = self.signatures[indices]
                y = self.signatures[indices[-1]]
                return torch.from_numpy(x), torch.from_numpy(y)

        return DataLoader(
            _SignatureCacheDataset(signatures, np.asarray(sequence_indices, dtype=np.int64)),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def _make_cached_loader_from_targets(self, arr: np.ndarray, target_ends: np.ndarray, shuffle: bool):
        if self.signature_cache_max_mb <= 0 or target_ends.size == 0:
            return None

        needed_ends = target_ends[:, None] + self._sequence_offsets()[None, :]
        unique_ends = np.unique(needed_ends.reshape(-1))
        estimated_mb = self._signature_cache_estimate_mb(int(unique_ends.size))
        if estimated_mb > self.signature_cache_max_mb:
            return None

        signatures = self._make_signatures_for_ends(arr, unique_ends)
        sequence_indices = np.searchsorted(unique_ends, needed_ends).astype(np.int64)
        return self._make_loader_from_signature_cache(signatures, sequence_indices, shuffle)

    def _make_loader_from_targets(self, arr: np.ndarray, target_ends: np.ndarray, shuffle: bool):
        import torch
        from torch.utils.data import DataLoader, Dataset

        target_ends = np.asarray(target_ends, dtype=np.int64)
        if arr is None or arr.shape[0] == 0 or len(target_ends) == 0:
            return None

        cached_loader = self._make_cached_loader_from_targets(arr, target_ends, shuffle)
        if cached_loader is not None:
            return cached_loader

        outer = self

        class _SignatureDataset(Dataset):
            def __init__(self, values: np.ndarray, ends: np.ndarray):
                self.values = values
                self.ends = ends

            def __len__(self) -> int:
                return int(self.ends.size)

            def __getitem__(self, idx: int):
                target_end = int(self.ends[int(idx)])
                seq = np.empty(
                    (
                        outer.step_max,
                        outer._n_signature_channels,
                        outer._n_features,
                        outer._n_features,
                    ),
                    dtype=np.float32,
                )
                for pos, offset in enumerate(outer._sequence_offsets()):
                    seq[pos] = outer._signature_at(self.values, target_end + int(offset))
                return torch.from_numpy(seq), torch.from_numpy(seq[-1])

        return DataLoader(
            _SignatureDataset(arr, target_ends),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    @staticmethod
    def _expand_step_values(values: np.ndarray, ends: np.ndarray, target_len: int, dtype=float) -> np.ndarray:
        out = np.zeros(int(target_len), dtype=dtype)
        values = np.asarray(values).reshape(-1)
        ends = np.asarray(ends, dtype=np.int64).reshape(-1)
        usable = min(values.size, ends.size)
        if usable == 0:
            return out
        values = values[:usable]
        ends = ends[:usable]
        first = int(max(0, min(ends[0], target_len - 1)))
        out[: first + 1] = values[0]
        prev = first
        for value, end in zip(values[1:], ends[1:]):
            end = int(max(0, min(end, target_len - 1)))
            if end >= prev + 1:
                out[prev + 1 : end + 1] = value
            prev = end
        if prev + 1 < target_len:
            out[prev + 1 :] = values[-1]
        return out

    def _raw_scores_from_scaled_array(self, arr: np.ndarray, stride: int = 1):
        import torch

        target_ends = self._make_target_ends(arr, target_stride=stride)
        if target_ends.size == 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=np.int64)

        loader = self._make_loader_from_targets(arr, target_ends, shuffle=False)
        if loader is None:
            return np.zeros(0, dtype=float), target_ends

        chunks = []
        self._network.eval()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.float().to(self._resolve_device())
                batch_y = batch_y.float().to(self._resolve_device())
                recon = self._network(batch_x)
                scores = self._network.reconstruction_scores(recon, batch_y)
                chunks.append(scores.detach().cpu().numpy())
        raw_scores = np.concatenate(chunks, axis=0) if chunks else np.zeros(0)
        raw_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
        return raw_scores, target_ends

    # ------------------------------------------------------------------
    # Benchmark interface
    # ------------------------------------------------------------------

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
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        arr = self._fit_array(train_data)
        if arr.shape[0] <= self._first_target_end() or arr.shape[2] == 0:
            self._network = None
            self._train_scores = None
            return

        self._n_stat_channels = int(arr.shape[1])
        self._n_features = int(arr.shape[2])
        arr = self._scale_fit(arr)

        target_ends = self._make_target_ends(arr, target_stride=1)
        if len(target_ends) == 0:
            self._network = None
            self._train_scores = None
            return

        val_n = int(len(target_ends) * self.val_ratio)
        if val_n > 0 and len(target_ends) - val_n > 0:
            train_ends = target_ends[: len(target_ends) - val_n]
            val_ends = target_ends[len(target_ends) - val_n :]
        else:
            train_ends = target_ends
            val_ends = target_ends

        train_loader = self._make_loader_from_targets(arr, train_ends, shuffle=True)
        val_loader = self._make_loader_from_targets(arr, val_ends, shuffle=False)
        if train_loader is None:
            self._network = None
            self._train_scores = None
            return

        logger.info(
            "[MSCRED] source=7fantasysz/MSCRED pytorch-port points=%d sensors=%d "
            "stats=%s channels=%d sequences=%d gap_time=%d batch=%d",
            arr.shape[0],
            self._n_features,
            ",".join(self.sensor_stats),
            self._n_signature_channels,
            len(target_ends),
            self.gap_time,
            self.batch_size,
        )

        device = self._resolve_device()
        self._network = MSCREDNetwork(
            n_scales=self._n_signature_channels,
            step_max=self.step_max,
        ).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.lr)

        best_state = None
        best_loss = math.inf
        stale = 0
        for _epoch in range(self.num_epochs):
            self._network.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                recon = self._network(batch_x)
                loss = criterion(recon, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_loss = self._validate(val_loader, criterion)
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

        self._train_scores, _train_ends = self._raw_scores_from_scaled_array(
            arr,
            stride=1,
        )

    def _validate(self, loader, criterion) -> float:
        import torch

        if loader is None:
            return math.inf
        losses = []
        self._network.eval()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.float().to(self._resolve_device())
                batch_y = batch_y.float().to(self._resolve_device())
                losses.append(float(criterion(self._network(batch_x), batch_y).detach().cpu()))
        return float(np.mean(losses)) if losses else math.inf

    def detect_score(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        n = len(test_data) if test_data is not None else 0
        if self._network is None or n == 0 or self._n_features == 0:
            return np.zeros(n, dtype=float)
        arr = self._inference_array(test_data)
        raw_scores, ends = self._raw_scores_from_scaled_array(arr, stride=1)
        return self._expand_step_values(raw_scores, ends, n, dtype=float)

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

        arr = self._inference_array(test_data)
        raw_scores, ends = self._raw_scores_from_scaled_array(arr, stride=1)
        test_scores = self._expand_step_values(raw_scores, ends, n, dtype=float)
        raw_preds = percentile_label_maps(raw_scores, self._train_scores, self.anomaly_ratio)
        preds = {
            ratio: self._expand_step_values(labels, ends, n, dtype=np.int32)
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


def make_mscred(**kwargs) -> MSCREDModel:
    return MSCREDModel(**kwargs)


__all__ = ["MSCREDModel", "make_mscred"]
