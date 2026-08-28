# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import logging
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tsad_benchmark.baselines._official_ad import (
    non_overlapping_timeline_scores,
    percentile_labels,
    rolling_windows,
    to_2d_array,
)
from tsad_benchmark.baselines._thresholding import normalize_anomaly_ratios
from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability


logger = logging.getLogger(__name__)


class _GPT4TSNetwork:
    @staticmethod
    def build(configs):
        import torch
        import torch.nn as nn
        from einops import rearrange

        from transformers.models.gpt2.modeling_gpt2 import GPT2Model

        class Model(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.seq_len = int(cfg.seq_len)
                self.segment_len = int(cfg.segment_len)
                self.d_ff = int(cfg.d_ff)
                self.c_out = int(cfg.c_out)
                if self.seq_len % self.segment_len != 0:
                    raise ValueError(
                        "Official GPT4TS anomaly_detection uses fixed segment "
                        f"length {self.segment_len}; win_size/seq_len must be divisible by it."
                    )
                self.gpt2 = GPT2Model.from_pretrained(
                    cfg.backbone,
                    output_attentions=False,
                    output_hidden_states=False,
                    local_files_only=bool(cfg.local_files_only),
                )
                self.gpt2.h = self.gpt2.h[: int(cfg.gpt_layers)]
                for name, param in self.gpt2.named_parameters():
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    elif "mlp" in name and int(cfg.mlp) == 1:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                self.out_layer = nn.Linear(self.d_ff, self.c_out, bias=True)

            def forward(self, x_enc, *_args, **_kwargs):
                bsz, length, n_features = x_enc.shape
                seg = self.segment_len
                x = rearrange(x_enc, "b (n s) m -> b n s m", s=seg)
                means = x.mean(2, keepdim=True).detach()
                x = x - means
                stdev = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + 1e-5)
                x = x / stdev
                x = rearrange(x, "b n s m -> b (n s) m")

                if n_features > 768:
                    raise ValueError("Official GPT4TS pads features to GPT-2 width 768; enc_in must be <= 768.")
                enc_out = torch.nn.functional.pad(x, (0, 768 - n_features))
                outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state
                outputs = outputs[:, :, : self.d_ff]
                dec_out = self.out_layer(outputs)

                dec_out = rearrange(dec_out, "b (n s) m -> b n s m", s=seg)
                dec_out = dec_out * stdev[:, :, 0, :].unsqueeze(2).repeat(1, 1, seg, 1)
                dec_out = dec_out + means[:, :, 0, :].unsqueeze(2).repeat(1, 1, seg, 1)
                return rearrange(dec_out, "b n s m -> b (n s) m")

        return Model(configs)


class GPT4TSModel(AnomalyModelBase):
    """GPT4TS / One-Fits-All anomaly-detection model."""

    model_name = "GPT4TS"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        backbone: str = "gpt2",
        win_size: int = 100,
        batch_size: int = 64,
        num_epochs: int = 20,
        lr: float = 1e-4,
        val_ratio: float = 0.2,
        patience: int = 3,
        lradj: str = "type1",
        sampling_rate: float = 0.05,
        sampling_strategy: str = "uniform",
        anomaly_ratio=None,
        gpt_layers: int = 3,
        d_ff: int = 32,
        mlp: int = 0,
        segment_len: int = 25,
        local_files_only: bool = False,
        device: str = "auto",
        max_features: int | None = None,
        feature_select: str = "none",
        channel_independent: bool = False,
        channel_chunk_size: int = 32,
        score_step: int = 1,
    ) -> None:
        self.win_size = int(win_size)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.lr = float(lr)
        self.val_ratio = float(val_ratio)
        self.patience = int(patience)
        self.lradj = str(lradj)
        self.sampling_rate = float(sampling_rate)
        self.sampling_strategy = str(sampling_strategy)
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.backbone = backbone
        self.gpt_layers = int(gpt_layers)
        self.d_ff = int(d_ff)
        self.mlp = int(mlp)
        self.segment_len = int(segment_len)
        self.local_files_only = bool(local_files_only)
        self.device = str(device)
        self.max_features = None if max_features is None else int(max_features)
        if self.max_features is not None and self.max_features <= 0:
            self.max_features = None
        self.feature_select = str(feature_select)
        self.channel_independent = bool(channel_independent)
        self.channel_chunk_size = max(1, int(channel_chunk_size))
        self.score_step = max(1, int(score_step))
        self.model_hyper_params = {
            "win_size": self.win_size,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "val_ratio": self.val_ratio,
            "patience": self.patience,
            "lradj": self.lradj,
            "sampling_rate": self.sampling_rate,
            "sampling_strategy": self.sampling_strategy,
            "anomaly_ratio": self.anomaly_ratio,
            "backbone": self.backbone,
            "gpt_layers": self.gpt_layers,
            "d_ff": self.d_ff,
            "mlp": self.mlp,
            "segment_len": self.segment_len,
            "local_files_only": self.local_files_only,
            "device": self.device,
            "max_features": self.max_features,
            "feature_select": self.feature_select,
            "channel_independent": self.channel_independent,
            "channel_chunk_size": self.channel_chunk_size,
            "score_step": self.score_step,
        }
        self._model = None
        self._scaler = None
        self._device = None
        self._selected_indices = None
        self._n_features = 0
        self._train_scores = None

    def _resolve_device(self):
        import torch

        if self._device is None:
            if self.device.lower() == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(self.device)
                if self._device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("GPT4TSModel was configured with device='cuda', but CUDA is not available.")
        return self._device

    def _adjust_learning_rate(self, optimizer, epoch: int) -> None:
        if self.lradj != "type1":
            return
        lr = self.lr * (0.5 ** ((int(epoch) - 1) // 1))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _build_network(self, n_features: int):
        configs = SimpleNamespace(
            backbone=self.backbone,
            seq_len=self.win_size,
            c_out=n_features,
            d_ff=self.d_ff,
            gpt_layers=self.gpt_layers,
            mlp=self.mlp,
            segment_len=self.segment_len,
            local_files_only=self.local_files_only,
        )
        return _GPT4TSNetwork.build(configs)

    def _sample_window_starts(self, length: int) -> np.ndarray:
        n_windows = max(0, int(length) - self.win_size + 1)
        starts = np.arange(n_windows, dtype=np.int64)
        if starts.size == 0:
            return starts
        if 0.0 < self.sampling_rate < 1.0:
            sample_n = max(1, int(starts.size * self.sampling_rate))
            if self.sampling_strategy == "uniform" and sample_n < starts.size:
                return starts[np.linspace(0, starts.size - 1, num=sample_n, dtype=np.int64)]
            return starts[:sample_n]
        return starts

    def _make_channel_chunk_loader(self, arr: np.ndarray, starts: np.ndarray, shuffle: bool):
        import torch
        from torch.utils.data import DataLoader, Dataset

        win_size = self.win_size
        chunk_size = self.channel_chunk_size
        n_features = int(arr.shape[1])
        n_chunks = (n_features + chunk_size - 1) // chunk_size

        class _ChannelChunkDataset(Dataset):
            def __init__(self, values: np.ndarray, window_starts: np.ndarray):
                self.values = values
                self.starts = np.asarray(window_starts, dtype=np.int64)

            def __len__(self):
                return int(self.starts.size) * n_chunks

            def __getitem__(self, idx):
                start_idx = int(idx) // n_chunks
                chunk_idx = int(idx) % n_chunks
                start = int(self.starts[start_idx])
                c0 = chunk_idx * chunk_size
                c1 = min(c0 + chunk_size, n_features)
                width = c1 - c0
                window = np.zeros((chunk_size, win_size, 1), dtype=np.float32)
                mask = np.zeros(chunk_size, dtype=bool)
                if width > 0:
                    window[:width, :, 0] = self.values[start : start + win_size, c0:c1].T
                    mask[:width] = True
                return torch.from_numpy(window), torch.from_numpy(mask)

        if starts.size == 0 or n_features == 0:
            return None
        return DataLoader(
            _ChannelChunkDataset(arr, starts),
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

    def _select_fit_features(self, arr: np.ndarray) -> np.ndarray:
        self._selected_indices = None
        if self.max_features is None or arr.shape[1] <= self.max_features:
            return arr.astype(np.float32)
        if self.feature_select != "train_variance":
            raise ValueError(
                "GPT4TS got more features than GPT-2 width. Set "
                "feature_select='train_variance' with max_features<=768, "
                "or enable channel_independent=True."
            )
        variances = np.nanvar(arr.astype(np.float64), axis=0)
        variances = np.nan_to_num(variances, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        self._selected_indices = np.argsort(variances)[-self.max_features :][::-1]
        logger.info(
            "GPT4TS train-only feature selection: %d -> %d features by variance",
            arr.shape[1],
            len(self._selected_indices),
        )
        return arr[:, self._selected_indices].astype(np.float32)

    def _select_apply_features(self, arr: np.ndarray) -> np.ndarray:
        if self._selected_indices is None:
            return arr.astype(np.float32)
        out = np.zeros((arr.shape[0], len(self._selected_indices)), dtype=np.float32)
        valid = self._selected_indices < arr.shape[1]
        if np.any(valid):
            out[:, valid] = arr[:, self._selected_indices[valid]]
        return out

    def _prepare_inference_array(self, data) -> np.ndarray:
        arr = to_2d_array(data)
        arr = self._select_apply_features(arr)
        if arr.shape[1] < self._n_features:
            pad = np.zeros((arr.shape[0], self._n_features - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > self._n_features:
            arr = arr[:, : self._n_features]
        return self._scale_apply(arr)

    def _channel_batch_loss(self, batch, mask, criterion):
        batch = batch.float().to(self._resolve_device())
        mask = mask.bool().to(self._resolve_device())
        flat = batch.reshape(-1, self.win_size, 1)
        flat_mask = mask.reshape(-1)
        if not bool(flat_mask.any()):
            return None, 0
        flat = flat[flat_mask]
        outputs = self._model(flat)
        return criterion(outputs, flat), int(flat.shape[0])

    def _evaluate_channel_loss(self, loader, criterion) -> float:
        import torch

        if loader is None:
            return math.nan
        losses = []
        weights = []
        self._model.eval()
        with torch.no_grad():
            for batch, mask in loader:
                loss, n = self._channel_batch_loss(batch, mask, criterion)
                if loss is not None and n > 0:
                    losses.append(float(loss.detach().cpu()) * n)
                    weights.append(n)
        return float(np.sum(losses) / max(np.sum(weights), 1)) if weights else math.nan

    def _channel_independent_window_scores(self, arr: np.ndarray, step: int = 1) -> np.ndarray:
        import torch

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < self.win_size or arr.shape[1] == 0:
            return np.zeros((0, self.win_size), dtype=float)

        starts = np.arange(0, arr.shape[0] - self.win_size + 1, max(1, int(step)), dtype=np.int64)
        if starts.size == 0:
            return np.zeros((0, self.win_size), dtype=float)

        n_features = int(arr.shape[1])
        chunk_size = self.channel_chunk_size
        offsets = np.arange(self.win_size, dtype=np.int64)
        out = np.zeros((starts.size, self.win_size), dtype=np.float64)
        device = self._resolve_device()

        self._model.eval()
        with torch.no_grad():
            for b0 in range(0, starts.size, self.batch_size):
                batch_starts = starts[b0 : b0 + self.batch_size]
                batch_sum = np.zeros((batch_starts.size, self.win_size), dtype=np.float64)
                row_index = batch_starts[:, None] + offsets[None, :]
                for c0 in range(0, n_features, chunk_size):
                    c1 = min(c0 + chunk_size, n_features)
                    chunk = arr[row_index, c0:c1]  # (B, L, C)
                    width = int(c1 - c0)
                    flat = (
                        np.transpose(chunk, (0, 2, 1))
                        .reshape(batch_starts.size * width, self.win_size, 1)
                        .astype(np.float32, copy=False)
                    )
                    batch = torch.from_numpy(flat).float().to(device)
                    outputs = self._model(batch)
                    score = torch.mean((batch - outputs) ** 2, dim=-1)
                    score = score.reshape(batch_starts.size, width, self.win_size)
                    batch_sum += score.detach().cpu().numpy().sum(axis=1)
                out[b0 : b0 + batch_starts.size] = batch_sum / max(n_features, 1)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _window_scores(self, arr: np.ndarray, step: int = 1) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self.channel_independent:
            return self._channel_independent_window_scores(arr, step=step)

        windows = rolling_windows(arr, self.win_size, step=step)
        if len(windows) == 0:
            return np.zeros((0, self.win_size), dtype=float)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(windows.astype(np.float32))),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        chunks = []
        device = self._resolve_device()
        self._model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.float().to(device)
                outputs = self._model(batch)
                score = torch.mean((batch - outputs) ** 2, dim=-1)
                chunks.append(score.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, self.win_size))

    def fit(
        self,
        train_data: pd.DataFrame,
        train_label=None,
        covariates=None,
        **kwargs,
    ) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        arr = to_2d_array(train_data)
        if arr.shape[0] < self.win_size or arr.shape[1] == 0:
            self._model = None
            self._train_scores = None
            return
        arr = self._select_fit_features(arr)
        self._n_features = arr.shape[1]
        arr = self._scale_fit(arr)

        val_n = int(arr.shape[0] * self.val_ratio)
        if val_n >= self.win_size + 1 and (arr.shape[0] - val_n) >= self.win_size:
            train_arr, val_arr = arr[: arr.shape[0] - val_n], arr[arr.shape[0] - val_n :]
        else:
            train_arr, val_arr = arr, None

        device = self._resolve_device()
        model_features = 1 if self.channel_independent else self._n_features
        self._model = self._build_network(model_features).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        best_val = None
        best_state = None
        bad_epochs = 0

        if self.channel_independent:
            train_starts = self._sample_window_starts(len(train_arr))
            if train_starts.size == 0:
                self._model = None
                self._train_scores = None
                return
            val_starts = (
                self._sample_window_starts(len(val_arr))
                if val_arr is not None
                else np.zeros(0, dtype=np.int64)
            )
            train_loader = self._make_channel_chunk_loader(train_arr, train_starts, shuffle=True)
            val_loader = (
                self._make_channel_chunk_loader(val_arr, val_starts, shuffle=False)
                if val_arr is not None and val_starts.size > 0
                else None
            )
            n_chunks = (self._n_features + self.channel_chunk_size - 1) // self.channel_chunk_size
            logger.info(
                "GPT4TS fit start: mode=channel_independent train_len=%d val_len=%d "
                "n_features=%d sampled_train_windows=%d sampled_val_windows=%d "
                "channel_chunk_size=%d channel_chunks=%d epochs=%d batch_size=%d score_step=%d",
                len(train_arr),
                0 if val_arr is None else len(val_arr),
                self._n_features,
                int(train_starts.size),
                int(val_starts.size),
                self.channel_chunk_size,
                n_chunks,
                self.num_epochs,
                self.batch_size,
                self.score_step,
            )

            for _epoch in range(self.num_epochs):
                losses = []
                weights = []
                self._model.train()
                for batch, mask in train_loader:
                    loss, n = self._channel_batch_loss(batch, mask, criterion)
                    if loss is None or n <= 0:
                        continue
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()) * n)
                    weights.append(n)
                train_loss = float(np.sum(losses) / max(np.sum(weights), 1)) if weights else math.nan
                val_loss = self._evaluate_channel_loss(val_loader, criterion)
                improved = False
                if not math.isnan(val_loss):
                    if best_val is None or val_loss < best_val:
                        best_val = val_loss
                        best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                        bad_epochs = 0
                        improved = True
                    else:
                        bad_epochs += 1
                logger.info(
                    "GPT4TS epoch %d/%d: train_loss=%.6f val_loss=%s lr=%.6g bad_epochs=%d%s",
                    _epoch + 1,
                    self.num_epochs,
                    train_loss,
                    "nan" if math.isnan(val_loss) else f"{val_loss:.6f}",
                    optimizer.param_groups[0]["lr"],
                    bad_epochs,
                    " best" if improved else "",
                )
                if not math.isnan(val_loss) and bad_epochs >= self.patience:
                    logger.info("GPT4TS early stopping at epoch %d", _epoch + 1)
                    break
                self._adjust_learning_rate(optimizer, _epoch + 1)
        else:
            train_windows = rolling_windows(train_arr, self.win_size, step=1)
            if len(train_windows) == 0:
                self._model = None
                self._train_scores = None
                return
            if 0.0 < self.sampling_rate < 1.0:
                sample_n = max(1, int(len(train_windows) * self.sampling_rate))
                if self.sampling_strategy == "uniform" and sample_n < len(train_windows):
                    indices = np.linspace(0, len(train_windows) - 1, num=sample_n, dtype=int)
                    train_windows_for_fit = train_windows[indices]
                else:
                    train_windows_for_fit = train_windows[:sample_n]
            else:
                train_windows_for_fit = train_windows
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_windows_for_fit.astype(np.float32))),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
            val_windows = (
                rolling_windows(val_arr, self.win_size, step=1)
                if val_arr is not None
                else np.zeros((0, self.win_size, self._n_features), dtype=np.float32)
            )
            val_loader = (
                DataLoader(
                    TensorDataset(torch.from_numpy(val_windows.astype(np.float32))),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                    drop_last=False,
                )
                if len(val_windows) > 0
                else None
            )
            logger.info(
                "GPT4TS fit start: mode=channel_mixed train_len=%d val_len=%d "
                "n_features=%d sampled_train_windows=%d val_windows=%d epochs=%d batch_size=%d",
                len(train_arr),
                0 if val_arr is None else len(val_arr),
                self._n_features,
                int(len(train_windows_for_fit)),
                int(len(val_windows)),
                self.num_epochs,
                self.batch_size,
            )
            for _epoch in range(self.num_epochs):
                losses = []
                self._model.train()
                for (batch,) in train_loader:
                    batch = batch.float().to(device)
                    outputs = self._model(batch)
                    loss = criterion(outputs, batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                train_loss = float(np.mean(losses)) if losses else math.nan
                val_loss = math.nan
                improved = False
                if val_loader is not None:
                    val_losses = []
                    self._model.eval()
                    with torch.no_grad():
                        for (batch,) in val_loader:
                            batch = batch.float().to(device)
                            outputs = self._model(batch)
                            val_losses.append(float(criterion(outputs, batch).detach().cpu()))
                    val_loss = float(np.mean(val_losses)) if val_losses else math.nan
                    if best_val is None or val_loss < best_val:
                        best_val = val_loss
                        best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                        bad_epochs = 0
                        improved = True
                    else:
                        bad_epochs += 1
                logger.info(
                    "GPT4TS epoch %d/%d: train_loss=%.6f val_loss=%s lr=%.6g bad_epochs=%d%s",
                    _epoch + 1,
                    self.num_epochs,
                    train_loss,
                    "nan" if math.isnan(val_loss) else f"{val_loss:.6f}",
                    optimizer.param_groups[0]["lr"],
                    bad_epochs,
                    " best" if improved else "",
                )
                if not math.isnan(val_loss) and bad_epochs >= self.patience:
                    logger.info("GPT4TS early stopping at epoch %d", _epoch + 1)
                    break
                self._adjust_learning_rate(optimizer, _epoch + 1)

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._train_scores = self._window_scores(train_arr, step=self.score_step).reshape(-1)

    def detect_score(self, test_data, covariates=None, **kwargs) -> np.ndarray:
        n = len(test_data) if test_data is not None else 0
        if self._model is None or n == 0:
            return np.zeros(n, dtype=float)
        arr = self._prepare_inference_array(test_data)
        window_scores = self._window_scores(arr, step=self.win_size)
        tail_scores = self._window_scores(arr[-self.win_size :], step=1) if n >= self.win_size else None
        return non_overlapping_timeline_scores(window_scores, n, self.win_size, tail_scores)

    def detect_label(self, test_data, covariates=None, test_label=None, **kwargs):
        n = len(test_data) if test_data is not None else 0
        if self._model is None or n == 0:
            empty = np.zeros(n, dtype=np.int32)
            return {str(r): empty.copy() for r in self.anomaly_ratio}, np.zeros(n, dtype=float)
        arr = self._prepare_inference_array(test_data)
        raw_window_scores = self._window_scores(arr, step=self.win_size)
        tail_scores = self._window_scores(arr[-self.win_size :], step=1) if n >= self.win_size else None
        test_scores = non_overlapping_timeline_scores(raw_window_scores, n, self.win_size, tail_scores)
        test_threshold_scores = self._window_scores(arr, step=self.score_step).reshape(-1)
        preds = percentile_labels(
            test_scores,
            self._train_scores,
            self.anomaly_ratio,
            test_label=test_label,
            apply_adjustment=False,
            threshold_test_scores=test_threshold_scores,
        )
        return preds, test_scores

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        return math.nan

    def estimate_n_params(self) -> float:
        if self._model is None:
            return math.nan
        try:
            return float(sum(p.numel() for p in self._model.parameters()))
        except Exception:
            return math.nan

    def estimate_model_size_mb(self) -> float:
        if self._model is None:
            return math.nan
        import torch

        buf = io.BytesIO()
        torch.save(self._model.state_dict(), buf)
        return float(buf.tell() / (1024.0 * 1024.0))


def make_gpt4ts(**kwargs) -> GPT4TSModel:
    return GPT4TSModel(**kwargs)


__all__ = ["GPT4TSModel", "make_gpt4ts"]
