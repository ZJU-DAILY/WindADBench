# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import importlib
import logging
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from tsad_benchmark.baselines._official_ad import (
    non_overlapping_timeline_scores,
    percentile_labels,
    rolling_windows,
    to_2d_array,
)
from tsad_benchmark.baselines._thresholding import normalize_anomaly_ratios
from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability

logger = logging.getLogger(__name__)


class UniTimeModel(AnomalyModelBase):
    """UniTime model wrapped as a reconstruction-based anomaly detector."""

    model_name = "UniTime"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        model_path: str = "gpt2",
        official_repo_path: str = "tsad_benchmark/baselines/llm_based/UniTime",
        win_size: int = 96,
        batch_size: int = 64,
        num_epochs: int = 10,
        lr: float = 1e-4,
        val_ratio: float = 0.2,
        patience: int = 10,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
        sampling_rate: float = 0.05,
        val_sample_rate: float = 1.0,
        train_score_sample_rate: float = 1.0,
        lradj: str = "type1",
        sampling_strategy: str = "uniform",
        local_files_only: bool = False,
        mask_rate: float = 0.5,
        patch_len: int = 16,
        stride: Optional[int] = 8,
        max_token_num: int = 17,
        max_backcast_len: int = 96,
        max_forecast_len: int = 0,
        ts_embed_dropout: float = 0.3,
        lm_ft_type: str = "fpt",
        lm_layer_num: int = 6,
        dec_trans_layer_num: int = 2,
        dec_head_dropout: float = 0.1,
        instruct_path: str = "data_configs/instruct_empty.json",
        instruction: str = "",
        device: str = "auto",
    ) -> None:
        self.win_size = int(win_size)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.lr = float(lr)
        self.val_ratio = float(val_ratio)
        self.patience = int(patience)
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.sampling_rate = float(sampling_rate)
        self.val_sample_rate = float(np.clip(val_sample_rate, 0.0, 1.0))
        self.train_score_sample_rate = float(np.clip(train_score_sample_rate, 0.0, 1.0))
        self.lradj = str(lradj)
        self.sampling_strategy = str(sampling_strategy)
        self.model_path = model_path
        self.official_repo_path = official_repo_path
        self.local_files_only = bool(local_files_only)
        self.mask_rate = float(mask_rate)
        self.patch_len = int(patch_len)
        self.stride = int(stride) if stride is not None else int(patch_len)
        self.max_token_num = int(max_token_num)
        self.max_backcast_len = int(max_backcast_len)
        self.max_forecast_len = int(max_forecast_len)
        self.ts_embed_dropout = float(ts_embed_dropout)
        self.lm_ft_type = lm_ft_type
        self.lm_layer_num = int(lm_layer_num)
        self.dec_trans_layer_num = int(dec_trans_layer_num)
        self.dec_head_dropout = float(dec_head_dropout)
        self.instruct_path = instruct_path
        self.instruction = instruction
        self.device = str(device)
        self.model_hyper_params = {
            "model_path": self.model_path,
            "official_repo_path": self.official_repo_path,
            "win_size": self.win_size,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "val_ratio": self.val_ratio,
            "patience": self.patience,
            "anomaly_ratio": self.anomaly_ratio,
            "sampling_rate": self.sampling_rate,
            "val_sample_rate": self.val_sample_rate,
            "train_score_sample_rate": self.train_score_sample_rate,
            "lradj": self.lradj,
            "sampling_strategy": self.sampling_strategy,
            "local_files_only": self.local_files_only,
            "mask_rate": self.mask_rate,
            "patch_len": self.patch_len,
            "stride": self.stride,
            "max_token_num": self.max_token_num,
            "max_backcast_len": self.max_backcast_len,
            "max_forecast_len": self.max_forecast_len,
            "ts_embed_dropout": self.ts_embed_dropout,
            "lm_ft_type": self.lm_ft_type,
            "lm_layer_num": self.lm_layer_num,
            "dec_trans_layer_num": self.dec_trans_layer_num,
            "dec_head_dropout": self.dec_head_dropout,
            "instruct_path": self.instruct_path,
            "instruction": self.instruction,
            "device": self.device,
        }
        self._model = None
        self._scaler = None
        self._device = None
        self._n_features = 0
        self._train_scores = None
        self._train_score_windows = None

    @staticmethod
    def _sample_windows(windows: np.ndarray, sample_rate: float) -> np.ndarray:
        sample_rate = float(np.clip(sample_rate, 0.0, 1.0))
        if len(windows) == 0 or sample_rate <= 0.0 or sample_rate >= 1.0:
            return windows
        keep = max(1, int(math.ceil(len(windows) * sample_rate)))
        indices = np.linspace(0, len(windows) - 1, num=keep, dtype=int)
        return windows[np.unique(indices)]

    def _window_starts(self, arr_len: int, step: int = 1) -> np.ndarray:
        if int(arr_len) < self.win_size:
            return np.zeros(0, dtype=int)
        return np.arange(0, int(arr_len) - self.win_size + 1, int(step), dtype=int)

    @staticmethod
    def _sample_starts(starts: np.ndarray, sample_rate: float) -> np.ndarray:
        sample_rate = float(np.clip(sample_rate, 0.0, 1.0))
        if len(starts) == 0 or sample_rate <= 0.0 or sample_rate >= 1.0:
            return starts
        keep = max(1, int(math.ceil(len(starts) * sample_rate)))
        indices = np.linspace(0, len(starts) - 1, num=keep, dtype=int)
        return starts[np.unique(indices)]

    def _windows_from_starts(self, arr: np.ndarray, starts: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        starts = np.asarray(starts, dtype=int)
        n_features = max(arr.shape[1] if arr.ndim == 2 else self._n_features, 1)
        if arr.ndim != 2 or len(starts) == 0 or arr.shape[0] < self.win_size:
            return np.zeros((0, self.win_size, n_features), dtype=np.float32)
        offsets = np.arange(self.win_size, dtype=int)
        return arr[starts[:, None] + offsets[None, :]].astype(np.float32, copy=False)

    def _adjust_learning_rate(self, optimizer, epoch: int) -> None:
        if self.lradj != "type1":
            return
        lr = self.lr * (0.5 ** ((int(epoch) - 1) // 1))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _resolve_device(self):
        if self._device is None:
            if self.device.lower() == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(self.device)
                if self._device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("UniTimeModel was configured with device='cuda', but CUDA is not available.")
        return self._device

    def _resolve_model_class(self):
        repo = Path(self.official_repo_path).expanduser()
        if not repo.is_absolute():
            repo = Path.cwd() / repo
        repo = repo.resolve()
        if not (repo / "models" / "unitime.py").exists():
            raise ImportError(
                "UniTimeModel requires the official UniTime source tree with "
                f"`models/unitime.py` at `{repo}`. Pass `official_repo_path` "
                "to the cloned UniTime directory if it lives elsewhere."
            )
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            module = importlib.import_module("models.unitime")
        except Exception as exc:
            raise ImportError(
                "Could not import `models.unitime` from the official UniTime source tree "
                f"at `{repo}`. Make sure it contains `models/unitime.py` and "
                "`models/unitimegpt2.py`."
            ) from exc
        try:
            return module.UniTime
        except AttributeError as exc:
            raise ImportError("Official UniTime module `models.unitime` does not expose `UniTime`.") from exc

    def _max_token_num_for_window(self) -> int:
        if self.max_token_num > 0:
            return self.max_token_num
        token_num = 1 if self.win_size <= self.patch_len else math.ceil((self.win_size - self.patch_len) / self.stride) + 1
        instruct_tokens = len(self.instruction.split()) if self.instruction else 0
        return token_num + instruct_tokens

    def _build_network(self, n_features: int):
        model_cls = self._resolve_model_class()
        cfg = SimpleNamespace(
            task_name="anomaly_detection",
            seq_len=self.win_size,
            pred_len=0,
            horizon=0,
            enc_in=n_features,
            dec_in=n_features,
            c_out=n_features,
            mask_rate=self.mask_rate,
            patch_len=self.patch_len,
            max_token_num=self._max_token_num_for_window(),
            max_backcast_len=max(self.max_backcast_len, self.win_size),
            max_forecast_len=self.max_forecast_len,
            model_path=self.model_path,
            local_files_only=self.local_files_only,
            ts_embed_dropout=self.ts_embed_dropout,
            lm_ft_type=self.lm_ft_type,
            lm_layer_num=self.lm_layer_num,
            dec_trans_layer_num=self.dec_trans_layer_num,
            dec_head_dropout=self.dec_head_dropout,
            instruct_path=self.instruct_path,
            logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
        )
        return model_cls(cfg)

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

    def _model_output(self, batch):
        mask = (torch.rand_like(batch) >= self.mask_rate).to(dtype=batch.dtype)
        masked_batch = batch.masked_fill(mask == 0, 0.0)
        info = (0, self.win_size, self.stride, self.instruction)
        return self._model(info, masked_batch, mask)

    def _align_output(self, output, batch):
        if isinstance(output, dict):
            for key in ("reconstruction", "recon", "outputs", "output", "prediction", "pred"):
                if key in output:
                    output = output[key]
                    break
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.shape == batch.shape:
            return output
        if output.ndim == 3 and output.transpose(1, 2).shape == batch.shape:
            return output.transpose(1, 2)
        if output.ndim == 3 and output.shape[0] == batch.shape[0] and output.shape[2] == batch.shape[2]:
            return output[:, : batch.shape[1], :]
        if output.ndim == 3 and output.shape[0] == batch.shape[0] and output.shape[1] == batch.shape[2]:
            transposed = output.transpose(1, 2)
            if transposed.shape[1] >= batch.shape[1]:
                return transposed[:, : batch.shape[1], :]
        raise RuntimeError(
            f"{self.model_name} output shape {tuple(output.shape)} cannot be aligned "
            f"with input shape {tuple(batch.shape)}."
        )

    def _forward_reconstruction(self, batch):
        return self._align_output(self._model_output(batch), batch)

    def _score_windows(self, windows: np.ndarray) -> np.ndarray:
        from torch.utils.data import DataLoader, TensorDataset

        if len(windows) == 0:
            return np.zeros((0, self.win_size), dtype=float)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(windows.astype(np.float32, copy=False))),
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
                recon = self._forward_reconstruction(batch)
                score = torch.mean((batch - recon) ** 2, dim=-1)
                chunks.append(score.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, self.win_size))

    def _window_scores(self, arr: np.ndarray, step: int = 1) -> np.ndarray:
        windows = rolling_windows(arr, self.win_size, step=step)
        return self._score_windows(windows)

    def fit(self, train_data: pd.DataFrame, train_label=None, covariates=None, **kwargs) -> None:
        from torch.utils.data import DataLoader, TensorDataset

        arr = to_2d_array(train_data)
        if arr.shape[0] < self.win_size or arr.shape[1] == 0:
            self._model = None
            self._train_scores = None
            self._train_score_windows = None
            return
        self._n_features = arr.shape[1]
        arr = self._scale_fit(arr)
        val_n = int(arr.shape[0] * self.val_ratio)
        if val_n >= self.win_size + 1 and (arr.shape[0] - val_n) >= self.win_size:
            train_arr, val_arr = arr[: arr.shape[0] - val_n], arr[arr.shape[0] - val_n :]
        else:
            train_arr, val_arr = arr, None
        train_starts = self._window_starts(train_arr.shape[0], step=1)
        if len(train_starts) == 0:
            self._model = None
            self._train_scores = None
            self._train_score_windows = None
            return
        if 0.0 < self.sampling_rate < 1.0:
            sample_n = max(1, int(len(train_starts) * self.sampling_rate))
            if self.sampling_strategy == "uniform" and sample_n < len(train_starts):
                indices = np.linspace(0, len(train_starts) - 1, num=sample_n, dtype=int)
                train_starts_for_fit = train_starts[np.unique(indices)]
            else:
                train_starts_for_fit = train_starts[:sample_n]
        else:
            train_starts_for_fit = train_starts
        train_windows_for_fit = self._windows_from_starts(train_arr, train_starts_for_fit)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_windows_for_fit.astype(np.float32, copy=False))),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        val_starts = (
            self._sample_starts(self._window_starts(val_arr.shape[0], step=1), self.val_sample_rate)
            if val_arr is not None
            else np.zeros(0, dtype=int)
        )
        val_windows = (
            self._windows_from_starts(val_arr, val_starts)
            if val_arr is not None
            else np.zeros((0, self.win_size, self._n_features), dtype=np.float32)
        )
        val_loader = (
            DataLoader(
                TensorDataset(torch.from_numpy(val_windows.astype(np.float32, copy=False))),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
            if len(val_windows) > 0
            else None
        )
        device = self._resolve_device()
        self._model = self._build_network(self._n_features).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        logger.info(
            "[UniTime] fit windows train=%d val=%d train_score=%d batch=%d epochs=%d win=%d stride=%d features=%d",
            len(train_windows_for_fit),
            len(val_windows),
            len(self._sample_starts(train_starts, self.train_score_sample_rate)),
            self.batch_size,
            self.num_epochs,
            self.win_size,
            self.stride,
            self._n_features,
        )
        best_val = None
        best_state = None
        bad_epochs = 0
        for _epoch in range(self.num_epochs):
            epoch_start = time.time()
            train_loss_sum = None
            train_loss_count = 0
            self._model.train()
            for (batch,) in train_loader:
                batch = batch.float().to(device)
                recon = self._forward_reconstruction(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                detached_loss = loss.detach()
                train_loss_sum = detached_loss if train_loss_sum is None else train_loss_sum + detached_loss
                train_loss_count += 1
            train_elapsed = time.time() - epoch_start
            train_loss = (
                float((train_loss_sum / max(train_loss_count, 1)).detach().cpu())
                if train_loss_sum is not None
                else 0.0
            )
            if val_loader is None:
                logger.info(
                    "[UniTime] epoch=%d/%d train_loss=%.6f train_time=%.1fs",
                    _epoch + 1,
                    self.num_epochs,
                    train_loss,
                    train_elapsed,
                )
                self._adjust_learning_rate(optimizer, _epoch + 1)
                continue
            val_start = time.time()
            losses = []
            self._model.eval()
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.float().to(device)
                    recon = self._forward_reconstruction(batch)
                    losses.append(float(criterion(recon, batch).detach().cpu()))
            val_loss = float(np.mean(losses)) if losses else 0.0
            logger.info(
                "[UniTime] epoch=%d/%d train_loss=%.6f val_loss=%.6f train_time=%.1fs val_time=%.1fs",
                _epoch + 1,
                self.num_epochs,
                train_loss,
                val_loss,
                train_elapsed,
                time.time() - val_start,
            )
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
            self._adjust_learning_rate(optimizer, _epoch + 1)
        if best_state is not None:
            self._model.load_state_dict(best_state)
        train_score_starts = self._sample_starts(train_starts, self.train_score_sample_rate)
        self._train_score_windows = self._windows_from_starts(train_arr, train_score_starts)
        self._train_scores = None

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
        threshold_test_scores = self._window_scores(arr, step=1).reshape(-1)
        if self._train_scores is None:
            score_windows = (
                self._train_score_windows
                if self._train_score_windows is not None
                else np.zeros((0, self.win_size, self._n_features), dtype=np.float32)
            )
            self._train_scores = self._score_windows(score_windows).reshape(-1)
        preds = percentile_labels(
            test_scores,
            self._train_scores,
            self.anomaly_ratio,
            test_label=test_label,
            apply_adjustment=False,
            threshold_test_scores=threshold_test_scores,
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
        buf = io.BytesIO()
        torch.save(self._model.state_dict(), buf)
        return float(buf.tell() / (1024.0 * 1024.0))


def make_unitime(**kwargs) -> UniTimeModel:
    return UniTimeModel(**kwargs)


__all__ = ["UniTimeModel", "make_unitime"]
