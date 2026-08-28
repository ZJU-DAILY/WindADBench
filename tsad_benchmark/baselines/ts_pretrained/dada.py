# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

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


class DADAModel(AnomalyModelBase):

    model_name = "DADA"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        model_id: str = "models/DADA",
        win_size: int = 100,
        batch_size: int = 64,
        norm: bool = False,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
        local_files_only: bool = False,
        mode: str = "zero_shot",
        score_mode: str = "mse",
        copies: int = 10,
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.win_size = int(win_size)
        self.batch_size = int(batch_size)
        self.norm = bool(norm)
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.local_files_only = bool(local_files_only)
        self.mode = mode
        self.score_mode = str(score_mode).lower()
        if self.score_mode == "var":
            self.score_mode = "variance"
        if self.score_mode not in {"variance", "mse"}:
            raise ValueError("DADA score_mode must be either 'variance' or 'mse'.")
        self.copies = int(copies)
        if self.copies <= 0:
            raise ValueError("DADA copies must be positive.")
        if self.copies % 2 != 0:
            raise ValueError("DADA symmetric masking requires an even `copies` value.")
        self.device = str(device)
        self.model_hyper_params = {
            "model_id": self.model_id,
            "win_size": self.win_size,
            "batch_size": self.batch_size,
            "norm": self.norm,
            "anomaly_ratio": self.anomaly_ratio,
            "local_files_only": self.local_files_only,
            "mode": self.mode,
            "score_mode": self.score_mode,
            "copies": self.copies,
            "device": self.device,
        }
        self._model = None
        self._scaler = None
        self._device = None
        self._n_features = 0
        self._train_arr = None
        self._train_scores = None

    def _resolve_device(self):
        import torch

        if self._device is None:
            if self.device and self.device.lower() != "auto":
                self._device = torch.device(self.device)
                if self._device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("DADAModel was configured with device='cuda', but CUDA is not available.")
            else:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def _load_model(self):
        try:
            from transformers import AutoModel, PreTrainedModel
        except Exception as exc:
            raise ImportError("DADAModel requires `transformers`.") from exc
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = property(lambda _self: {})
        model_path = Path(self.model_id).expanduser()
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        if self.model_id == "models/DADA" and not (model_path / "config.json").exists():
            raise FileNotFoundError(
                "DADA model directory not found. Copy a remote-code model directory "
                "containing `config.json`, "
                "`configuration_DADA.py`, `modeling_DADA.py`, and `pytorch_model.bin` "
                f"to `{model_path}`, or pass `model_id` to that directory."
            )
        model = AutoModel.from_pretrained(
            str(model_path) if (model_path / "config.json").exists() else self.model_id,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        official_seq_len = getattr(getattr(model, "config", None), "seq_len", None)
        if isinstance(official_seq_len, (list, tuple)):
            official_seq_len = official_seq_len[0] if official_seq_len else None
        if official_seq_len is not None and int(official_seq_len) != self.win_size:
            raise ValueError(
                "DADA remote-code checkpoints use a fixed `seq_len`; "
                f"this checkpoint expects win_size={int(official_seq_len)}, "
                f"but got win_size={self.win_size}."
            )
        model.to(self._resolve_device())
        model.eval()
        return model

    def _call_infer(self, batch):
        nested = getattr(self._model, "model", None)
        nested_infer = getattr(nested, "infer", None)
        if callable(nested_infer):
            return nested_infer(batch, norm=self.norm, copies=self.copies)
        infer = getattr(self._model, "infer", None)
        if not callable(infer):
            raise RuntimeError("DADA remote-code model must expose an `infer` method.")
        try:
            return infer(batch, norm=self.norm)
        except TypeError:
            return infer(batch)

    def _infer_outputs(self, batch):
        return batch, self._call_infer(batch)

    def _score_from_outputs(self, batch, outputs):
        import torch

        if outputs.ndim == 4:
            if self.score_mode == "variance":
                nested = getattr(self._model, "model", None)
                scorer = getattr(nested, "cal_anomaly_score", None)
                if callable(scorer):
                    score = scorer(
                        batch_x=batch,
                        batch_out_copies=outputs,
                        anomaly_criterion=None,
                        L=1,
                    )
                    return score
                return torch.mean(torch.var(outputs, dim=0), dim=-1)
            outputs = torch.median(outputs, dim=0).values
        if (
            outputs.shape != batch.shape
            and getattr(outputs, "ndim", 0) == 3
            and outputs.transpose(1, 2).shape == batch.shape
        ):
            outputs = outputs.transpose(1, 2)
        if outputs.shape != batch.shape:
            raise RuntimeError(
                f"DADA infer output shape {tuple(outputs.shape)} cannot be aligned "
                f"with input shape {tuple(batch.shape)}."
            )
        return torch.mean((batch - outputs) ** 2, dim=-1)

    def _score_batch(self, batch):
        batch, outputs = self._infer_outputs(batch)
        return self._score_from_outputs(batch, outputs)

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

    def _raw_scores(self, arr: np.ndarray, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        windows = rolling_windows(arr, self.win_size, step=step)
        if len(windows) == 0:
            return np.zeros(0, dtype=float), np.zeros((0, self.win_size), dtype=float)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(windows.astype(np.float32))),
            batch_size=self.batch_size,
            shuffle=False,
        )
        chunks = []
        device = self._resolve_device()
        self._model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.float().to(device)
                score = self._score_batch(batch)
                chunks.append(score.detach().cpu().numpy())
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        raw = np.concatenate(chunks, axis=0).reshape(-1) if chunks else np.zeros(0)
        if raw.size == len(windows) * self.win_size:
            window_scores = raw.reshape(len(windows), self.win_size)
        elif raw.size == len(windows):
            window_scores = np.repeat(raw[:, None], self.win_size, axis=1)
        else:
            window_scores = np.zeros((len(windows), self.win_size), dtype=float)
            usable = min(raw.size, window_scores.size)
            window_scores.reshape(-1)[:usable] = raw[:usable]
        return raw, np.nan_to_num(window_scores, nan=0.0, posinf=0.0, neginf=0.0)

    def _timeline_scores(self, arr: np.ndarray, target_len: Optional[int] = None) -> np.ndarray:
        target_len = int(arr.shape[0] if target_len is None else target_len)
        _, window_scores = self._raw_scores(arr, step=self.win_size)
        _, tail_scores = (
            self._raw_scores(arr[-self.win_size :], step=1)
            if target_len >= self.win_size
            else (None, None)
        )
        return non_overlapping_timeline_scores(window_scores, target_len, self.win_size, tail_scores)

    def fit(self, train_data: pd.DataFrame, train_label=None, covariates=None, **kwargs) -> None:
        arr = to_2d_array(train_data)
        self._n_features = arr.shape[1] if arr.ndim == 2 else 0
        if arr.shape[0] < self.win_size or self._n_features == 0:
            self._model = None
            self._train_arr = None
            self._train_scores = None
            return
        arr = self._scale_fit(arr)
        self._model = self._load_model()
        self._train_arr = arr
        self._train_scores = None

    def detect_score(self, test_data: pd.DataFrame, covariates=None, **kwargs) -> np.ndarray:
        n = len(test_data) if test_data is not None else 0
        if self._model is None or n == 0:
            return np.zeros(n, dtype=float)
        arr = self._prepare_inference_array(test_data)
        return self._timeline_scores(arr, n)

    def detect_label(self, test_data: pd.DataFrame, covariates=None, test_label=None, **kwargs):
        n = len(test_data) if test_data is not None else 0
        if self._model is None or n == 0:
            empty = np.zeros(n, dtype=np.int32)
            return {str(r): empty.copy() for r in self.anomaly_ratio}, np.zeros(n)
        arr = self._prepare_inference_array(test_data)
        if self._train_scores is None and self._train_arr is not None:
            self._train_scores = self._timeline_scores(self._train_arr)
        scores = self._timeline_scores(arr, n)
        preds = percentile_labels(
            scores,
            self._train_scores,
            self.anomaly_ratio,
            apply_adjustment=False,
        )
        return preds, scores

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
        try:
            import io
            import torch

            buf = io.BytesIO()
            torch.save(self._model.state_dict(), buf)
            return float(buf.tell() / (1024.0 * 1024.0))
        except Exception:
            return math.nan


def make_dada(**kwargs) -> DADAModel:
    return DADAModel(**kwargs)


__all__ = ["DADAModel", "make_dada"]
