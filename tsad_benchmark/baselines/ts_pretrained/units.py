# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
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


class UniTSModel(AnomalyModelBase):

    model_name = "UniTS"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        model_id: str = "mims-harvard/UniTS",
        checkpoint_path: Optional[str] = None,
        dataset_name: str = "DEFAULT",
        use_p: bool = True,
        win_size: int = 96,
        batch_size: int = 32,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
        local_files_only: bool = False,
        mode: str = "zero_shot",
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path
        self.dataset_name = dataset_name
        self.use_p = bool(use_p)
        self.win_size = int(win_size)
        self.batch_size = int(batch_size)
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.local_files_only = bool(local_files_only)
        self.mode = mode
        self.device = str(device)
        self.model_hyper_params = {
            "model_id": self.model_id,
            "checkpoint_path": self.checkpoint_path,
            "dataset_name": self.dataset_name,
            "use_p": self.use_p,
            "win_size": self.win_size,
            "batch_size": self.batch_size,
            "anomaly_ratio": self.anomaly_ratio,
            "local_files_only": self.local_files_only,
            "mode": self.mode,
            "device": self.device,
        }
        self._model = None
        self._scaler = None
        self._device = None
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
                    raise RuntimeError("UniTSModel was configured with device='cuda', but CUDA is not available.")
        return self._device

    def _resolve_checkpoint_path(self) -> str:
        path = self.checkpoint_path
        if path is None and self.model_id and self.model_id.endswith(".pth"):
            path = self.model_id
        if path is None:
            path = "models/UniTS/units_x32_pretrain_checkpoint.pth"
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        if not resolved.exists():
            raise FileNotFoundError(
                "UniTS checkpoint not found. Download `units_x32_pretrain_checkpoint.pth` "
                f"and pass `checkpoint_path`, or place it at `{resolved}`."
            )
        return str(resolved)

    def _load_model(self, n_features: int):
        try:
            import torch
            from tsad_benchmark.baselines.ts_pretrained._units_official import Model as OfficialUniTS
        except Exception as exc:
            raise ImportError(
                "UniTSModel requires the vendored official UniTS implementation and `timm`. "
                "Install `timm` if this import failed because of a missing dependency."
            ) from exc

        args = SimpleNamespace(
            d_model=32,
            n_heads=8,
            e_layers=3,
            prompt_num=10,
            dropout=0.1,
            patch_len=16,
            stride=16,
            batch_size=self.batch_size,
        )
        dataset_name = str(self.dataset_name) if self.dataset_name else "DEFAULT"
        task_name = f"AD_{dataset_name}_p0"
        task_config = {
            "task_name": "anomaly_detection",
            "dataset_name": dataset_name,
            "dataset": dataset_name,
            "data": dataset_name,
            "embed": "timeF",
            "features": "M",
            "seq_len": self.win_size,
            "label_len": 0,
            "pred_len": 0,
            "enc_in": int(n_features),
            "dec_in": int(n_features),
            "c_out": int(n_features),
            "max_batch": self.batch_size,
        }
        model = OfficialUniTS(args, [[task_name, task_config]], pretrain=False)

        checkpoint_path = self._resolve_checkpoint_path()
        try:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            raw = torch.load(checkpoint_path, map_location="cpu")
        state_dict = raw.get("student", raw) if isinstance(raw, dict) else raw
        ckpt = {}
        for key, value in state_dict.items():
            if "cls_prompts" in key:
                continue
            key = key.replace("module.", "", 1) if key.startswith("module.") else key
            ckpt[key] = value
        msg = model.load_state_dict(ckpt, strict=False)
        if msg.missing_keys:
            print(
                "Warning: UniTS checkpoint missing keys: "
                f"{msg.missing_keys}. Prediction quality may be affected."
            )

        if not self.use_p:
            for param in model.parameters():
                param.data.uniform_(-0.02, 0.02)

        model.to(self._resolve_device())
        model.eval()
        return model

    @staticmethod
    def _extract_reconstruction(output):
        if hasattr(output, "reconstruction"):
            return output.reconstruction
        if isinstance(output, dict):
            for key in ("reconstruction", "recon", "outputs", "output"):
                if key in output:
                    return output[key]
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _forward_reconstruction(self, batch):
        return self._extract_reconstruction(
            self._model(batch, None, None, None, task_id=0, task_name="anomaly_detection")
        )

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

    def _window_scores(self, arr: np.ndarray, step: int = 1) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        windows = rolling_windows(arr, self.win_size, step=step)
        if len(windows) == 0:
            return np.zeros((0, self.win_size), dtype=float)
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
                recon = self._forward_reconstruction(batch)
                if recon.shape != batch.shape and recon.transpose(1, 2).shape == batch.shape:
                    recon = recon.transpose(1, 2)
                score = torch.mean((batch - recon) ** 2, dim=-1)
                chunks.append(score.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, self.win_size))

    def fit(self, train_data: pd.DataFrame, train_label=None, covariates=None, **kwargs) -> None:
        arr = to_2d_array(train_data)
        self._n_features = arr.shape[1] if arr.ndim == 2 else 0
        if arr.shape[0] < self.win_size or self._n_features == 0:
            self._model = None
            self._train_scores = None
            return
        arr = self._scale_fit(arr)
        self._model = self._load_model(self._n_features)
        self._train_scores = self._window_scores(arr).reshape(-1)

    def detect_score(self, test_data: pd.DataFrame, covariates=None, **kwargs) -> np.ndarray:
        n = len(test_data) if test_data is not None else 0
        if self._model is None or n == 0:
            return np.zeros(n, dtype=float)
        arr = self._prepare_inference_array(test_data)
        window_scores = self._window_scores(arr, step=self.win_size)
        tail_scores = self._window_scores(arr[-self.win_size :], step=1) if n >= self.win_size else None
        return non_overlapping_timeline_scores(window_scores, n, self.win_size, tail_scores)

    def detect_label(self, test_data: pd.DataFrame, covariates=None, test_label=None, **kwargs):
        n = len(test_data) if test_data is not None else 0
        if self._model is None or n == 0:
            empty = np.zeros(n, dtype=np.int32)
            return {str(r): empty.copy() for r in self.anomaly_ratio}, np.zeros(n)
        arr = self._prepare_inference_array(test_data)
        raw_window_scores = self._window_scores(arr, step=self.win_size)
        tail_scores = self._window_scores(arr[-self.win_size :], step=1) if n >= self.win_size else None
        scores = non_overlapping_timeline_scores(raw_window_scores, n, self.win_size, tail_scores)
        threshold_window_scores = self._window_scores(arr, step=1)
        preds = percentile_labels(
            scores,
            self._train_scores,
            self.anomaly_ratio,
            test_label=test_label,
            apply_adjustment=False,
            threshold_test_scores=threshold_window_scores.reshape(-1),
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


def make_units(**kwargs) -> UniTSModel:
    return UniTSModel(**kwargs)


__all__ = ["UniTSModel", "make_units"]
