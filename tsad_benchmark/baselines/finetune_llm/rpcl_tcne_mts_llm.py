# -*- coding: utf-8 -*-


from __future__ import annotations

import io
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from tsad_benchmark.baselines._official_ad import labels_1d, to_2d_array
from tsad_benchmark.baselines._thresholding import (
    normalize_anomaly_ratios,
    percentile_label_maps,
)
from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability

logger = logging.getLogger(__name__)


def _torch_modules():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils import weight_norm

    return torch, nn, F, weight_norm


def _configure_torch_runtime(torch) -> None:
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


class _MinMaxScaler:
    def __init__(self) -> None:
        self.min_: Optional[np.ndarray] = None
        self.range_: Optional[np.ndarray] = None

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.min_ = np.nanmin(x, axis=0)
        max_ = np.nanmax(x, axis=0)
        self.range_ = np.where(max_ - self.min_ == 0.0, 1.0, max_ - self.min_)
        return self.transform(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.range_ is None:
            return x.astype(np.float32)
        return ((x - self.min_) / self.range_).astype(np.float32)


def _window_starts(
    n_obs: int,
    win_size: int,
    stride: int = 1,
    label: Optional[np.ndarray] = None,
) -> np.ndarray:
    if n_obs < win_size:
        return np.zeros(0, dtype=np.int64)
    stride = max(1, int(stride))
    starts = np.arange(0, int(n_obs) - int(win_size) + 1, stride, dtype=np.int64)
    if stride <= 1 or label is None:
        return starts
    y = np.asarray(label, dtype=np.int64).reshape(-1)
    if y.size < n_obs:
        return starts
    positive_starts = np.flatnonzero(y[win_size - 1 :] > 0).astype(np.int64)
    if positive_starts.size == 0:
        return starts
    return np.unique(np.concatenate([starts, positive_starts]))


def _window_labels(
    label: Optional[np.ndarray],
    win_size: int,
    stride: int = 1,
    starts: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if label is None:
        return None
    y = np.asarray(label, dtype=np.int64).reshape(-1)
    if y.size < win_size:
        return None
    if starts is None:
        starts = _window_starts(y.size, win_size, stride)
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    if starts.size == 0:
        return np.zeros(0, dtype=np.int64)
    # The paper assigns one label to each subsequence created from
    # x_m...x_{m+T-1}; use the endpoint label.
    return y[starts + win_size - 1].astype(np.int64)


def _window_array(
    x: np.ndarray,
    win_size: int,
    stride: int = 1,
    starts: Optional[np.ndarray] = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < win_size or x.shape[1] == 0:
        n_feat = max(x.shape[1] if x.ndim == 2 else 1, 1)
        return np.zeros((0, win_size, n_feat), dtype=np.float32)
    if starts is None:
        starts = _window_starts(x.shape[0], win_size, stride)
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    offsets = np.arange(win_size)
    return x[starts[:, None] + offsets[None, :]].astype(np.float32)


def _align_endpoint_scores(window_scores: np.ndarray, target_len: int, win_size: int) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(window_scores, dtype=float).reshape(-1))
    out = np.zeros(int(target_len), dtype=float)
    if target_len <= 0:
        return out
    if scores.size == 0:
        return out
    first = min(win_size - 1, target_len - 1)
    usable = min(scores.size, target_len - first)
    out[:first] = scores[0]
    out[first : first + usable] = scores[:usable]
    if first + usable < target_len:
        out[first + usable :] = scores[-1]
    return out


def _align_endpoint_labels(window_labels: np.ndarray, target_len: int, win_size: int) -> np.ndarray:
    labels = np.asarray(window_labels, dtype=np.int32).reshape(-1)
    out = np.zeros(int(target_len), dtype=np.int32)
    if target_len <= 0 or labels.size == 0:
        return out
    first = min(win_size - 1, target_len - 1)
    usable = min(labels.size, target_len - first)
    out[:first] = labels[0]
    out[first : first + usable] = labels[:usable]
    if first + usable < target_len:
        out[first + usable :] = labels[-1]
    return out


class _CausalConv1d:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        torch, nn, _F, weight_norm = _torch_modules()

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.pad = (int(kernel_size) - 1) * int(dilation)
                self.conv = weight_norm(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=int(kernel_size),
                        padding=self.pad,
                        dilation=int(dilation),
                    )
                )

            def forward(self, x):
                y = self.conv(x)
                if self.pad:
                    y = y[:, :, : -self.pad]
                return y

        self.module = Module()


def _build_residual_block_cls():
    torch, nn, _F, weight_norm = _torch_modules()

    class TCNEResidualBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.conv1 = _CausalConv1d(
                in_channels, out_channels, kernel_size, dilation
            ).module
            self.conv2 = _CausalConv1d(
                out_channels, out_channels, kernel_size, dilation
            ).module
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv1d(in_channels, out_channels, kernel_size=1)
            )
            self.act = nn.LeakyReLU()
            self.dropout = nn.Dropout(float(dropout))

        def forward(self, x):
            y = self.conv1(x)
            y = self.act(y)
            y = self.dropout(y)
            y = self.conv2(y)
            y = self.act(y)
            y = self.dropout(y)
            return y + self.skip(x)

    return TCNEResidualBlock


def _build_tcne_cls():
    torch, nn, _F, _weight_norm = _torch_modules()
    TCNEResidualBlock = _build_residual_block_cls()

    class TCNE(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            n_blocks: int,
            kernel_size: int,
            dropout: float,
        ) -> None:
            super().__init__()
            blocks = []
            in_dim = int(input_dim)
            for i in range(int(n_blocks)):
                out_dim = int(hidden_dim)
                blocks.append(
                    TCNEResidualBlock(
                        in_dim,
                        out_dim,
                        kernel_size=int(kernel_size),
                        dilation=2**i,
                        dropout=float(dropout),
                    )
                )
                in_dim = out_dim
            self.blocks = nn.Sequential(*blocks)
            self.output = nn.Conv1d(in_dim, int(output_dim), kernel_size=1)

        def forward(self, x):
            # x: B x T x C -> H_IE: B x T x E
            y = x.transpose(1, 2)
            y = self.blocks(y)
            y = self.output(y)
            return y.transpose(1, 2)

    return TCNE


def _build_lora_c_attn_cls():
    torch, nn, _F, _weight_norm = _torch_modules()

    class LoRACAttn(nn.Module):
        def __init__(self, base, hidden_size: int, rank: int, alpha: float):
            super().__init__()
            self.base = base
            self.hidden_size = int(hidden_size)
            self.rank = int(rank)
            self.scale = float(alpha) / float(max(rank, 1))
            self.q_a = nn.Linear(self.hidden_size, self.rank, bias=False)
            self.q_b = nn.Linear(self.rank, self.hidden_size, bias=False)
            self.v_a = nn.Linear(self.hidden_size, self.rank, bias=False)
            self.v_b = nn.Linear(self.rank, self.hidden_size, bias=False)
            nn.init.kaiming_uniform_(self.q_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.q_b.weight)
            nn.init.kaiming_uniform_(self.v_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.v_b.weight)

        def forward(self, x):
            y = self.base(x)
            q_delta = self.q_b(self.q_a(x)) * self.scale
            v_delta = self.v_b(self.v_a(x)) * self.scale
            y = y.clone()
            hs = self.hidden_size
            y[..., :hs] = y[..., :hs] + q_delta
            y[..., 2 * hs : 3 * hs] = y[..., 2 * hs : 3 * hs] + v_delta
            return y

    return LoRACAttn


@dataclass
class _RPCLConfig:
    backbone: str
    input_dim: int
    win_size: int
    n_classes: int
    tcne_hidden_dim: int
    tcne_blocks: int
    kernel_size: int
    dropout: float
    lora_rank: int
    lora_alpha: float
    local_files_only: bool


def _build_network_cls():
    torch, nn, F, _weight_norm = _torch_modules()
    TCNE = _build_tcne_cls()
    LoRACAttn = _build_lora_c_attn_cls()

    class RPCLTCNEMTSLLMNet(nn.Module):
        def __init__(self, cfg: _RPCLConfig) -> None:
            super().__init__()
            try:
                from transformers.models.gpt2.modeling_gpt2 import GPT2Model
            except Exception as exc:  # pragma: no cover - dependency optional at import time
                raise ImportError(
                    "RPCLTCNEMTSLLMModel requires `transformers` and GPT-2 "
                    "weights. Install transformers and make the requested "
                    "backbone available."
                ) from exc

            self.gpt2 = GPT2Model.from_pretrained(
                cfg.backbone,
                output_attentions=False,
                output_hidden_states=False,
                local_files_only=bool(cfg.local_files_only),
            )
            self.gpt2.config.use_cache = False
            hidden = int(self.gpt2.config.n_embd)
            self.tcne = TCNE(
                input_dim=int(cfg.input_dim),
                hidden_dim=int(cfg.tcne_hidden_dim),
                output_dim=hidden,
                n_blocks=int(cfg.tcne_blocks),
                kernel_size=int(cfg.kernel_size),
                dropout=float(cfg.dropout),
            )
            self.classifier = nn.Linear(hidden, int(cfg.n_classes))
            self._freeze_and_attach_lora(hidden, int(cfg.lora_rank), float(cfg.lora_alpha))

        def _freeze_and_attach_lora(self, hidden: int, rank: int, alpha: float) -> None:
            for param in self.gpt2.parameters():
                param.requires_grad = False
            for name, param in self.gpt2.named_parameters():
                if ".ln_" in name or name.endswith(".ln_f.weight") or name.endswith(".ln_f.bias") or "wpe" in name:
                    param.requires_grad = True
            if rank > 0:
                for block in self.gpt2.h:
                    block.attn.c_attn = LoRACAttn(block.attn.c_attn, hidden, rank, alpha)

        def forward(self, x):
            h_ie = self.tcne(x)
            out = self.gpt2(inputs_embeds=h_ie, use_cache=False).last_hidden_state
            pooled = out[:, -1, :]
            logits = self.classifier(pooled)
            return logits, h_ie

    return RPCLTCNEMTSLLMNet


def _rpcl_loss(h_ie, noise_std: float):
    torch, _nn, F, _weight_norm = _torch_modules()
    # H_IE: B x T x E. Eq. (2): max pooling along feature dimension yields
    # V: B x T. Relative positions are columns of V.
    v = torch.max(h_ie, dim=-1).values
    if v.ndim != 2 or v.shape[1] < 2:
        return v.new_tensor(0.0)
    u = v.transpose(0, 1).contiguous()
    u_norm = F.normalize(u, p=2, dim=1, eps=1e-8)
    sim = u_norm @ u_norm.transpose(0, 1)
    n = u.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=u.device)
    neg = sim.masked_select(~eye).sum()
    noise = torch.randn_like(u) * float(noise_std)
    pos_sim = F.cosine_similarity(u, u + noise, dim=1, eps=1e-8)
    pos = (1.0 - pos_sim).sum()
    denom = max(float(n * (n - 1) / 2.0), 1.0)
    return (neg + pos) / denom


class RPCLTCNEMTSLLMModel(AnomalyModelBase):
    """Strict paper-structured RPCL-updated TCNE-MTS-LLM adapter."""

    model_name = "RPCL-TCNE-MTS-LLM"
    capability = ModelCapability.score_and_label(training_paradigm="supervised")

    def __init__(
        self,
        backbone: str = "gpt2",
        win_size: int = 22,
        train_stride: int = 1,
        batch_size: int = 1000,
        num_epochs: int = 300,
        lr: float = 1e-4,
        tcne_hidden_dim: int = 44,
        tcne_blocks: int = 10,
        kernel_size: int = 3,
        dropout: float = 0.2,
        lora_rank: int = 4,
        lora_alpha: float = 4.0,
        gaussian_noise_std: float = 0.05,
        progress_log_interval: float = 60.0,
        anomaly_ratio: Optional[float | Sequence[float]] = None,
        local_files_only: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.backbone = str(backbone)
        self.win_size = int(win_size)
        self.train_stride = max(1, int(train_stride))
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.lr = float(lr)
        self.tcne_hidden_dim = int(tcne_hidden_dim)
        self.tcne_blocks = int(tcne_blocks)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.progress_log_interval = max(0.0, float(progress_log_interval))
        self.anomaly_ratio = normalize_anomaly_ratios(anomaly_ratio)
        self.local_files_only = bool(local_files_only)
        self.seed = None if seed is None else int(seed)
        self.model_hyper_params = {
            "backbone": self.backbone,
            "win_size": self.win_size,
            "train_stride": self.train_stride,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "tcne_hidden_dim": self.tcne_hidden_dim,
            "tcne_blocks": self.tcne_blocks,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "gaussian_noise_std": self.gaussian_noise_std,
            "progress_log_interval": self.progress_log_interval,
            "anomaly_ratio": self.anomaly_ratio,
            "local_files_only": self.local_files_only,
            "seed": self.seed,
        }
        self._model = None
        self._scaler = _MinMaxScaler()
        self._device = None
        self._n_features = 0
        self._n_classes = 2
        self._train_scores: Optional[np.ndarray] = None
        self._train_array: Optional[np.ndarray] = None

    def _resolve_device(self):
        torch, _nn, _F, _weight_norm = _torch_modules()
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def _set_seed(self) -> None:
        if self.seed is None:
            return
        import random

        torch, _nn, _F, _weight_norm = _torch_modules()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _labels(self, label) -> Optional[np.ndarray]:
        y = labels_1d(label)
        if y is None:
            return None
        return (y > 0).astype(np.int64)

    def _build_model(self, n_features: int, n_classes: int):
        RPCLTCNEMTSLLMNet = _build_network_cls()
        cfg = _RPCLConfig(
            backbone=self.backbone,
            input_dim=int(n_features),
            win_size=self.win_size,
            n_classes=int(max(n_classes, 2)),
            tcne_hidden_dim=self.tcne_hidden_dim,
            tcne_blocks=self.tcne_blocks,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            local_files_only=self.local_files_only,
        )
        return RPCLTCNEMTSLLMNet(cfg)

    def fit(self, train_data: pd.DataFrame, train_label=None, covariates=None, **kwargs) -> None:
        torch, nn, _F, _weight_norm = _torch_modules()
        from torch.utils.data import DataLoader, TensorDataset

        _configure_torch_runtime(torch)
        self._set_seed()
        x = to_2d_array(train_data)
        y = self._labels(train_label)
        if x.shape[0] < self.win_size or x.shape[1] == 0:
            self._model = None
            self._train_scores = None
            self._train_array = None
            return
        self._n_features = int(x.shape[1])
        x = self._scaler.fit_transform(x)
        train_starts = _window_starts(x.shape[0], self.win_size, self.train_stride, label=y)
        windows = _window_array(x, self.win_size, starts=train_starts)
        win_labels = _window_labels(y, self.win_size, starts=train_starts)
        if windows.shape[0] == 0:
            self._model = None
            self._train_scores = None
            self._train_array = None
            return
        if win_labels is None:
            win_labels = np.zeros(windows.shape[0], dtype=np.int64)
        self._n_classes = int(max(2, int(np.max(win_labels)) + 1))
        logger.info(
            "[RPCL] preparing fit windows=%d batch=%d epochs=%d win=%d train_stride=%d features=%d classes=%d",
            int(windows.shape[0]),
            self.batch_size,
            self.num_epochs,
            self.win_size,
            self.train_stride,
            self._n_features,
            self._n_classes,
        )

        ds = TensorDataset(
            torch.from_numpy(windows.astype(np.float32)),
            torch.from_numpy(win_labels.astype(np.int64)),
        )
        device = self._resolve_device()
        pin_memory = device.type == "cuda"
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            pin_memory=pin_memory,
        )
        total_batches = len(loader)
        logger.info(
            "[RPCL] building model device=%s total_batches=%d",
            str(device),
            int(total_batches),
        )
        self._model = self._build_model(self._n_features, self._n_classes).to(device)
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        logger.info(
            "[RPCL] model ready device=%s trainable_params=%d",
            str(device),
            int(trainable_params),
        )
        optimizer = torch.optim.RAdam(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=self.lr,
        )
        ce = nn.CrossEntropyLoss()
        log_every = max(1, self.num_epochs // 10)

        for _epoch in range(self.num_epochs):
            t0 = time.perf_counter()
            last_progress = t0
            loss_sum = None
            loss_count = 0
            self._model.train()
            for batch_idx, (batch_x, batch_y) in enumerate(loader, start=1):
                batch_x = batch_x.to(device, non_blocking=pin_memory)
                batch_y = batch_y.to(device, non_blocking=pin_memory)
                logits, h_ie = self._model(batch_x)
                l_c = ce(logits, batch_y)
                l_s = _rpcl_loss(h_ie, self.gaussian_noise_std)
                loss = l_s + l_c / 2.0
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                detached_loss = loss.detach()
                loss_sum = detached_loss if loss_sum is None else loss_sum + detached_loss
                loss_count += 1
                now = time.perf_counter()
                should_log_progress = (
                    batch_idx == 1
                    or batch_idx == total_batches
                    or (
                        self.progress_log_interval > 0.0
                        and now - last_progress >= self.progress_log_interval
                    )
                )
                if should_log_progress:
                    logger.info(
                        "[RPCL] epoch=%d/%d batch=%d/%d loss=%.6f elapsed=%.1fs",
                        _epoch + 1,
                        self.num_epochs,
                        int(batch_idx),
                        int(total_batches),
                        float(detached_loss.detach().cpu()),
                        now - t0,
                    )
                    last_progress = now
            if _epoch == 0 or _epoch + 1 == self.num_epochs or (_epoch + 1) % log_every == 0:
                mean_loss = 0.0
                if loss_sum is not None and loss_count:
                    mean_loss = float((loss_sum / loss_count).detach().cpu())
                logger.info(
                    "[RPCL] epoch=%d/%d loss=%.6f time=%.1fs",
                    _epoch + 1,
                    self.num_epochs,
                    mean_loss,
                    time.perf_counter() - t0,
                )

        self._train_array = x
        self._train_scores = None

    def _prepare_inference_array(self, data) -> np.ndarray:
        x = to_2d_array(data)
        if self._n_features:
            if x.shape[1] < self._n_features:
                pad = np.zeros((x.shape[0], self._n_features - x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)
            elif x.shape[1] > self._n_features:
                x = x[:, : self._n_features]
        return self._scaler.transform(x)

    def _window_scores(self, arr: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.zeros(0, dtype=float)
        torch, _nn, F, _weight_norm = _torch_modules()
        from torch.utils.data import DataLoader, TensorDataset

        _configure_torch_runtime(torch)
        windows = _window_array(arr, self.win_size)
        if windows.shape[0] == 0:
            return np.zeros(0, dtype=float)
        device = self._resolve_device()
        pin_memory = device.type == "cuda"
        loader = DataLoader(
            TensorDataset(torch.from_numpy(windows.astype(np.float32))),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=pin_memory,
        )
        self._model.eval()
        chunks = []
        with torch.inference_mode():
            for (batch_x,) in loader:
                logits, _h_ie = self._model(batch_x.to(device, non_blocking=pin_memory))
                prob = F.softmax(logits, dim=-1)
                if prob.shape[1] <= 1:
                    score = 1.0 - prob[:, 0]
                else:
                    score = prob[:, 1:].sum(dim=1)
                chunks.append(score.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.zeros(0, dtype=float)

    def detect_score(self, test_data, covariates=None, **kwargs) -> np.ndarray:
        n = len(test_data) if test_data is not None else 0
        if self._model is None or n == 0:
            return np.zeros(n, dtype=float)
        arr = self._prepare_inference_array(test_data)
        return _align_endpoint_scores(self._window_scores(arr), n, self.win_size)

    def detect_label(self, test_data, covariates=None, test_label=None, **kwargs):
        n = len(test_data) if test_data is not None else 0
        if self._model is None or n == 0:
            empty = np.zeros(n, dtype=np.int32)
            aux = np.zeros(n, dtype=float)
            return {str(r): empty.copy() for r in self.anomaly_ratio}, aux

        arr = self._prepare_inference_array(test_data)
        raw_scores = self._window_scores(arr)
        scores = _align_endpoint_scores(raw_scores, n, self.win_size)
        if self._train_scores is None and self._train_array is not None:
            self._train_scores = self._window_scores(self._train_array)
        raw_preds = percentile_label_maps(raw_scores, self._train_scores, self.anomaly_ratio)
        preds = {
            ratio: _align_endpoint_labels(labels, n, self.win_size)
            for ratio, labels in raw_preds.items()
        }
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
        torch, _nn, _F, _weight_norm = _torch_modules()
        buf = io.BytesIO()
        torch.save(self._model.state_dict(), buf)
        return float(buf.tell() / (1024.0 * 1024.0))


def make_rpcl_tcne_mts_llm(**kwargs) -> RPCLTCNEMTSLLMModel:
    return RPCLTCNEMTSLLMModel(**kwargs)


__all__ = ["RPCLTCNEMTSLLMModel", "make_rpcl_tcne_mts_llm"]
