# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Optional

import einops
import torch
from torch import Tensor, nn


class SpatialEncoding(nn.Module):
    def __init__(self, input_size: int, model_size: int, requires_grad: bool = True):
        super().__init__()
        self.se = nn.Parameter(
            torch.randn(1, 1, input_size, model_size),
            requires_grad=requires_grad,
        )

    def forward(self) -> Tensor:
        return self.se


class Embedding(nn.Module):
    def __init__(
        self,
        input_size: int,
        patch_size: int,
        model_size: int,
        dropout: float,
    ):
        super().__init__()
        self.encoding = nn.Linear(patch_size, model_size)
        self.spatial_encoding = SpatialEncoding(input_size, model_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoding(x) + self.spatial_encoding()
        return self.dropout(x)


class Patching(nn.Module):
    def __init__(self, num_patches: int):
        super().__init__()
        self.num_patches = int(num_patches)

    def forward(self, x: Tensor) -> Tensor:
        return einops.rearrange(x, "b (p s) i -> b p i s", p=self.num_patches)


class Unpatching(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return einops.rearrange(x, "b p i s -> b (p s) i")


class Attention(nn.Module):
    def __init__(
        self,
        input_size: int,
        model_size: int,
        n_heads: int,
        dropout: float,
        bias: bool,
        is_diagnoal_masked: bool,
    ):
        super().__init__()
        self.model_size = int(model_size)
        self.n_heads = int(n_heads)
        self.head_size = self.model_size // self.n_heads
        assert self.model_size % self.n_heads == 0, "model_size must be divisible by n_heads"

        self.Q = nn.Linear(self.model_size, self.model_size, bias)
        self.K = nn.Linear(self.model_size, self.model_size, bias)
        self.V = nn.Linear(self.model_size, self.model_size, bias)
        self.linear = nn.Linear(self.model_size, self.model_size)
        self.dropout = nn.Dropout(dropout)
        self.is_diagnoal_masked = bool(is_diagnoal_masked)

        diag_mask = 1.0 - torch.eye(input_size, input_size).unsqueeze(0).unsqueeze(0)
        self.register_buffer("diag_mask", diag_mask)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, s: Optional[Tensor] = None):
        batch_size, input_size, _ = q.size()
        v = self.V(v).view(batch_size, input_size, self.n_heads, self.head_size)

        if s is None:
            q = self.Q(q).view(batch_size, input_size, self.n_heads, self.head_size)
            k = self.K(k).view(batch_size, input_size, self.n_heads, self.head_size)
            scores = torch.einsum("bqhe,bkhe->bhqk", [q, k]) / math.sqrt(self.head_size)
            s = torch.softmax(scores, dim=-1)
            if self.is_diagnoal_masked:
                s = s * self.diag_mask
                s = s / (s.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            if self.is_diagnoal_masked:
                s = s * self.diag_mask
            s = s - s.min(dim=1, keepdim=True)[0]
            s = s / (s.sum(dim=-1, keepdim=True) + 1e-6)

        s_d = self.dropout(s)
        attention = torch.einsum("bhql,blhd->bqhd", [s_d, v])
        attention = attention.reshape(batch_size, input_size, self.model_size)
        attention = self.linear(attention)
        return attention, s


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        model_size: int,
        feedforward_size: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        is_diagnoal_masked: bool,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.attention = Attention(
            input_size,
            model_size,
            num_heads,
            dropout,
            bias,
            is_diagnoal_masked,
        )
        self.norm1 = nn.LayerNorm(model_size)
        self.norm2 = nn.LayerNorm(model_size)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_size, feedforward_size, bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, model_size, bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, s: Optional[Tensor] = None):
        if x.dim() == 4:
            batch_size, num_patches, input_size, model_size = x.size()
            z = x.view(batch_size * num_patches, input_size, model_size)
        else:
            z = x
            batch_size = num_patches = input_size = model_size = None

        attention, s = self.attention(z, z, z, s)
        z = self.norm1(z + self.dropout(attention))
        z = self.norm2(z + self.feed_forward(z))

        if x.dim() == 4:
            z = z.view(batch_size, num_patches, input_size, model_size)
            s = s.view(batch_size, num_patches, self.num_heads, input_size, input_size)
        return z, s


class Decoder(nn.Module):
    def __init__(self, patch_size: int, model_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(model_size)
        self.linear = nn.Linear(model_size, patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.ln(x))


class SAR73(nn.Module):
    def __init__(
        self,
        input_size: int,
        window_size: int,
        model_size: int,
        num_layers: int,
        num_heads: int,
        num_patches: int,
        dropout: float,
        is_diagnoal_masked: bool = True,
    ):
        super().__init__()
        feedforward_size = 4 * model_size
        patch_size = window_size // num_patches
        self.patching = Patching(num_patches)
        self.embedding = Embedding(input_size, patch_size, model_size, dropout)
        self.encoders = nn.ModuleList(
            [
                SpatialEncoder(
                    input_size,
                    model_size,
                    feedforward_size,
                    num_heads,
                    dropout,
                    bias=True,
                    is_diagnoal_masked=is_diagnoal_masked,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = Decoder(patch_size, model_size)
        self.unpatching = Unpatching()

    def forward(self, x: Tensor):
        x = self.patching(x)
        z = self.embedding(x)
        s_all = []
        for encoder in self.encoders:
            z, s_layer = encoder(z)
            s_all.append(s_layer)
        s_all = torch.stack(s_all, dim=1)
        x_hat = self.decoder(z)
        x_hat = self.unpatching(x_hat)
        return x_hat, s_all


class SAR76(SAR73):
    def __init__(
        self,
        input_size: int,
        window_size: int,
        model_size: int,
        num_layers: int,
        num_heads: int,
        num_patches: int,
        detector_size: int,
        dropout: float,
        is_diagnoal_masked: bool = False,
    ):
        assert num_patches == 2, "Number of patches must be 2."
        super().__init__(
            input_size,
            window_size,
            model_size,
            num_layers,
            num_heads,
            num_patches,
            dropout,
            is_diagnoal_masked,
        )
        self.input_size = int(input_size)
        self.detector = nn.Sequential(
            nn.Linear(num_heads * input_size, detector_size),
            nn.ReLU(),
            nn.Linear(detector_size, num_heads * input_size),
            nn.ReLU(),
        )

        self.register_buffer("detec_avg", torch.tensor(0.0))
        self.register_buffer("detec_std", torch.tensor(1.0))
        self.register_buffer("recon_avg", torch.tensor(0.0))
        self.register_buffer("recon_std", torch.tensor(1.0))
        self.register_buffer("rdiag_avg", torch.zeros(input_size))
        self.register_buffer("rdiag_std", torch.ones(input_size))
        self.register_buffer("ddiag_avg", torch.zeros(input_size))
        self.register_buffer("ddiag_std", torch.ones(input_size))

    def forward(self, x: Tensor):
        x = self.patching(x)
        z = self.embedding(x)
        s_all = []
        for encoder in self.encoders:
            z, s_layer = encoder(z)
            s_all.append(s_layer)
        s_all = torch.stack(s_all, dim=1)
        x_hat = self.decoder(z)
        x_hat = self.unpatching(x_hat)

        s_reduced = s_all.detach()[:, :, 0] - s_all.detach()[:, :, 1]
        s_reduced = torch.nn.functional.relu(s_reduced)
        q = s_reduced[:, -1].sum(-2)
        q_bar = self.detector(q.flatten(1)).reshape_as(q)
        return x_hat, s_all, q, q_bar


__all__ = [
    "Attention",
    "Decoder",
    "Embedding",
    "Patching",
    "SAR73",
    "SAR76",
    "SpatialEncoder",
    "SpatialEncoding",
    "Unpatching",
]
