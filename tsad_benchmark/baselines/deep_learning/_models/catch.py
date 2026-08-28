# -*- coding: utf-8 -*-

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
from torch.nn.functional import gumbel_softmax


# ---------------------------------------------------------------------------
# RevIN
# ---------------------------------------------------------------------------


class RevIN(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        if mode == "transform":
            return self._normalize(x)
        raise NotImplementedError(mode)

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# ---------------------------------------------------------------------------
# Channel-mask generator
# ---------------------------------------------------------------------------


class channel_mask_generator(nn.Module):
    def __init__(self, input_size: int, n_vars: int):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_size * 2, n_vars, bias=False), nn.Sigmoid()
        )
        with torch.no_grad():
            self.generator[0].weight.zero_()
        self.n_vars = n_vars

    def forward(self, x):
        distribution_matrix = self.generator(x)
        resample_matrix = self._bernoulli_gumbel_rsample(distribution_matrix)

        inverse_eye = 1 - torch.eye(self.n_vars).to(x.device)
        diag = torch.eye(self.n_vars).to(x.device)
        resample_matrix = (
            torch.einsum("bcd,cd->bcd", resample_matrix, inverse_eye) + diag
        )
        return resample_matrix

    def _bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape
        flatten_matrix = rearrange(distribution_matrix, "b c d -> (b c d) 1")
        r_flatten_matrix = 1 - flatten_matrix
        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)
        new_matrix = torch.cat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)
        resample_matrix = rearrange(
            resample_matrix[..., 0], "(b c d) -> b c d", b=b, c=c, d=d
        )
        return resample_matrix


# ---------------------------------------------------------------------------
# Cross-channel Transformer
# ---------------------------------------------------------------------------


class DynamicalContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, k: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.k = k

    def forward(self, scores, attn_mask, norm_matrix):
        b = scores.shape[0]
        n_vars = scores.shape[-1]
        cosine = (scores / norm_matrix).mean(1)
        pos_scores = torch.exp(cosine / self.temperature) * attn_mask
        all_scores = torch.exp(cosine / self.temperature)
        clustering_loss = -torch.log(
            pos_scores.sum(dim=-1) / all_scores.sum(dim=-1)
        )
        eye = (
            torch.eye(attn_mask.shape[-1])
            .unsqueeze(0)
            .repeat(b, 1, 1)
            .to(attn_mask.device)
        )
        regular_loss = (
            1
            / (n_vars * (n_vars - 1))
            * torch.norm(
                eye.reshape(b, -1) - attn_mask.reshape((b, -1)), p=1, dim=-1
            )
        )
        loss = clustering_loss.mean(1) + self.k * regular_loss
        return loss.mean()


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class c_Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.8,
        regular_lambda: float = 0.3,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head * heads
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.dynamicalContranstiveLoss = DynamicalContrastiveLoss(
            k=regular_lambda, temperature=temperature
        )

    def forward(self, x, attn_mask=None):
        h = self.heads
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=h)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=h)
        scale = 1 / self.d_k

        scores = einsum("b h i d, b h j d -> b h i j", q, k)
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        norm_matrix = torch.einsum("bhid,bhjd->bhij", q_norm, k_norm)

        dynamical_contrastive_loss = None
        if attn_mask is not None:
            large_negative = -math.log(1e10)
            attention_mask = torch.where(attn_mask == 0, large_negative, 0.0)
            masked_scores = scores * attn_mask.unsqueeze(1) + attention_mask.unsqueeze(1)
            dynamical_contrastive_loss = self.dynamicalContranstiveLoss(
                scores, attn_mask, norm_matrix
            )
        else:
            masked_scores = scores

        attn = self.attend(masked_scores * scale)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), attn, dynamical_contrastive_loss


class c_Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.8,
        regular_lambda: float = 0.3,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            c_Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                regular_lambda=regular_lambda,
                                temperature=temperature,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, attn_mask=None):
        total_loss = 0
        attn = None
        for attn_layer, ff in self.layers:
            x_n, attn, dcloss = attn_layer(x, attn_mask=attn_mask)
            total_loss = total_loss + dcloss
            x = x_n + x
            x = ff(x) + x
        dcloss = total_loss / len(self.layers)
        return x, attn, dcloss


class Trans_C(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int,
        dropout: float,
        patch_dim: int,
        horizon: int,
        d_model: int,
        regular_lambda: float = 0.3,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim), nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            regular_lambda=regular_lambda,
            temperature=temperature,
        )
        self.mlp_head = nn.Linear(dim, d_model)

    def forward(self, x, attn_mask=None):
        x = self.to_patch_embedding(x)
        x, _, dcloss = self.transformer(x, attn_mask)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x, dcloss


# ---------------------------------------------------------------------------
# CATCHModel + Flatten_Head
# ---------------------------------------------------------------------------


class Flatten_Head(nn.Module):
    def __init__(self, individual: int, n_vars: int, nf: int, seq_len: int, head_dropout: float = 0.0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        if self.individual:
            self.linears1 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, seq_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, seq_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears1[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)
        return x


class CATCHModel(nn.Module):
    name = "CATCH"

    def __init__(self, configs, **kwargs):
        super().__init__()
        self.revin_layer = RevIN(
            configs.c_in, affine=configs.affine, subtract_last=configs.subtract_last
        )
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.seq_len = configs.seq_len
        self.horizon = self.seq_len
        patch_num = int(
            (configs.seq_len - configs.patch_size) / configs.patch_stride + 1
        )
        self.norm = nn.LayerNorm(self.patch_size)
        self.re_attn = True
        self.mask_generator = channel_mask_generator(
            input_size=configs.patch_size, n_vars=configs.c_in
        )
        self.frequency_transformer = Trans_C(
            dim=configs.cf_dim,
            depth=configs.e_layers,
            heads=configs.n_heads,
            mlp_dim=configs.d_ff,
            dim_head=configs.head_dim,
            dropout=configs.dropout,
            patch_dim=configs.patch_size * 2,
            horizon=self.horizon * 2,
            d_model=configs.d_model * 2,
            regular_lambda=configs.regular_lambda,
            temperature=configs.temperature,
        )
        self.head_nf_f = configs.d_model * 2 * patch_num
        self.n_vars = configs.c_in
        self.individual = configs.individual
        self.head_f1 = Flatten_Head(
            self.individual,
            self.n_vars,
            self.head_nf_f,
            configs.seq_len,
            head_dropout=configs.head_dropout,
        )
        self.head_f2 = Flatten_Head(
            self.individual,
            self.n_vars,
            self.head_nf_f,
            configs.seq_len,
            head_dropout=configs.head_dropout,
        )
        self.ircom = nn.Linear(self.seq_len * 2, self.seq_len)
        self.rfftlayer = nn.Linear(self.seq_len * 2 - 2, self.seq_len)
        self.final = nn.Linear(self.seq_len * 2, self.seq_len)
        self.get_r = nn.Linear(configs.d_model * 2, configs.d_model * 2)
        self.get_i = nn.Linear(configs.d_model * 2, configs.d_model * 2)

    def forward(self, z):
        z = self.revin_layer(z, "norm")
        z = z.permute(0, 2, 1)
        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag

        z1 = z1.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        z2 = z2.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)

        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]

        z1 = torch.reshape(z1, (batch_size * patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size * patch_num, c_in, z2.shape[-1]))
        z_cat = torch.cat((z1, z2), -1)

        channel_mask = self.mask_generator(z_cat)
        z, dcloss = self.frequency_transformer(z_cat, channel_mask)
        z1 = self.get_r(z)
        z2 = self.get_i(z)

        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)
        z1 = self.head_f1(z1)
        z2 = self.head_f2(z2)

        complex_z = torch.complex(z1, z2)
        z = torch.fft.ifft(complex_z)
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))

        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, "denorm")
        return z, complex_z.permute(0, 2, 1), dcloss


# ---------------------------------------------------------------------------
# Frequency reconstruction loss (auxiliary)
# ---------------------------------------------------------------------------


class frequency_loss(nn.Module):
    def __init__(self, configs, keep_dim: bool = False, dim=None):
        super().__init__()
        self.keep_dim = keep_dim
        self.dim = dim
        if configs.auxi_mode == "fft":
            self.fft = torch.fft.fft
        elif configs.auxi_mode == "rfft":
            self.fft = torch.fft.rfft
        else:
            raise NotImplementedError(configs.auxi_mode)
        self.configs = configs
        self.mask = None  # default: no mask (configs.mask=False)

    def forward(self, outputs, batch_y):
        if outputs.is_complex():
            frequency_outputs = outputs
        else:
            frequency_outputs = self.fft(outputs, dim=1)

        if self.configs.auxi_type == "complex":
            loss_auxi = frequency_outputs - self.fft(batch_y, dim=1)
        elif self.configs.auxi_type == "complex-phase":
            loss_auxi = (frequency_outputs - self.fft(batch_y, dim=1)).angle()
        elif self.configs.auxi_type == "phase":
            loss_auxi = frequency_outputs.angle() - self.fft(batch_y, dim=1).angle()
        elif self.configs.auxi_type == "mag":
            loss_auxi = frequency_outputs.abs() - self.fft(batch_y, dim=1).abs()
        else:
            raise NotImplementedError(self.configs.auxi_type)

        if self.mask is not None:
            loss_auxi = loss_auxi * self.mask
        if self.configs.auxi_loss == "MAE":
            loss_auxi = (
                loss_auxi.abs().mean(dim=self.dim, keepdim=self.keep_dim)
                if self.configs.module_first
                else loss_auxi.mean(dim=self.dim, keepdim=self.keep_dim).abs()
            )
        elif self.configs.auxi_loss == "MSE":
            loss_auxi = (
                (loss_auxi.abs() ** 2).mean(dim=self.dim, keepdim=self.keep_dim)
                if self.configs.module_first
                else (loss_auxi ** 2).mean(dim=self.dim, keepdim=self.keep_dim).abs()
            )
        else:
            raise NotImplementedError(self.configs.auxi_loss)
        return loss_auxi


class frequency_criterion(nn.Module):
    def __init__(self, configs):
        super().__init__()
        import numpy as np

        self._np = np
        self.metric = frequency_loss(configs, dim=1, keep_dim=True)
        self.patch_size = configs.inference_patch_size
        self.patch_stride = configs.inference_patch_stride
        self.win_size = configs.seq_len
        self.patch_num = int(
            (self.win_size - self.patch_size) / self.patch_stride + 1
        )
        self.padding_length = (
            self.win_size - (self.patch_size + (self.patch_num - 1) * self.patch_stride)
        )

    def forward(self, outputs, batch_y):
        np = self._np
        output_patch = outputs.unfold(
            dimension=1, size=self.patch_size, step=self.patch_stride
        )
        b, n, c, p = output_patch.shape
        output_patch = rearrange(output_patch, "b n c p -> (b n) p c")
        y_patch = batch_y.unfold(
            dimension=1, size=self.patch_size, step=self.patch_stride
        )
        y_patch = rearrange(y_patch, "b n c p -> (b n) p c")

        main_part_loss = self.metric(output_patch, y_patch)
        main_part_loss = main_part_loss.repeat(1, self.patch_size, 1)
        main_part_loss = rearrange(main_part_loss, "(b n) p c -> b n p c", b=b)

        end_point = self.patch_size + (self.patch_num - 1) * self.patch_stride - 1
        start_indices = np.array(range(0, end_point, self.patch_stride))
        end_indices = start_indices + self.patch_size

        indices = (
            torch.tensor(
                [range(start_indices[i], end_indices[i]) for i in range(n)]
            )
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        indices = indices.repeat(b, 1, 1, c).to(main_part_loss.device)
        main_loss = torch.zeros(
            (b, n, self.win_size - self.padding_length, c)
        ).to(main_part_loss.device)
        main_loss.scatter_(dim=2, index=indices, src=main_part_loss)

        non_zero_cnt = torch.count_nonzero(main_loss, dim=1)
        main_loss = main_loss.sum(1) / non_zero_cnt

        if self.padding_length > 0:
            padding_loss = self.metric(
                outputs[:, -self.padding_length :, :], batch_y[:, -self.padding_length :, :]
            )
            padding_loss = padding_loss.repeat(1, self.padding_length, 1)
            total_loss = torch.cat([main_loss, padding_loss], dim=1)
        else:
            total_loss = main_loss
        return total_loss
