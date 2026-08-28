# -*- coding: utf-8 -*-

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_GCONST = -0.9189385332046727


def create_masks(input_size, hidden_size, n_hidden, input_order="sequential", input_degrees=None):
    degrees = []
    if input_size > 1:
        if input_order == "sequential":
            degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
            for _ in range(n_hidden + 1):
                degrees += [torch.arange(hidden_size) % (input_size - 1)]
            degrees += [
                torch.arange(input_size) % input_size - 1
                if input_degrees is None
                else input_degrees % input_size - 1
            ]
        elif input_order == "random":
            degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
            for _ in range(n_hidden + 1):
                min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
                degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [
                torch.randint(min_prev_degree, input_size, (input_size,)) - 1
                if input_degrees is None
                else input_degrees - 1
            ]
        else:
            raise ValueError("input_order must be sequential or random")
    else:
        degrees += [torch.zeros([1]).long()]
        for _ in range(n_hidden + 1):
            degrees += [torch.zeros([hidden_size]).long()]
        degrees += [torch.zeros([input_size]).long()]
    masks = [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float() for d0, d1 in zip(degrees[:-1], degrees[1:])]
    return masks, degrees[0]


class MaskedLinear(nn.Linear):
    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)
        self.register_buffer("mask", mask)
        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size))

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out


class BatchNorm(nn.Module):
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))
        self.register_buffer("running_mean", torch.zeros(input_size))
        self.register_buffer("running_var", torch.ones(input_size))

    def forward(self, x, y=None):
        if self.training and x.shape[0] > 1:
            mean = x.mean(0)
            var = x.var(0, unbiased=False)
            self.running_mean.mul_(self.momentum).add_(mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(var.data * (1 - self.momentum))
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        out = self.log_gamma.exp() * x_hat + self.beta
        log_abs_det = self.log_gamma - 0.5 * torch.log(var + self.eps)
        return out, log_abs_det.expand_as(x)

    def inverse(self, y, cond_y=None):
        mean = self.running_mean
        var = self.running_var
        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean
        log_abs_det = 0.5 * torch.log(var + self.eps) - self.log_gamma
        return x, log_abs_det.expand_as(y)


class MADE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="tanh",
        input_order="sequential",
        input_degrees=None,
    ):
        super().__init__()
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)
        activation_fn = nn.Tanh() if activation == "tanh" else nn.ReLU()
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        layers = []
        for mask in masks[1:-1]:
            layers += [activation_fn, MaskedLinear(hidden_size, hidden_size, mask)]
        layers += [activation_fn, MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x, y=None):
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
        u = (x - m) * torch.exp(-loga)
        return u, -loga


class FlowSequential(nn.Sequential):
    def forward(self, x, y=None):
        total_log_abs_det = 0
        for module in self:
            x, log_abs_det = module(x, y)
            total_log_abs_det = total_log_abs_det + log_abs_det
        return x, total_log_abs_det


class MAF(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_sensor,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="tanh",
        batch_norm=False,
    ):
        super().__init__()
        self.register_buffer("base_dist_mean", torch.randn(n_sensor, input_size))
        self.register_buffer("base_dist_var", torch.ones(n_sensor, input_size))
        modules = []
        input_degrees = None
        for _ in range(n_blocks):
            made = MADE(input_size, hidden_size, n_hidden, cond_label_size, activation, "sequential", input_degrees)
            modules.append(made)
            input_degrees = made.input_degrees.flip(0)
            if batch_norm:
                modules.append(BatchNorm(input_size))
        self.net = FlowSequential(*modules)

    def base_log_prob(self, z, n_sensor, window_size):
        n = z.shape[0] // n_sensor // window_size
        mean = self.base_dist_mean.repeat_interleave(window_size, dim=0).repeat(n, 1)
        return -0.5 * (z - mean) ** 2

    def log_prob(self, x, n_sensor, window_size, y=None):
        u, log_abs_det = self.net(x, y)
        return torch.sum(self.base_log_prob(u, n_sensor, window_size) + log_abs_det, dim=1) + x.shape[1] * _GCONST


class GNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, graph):
        h_n = self.lin_n(torch.einsum("bkld,bkj->bjld", h, graph))
        h_r = self.lin_r(h[:, :, :-1])
        h_n[:, :, 1:] += h_r
        return self.lin_2(F.relu(h_n))


class ScaleDotProductAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.w_q = nn.Linear(channels, channels)
        self.w_k = nn.Linear(channels, channels)
        self.w_v = nn.Linear(channels, channels)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask=None):
        shape = x.shape
        flat = x.reshape((shape[0], shape[1], -1))
        batch_size, length, channels = flat.size()
        q = self.w_q(flat)
        k = self.w_k(flat)
        score = (q @ k.view(batch_size, channels, length)) / math.sqrt(channels)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        return self.dropout(self.softmax(score))


class MTGFlowNetwork(nn.Module):
    def __init__(
        self,
        n_blocks,
        input_size,
        hidden_size,
        n_hidden,
        window_size,
        n_sensor,
        dropout=0.0,
        batch_norm=False,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.n_sensor = int(n_sensor)
        self.input_size = int(input_size)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.0 if dropout <= 0 else dropout,
        )
        self.gcn = GNN(hidden_size, hidden_size)
        self.nf = MAF(
            n_blocks,
            n_sensor,
            input_size,
            hidden_size,
            n_hidden,
            cond_label_size=hidden_size,
            batch_norm=batch_norm,
            activation="tanh",
        )
        self.attention = ScaleDotProductAttention(window_size * input_size)
        self.graph = None

    def log_prob_matrix(self, x):
        # x: (B, K, L, D)
        full_shape = x.shape
        graph = self.attention(x)
        self.graph = graph
        flat_seq = x.reshape((full_shape[0] * full_shape[1], full_shape[2], full_shape[3]))
        h, _ = self.rnn(flat_seq)
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h = self.gcn(h, graph).reshape((-1, h.shape[3]))
        flat_x = flat_seq.reshape((-1, full_shape[3]))
        return self.nf.log_prob(flat_x, full_shape[1], full_shape[2], h).reshape([full_shape[0], -1])

    def forward(self, x):
        return self.log_prob_matrix(x).mean()

    def anomaly_scores(self, x):
        return -self.log_prob_matrix(x).mean(dim=1)


__all__ = [
    "BatchNorm",
    "GNN",
    "MADE",
    "MAF",
    "MTGFlowNetwork",
    "ScaleDotProductAttention",
]
