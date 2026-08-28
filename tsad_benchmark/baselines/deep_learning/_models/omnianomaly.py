# -*- coding: utf-8 -*-
# Adapted from a public PyTorch OmniAnomaly reproduction in
# https://huggingface.co/spaces/thu-sail-lab/Time_RCD/blob/main/models/OmniAnomaly.py
# which itself cites the original NetManAIOps/OmniAnomaly implementation.

from __future__ import annotations

import torch
import torch.nn as nn


class OmniAnomalyNetwork(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        n_layers: int = 2,
        beta: float = 0.01,
    ):
        super().__init__()
        self.name = "OmniAnomaly"
        self.n_features = int(n_features)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.n_layers = int(n_layers)
        self.beta = float(beta)

        self.rnn = nn.GRU(self.n_features, self.hidden_dim, self.n_layers)
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.PReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.PReLU(),
            nn.Linear(self.hidden_dim, 2 * self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.PReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.PReLU(),
            nn.Linear(self.hidden_dim, self.n_features),
            nn.Sigmoid(),
        )

    def forward(self, x, hidden=None):
        batch_size, window_size, _ = x.shape
        if hidden is not None:
            hidden = torch.rand(
                self.n_layers,
                batch_size,
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype,
            )
        out, hidden = self.rnn(x.view(-1, batch_size, self.n_features), hidden)
        encoded = self.encoder(out)
        mu, logvar = torch.split(encoded, [self.latent_dim, self.latent_dim], dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decoder(z)
        return (
            recon.reshape(batch_size, window_size * self.n_features),
            mu.reshape(batch_size, window_size * self.latent_dim),
            logvar.reshape(batch_size, window_size * self.latent_dim),
            hidden,
        )

    def loss(self, batch, criterion, hidden=None):
        y_pred, mu, logvar, hidden = self.forward(batch, hidden)
        target = batch.reshape(batch.shape[0], -1)
        mse = torch.mean(criterion(y_pred, target), dim=-1)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return torch.mean(mse + self.beta * kld), hidden

    def window_scores(self, batch, criterion, hidden=None):
        y_pred, _mu, _logvar, hidden = self.forward(batch, hidden)
        target = batch.reshape(batch.shape[0], -1)
        scores = torch.mean(criterion(y_pred, target), dim=-1)
        return scores, hidden


__all__ = ["OmniAnomalyNetwork"]
