# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from tsad_benchmark.baselines._dl_base import DLBaseModel
from tsad_benchmark.baselines.deep_learning._models.anomaly_transformer import (
    AnomalyTransformer,
    my_kl_loss,
)
from tsad_benchmark.models.base import ModelCapability


class AnomalyTransformerModel(DLBaseModel):
 
    model_name = "AnomalyTransformer"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        win_size: int = 100,
        batch_size: int = 64,
        num_epochs: int = 3,
        lr: float = 1e-4,
        val_ratio: float = 0.2,
        patience: int = 3,
        anomaly_ratio=None,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        k: float = 3.0,
        temperature: float = 50.0,
    ) -> None:
        super().__init__(
            win_size=win_size,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            val_ratio=val_ratio,
            patience=patience,
            anomaly_ratio=anomaly_ratio,
        )
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.e_layers = int(e_layers)
        self.d_ff = int(d_ff)
        self.k = float(k)
        self.temperature = float(temperature)
        self.model_hyper_params.update(
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            k=self.k,
        )

    # ------------------------------------------------------------------
    # DLBaseModel hooks
    # ------------------------------------------------------------------

    def _build_network(self, n_features: int):
        return AnomalyTransformer(
            win_size=self.win_size,
            enc_in=n_features,
            c_out=n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            output_attention=True,
        )

    # ----- per-layer association-discrepancy helpers --------------------

    def _series_prior_losses(self, series, prior):
        """Author's two-direction symmetric KL summed across encoder layers."""
        import torch

        win = self.win_size
        s_loss = 0.0
        p_loss = 0.0
        for u in range(len(prior)):
            prior_norm = (
                prior[u]
                / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                    1, 1, 1, win
                )
            )
            s_loss = s_loss + (
                torch.mean(my_kl_loss(series[u], prior_norm.detach()))
                + torch.mean(my_kl_loss(prior_norm.detach(), series[u]))
            )
            p_loss = p_loss + (
                torch.mean(my_kl_loss(prior_norm, series[u].detach()))
                + torch.mean(my_kl_loss(series[u].detach(), prior_norm))
            )
        s_loss = s_loss / len(prior)
        p_loss = p_loss / len(prior)
        return s_loss, p_loss

    def _train_one_step(self, model, batch_x, criterion, epoch, optimizer) -> float:
        import torch

        # batch_x: (B, W, F) — already what AnomalyTransformer expects.
        output, series, prior, _ = model(batch_x)
        rec_loss = criterion(output, batch_x)
        s_loss, p_loss = self._series_prior_losses(series, prior)
        loss1 = rec_loss - self.k * s_loss
        loss2 = rec_loss + self.k * p_loss

        optimizer.zero_grad()
        loss1.backward(retain_graph=True)
        loss2.backward()
        optimizer.step()
        return float(loss1.detach().cpu())

    def _compute_window_scores(self, model, batch_x, criterion) -> np.ndarray:
        import torch

        with torch.no_grad():
            output, series, prior, _ = model(batch_x)
            # MSE per (B, W, F) → mean over feature axis → (B, W).
            mse_per_t = torch.mean((batch_x - output) ** 2, dim=-1)

            # Per-timestep two-direction symmetric KL summed across layers
            # (no batch- / time-axis reduction → shape (B, W)).
            win = self.win_size
            s = None
            p = None
            for u in range(len(prior)):
                prior_norm = (
                    prior[u]
                    / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                        1, 1, 1, win
                    )
                )
                # res_p: (B, H, W, W) → sum over last → (B, H, W) → mean over H → (B, W)
                res_p = series[u] * (
                    torch.log(series[u] + 1e-4) - torch.log(prior_norm.detach() + 1e-4)
                )
                s_u = res_p.sum(dim=-1).mean(dim=1)
                res_q = prior_norm * (
                    torch.log(prior_norm + 1e-4) - torch.log(series[u].detach() + 1e-4)
                )
                p_u = res_q.sum(dim=-1).mean(dim=1)
                s = s_u if s is None else s + s_u
                p = p_u if p is None else p + p_u
            s = s * self.temperature
            p = p * self.temperature
            metric = torch.softmax(-(s + p), dim=-1)  # (B, W)
            cri = metric * mse_per_t                  # (B, W)
        return cri.detach().cpu().numpy()


def make_anomaly_transformer(**kwargs) -> AnomalyTransformerModel:
    return AnomalyTransformerModel(**kwargs)


__all__ = ["AnomalyTransformerModel", "make_anomaly_transformer"]
