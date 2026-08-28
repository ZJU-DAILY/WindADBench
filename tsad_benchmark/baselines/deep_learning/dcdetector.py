# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from tsad_benchmark.baselines._dl_base import DLBaseModel
from tsad_benchmark.baselines.deep_learning._models.dcdetector import (
    DCdetector,
    my_kl_loss,
)
from tsad_benchmark.models.base import ModelCapability


class DCdetectorModel(DLBaseModel):
   

    model_name = "DCdetector"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        win_size: int = 105,
        batch_size: int = 64,
        num_epochs: int = 3,
        lr: float = 1e-4,
        val_ratio: float = 0.2,
        patience: int = 5,
        anomaly_ratio=None,
        d_model: int = 256,
        n_heads: int = 1,
        e_layers: int = 3,
        patch_size: Optional[Sequence[int]] = None,
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
        if patch_size is None:
            self.patch_size = [3, 5, 7]
        elif isinstance(patch_size, int):
            self.patch_size = [int(patch_size)]
        else:
            self.patch_size = [int(p) for p in patch_size]
        self.temperature = float(temperature)
        # Sanity: every patch must divide win_size — enforce here so
        # we surface a clear error, not a deep einops crash.
        for p in self.patch_size:
            if self.win_size % int(p) != 0:
                raise ValueError(
                    f"DCdetector requires win_size % patch == 0; "
                    f"got win_size={self.win_size}, patch={self.patch_size}."
                )
        self.model_hyper_params.update(
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            patch_size=self.patch_size,
        )

    # ------------------------------------------------------------------
    # DLBaseModel hooks
    # ------------------------------------------------------------------

    def _build_network(self, n_features: int):
        return DCdetector(
            win_size=self.win_size,
            enc_in=n_features,
            c_out=n_features,
            n_heads=self.n_heads,
            d_model=self.d_model,
            e_layers=self.e_layers,
            patch_size=self.patch_size,
            channel=n_features,
            output_attention=True,
        )

    # --- scalar association-discrepancy losses --------------------------

    def _series_prior_losses(self, series, prior):
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
        # batch_x: (B, W, F)
        series, prior = model(batch_x)
        s_loss, p_loss = self._series_prior_losses(series, prior)
        loss = p_loss - s_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    def _compute_window_scores(self, model, batch_x, criterion) -> np.ndarray:
        import torch

        with torch.no_grad():
            series, prior = model(batch_x)
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
                # Per-timestep KL — sum over key axis, mean over heads → (B, W)
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
        return metric.detach().cpu().numpy()


def make_dcdetector(**kwargs) -> DCdetectorModel:
    return DCdetectorModel(**kwargs)


__all__ = ["DCdetectorModel", "make_dcdetector"]
