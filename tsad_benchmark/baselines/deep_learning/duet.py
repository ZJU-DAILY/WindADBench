# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from tsad_benchmark.baselines._dl_base import DLBaseModel
from tsad_benchmark.baselines.deep_learning._models.duet.duet_model import DUETModel
from tsad_benchmark.models.base import ModelCapability


class DUETAnomalyModel(DLBaseModel):
    """DUET (Qiu et al., KDD 2025) — anomaly_detection task."""

    model_name = "DUET"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        win_size: int = 96,
        batch_size: int = 256,
        num_epochs: int = 3,
        lr: float = 0.02,
        val_ratio: float = 0.2,
        patience: int = 10,
        anomaly_ratio=None,
        d_model: int = 512,
        d_ff: int = 2048,
        e_layers: int = 2,
        n_heads: int = 8,
        factor: int = 1,
        dropout: float = 0.2,
        fc_dropout: float = 0.2,
        activation: str = "gelu",
        moving_avg: int = 25,
        num_experts: int = 4,
        k: int = 1,
        hidden_size: int = 256,
        noisy_gating: bool = True,
        CI: bool = True,
        output_attention: int = 0,
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
        self.d_ff = int(d_ff)
        self.e_layers = int(e_layers)
        self.n_heads = int(n_heads)
        self.factor = int(factor)
        self.dropout = float(dropout)
        self.fc_dropout = float(fc_dropout)
        self.activation = str(activation)
        self.moving_avg = int(moving_avg)
        self.num_experts = int(num_experts)
        self.k = int(k)
        self.hidden_size = int(hidden_size)
        self.noisy_gating = bool(noisy_gating)
        self.CI = bool(CI)
        self.output_attention = int(output_attention)
        self.model_hyper_params.update(
            d_model=self.d_model,
            d_ff=self.d_ff,
            e_layers=self.e_layers,
            n_heads=self.n_heads,
            num_experts=self.num_experts,
            k=self.k,
            CI=self.CI,
        )

    # ------------------------------------------------------------------
    # DLBaseModel hooks
    # ------------------------------------------------------------------

    def _build_network(self, n_features: int):
        configs = SimpleNamespace(
            seq_len=self.win_size,
            pred_len=self.win_size,            # anomaly_detection: horizon == seq_len
            enc_in=n_features,
            dec_in=n_features,
            c_out=n_features,
            d_model=self.d_model,
            d_ff=self.d_ff,
            e_layers=self.e_layers,
            n_heads=self.n_heads,
            factor=self.factor,
            dropout=self.dropout,
            fc_dropout=self.fc_dropout,
            activation=self.activation,
            moving_avg=self.moving_avg,
            num_experts=self.num_experts,
            k=self.k,
            hidden_size=self.hidden_size,
            noisy_gating=self.noisy_gating,
            CI=self.CI,
            output_attention=self.output_attention,
        )
        return DUETModel(configs)

    def _train_one_step(self, model, batch_x, criterion, epoch, optimizer) -> float:
        # batch_x: (B, W, F)
        output, _ = model(batch_x)
        loss = criterion(output, batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    def _compute_window_scores(self, model, batch_x, criterion) -> np.ndarray:
        import torch

        with torch.no_grad():
            output, _ = model(batch_x)
            score = torch.mean((batch_x - output) ** 2, dim=-1)  # (B, W)
        return score.detach().cpu().numpy()


def make_duet(**kwargs) -> DUETAnomalyModel:
    return DUETAnomalyModel(**kwargs)


__all__ = ["DUETAnomalyModel", "make_duet"]
