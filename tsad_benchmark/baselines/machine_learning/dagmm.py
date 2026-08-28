# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

from tsad_benchmark.baselines._merlion_base import MerlionBaseModel


class DAGMMModel(MerlionBaseModel):
 
    model_name = "DAGMM"

    def __init__(
        self,
        hidden_dim: int = 64,
        n_gmm: int = 4,
        sequence_len: int = 1,
        num_epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_gmm = int(n_gmm)
        self.sequence_len = int(sequence_len)
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.model_hyper_params.update(
            {
                "hidden_dim": self.hidden_dim,
                "n_gmm": self.n_gmm,
                "sequence_len": self.sequence_len,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
            }
        )

    def _build_merlion_model(self):
        from merlion.models.anomaly.dagmm import DAGMM, DAGMMConfig
        config = DAGMMConfig(
            hidden_size=self.hidden_dim,
            sequence_len=self.sequence_len,
            gmm_k=self.n_gmm,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
        )
        return DAGMM(config)
