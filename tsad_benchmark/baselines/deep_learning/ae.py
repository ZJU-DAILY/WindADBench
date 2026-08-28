# -*- coding: utf-8 -*-


from __future__ import annotations

from tsad_benchmark.baselines._merlion_base import MerlionBaseModel


class AutoEncoderModel(MerlionBaseModel):
    """Wrapper around ``merlion.models.anomaly.autoencoder.AutoEncoder``."""

    model_name = "AutoEncoder"

    def __init__(
        self,
        hidden_dim: int = 5,
        sequence_len: int = 1,
        num_epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.sequence_len = int(sequence_len)
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.model_hyper_params.update(
            {
                "hidden_dim": self.hidden_dim,
                "sequence_len": self.sequence_len,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
            }
        )

    def _build_merlion_model(self):
        from merlion.models.anomaly.autoencoder import AutoEncoder, AutoEncoderConfig

        config = AutoEncoderConfig(
            hidden_size=self.hidden_dim,
            sequence_len=self.sequence_len,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
        )
        return AutoEncoder(config)
