# -*- coding: utf-8 -*-

from __future__ import annotations

from tsad_benchmark.baselines._merlion_base import MerlionBaseModel


class _SafeLSTMEDMixin:
    """Merlion LSTMED with scalar loss accounting and no-grad scoring."""

    def _train(self, train_data, train_config=None):
        import torch
        from merlion.models.utils.rolling_window_dataset import RollingWindowDataset
        from merlion.utils.misc import ProgressBar

        train_loader = RollingWindowDataset(
            train_data,
            target_seq_index=None,
            shuffle=True,
            flatten=False,
            n_past=self.sequence_length,
            n_future=0,
            batch_size=self.batch_size,
        )
        self.data_dim = train_data.shape[1]
        self.lstmed = self._build_model(train_data.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr)
        loss_func = torch.nn.MSELoss(reduction="sum")
        bar = ProgressBar(total=self.num_epochs)

        self.lstmed.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for batch, _, _, _ in train_loader:
                batch = torch.as_tensor(batch, dtype=torch.float, device=self.device)
                optimizer.zero_grad(set_to_none=True)
                output = self.lstmed(batch)
                loss = loss_func(output, batch)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
            if bar is not None:
                bar.print(
                    epoch + 1,
                    prefix="",
                    suffix="Complete, Loss {:.4f}".format(
                        total_loss / len(train_loader)
                    ),
                )
        return self._get_anomaly_score(train_data)

    def _get_anomaly_score(self, time_series, time_series_prev=None):
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        from merlion.models.utils.rolling_window_dataset import RollingWindowDataset

        self.lstmed.eval()
        ts = (
            time_series
            if time_series_prev is None
            else pd.concat((time_series_prev, time_series))
        )
        data_loader = RollingWindowDataset(
            ts,
            target_seq_index=None,
            shuffle=False,
            flatten=False,
            n_past=self.sequence_length,
            n_future=0,
            batch_size=self.batch_size,
        )

        scores = []
        with torch.no_grad():
            for batch, _, _, _ in data_loader:
                batch = torch.as_tensor(batch, dtype=torch.float, device=self.device)
                output = self.lstmed(batch)
                error = nn.L1Loss(reduction="none")(output, batch)
                score = error.reshape(-1, ts.shape[1]).mean(dim=1)
                scores.append(
                    score.detach()
                    .cpu()
                    .numpy()
                    .reshape(batch.shape[0], self.sequence_length)
                )

        if not scores:
            return pd.DataFrame(np.zeros(len(time_series)), index=time_series.index)

        scores = np.concatenate(scores)
        lattice = np.full((self.sequence_length, ts.shape[0]), np.nan)
        for i, score in enumerate(scores):
            lattice[i % self.sequence_length, i : i + self.sequence_length] = score
        scores = np.nanmean(lattice, axis=0)
        return pd.DataFrame(scores[-len(time_series) :], index=time_series.index)


def _safe_lstmed_class(base_cls):
    cls = globals().get("_SafeLSTMED")
    if cls is None or not issubclass(cls, base_cls):
        cls = type(
            "_SafeLSTMED",
            (_SafeLSTMEDMixin, base_cls),
            {
                "__module__": __name__,
                "__doc__": "Safe local subclass of merlion.models.anomaly.lstm_ed.LSTMED.",
            },
        )
        globals()["_SafeLSTMED"] = cls
    return cls


class LSTMEDModel(MerlionBaseModel):
    """Wrapper around ``merlion.models.anomaly.lstm_ed.LSTMED``."""

    model_name = "LSTMED"

    def __init__(
        self,
        hidden_dim: int = 5,
        sequence_len: int = 20,
        num_epochs: int = 10,
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
        from merlion.models.anomaly.lstm_ed import LSTMED, LSTMEDConfig

        config = LSTMEDConfig(
            hidden_size=self.hidden_dim,
            sequence_len=self.sequence_len,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
        )
        return _safe_lstmed_class(LSTMED)(config)
