from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from tsad_benchmark.baselines._dl_base import DLBaseModel
from tsad_benchmark.baselines.deep_learning._models.timesnet import Model
from tsad_benchmark.models.base import ModelCapability


class TimesNetModel(DLBaseModel):
    model_name = "TimesNet"
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
        d_model: int = 64,
        d_ff: int = 64,
        e_layers: int = 2,
        top_k: int = 5,
        num_kernels: int = 6,
        dropout: float = 0.1,
        score_protocol: str = "independent_window_v1",
        score_batch_size: int = 1,
    ) -> None:
        if score_protocol != "independent_window_v1":
            raise ValueError("TimesNet supports score_protocol='independent_window_v1'.")
        if isinstance(score_batch_size, bool) or score_batch_size != 1:
            raise ValueError("TimesNet independent-window scoring requires score_batch_size=1.")
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
        self.top_k = int(top_k)
        self.num_kernels = int(num_kernels)
        self.dropout = float(dropout)
        self.score_protocol = score_protocol
        self.score_batch_size = 1
        self.model_hyper_params.update(
            d_model=self.d_model,
            d_ff=self.d_ff,
            e_layers=self.e_layers,
            top_k=self.top_k,
            num_kernels=self.num_kernels,
            score_protocol=self.score_protocol,
            score_batch_size=self.score_batch_size,
        )

    def _make_loader(self, arr: np.ndarray, mode: str, shuffle: bool):
        loader = super()._make_loader(arr, mode, shuffle)
        if (loader is None or mode != "thre"
                or getattr(self, "score_protocol", None) != "independent_window_v1"):

            return loader
        from torch.utils.data import DataLoader

        return DataLoader(loader.dataset, batch_size=self.score_batch_size,
                          shuffle=shuffle, num_workers=0, drop_last=False)

    def _build_network(self, n_features: int):
        configs = SimpleNamespace(
            task_name="anomaly_detection",
            seq_len=self.win_size,
            label_len=0,
            pred_len=0,
            enc_in=n_features,
            c_out=n_features,
            d_model=self.d_model,
            d_ff=self.d_ff,
            e_layers=self.e_layers,
            top_k=self.top_k,
            num_kernels=self.num_kernels,
            dropout=self.dropout,
        )
        return Model(configs)

    def _train_one_step(self, model, batch_x, criterion, epoch, optimizer) -> float:
        outputs = model(batch_x, None, None, None)
        loss = criterion(outputs, batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    def _compute_window_scores(self, model, batch_x, criterion) -> np.ndarray:
        import torch

        with torch.no_grad():
            outputs = model(batch_x, None, None, None)
            score = torch.mean((batch_x - outputs) ** 2, dim=-1)
        return score.detach().cpu().numpy()


def make_timesnet(**kwargs) -> TimesNetModel:
    return TimesNetModel(**kwargs)


__all__ = ["TimesNetModel", "make_timesnet"]
