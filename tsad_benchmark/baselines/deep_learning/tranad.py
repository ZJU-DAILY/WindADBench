# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from tsad_benchmark.baselines._dl_base import DLBaseModel
from tsad_benchmark.baselines._thresholding import pot_label_maps
from tsad_benchmark.baselines.deep_learning._models.tranad import TranAD
from tsad_benchmark.models.base import ModelCapability


class TranADModel(DLBaseModel):
    """TranAD wrapper (Tuli et al., VLDB 2022)."""

    model_name = "TranAD"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        win_size: int = 10,
        batch_size: int = 128,
        num_epochs: int = 5,
        lr: float = 1e-3,
        val_ratio: float = 0.2,
        patience: int = 3,
        anomaly_ratio=None,
        pot_level: float = 0.98,
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
        self.pot_level = float(pot_level)
        self.model_hyper_params.update(pot_level=self.pot_level)

    # ------------------------------------------------------------------
    # DLBaseModel hooks
    # ------------------------------------------------------------------

    def _build_network(self, n_features: int):
        # TranAD requires d_model = 2 * feats and nhead = feats.
        return TranAD(feats=n_features, n_window=self.win_size)

    def _train_one_step(self, model, batch_x, criterion, epoch, optimizer) -> float:
        import torch

        # batch_x: (B, W, F); the author implementation expects (W, B, F).
        window = batch_x.permute(1, 0, 2).contiguous()
        elem = window[-1:, :, :].contiguous()  # (1, B, F)
        x1, x2 = model(window, elem)
        n = float(epoch + 1)
        loss = (1.0 / n) * torch.mean((x1 - elem) ** 2) + (
            1.0 - 1.0 / n
        ) * torch.mean((x2 - elem) ** 2)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        return float(loss.detach().cpu())

    def _compute_window_scores(self, model, batch_x, criterion) -> np.ndarray:
        import torch

        with torch.no_grad():
            window = batch_x.permute(1, 0, 2).contiguous()  # (W, B, F)
            elem = window[-1:, :, :].contiguous()  # (1, B, F)
            _, x2 = model(window, elem)  # (1, B, F)
            # Keep DLBaseModel's per-window/per-timestep score contract.
            err = (window - x2) ** 2  # (W, B, F)
            score = err.mean(dim=-1).permute(1, 0)  # (B, W)
        return score.detach().cpu().numpy()

    def detect_label(self, test_data, covariates=None, **kwargs):
        n = len(test_data) if test_data is not None else 0
        if self._network is None or n == 0:
            empty = np.zeros(n, dtype=np.int32)
            aux = np.zeros(n, dtype=float)
            return {str(r): empty.copy() for r in self.anomaly_ratio}, aux

        import torch

        arr = self._prepare_inference_array(test_data)
        criterion = torch.nn.MSELoss()
        raw_scores = self._inference_scores(arr, criterion, mode="thre")
        test_scores = self._align_length(raw_scores, n)
        preds = pot_label_maps(
            raw_scores,
            self._train_scores,
            self.anomaly_ratio,
            level=self.pot_level,
        )
        preds = {
            ratio: self._align_label_length(labels, n)
            for ratio, labels in preds.items()
        }
        return preds, test_scores


def make_tranad(**kwargs) -> TranADModel:
    """Factory for ``--model-path`` use."""
    return TranADModel(**kwargs)


__all__ = ["TranADModel", "make_tranad"]

