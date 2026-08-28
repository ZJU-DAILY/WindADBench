# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd

from tsad_benchmark.baselines._dl_base import DLBaseModel, _EarlyStopping
from tsad_benchmark.baselines.deep_learning._models.catch import (
    CATCHModel,
    frequency_criterion,
    frequency_loss,
)
from tsad_benchmark.models.base import ModelCapability

logger = logging.getLogger(__name__)


class CATCHAnomalyModel(DLBaseModel):
    """CATCH (Wu et al., ICLR 2025) — anomaly_detection task."""

    model_name = "CATCH"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        win_size: int = 192,
        batch_size: int = 128,
        num_epochs: int = 3,
        lr: float = 1e-4,
        Mlr: float = 1e-5,
        val_ratio: float = 0.2,
        patience: int = 3,
        anomaly_ratio=None,
        e_layers: int = 3,
        n_heads: int = 2,
        cf_dim: int = 64,
        d_ff: int = 256,
        d_model: int = 128,
        head_dim: int = 64,
        individual: int = 0,
        dropout: float = 0.2,
        head_dropout: float = 0.1,
        patch_size: int = 16,
        patch_stride: int = 8,
        inference_patch_size: int = 32,
        inference_patch_stride: int = 1,
        regular_lambda: float = 0.5,
        temperature: float = 0.07,
        dc_lambda: float = 0.005,
        auxi_lambda: float = 0.005,
        score_lambda: float = 0.05,
        affine: int = 0,
        subtract_last: int = 0,
        pct_start: float = 0.3,
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
        self.Mlr = float(Mlr)
        self.e_layers = int(e_layers)
        self.n_heads = int(n_heads)
        self.cf_dim = int(cf_dim)
        self.d_ff = int(d_ff)
        self.d_model = int(d_model)
        self.head_dim = int(head_dim)
        self.individual = int(individual)
        self.dropout = float(dropout)
        self.head_dropout = float(head_dropout)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.inference_patch_size = int(inference_patch_size)
        self.inference_patch_stride = int(inference_patch_stride)
        self.regular_lambda = float(regular_lambda)
        self.temperature = float(temperature)
        self.dc_lambda = float(dc_lambda)
        self.auxi_lambda = float(auxi_lambda)
        self.score_lambda = float(score_lambda)
        self.affine = int(affine)
        self.subtract_last = int(subtract_last)
        self.pct_start = float(pct_start)
        self.model_hyper_params.update(
            d_model=self.d_model,
            d_ff=self.d_ff,
            e_layers=self.e_layers,
            n_heads=self.n_heads,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
        )
        self._freq_criterion: Optional[frequency_criterion] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_configs(self, n_features: int) -> SimpleNamespace:
        return SimpleNamespace(
            seq_len=self.win_size,
            c_in=n_features,
            enc_in=n_features,
            c_out=n_features,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            inference_patch_size=self.inference_patch_size,
            inference_patch_stride=self.inference_patch_stride,
            e_layers=self.e_layers,
            n_heads=self.n_heads,
            cf_dim=self.cf_dim,
            d_ff=self.d_ff,
            d_model=self.d_model,
            head_dim=self.head_dim,
            individual=self.individual,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
            regular_lambda=self.regular_lambda,
            temperature=self.temperature,
            affine=self.affine,
            subtract_last=self.subtract_last,
            auxi_loss="MAE",
            auxi_type="complex",
            auxi_mode="fft",
            module_first=True,
            mask=False,
        )

    def _build_network(self, n_features: int):
        return CATCHModel(self._make_configs(n_features))

    # ------------------------------------------------------------------
    # Overrides: fit (CATCH needs two optimisers and a composite loss)
    # ------------------------------------------------------------------

    def fit(
        self,
        train_data: pd.DataFrame,
        train_label: Optional[pd.DataFrame] = None,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> None:
        import torch
        from torch.optim import lr_scheduler

        arr = self._to_2d(train_data)
        n = arr.shape[0]
        if n < self.win_size or arr.shape[1] == 0:
            self._network = None
            self._train_scores = None
            return
        self._n_features = arr.shape[1]
        arr = self._scale_fit(arr)

        # train/val split — same pattern as base, only for early stopping.
        val_n = int(n * self.val_ratio)
        if val_n >= self.win_size + 1 and (n - val_n) >= self.win_size:
            train_arr, val_arr = arr[: n - val_n], arr[n - val_n :]
        else:
            train_arr, val_arr = arr, None

        train_loader = self._make_loader(train_arr, mode="train", shuffle=True)
        if train_loader is None:
            self._network = None
            self._train_scores = None
            return
        val_loader = (
            self._make_loader(val_arr, mode="val", shuffle=False)
            if val_arr is not None
            else None
        )

        device = self._resolve_device()
        self._network = self._build_network(self._n_features).to(device)
        configs = self._make_configs(self._n_features)
        criterion = torch.nn.MSELoss()
        auxi_loss_fn = frequency_loss(configs)
        self._freq_criterion = frequency_criterion(configs)

        # Two optimisers — main params and mask_generator — exactly as
        # in the upstream training script.
        main_params = [
            p for n_, p in self._network.named_parameters() if "mask_generator" not in n_
        ]
        optimizer = torch.optim.Adam(main_params, lr=self.lr)
        optimizerM = torch.optim.Adam(
            self._network.mask_generator.parameters(), lr=self.Mlr
        )

        steps_per_epoch = max(1, len(train_loader))
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.pct_start,
            epochs=self.num_epochs,
            max_lr=self.lr,
        )
        schedulerM = lr_scheduler.OneCycleLR(
            optimizerM,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.pct_start,
            epochs=self.num_epochs,
            max_lr=self.Mlr,
        )
        early = _EarlyStopping(patience=self.patience)

        # Cadence at which the mask generator is updated, copied from
        # the official trainer: ``min(len(train_loader)/10, 100)``.
        m_step = max(1, min(int(steps_per_epoch / 10), 100))

        for epoch in range(self.num_epochs):
            self._network.train()
            for i, batch in enumerate(train_loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                output, output_complex, dcloss = self._network(batch)
                rec_loss = criterion(output, batch)
                norm_input = self._network.revin_layer(batch, "transform")
                auxi_loss_val = auxi_loss_fn(output_complex, norm_input)
                loss = (
                    rec_loss
                    + self.dc_lambda * dcloss
                    + self.auxi_lambda * auxi_loss_val
                )
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i + 1) % m_step == 0:
                    optimizerM.step()
                    optimizerM.zero_grad()
                    schedulerM.step()

            if val_loader is not None:
                val_loss = self._validate(val_loader, criterion, epoch)
                early.step(val_loss)
                if early.early_stop:
                    logger.debug(
                        "[CATCH] EarlyStopping at epoch %d (val=%.6f).",
                        epoch + 1, val_loss,
                    )
                    break

        # Cache per-timestep training scores for the threshold rule.
        self._train_scores = self._inference_scores(train_arr, criterion, mode="train")

    # ------------------------------------------------------------------
    # Inference hooks
    # ------------------------------------------------------------------

    def _train_one_step(self, model, batch_x, criterion, epoch, optimizer) -> float:  # pragma: no cover
        # ``fit`` is fully overridden, so this hook is never reached.
        raise NotImplementedError("CATCH manages its own training loop in fit().")

    def _compute_window_scores(self, model, batch_x, criterion) -> np.ndarray:
        import torch

        if self._freq_criterion is None:
            self._freq_criterion = frequency_criterion(
                self._make_configs(self._n_features)
            )
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs, _, _ = model(batch_x)
            temp_score = torch.mean((batch_x - outputs) ** 2, dim=-1)  # (B, W)
            freq_score = self._freq_criterion(batch_x, outputs)        # (B, W, C)
            freq_score = torch.mean(freq_score, dim=-1)                # (B, W)
            score = temp_score + self.score_lambda * freq_score
        return score.detach().cpu().numpy()


def make_catch(**kwargs) -> CATCHAnomalyModel:
    return CATCHAnomalyModel(**kwargs)


__all__ = ["CATCHAnomalyModel", "make_catch"]
