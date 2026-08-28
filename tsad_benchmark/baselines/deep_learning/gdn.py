# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import logging
import math
import random
from typing import Optional

import numpy as np
import pandas as pd

from tsad_benchmark.baselines._dl_base import DLBaseModel
from tsad_benchmark.baselines._thresholding import percentile_label_maps
from tsad_benchmark.models.base import ModelCapability

logger = logging.getLogger(__name__)


class _GDNWindowDataset:
    """Sliding-window dataset matching official ``TimeDataset.process``."""

    def __init__(self, data: np.ndarray, win_size: int, stride: int, train: bool):
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __init__(self, arr, win, step, is_train):
                self.arr = np.asarray(arr, dtype=np.float32)
                self.win = int(win)
                self.step = int(step)
                self.is_train = bool(is_train)
                stop = self.arr.shape[0]
                if self.is_train:
                    self.targets = list(range(self.win, stop, self.step))
                else:
                    self.targets = list(range(self.win, stop))

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, idx):
                i = self.targets[idx]
                x = self.arr[i - self.win : i].T
                y = self.arr[i]
                return torch.from_numpy(x), torch.from_numpy(y)

        self.dataset = _Dataset(data, win_size, stride, train)


class GDNModel(DLBaseModel):
   

    model_name = "GDN"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        win_size: int = 15,
        batch_size: int = 128,
        num_epochs: int = 100,
        lr: float = 1e-3,
        val_ratio: float = 0.1,
        patience: int = 15,
        anomaly_ratio=None,
        dim: int = 64,
        topk: int = 20,
        eval_topk: int = 1,
        out_layer_num: int = 1,
        out_layer_inter_dim: int = 256,
        slide_stride: int = 5,
        decay: float = 0.0,
        seed: int = 0,
        label_rule: str = "official",
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
        self.dim = int(dim)
        self.topk = int(topk)
        self.eval_topk = int(eval_topk)
        self.out_layer_num = int(out_layer_num)
        self.out_layer_inter_dim = int(out_layer_inter_dim)
        self.slide_stride = int(slide_stride)
        self.decay = float(decay)
        self.seed = int(seed)
        self.label_rule = str(label_rule).lower()
        self._edge_index = None
        self._val_result = None
        self._normal_scores = None
        self._gdn_threshold = math.nan
        self.model_hyper_params.update(
            dim=self.dim,
            topk=self.topk,
            eval_topk=self.eval_topk,
            out_layer_num=self.out_layer_num,
            out_layer_inter_dim=self.out_layer_inter_dim,
            slide_stride=self.slide_stride,
            decay=self.decay,
            seed=self.seed,
            label_rule=self.label_rule,
        )

    # ------------------------------------------------------------------
    # DLBaseModel hooks are unused because GDN owns its train/test loop.
    # ------------------------------------------------------------------

    def _build_network(self, n_features: int):
        from tsad_benchmark.baselines.deep_learning._models.gdn import GDN

        edge_index = self._fully_connected_edge_index(n_features)
        self._edge_index = edge_index
        return GDN(
            [edge_index],
            n_features,
            dim=self.dim,
            input_dim=self.win_size,
            out_layer_num=self.out_layer_num,
            out_layer_inter_dim=self.out_layer_inter_dim,
            topk=min(self.topk, n_features),
        )

    def _train_one_step(self, model, batch_x, criterion, epoch, optimizer) -> float:  # pragma: no cover
        raise NotImplementedError("GDN manages its own official training loop.")

    def _compute_window_scores(self, model, batch_x, criterion) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("GDN computes scores with official helpers.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fully_connected_edge_index(n_features: int):
        import torch

        edge_indexes = [[], []]
        for parent in range(n_features):
            for child in range(n_features):
                if child == parent:
                    continue
                edge_indexes[0].append(child)
                edge_indexes[1].append(parent)
        return torch.tensor(edge_indexes, dtype=torch.long)

    def _make_loader(self, arr: np.ndarray, train: bool, shuffle: bool):
        import torch
        from torch.utils.data import DataLoader

        if arr is None or arr.shape[0] <= self.win_size:
            return None
        ds = _GDNWindowDataset(
            arr,
            win_size=self.win_size,
            stride=max(1, self.slide_stride),
            train=train,
        ).dataset
        if len(ds) == 0:
            return None
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def _run_model(self, loader):
        import torch

        device = self._resolve_device()
        predicted_chunks = []
        ground_chunks = []
        losses = []
        criterion = torch.nn.MSELoss(reduction="mean")
        self._network.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.float().to(device)
                y = y.float().to(device)
                pred = self._network(x, self._edge_index).float().to(device)
                losses.append(float(criterion(pred, y).detach().cpu()))
                predicted_chunks.append(pred.detach().cpu().numpy())
                ground_chunks.append(y.detach().cpu().numpy())
        if not predicted_chunks:
            empty = np.zeros((0, self._n_features), dtype=float)
            return 0.0, [empty, empty.copy(), empty.copy()]
        predicted = np.concatenate(predicted_chunks, axis=0)
        ground = np.concatenate(ground_chunks, axis=0)
        labels = np.zeros_like(ground)
        return float(np.mean(losses)) if losses else 0.0, [predicted, ground, labels]

    def _official_scores(self, arr: np.ndarray):
        from tsad_benchmark.baselines.deep_learning._models.gdn import (
            aggregate_topk_scores,
            get_full_err_scores,
        )

        loader = self._make_loader(arr, train=False, shuffle=False)
        if loader is None or self._val_result is None:
            return np.zeros(0, dtype=float), None
        _, test_result = self._run_model(loader)
        full_scores, normal_scores = get_full_err_scores(test_result, self._val_result)
        topk_scores = aggregate_topk_scores(full_scores, topk=self.eval_topk)
        return np.nan_to_num(topk_scores, nan=0.0, posinf=0.0, neginf=0.0), normal_scores

    def _align_gdn_length(self, scores: np.ndarray, target_len: int) -> np.ndarray:
        out = np.zeros(target_len, dtype=float)
        if scores.size == 0:
            return out
        start = min(self.win_size, target_len)
        usable = min(scores.size, max(target_len - start, 0))
        if usable > 0:
            out[start : start + usable] = scores[:usable]
        return out

    # ------------------------------------------------------------------
    # AnomalyModelBase interface
    # ------------------------------------------------------------------

    def fit(
        self,
        train_data: pd.DataFrame,
        train_label: Optional[pd.DataFrame] = None,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> None:
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        arr = self._to_2d(train_data)
        n = arr.shape[0]
        if n <= self.win_size or arr.shape[1] == 0:
            self._network = None
            self._train_scores = None
            return

        self._n_features = arr.shape[1]
        arr = self._scale_fit(arr)

        device = self._resolve_device()
        self._network = self._build_network(self._n_features).to(device)
        if self._edge_index is not None:
            self._edge_index = self._edge_index.to(device)

        full_train_dataset = _GDNWindowDataset(
            arr,
            win_size=self.win_size,
            stride=max(1, self.slide_stride),
            train=True,
        ).dataset
        dataset_len = len(full_train_dataset)
        if dataset_len == 0:
            self._network = None
            self._train_scores = None
            return

        from torch.utils.data import DataLoader, Subset

        val_use_len = int(dataset_len * self.val_ratio)
        if val_use_len > 0 and dataset_len - val_use_len > 0:
            train_use_len = dataset_len - val_use_len
            val_start_index = random.randrange(train_use_len)
            indices = torch.arange(dataset_len)
            train_sub_indices = torch.cat(
                [indices[:val_start_index], indices[val_start_index + val_use_len :]]
            )
            val_sub_indices = indices[val_start_index : val_start_index + val_use_len]
            train_ds = Subset(full_train_dataset, train_sub_indices)
            val_ds = Subset(full_train_dataset, val_sub_indices)
        else:
            train_ds = full_train_dataset
            val_ds = full_train_dataset

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        optimizer = torch.optim.Adam(
            self._network.parameters(), lr=self.lr, weight_decay=self.decay
        )
        criterion = torch.nn.MSELoss(reduction="mean")
        best_state = None
        best_loss = float("inf")
        stale = 0

        for epoch in range(self.num_epochs):
            self._network.train()
            losses = []
            for x, y in train_loader:
                x = x.float().to(device)
                y = y.float().to(device)
                optimizer.zero_grad()
                out = self._network(x, self._edge_index).float().to(device)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            val_loss, _ = self._run_model(val_loader)
            monitor_loss = val_loss if math.isfinite(val_loss) else float(np.mean(losses))
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                best_state = copy.deepcopy(self._network.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    logger.debug("[GDN] EarlyStopping at epoch %d.", epoch + 1)
                    break

        if best_state is not None:
            self._network.load_state_dict(best_state)

        _, self._val_result = self._run_model(val_loader)
        self._train_scores, self._normal_scores = self._official_scores(arr)
        if self._normal_scores is not None and np.asarray(self._normal_scores).size:
            self._gdn_threshold = float(np.max(self._normal_scores))

    def detect_score(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        n = len(test_data) if test_data is not None else 0
        if self._network is None or n == 0:
            return np.zeros(n, dtype=float)
        arr = self._prepare_inference_array(test_data)
        raw_scores, _ = self._official_scores(arr)
        return self._align_gdn_length(raw_scores, n)

    def detect_label(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ):
        from tsad_benchmark.baselines.deep_learning._models.gdn import (
            aggregate_topk_scores,
            val_threshold_labels,
        )

        n = len(test_data) if test_data is not None else 0
        if self._network is None or n == 0:
            empty = np.zeros(n, dtype=np.int32)
            if self.label_rule in {"percentile", "anomaly_ratio", "tab"}:
                return {str(r): empty.copy() for r in self.anomaly_ratio}, np.zeros(n, dtype=float)
            return {"val": empty}, np.zeros(n, dtype=float)

        arr = self._prepare_inference_array(test_data)
        loader = self._make_loader(arr, train=False, shuffle=False)
        if loader is None or self._val_result is None:
            empty = np.zeros(n, dtype=np.int32)
            if self.label_rule in {"percentile", "anomaly_ratio", "tab"}:
                return {str(r): empty.copy() for r in self.anomaly_ratio}, np.zeros(n, dtype=float)
            return {"val": empty}, np.zeros(n, dtype=float)

        from tsad_benchmark.baselines.deep_learning._models.gdn import get_full_err_scores

        _, test_result = self._run_model(loader)
        full_scores, normal_scores = get_full_err_scores(test_result, self._val_result)
        if self.label_rule in {"percentile", "anomaly_ratio", "tab"}:
            raw_scores = aggregate_topk_scores(full_scores, topk=self.eval_topk)
            raw_preds = percentile_label_maps(
                raw_scores,
                self._train_scores,
                self.anomaly_ratio,
            )
            preds = {
                key: self._align_label_length(
                    self._align_gdn_length(labels.astype(float), n).astype(np.int32), n
                )
                for key, labels in raw_preds.items()
            }
            return preds, self._align_gdn_length(raw_scores, n)

        if self.label_rule not in {"official", "val", "val_max", "validation"}:
            raise ValueError(
                "label_rule must be 'official'/'val_max' or 'percentile'/'anomaly_ratio'"
            )

        raw_labels, raw_scores, threshold = val_threshold_labels(
            full_scores,
            normal_scores,
            topk=self.eval_topk,
        )
        self._gdn_threshold = threshold
        labels = self._align_label_length(
            self._align_gdn_length(raw_labels.astype(float), n).astype(np.int32), n
        )
        scores = self._align_gdn_length(raw_scores, n)
        return {"val": labels}, scores

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        return math.nan


def make_gdn(**kwargs) -> GDNModel:
    return GDNModel(**kwargs)


__all__ = ["GDNModel", "make_gdn"]
