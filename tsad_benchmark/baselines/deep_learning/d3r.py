# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from tsad_benchmark.baselines._dl_base import DLBaseModel, _EarlyStopping
from tsad_benchmark.models.base import ModelCapability

logger = logging.getLogger(__name__)


class _D3RWindowDataset:
 
    def __init__(
        self,
        data: np.ndarray,
        time: np.ndarray,
        stable: np.ndarray,
        win_size: int,
    ):
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __init__(self, arr, tm, st, win):
                self.data = np.asarray(arr, dtype=np.float32)
                self.time = np.asarray(tm, dtype=np.float32)
                self.stable = np.asarray(st, dtype=np.float32)
                self.window_size = int(win)

            def __len__(self):
                return max(0, len(self.data) - self.window_size + 1)

            def __getitem__(self, index):
                end = index + self.window_size
                data = self.data[index:end, :]
                time = self.time[index:end, :]
                stable = self.stable[index:end, :]
                return (
                    torch.from_numpy(data),
                    torch.from_numpy(time),
                    torch.from_numpy(stable),
                )

        self.dataset = _Dataset(data, time, stable, win_size)


class D3RModel(DLBaseModel):
    

    model_name = "D3R"
    capability = ModelCapability.score_and_label()

    def __init__(
        self,
        win_size: int = 64,
        batch_size: int = 8,
        num_epochs: int = 8,
        lr: float = 1e-4,
        val_ratio: float = 0.2,
        patience: int = 3,
        anomaly_ratio=None,
        period: Optional[int] = None,
        points_per_day: int = 144,
        model_dim: int = 512,
        ff_dim: int = 2048,
        atten_dim: int = 64,
        block_num: int = 2,
        head_num: int = 8,
        dropout: float = 0.6,
        time_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        t: int = 500,
        p: float = 10.0,
        d: int = 30,
        q: float = 0.01,
        weight_decay: float = 1e-4,
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
        self.period = int(period) if period is not None else int(points_per_day)
        self.points_per_day = int(points_per_day)
        self.model_dim = int(model_dim)
        self.ff_dim = int(ff_dim)
        self.atten_dim = int(atten_dim)
        self.block_num = int(block_num)
        self.head_num = int(head_num)
        self.dropout = float(dropout)
        self.time_steps = int(time_steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.t = int(t)
        self.p = float(p)
        self.d = int(d)
        self.q = float(q)
        self.weight_decay = float(weight_decay)
        self._train_len = 0
        self.model_hyper_params.update(
            period=self.period,
            points_per_day=self.points_per_day,
            model_dim=self.model_dim,
            ff_dim=self.ff_dim,
            atten_dim=self.atten_dim,
            block_num=self.block_num,
            head_num=self.head_num,
            dropout=self.dropout,
            time_steps=self.time_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            t=self.t,
            p=self.p,
            d=self.d,
            q=self.q,
            weight_decay=self.weight_decay,
        )

    # ------------------------------------------------------------------
    # DLBaseModel hooks are unused because D3R has a custom train loop.
    # ------------------------------------------------------------------

    def _build_network(self, n_features: int):
        from tsad_benchmark.baselines.deep_learning._models.d3r import DDDR

        if self.t >= self.time_steps:
            raise ValueError(
                f"D3R requires t < time_steps; got t={self.t}, "
                f"time_steps={self.time_steps}."
            )
        return DDDR(
            time_steps=self.time_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            window_size=self.win_size,
            model_dim=self.model_dim,
            ff_dim=self.ff_dim,
            atten_dim=self.atten_dim,
            feature_num=n_features,
            time_num=5,
            block_num=self.block_num,
            head_num=self.head_num,
            dropout=self.dropout,
            device=self._resolve_device(),
            d=self.d,
            t=self.t,
        )

    def _train_one_step(self, model, batch_x, criterion, epoch, optimizer) -> float:  # pragma: no cover
        raise NotImplementedError("D3R manages its own official training loop.")

    def _compute_window_scores(self, model, batch_x, criterion) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("D3R computes last-step reconstruction scores.")

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _make_time_embedding(
        self,
        frame: Optional[pd.DataFrame],
        length: int,
        start_offset: int = 0,
    ) -> np.ndarray:
        index = getattr(frame, "index", None)
        dt_index = None
        if index is not None and not isinstance(index, pd.RangeIndex):
            parsed = pd.to_datetime(index, errors="coerce")
            if len(parsed) == length and not pd.isna(parsed).any():
                dt_index = parsed

        if dt_index is None:
            minutes_per_point = 1440.0 / max(float(self.points_per_day), 1.0)
            start = pd.Timestamp("2000-01-01") + pd.to_timedelta(
                start_offset * minutes_per_point, unit="m"
            )
            dt_index = pd.date_range(
                start=start,
                periods=length,
                freq=pd.to_timedelta(minutes_per_point, unit="m"),
            )

        df = pd.DataFrame({"time": pd.to_datetime(dt_index)})
        df["minute"] = df["time"].apply(lambda row: row.minute / 59 - 0.5)
        df["hour"] = df["time"].apply(lambda row: row.hour / 23 - 0.5)
        df["weekday"] = df["time"].apply(lambda row: row.weekday() / 6 - 0.5)
        df["day"] = df["time"].apply(lambda row: row.day / 30 - 0.5)
        df["month"] = df["time"].apply(lambda row: row.month / 365 - 0.5)
        return df[["minute", "hour", "weekday", "day", "month"]].values.astype(np.float32)

    def _stable_data_and_target(
        self,
        arr: np.ndarray,
        time_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        period = int(self.period)
        if period > 1 and len(arr) > period:
            trend = pd.DataFrame(arr).rolling(period, center=True).median().values
            stable = arr - trend
            start = period // 2
            end = -period // 2
            data = arr[start:end, :]
            time = time_arr[start:end, :]
            stable = stable[start:end, :]
        else:
            period = max(1, min(period, len(arr)))
            trend = (
                pd.DataFrame(arr)
                .rolling(period, center=True, min_periods=1)
                .median()
                .values
            )
            data = arr
            time = time_arr
            stable = arr - trend
        return (
            np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
            np.nan_to_num(time, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
            np.nan_to_num(stable, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        )

    def _make_loader(
        self,
        arr: np.ndarray,
        time_arr: np.ndarray,
        stable_arr: np.ndarray,
        shuffle: bool,
    ):
        from torch.utils.data import DataLoader

        if arr is None or arr.shape[0] < self.win_size:
            return None
        ds = _D3RWindowDataset(arr, time_arr, stable_arr, self.win_size).dataset
        if len(ds) == 0:
            return None
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    # ------------------------------------------------------------------
    # Training/scoring helpers
    # ------------------------------------------------------------------

    def _process_train_batch(self, batch_data, batch_time, batch_stable):
        import torch

        criterion = torch.nn.MSELoss(reduction="mean")
        stable, _, recon = self._network(batch_data, batch_time, self.p)
        return 0.5 * criterion(stable, batch_stable) + 0.5 * criterion(recon, batch_data)

    def _score_array(self, arr: np.ndarray, time_arr: np.ndarray) -> np.ndarray:
        import torch

        if self._network is None or arr.shape[0] < self.win_size:
            return np.zeros(0, dtype=float)
        stable = np.zeros_like(arr, dtype=np.float32)
        loader = self._make_loader(arr, time_arr, stable, shuffle=False)
        if loader is None:
            return np.zeros(0, dtype=float)
        device = self._resolve_device()
        scores = []
        self._network.eval()
        with torch.no_grad():
            for batch_data, batch_time, batch_stable in loader:
                batch_data = batch_data.float().to(device)
                batch_time = batch_time.float().to(device)
                _, _, recon = self._network(batch_data, batch_time, 0.0)
                mse = torch.mean((batch_data[:, -1, :] - recon[:, -1, :]) ** 2, dim=-1)
                scores.append(mse.detach().cpu().numpy())
        if not scores:
            return np.zeros(0, dtype=float)
        return np.nan_to_num(np.concatenate(scores, axis=0), nan=0.0, posinf=0.0, neginf=0.0)

    def _align_d3r_length(self, scores: np.ndarray, target_len: int) -> np.ndarray:
        out = np.zeros(target_len, dtype=float)
        if scores.size == 0:
            return out
        start = min(max(self.win_size - 1, 0), target_len)
        usable = min(scores.size, max(target_len - start, 0))
        if usable > 0:
            out[start : start + usable] = scores[:usable]
        return out

    def _official_spot_labels(self, raw_scores: np.ndarray) -> tuple[np.ndarray, float]:
        from tsad_benchmark.baselines.deep_learning._models.d3r import evaluate

        if raw_scores.size == 0:
            return np.zeros(0, dtype=np.int32), float("inf")
        init_scores = self._train_scores
        if init_scores is None or np.asarray(init_scores).size == 0:
            init_scores = raw_scores
        try:
            res = evaluate(
                np.asarray(init_scores, dtype=float).reshape(-1),
                np.asarray(raw_scores, dtype=float).reshape(-1),
                test_label=None,
                q=self.q,
            )
            return np.asarray(res["test_pred"], dtype=np.int32), float(res["threshold"])
        except Exception as exc:
            logger.warning("[D3R] Official SPOT threshold failed; using percentile fallback: %s", exc)
            threshold = float(np.percentile(init_scores, 100.0 * (1.0 - self.q)))
            return (raw_scores > threshold).astype(np.int32), threshold

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

        arr = self._to_2d(train_data)
        n = arr.shape[0]
        if n < self.win_size or arr.shape[1] == 0:
            self._network = None
            self._train_scores = None
            return

        self._n_features = arr.shape[1]
        self._train_len = n
        arr = self._scale_fit(arr)
        time_arr = self._make_time_embedding(train_data, n, start_offset=0)
        model_arr, model_time, stable_arr = self._stable_data_and_target(arr, time_arr)

        model_n = model_arr.shape[0]
        val_n = int(model_n * self.val_ratio)
        if val_n >= self.win_size and (model_n - val_n) >= self.win_size:
            train_arr, val_arr = model_arr[: model_n - val_n], model_arr[model_n - val_n :]
            train_time, val_time = model_time[: model_n - val_n], model_time[model_n - val_n :]
            train_stable, val_stable = stable_arr[: model_n - val_n], stable_arr[model_n - val_n :]
        else:
            train_arr, val_arr = model_arr, None
            train_time, val_time = model_time, None
            train_stable, val_stable = stable_arr, None

        train_loader = self._make_loader(train_arr, train_time, train_stable, shuffle=True)
        if train_loader is None:
            self._network = None
            self._train_scores = None
            return

        val_loader = (
            self._make_loader(val_arr, val_time, val_stable, shuffle=False)
            if val_arr is not None
            else None
        )

        device = self._resolve_device()
        self._network = self._build_network(self._n_features).to(device)
        optimizer = torch.optim.Adam(
            self._network.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        early = _EarlyStopping(patience=self.patience)
        best_state = None
        best_loss = float("inf")

        for epoch in range(self.num_epochs):
            self._network.train()
            for batch_data, batch_time, batch_stable in train_loader:
                batch_data = batch_data.float().to(device)
                batch_time = batch_time.float().to(device)
                batch_stable = batch_stable.float().to(device)
                optimizer.zero_grad()
                loss = self._process_train_batch(batch_data, batch_time, batch_stable)
                loss.backward()
                optimizer.step()

            monitor_loss = None
            if val_loader is not None:
                self._network.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_data, batch_time, batch_stable in val_loader:
                        batch_data = batch_data.float().to(device)
                        batch_time = batch_time.float().to(device)
                        batch_stable = batch_stable.float().to(device)
                        val_losses.append(
                            float(
                                self._process_train_batch(
                                    batch_data, batch_time, batch_stable
                                ).detach().cpu()
                            )
                        )
                monitor_loss = float(np.mean(val_losses)) if val_losses else None

            if monitor_loss is not None:
                if monitor_loss < best_loss:
                    best_loss = monitor_loss
                    best_state = copy.deepcopy(self._network.state_dict())
                early.step(monitor_loss)
                if early.early_stop:
                    logger.debug("[D3R] EarlyStopping at epoch %d.", epoch + 1)
                    break

        if best_state is not None:
            self._network.load_state_dict(best_state)

        self._train_scores = self._score_array(model_arr, model_time)

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
        time_arr = self._make_time_embedding(
            test_data,
            n,
            start_offset=self._train_len,
        )
        raw_scores = self._score_array(arr, time_arr)
        return self._align_d3r_length(raw_scores, n)

    def detect_label(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ):
        n = len(test_data) if test_data is not None else 0
        if self._network is None or n == 0:
            empty = np.zeros(n, dtype=np.int32)
            return {f"q={self.q:g}": empty}, np.zeros(n, dtype=float)
        arr = self._prepare_inference_array(test_data)
        time_arr = self._make_time_embedding(
            test_data,
            n,
            start_offset=self._train_len,
        )
        raw_scores = self._score_array(arr, time_arr)
        raw_labels, _threshold = self._official_spot_labels(raw_scores)
        labels = self._align_d3r_length(raw_labels.astype(float), n).astype(np.int32)
        scores = self._align_d3r_length(raw_scores, n)
        return {f"q={self.q:g}": labels}, scores

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        return math.nan


def make_d3r(**kwargs) -> D3RModel:
    return D3RModel(**kwargs)


__all__ = ["D3RModel", "make_d3r"]
