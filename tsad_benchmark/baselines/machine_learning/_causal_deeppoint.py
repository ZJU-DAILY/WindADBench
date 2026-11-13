from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from merlion.models.anomaly.deep_point_anomaly_detector import (
    DeepPointAnomalyDetector, MLPNet,
)
from merlion.utils import to_timestamp


class SourceFrozenDeepPointDetector(DeepPointAnomalyDetector):
    def __init__(self, config, source_segment_lengths=None):
        super().__init__(config)

        self._time_mean = None
        self._time_scale = None
        self._value_mean = None
        self._value_scale = None
        self._score_columns = None
        self._score_network = None
        self._source_segment_lengths = source_segment_lengths
        self._source_fit_rows = None

    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        if self._source_segment_lengths is not None:
            lengths = np.asarray(self._source_segment_lengths, dtype=int)
            if (lengths < 1).any() or int(lengths.sum()) - 1 != len(train_data):
                raise ValueError("DeepPoint source segments do not match differenced training rows.")
            keep = np.ones(len(train_data), dtype=bool)
            keep[np.cumsum(lengths)[:-1] - 1] = False
            train_data = train_data.iloc[keep]
        if len(train_data) < 2:
            raise ValueError("DeepPoint needs at least two source rows after differencing.")
        self._source_fit_rows = len(train_data)
        times = np.asarray(to_timestamp(train_data.index), dtype=float).reshape(-1, 1)
        values = train_data.to_numpy(dtype=float)
        self._time_mean = times.mean(axis=0, keepdims=True)
        self._time_scale = times.std(axis=0, keepdims=True) + 1e-7
        self._value_mean = values.mean(axis=0, keepdims=True)
        self._value_scale = values.std(axis=0, keepdims=True) + 1e-7
        self._score_columns = list(train_data.columns)
        device = torch.device("cuda" if self.use_cuda and torch.cuda.is_available() else "cpu")
        x = torch.as_tensor((times-self._time_mean)/self._time_scale, dtype=torch.float32, device=device)
        y = torch.as_tensor((values-self._value_mean)/self._value_scale, dtype=torch.float32, device=device)
        self._score_network = MLPNet(dim_inp=1, dim_out=y.shape[1], nhiddens=[400, 400, 400], bn=True).to(device)
        optimizer = torch.optim.Adam(self._score_network.parameters(), lr=0.001, weight_decay=1e-3)
        self._score_network.train()
        for _ in range(20):
            prediction = self._score_network(x)
            loss = ((y-prediction)**2).sum(1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self._score_network.eval()
        self._score_network.requires_grad_(False)
        return self._get_anomaly_score(train_data)

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev=None) -> pd.DataFrame:
        network = getattr(self, "_score_network", None)
        if network is None:
            raise RuntimeError("DeepPoint has no source-fitted network; fit the causal adapter first.")
        if list(time_series.columns) != self._score_columns:
            raise ValueError("DeepPoint scoring columns differ from source training columns.")
        times = np.asarray(to_timestamp(time_series.index), dtype=float).reshape(-1, 1)
        values = time_series.to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(times).all():
            raise ValueError("DeepPoint requires finite source-imputed observations.")
        device = next(network.parameters()).device
        x = torch.as_tensor((times-self._time_mean)/self._time_scale, dtype=torch.float32, device=device)
        y = torch.as_tensor((values-self._value_mean)/self._value_scale, dtype=torch.float32, device=device)
        network.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(x), 4096):
                prediction = network(x[start:start+4096])
                chunks.append(((y[start:start+4096]-prediction)**2).sum(1).cpu().numpy())
        scores = np.concatenate(chunks) if chunks else np.zeros(0, dtype=float)
        return pd.DataFrame(scores, index=time_series.index, columns=["anom_score"])
