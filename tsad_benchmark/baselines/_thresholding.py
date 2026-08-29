# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np


DEFAULT_ANOMALY_RATIOS = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 25.0]


def normalize_anomaly_ratios(
    anomaly_ratio: Optional[float | Sequence[float]],
) -> list[float]:
    if anomaly_ratio is None:
        return list(DEFAULT_ANOMALY_RATIOS)
    if isinstance(anomaly_ratio, (str, bytes)):
        return [float(anomaly_ratio)]
    if isinstance(anomaly_ratio, Sequence):
        return [float(r) for r in anomaly_ratio]
    return [float(anomaly_ratio)]


def percentile_label_maps(
    test_scores: np.ndarray,
    reference_scores: Optional[np.ndarray],
    anomaly_ratios: Sequence[float],
    m: str = "tt",
) -> dict[str, np.ndarray]:
    tes = _finite_1d(test_scores)
    trs = _finite_1d(reference_scores)
    pool = _cs(trs, tes, m=m)
    preds: dict[str, np.ndarray] = {}
    for ratio in anomaly_ratios:
        ratio = float(np.clip(ratio, 0.0, 100.0))
        if ratio <= 0.0 or tes.size == 0:
            labels = np.zeros(tes.shape[0], dtype=np.int32)
        elif ratio >= 100.0:
            labels = np.ones(tes.shape[0], dtype=np.int32)
        else:
            threshold = float(np.percentile(pool, 100.0 - ratio))
            labels = (tes > threshold).astype(np.int32)
        preds[str(ratio)] = labels
    return preds


def _cs(trs: Optional[np.ndarray], tes: Optional[np.ndarray], m: str = "tt") -> np.ndarray:
    trs = _finite_1d(trs)
    tes = _finite_1d(tes)
    if m == "t":
        if trs.size == 0:
            raise ValueError("Empty reference scores.")
        return trs
    if m == "tt":
        return tes if trs.size == 0 else np.concatenate([trs, tes], axis=0)
    raise ValueError(f"Unknown mode: {m}")


def pot_label_maps(
    test_scores: np.ndarray,
    init_scores: Optional[np.ndarray],
    anomaly_ratios: Sequence[float],
    level: float = 0.98,
) -> dict[str, np.ndarray]:
    test = _finite_1d(test_scores)
    init = _finite_1d(init_scores)
    if init.size == 0:
        init = test
    preds: dict[str, np.ndarray] = {}
    for ratio in anomaly_ratios:
        ratio = float(np.clip(ratio, 0.0, 100.0))
        if ratio <= 0.0 or test.size == 0:
            labels = np.zeros(test.shape[0], dtype=np.int32)
        elif ratio >= 100.0:
            labels = np.ones(test.shape[0], dtype=np.int32)
        else:
            q = max(ratio / 100.0, 1e-12)
            threshold = _pot_threshold(init, q=q, level=level)
            labels = (test > threshold).astype(np.int32)
        preds[str(ratio)] = labels
    return preds


def _pot_threshold(init_scores: np.ndarray, q: float, level: float) -> float:
    init = _finite_1d(init_scores)
    if init.size == 0:
        return float("inf")
    level = float(np.clip(level, 0.5, 0.999))
    fallback = float(np.percentile(init, 100.0 * (1.0 - q)))
    base = float(np.percentile(init, 100.0 * level))
    peaks = init[init > base] - base
    if peaks.size < 3 or not np.isfinite(peaks).all():
        return fallback
    try:
        from scipy.stats import genpareto

        shape, _, scale = genpareto.fit(peaks, floc=0.0)
        scale = float(scale)
        if scale <= 0.0 or not np.isfinite(scale):
            return fallback
        r = init.size * float(q) / float(peaks.size)
        if r <= 0.0 or not np.isfinite(r):
            return fallback
        shape = float(shape)
        if abs(shape) < 1e-8:
            threshold = base - scale * np.log(r)
        else:
            threshold = base + (scale / shape) * (pow(r, -shape) - 1.0)
        if not np.isfinite(threshold):
            return fallback
        return float(max(threshold, fallback))
    except Exception:
        return fallback


def _finite_1d(scores: Optional[np.ndarray]) -> np.ndarray:
    if scores is None:
        return np.zeros(0, dtype=float)
    arr = np.asarray(scores, dtype=float).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
