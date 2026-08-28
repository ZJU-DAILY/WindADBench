# -*- coding: utf-8 -*-


from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from tsad_benchmark.baselines._thresholding import normalize_anomaly_ratios


def to_2d_array(data) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        arr = data.values
    else:
        arr = np.asarray(data)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size and not np.all(np.isfinite(arr)):
        mask = ~np.isfinite(arr)
        col_mean = np.nanmean(np.where(mask, np.nan, arr), axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        arr = np.where(mask, col_mean, arr).astype(np.float32)
    return arr


def labels_1d(labels) -> Optional[np.ndarray]:
    if labels is None:
        return None
    if isinstance(labels, pd.DataFrame):
        arr = labels.values
    else:
        arr = np.asarray(labels)
    if arr.size == 0:
        return None
    return np.asarray(arr).reshape(-1).astype(np.int32)


def official_adjustment(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Point-adjustment copied from the GPT4TS/UniTS official utilities."""

    gt = np.asarray(gt, dtype=np.int32).reshape(-1)
    pred = np.asarray(pred, dtype=np.int32).reshape(-1).copy()
    n = min(gt.size, pred.size)
    gt = gt[:n]
    pred = pred[:n]
    anomaly_state = False
    for i in range(n):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                if pred[j] == 0:
                    pred[j] = 1
            for j in range(i, n):
                if gt[j] == 0:
                    break
                if pred[j] == 0:
                    pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred


def percentile_labels(
    test_scores: np.ndarray,
    train_scores: Optional[np.ndarray],
    anomaly_ratios: Sequence[float],
    test_label=None,
    apply_adjustment: bool = False,
    threshold_test_scores: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
  

    test = np.nan_to_num(np.asarray(test_scores, dtype=float).reshape(-1))
    train = (
        np.zeros(0, dtype=float)
        if train_scores is None
        else np.nan_to_num(np.asarray(train_scores, dtype=float).reshape(-1))
    )
    threshold_test = (
        test
        if threshold_test_scores is None
        else np.nan_to_num(np.asarray(threshold_test_scores, dtype=float).reshape(-1))
    )
    pool = threshold_test if train.size == 0 else np.concatenate([train, threshold_test], axis=0)
    gt = labels_1d(test_label)
    preds: dict[str, np.ndarray] = {}
    for ratio in normalize_anomaly_ratios(anomaly_ratios):
        ratio = float(np.clip(ratio, 0.0, 100.0))
        if ratio <= 0.0 or test.size == 0:
            pred = np.zeros(test.shape[0], dtype=np.int32)
        elif ratio >= 100.0:
            pred = np.ones(test.shape[0], dtype=np.int32)
        else:
            threshold = float(np.percentile(pool, 100.0 - ratio))
            pred = (test > threshold).astype(np.int32)
        if apply_adjustment and gt is not None:
            pred = official_adjustment(gt, pred)
        preds[str(ratio)] = pred
    return preds


def rolling_windows(arr: np.ndarray, win_size: int, step: int = 1) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < win_size or arr.shape[1] == 0:
        return np.zeros((0, int(win_size), max(arr.shape[1] if arr.ndim == 2 else 1, 1)), dtype=np.float32)
    starts = np.arange(0, arr.shape[0] - win_size + 1, int(step))
    offsets = np.arange(int(win_size))
    return arr[starts[:, None] + offsets[None, :]].astype(np.float32)


def average_overlapping_scores(window_scores: np.ndarray, target_len: int, win_size: int, step: int = 1) -> np.ndarray:
    scores = np.asarray(window_scores, dtype=float)
    out = np.zeros(int(target_len), dtype=float)
    counts = np.zeros(int(target_len), dtype=float)
    if scores.size == 0:
        return out
    scores = scores.reshape(scores.shape[0], -1)
    for i in range(scores.shape[0]):
        start = i * int(step)
        end = min(start + int(win_size), int(target_len))
        width = max(end - start, 0)
        if width:
            out[start:end] += scores[i, :width]
            counts[start:end] += 1.0
    seen = counts > 0
    out[seen] /= counts[seen]
    if np.any(seen):
        first = int(np.argmax(seen))
        last = int(len(seen) - 1 - np.argmax(seen[::-1]))
        out[:first] = out[first]
        out[last + 1 :] = out[last]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def non_overlapping_timeline_scores(
    window_scores: np.ndarray,
    target_len: int,
    win_size: int,
    tail_window_scores: Optional[np.ndarray] = None,
) -> np.ndarray:

    target_len = int(target_len)
    scores = np.asarray(window_scores, dtype=float).reshape(-1)
    if scores.size == target_len:
        out = scores
    elif scores.size == 0:
        out = np.zeros(target_len, dtype=float)
    elif scores.size < target_len:
        missing = target_len - scores.size
        tail = (
            np.asarray(tail_window_scores, dtype=float).reshape(-1)
            if tail_window_scores is not None
            else np.zeros(0, dtype=float)
        )
        if tail.size:
            scores = np.concatenate([scores, tail[-missing:]], axis=0)
        out = np.zeros(target_len, dtype=float)
        usable = min(scores.size, target_len)
        out[:usable] = scores[:usable]
    else:
        out = scores[:target_len]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
