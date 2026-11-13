from __future__ import annotations
import numpy as np
from tsad_benchmark.evaluation.score_timing import (
    score_timing, published_score_stream, timed_alarm_events, timing_status,
)
from tsad_benchmark.evaluation.thresholding import training_percentile_thresholds

ALARM_POLICY = 'batch_max_hold_v1'
THRESHOLD_METHOD = 'train_available_percentile_v2'
WARNING_HORIZON_POINTS = 288
WARNING_DELTA_POINTS = 72


def prepare_scores(scores, model_name, params, *, legacy_global=False):
    raw = np.asarray(scores, dtype=float).reshape(-1)
    if not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Raw scores must be non-empty and finite.')
    release = score_timing(model_name, len(raw), params, legacy_global=legacy_global)
    stream = published_score_stream(raw, release)
    status = timing_status(model_name, params, legacy_global=legacy_global)
    return {'raw': raw, 'release': release, 'stream': stream,
            'available': np.isfinite(stream), 'status': status}


def calibrate_scores(prepared, labels=None):
    valid = prepared['available'].copy()
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != len(valid):
            raise ValueError('Calibration labels and scores must align.')
        valid &= labels == 0
    values = prepared['stream'][valid]
    if not len(values):
        raise ValueError('No published normal calibration scores; cannot select a threshold.')
    lo, hi = float(values.min()), float(values.max())

    floor = lo - max(hi-lo, abs(lo), 1.0)
    return training_percentile_thresholds(values), floor, int(valid.sum())


def evaluate_threshold(prepared, threshold):
    alarms = (prepared['available'] & (prepared['stream'] > threshold)).astype(np.int8)
    warnings = np.zeros(len(alarms), dtype=np.int8)
    delivered = timed_alarm_events(prepared['raw'], threshold, prepared['release'])
    if len(delivered):
        warnings[delivered] = 1
    return alarms, warnings


def curve_scores(prepared, floor):
    return np.where(prepared['available'], prepared['stream'], float(floor))
