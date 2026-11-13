from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np

from tsad_benchmark.evaluation.online_protocol import (
    ALARM_POLICY, WARNING_HORIZON_POINTS, WARNING_DELTA_POINTS,
    evaluate_threshold,
    curve_scores,
)
from tsad_benchmark.evaluation.score_timing import (
    normalize_model_name, published_score_stream, score_timing, timing_status,
)
from .metrics import compute_label_metrics, compute_score_metrics, SCORE_METRICS
from .protocol import calibrate_at_fpr


CROSS_TIMING_VERSION = 'cross_context_publication_v1'
CROSS_THRESHOLD_METHOD = 'source_available_state_fpr_v1'


def adapter_context_rows(model_name: str, params: Mapping, adapter: str) -> int:
    model = normalize_model_name(model_name)
    if adapter == 'point_concat':
        return 0
    if adapter == 'dlbase_segmented':
        return int(params['win_size']) - 1
    if adapter == 'kmeans_segmented':
        return int(params['window_size']) - 1
    if adapter != 'model_segmented':
        raise ValueError(f'No audited cross-domain timing for adapter {adapter!r}')
    if model in {'ae', 'dagmm', 'deeppoint'}:
        return 0
    if model == 'torsk':
        return 1
    if model == 'lstmed':
        return int(params.get('sequence_len', params.get('sequence_length'))) - 1
    if model == 'chronos':
        return int(params['context_length'])
    if model in {'gdn', 'mtad_gat'}:
        return int(params['win_size'])
    if model == 'mscred':
        return max(params['signature_scales']) - 1 + (int(params['step_max']) - 1) * int(params['gap_time'])
    return max(int(params.get('win_size', 1)) - 1, 0)


def prepare_cross_scores(scores, valid_mask, model_name: str, params: Mapping,
                         adapter: str, *, legacy_global: bool = False) -> dict:
    raw = np.asarray(scores, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if raw.ndim != 1 or valid.shape != raw.shape or raw.size == 0:
        raise ValueError('scores and valid_mask must be nonempty equal-length vectors')
    if not np.isfinite(raw[valid]).all():
        raise ValueError('Adapter-valid scores must be finite')
    prefix = adapter_context_rows(model_name, params, adapter)
    combined_release = score_timing(model_name, len(raw) + prefix, params,
                                    legacy_global=legacy_global)[prefix:]
    release = np.where(combined_release >= 0, combined_release - prefix, -1)
    release[~valid] = -1

    if (release < -1).any():
        raise ValueError('Cross adapter schedule published a target score before its timeline')
    stream = published_score_stream(raw, release)
    status = timing_status(model_name, params, legacy_global=legacy_global)
    status.update(cross_timing_version=CROSS_TIMING_VERSION,
                  adapter=adapter, context_rows=prefix)
    return {'raw': raw, 'release': release, 'stream': stream,
            'available': np.isfinite(stream), 'status': status}


def calibrate_cross_segments(prepared_segments: Sequence[dict], budget: float):
    values = [item['stream'][item['available']] for item in prepared_segments]
    values = np.concatenate(values) if values else np.empty(0)
    if values.size == 0:
        raise ValueError('Source calibration has no published scores')
    threshold, _, realized = calibrate_at_fpr(values, budget)
    low, high = float(values.min()), float(values.max())
    floor = low - max(high - low, abs(low), 1.0)
    return threshold, realized, floor, int(values.size)


def evaluate_cross_labels(labels, prepared: dict, threshold: float, *,
                          event_label=None, points_per_day=144,
                          horizon_points=WARNING_HORIZON_POINTS,
                          lead_delta_points=WARNING_DELTA_POINTS,
                          empty_alarm_affiliation="legacy_zero"):
    predictions, warnings = evaluate_threshold(prepared, threshold)
    metrics = compute_label_metrics(
        labels, predictions, event_label=event_label,
        points_per_day=points_per_day, horizon_points=horizon_points,
        lead_delta_points=lead_delta_points, warning_predictions=warnings,
        empty_alarm_affiliation=empty_alarm_affiliation,
    )
    return metrics, predictions, warnings


def compute_cross_score_metrics(labels, prepared: dict, score_floor: float, **kwargs):
    if not prepared['available'].any():
        labels = np.asarray(labels)
        return dict(n_points=float(len(labels)), n_positive=float(labels.sum()),
                    n_negative=float(len(labels)-labels.sum()),
                    **{name: float('nan') for name in SCORE_METRICS})
    return compute_score_metrics(labels, curve_scores(prepared, score_floor), **kwargs)


def timing_metadata(prepared: dict) -> dict:
    status = prepared['status']
    return {
        'alarm_policy': ALARM_POLICY,
        'threshold_method': CROSS_THRESHOLD_METHOD,
        'score_timing_version': CROSS_TIMING_VERSION,
        'timing_status': status['status'],
        'pending_global_dependency': status['pending_global_dependency'],
        'runtime_certified': False,
        'adapter_context_rows': status['context_rows'],
    }
