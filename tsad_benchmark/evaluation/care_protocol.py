from pathlib import Path

import numpy as np
import pandas as pd

from tsad_benchmark.evaluation import metrics as native
from tsad_benchmark.evaluation import care_bc_metrics as bc
from tsad_benchmark.evaluation.care_bc_affiliation import label_metrics as affiliation
from tsad_benchmark.evaluation.online_protocol import curve_scores, evaluate_threshold

PROTOCOL = 'care_unified_v6_target_status_benchmark_empty_v1'
AFF = ('affiliation_precision', 'affiliation_recall', 'affiliation_f1')
CURVES = tuple(native.classification_metrics_score.__all__)
LABELS = tuple(native.classification_metrics_label.__all__)
H, DELTA, PPD = 288, 72, 144


def require_current(value):
    if value != PROTOCOL:
        raise ValueError('This launcher requires the current CARE evaluation protocol.')


def evaluation_mask(farm, status):
    status = np.asarray(status)
    if status.ndim != 1 or not np.isin(status, np.arange(6)).all():
        raise ValueError('Invalid CARE status vector')
    if farm == 'A':
        return np.ones(len(status), dtype=bool)
    if farm in ('B', 'C'):
        return np.isin(status, [0, 2])
    raise ValueError(f'Unknown wind farm: {farm}')


def current_truth(root, file_name, expected_ids, expected_asset):
    farm, event = Path(file_name).stem.split('_', 1)
    folder = Path(root) / f'Wind Farm {farm}'
    if not (folder / 'event_info.csv').is_file():
        folder /= f'Wind Farm {farm}'
    info = pd.read_csv(folder/'event_info.csv', sep=';')
    match = info[info.event_id.astype(int).eq(int(event))]
    if len(match) != 1:
        raise ValueError(f'Ambiguous current event: {file_name}')
    record = match.iloc[0]
    frame = pd.read_csv(folder/'datasets'/f'{event}.csv', sep=';',
                        usecols=['id', 'asset_id', 'train_test', 'status_type_id'])
    test = frame[frame.train_test.astype(str).str.strip().isin(['prediction', 'test'])]
    if not np.array_equal(test.id.to_numpy(), np.asarray(expected_ids)):
        raise ValueError(f'Current truth row IDs differ: {file_name}')
    if not test.asset_id.astype(str).eq(str(expected_asset)).all():
        raise ValueError(f'Current truth asset differs: {file_name}')
    kind = str(record.event_label).strip().lower()
    if kind not in ('normal', 'anomaly'):
        raise ValueError(f'Invalid current event label: {file_name}')
    labels = np.zeros(len(test), dtype=np.int8)
    if kind == 'anomaly':
        labels = test.id.between(int(record.event_start_id), int(record.event_end_id)).to_numpy(dtype=np.int8)
        if not labels.any():
            raise ValueError(f'Current anomaly is outside test rows: {file_name}')
    return labels, test.status_type_id.to_numpy(), kind


def score_metrics(y, scores, valid, farm, kind, *, available=True):
    if kind == 'normal' or not available:
        return dict.fromkeys(CURVES, np.nan)
    if farm in ('B', 'C'):
        return bc.curves(y, scores, valid)
    if farm != 'A':
        raise ValueError('Unknown farm')
    from tsad_benchmark.evaluation.metrics.vus_exact import exact_vus
    native.clear_score_metric_cache()
    try:
        out = {name: native.METRICS[name](y, scores) for name in CURVES if not name.startswith('vus_')}
        out['vus_roc'], out['vus_pr'] = exact_vus(np.asarray(y, dtype=np.int8), scores)
        return out
    finally:
        native.clear_score_metric_cache()


def in_farm_labels(y, alarms, warnings, valid, farm):
    if farm in ('B', 'C'):
        out = bc.label_metrics(y, alarms, valid, warning_predictions=warnings,
                               horizon_points=H, lead_delta_points=DELTA, points_per_day=PPD)
        core = affiliation(y, alarms, valid)
        for key in AFF:
            out[key+'_unmasked_reference'] = out[key]
            out[key] = core[key]
            out[key+'_core'] = core[key]
        for key, value in core.items():
            if key not in AFF:
                out['aff_support_'+key] = value
        out['affiliation_policy'] = 'care_bc_status_conditioned_affiliation_single_event_v1_native_empty_report'
        return out
    if farm != 'A':
        raise ValueError('Unknown farm')
    kw = dict(horizon_points=H, lead_delta_points=DELTA, points_per_day=PPD,
              warning_predictions=warnings)
    out = native._warning_summary(y, alarms, **kw)
    out.update(zip(['range_precision', 'range_recall', 'range_f1'], native._range_prf(y, alarms)))
    out.update(zip(AFF[:2], native._affiliation_pr(y, alarms)))
    p, r = out[AFF[0]], out[AFF[1]]
    out[AFF[2]] = 2*p*r/(p+r) if np.isfinite(p+r) and p+r else np.nan
    for name in LABELS:
        if name not in out:
            out[name] = native.METRICS[name](y, alarms, **kw)
    false = native._false_alarm_events(y, alarms, horizon=H)
    intervals = np.diff([s for s, e in false])
    negative = int(np.count_nonzero(y == 0))
    false_points = int(np.count_nonzero((y == 0) & alarms.astype(bool)))
    out.update(normal_points=negative, normal_alert_points=false_points,
               normal_point_fpr=false_points/negative if negative else np.nan,
               raw_n_points=len(y), valid_points=len(y), invalid_points=0,
               valid_positive_points=int(np.sum(y)), valid_negative_points=negative,
               false_interval_sum=int(intervals.sum()), false_interval_count=len(intervals),
               observable_eligible_event_count=out['eligible_event_count'],
               observable_early_detection_count=out['early_detection_count'],
               observable_early_detection_rate=out['early_detection_rate'],
               evaluation_policy='generic_A_status_ignored',
               affiliation_policy='native_unmasked_reference_benchmark')
    for key in AFF:
        out[key+'_unmasked_reference'] = out[key]
    return out


def timing_support(prepared, y, valid):
    release = prepared['release']
    published = np.unique(release[(release >= 0) & (release < len(y))])
    events = native._events_from_binary(y)
    if len(events) > 1:
        raise ValueError('CARE expects at most one original truth event')
    start = events[0][0] if events else None
    lower = max(0, start-H) if start is not None else 0
    right = start-DELTA if start is not None else -1
    updates = published[(published >= lower) & (published <= right)]
    updates = updates[valid[updates]]
    status = prepared['status']
    return dict(evaluation_protocol=PROTOCOL, evaluation_truth_version='CARE_v6',
                timing_status=status['status'], pending_global_dependency=status['pending_global_dependency'],
                legacy_identity=status['legacy_identity'], runtime_certified=False,
                eligible=bool(start is not None and start >= DELTA),
                event_count=len(events), available_clock_points=int(prepared['available'].sum()),
                valid_available_clock_points=int((prepared['available'] & valid).sum()),
                raw_test_points=len(y), excluded_status_points=int((~valid).sum()),
                available_warning_updates=len(updates), publication_opportunity=bool(len(updates)))


def _masked_ltu(y, pred, valid, delta=DELTA):
    y = np.asarray(y, dtype=bool); p = np.asarray(pred, dtype=bool); v = np.asarray(valid, dtype=bool)
    d = np.diff(np.r_[False, y, False].astype('int8'))
    starts = np.flatnonzero(d == 1); ends = np.flatnonzero(d == -1)-1
    values = []
    for start, end in zip(starts, ends):
        if not v[start:end+1].any():
            continue
        left = max(0, start-delta)
        active = np.flatnonzero((p & v)[left:end+1])
        if not len(active):
            values.append(0.); continue
        first = left+int(active[0])
        values.append(1. if first <= start or end == start else float((end-first)/(end-start)))
    return float(np.mean(values)) if values else np.nan


def transfer_metrics(y, prepared, threshold, status, farm, kind, floor):
    from experiments.cross_domain.metrics import compute_label_metrics, LABEL_METRICS
    y = np.asarray(y, dtype=np.int8)
    valid = evaluation_mask(farm, status)
    if len(valid) != len(y) or not valid.any():
        raise ValueError('Missing or unaligned valid evaluation exposure')
    p, w = evaluate_threshold(prepared, threshold)
    events = native._events_from_binary(y)
    if len(events) > 1 or bool(events) != (kind == 'anomaly'):
        raise ValueError('Event kind and current truth disagree')
    if farm == 'A':
        label = compute_label_metrics(y, p, event_label=kind, warning_predictions=w,
                                      horizon_points=H, lead_delta_points=DELTA,
                                      points_per_day=PPD, empty_alarm_affiliation='benchmark')
        intervals = max(int(label['false_alarm_events'])-1, 0)
        label.update(event_count=len(events), evaluable_event_count=len(events),
                     delay_support_count=sum(bool(p[s:e+1].any()) for s, e in events),
                     false_interval_count=intervals,
                     false_interval_sum=int(round(label['mtbfa']*intervals)) if intervals else 0,
                     observable_eligible_event_count=int(label['eligible_event_count']),
                     observable_early_detection_count=int(label['early_detection_count']),
                     observable_early_detection_rate=label['early_detection_rate'],
                     valid_warning_opportunity_points=sum(max(0, s-DELTA-max(0, s-H)+1) for s, e in events),
                     raw_alarm_count=len(native._events_from_binary(p)),
                     evaluable_alarm_count=len(native._events_from_binary(p)),
                     normal_points=int((y == 0).sum()), normal_alert_points=int(((y == 0) & (p == 1)).sum()),
                     normal_point_fpr=float(p[y == 0].mean()) if (y == 0).any() else np.nan,
                     affiliation_policy='native_A_benchmark_empty', evaluation_policy='A_original_full_grid')
        for key in AFF:
            label[key+'_core'] = label[key]
        label.update(affiliation_valid_zone_points=len(y), affiliation_valid_truth_points=int(y.sum()),
                     affiliation_valid_negative_points=int((y == 0).sum()), affiliation_valid_alarm_points=int(p.sum()))
    else:
        label = bc.label_metrics(y, p, valid, warning_predictions=w, horizon_points=H,
                                 lead_delta_points=DELTA, points_per_day=PPD, empty_alarm_affiliation='benchmark')
        core = affiliation(y, p, valid)
        for key in AFF:
            label[key+'_unmasked_reference'] = label[key]
            label[key] = core[key]
            label[key+'_core'] = core[key]
        label.update({key: value for key, value in core.items() if key not in AFF})
        label.update(lead_time_utility=_masked_ltu(y, p, valid), ltu=_masked_ltu(y, p, valid))
        label['miss_rate'] = 1.-label['event_recall']
    label.update(n_points=int(valid.sum()), total_points=int(valid.sum()),
                 n_positive=int((y & valid).sum()), n_negative=int(((y == 0) & valid).sum()),
                 excluded_status_points=int((~valid).sum()), raw_n_points=len(y),
                 valid_points=int(valid.sum()), raw_positive_points=int(y.sum()),
                 valid_positive_points=int((y & valid).sum()), valid_negative_points=int(((y == 0) & valid).sum()),
                 raw_alarm_positive_points=int(p.sum()), raw_warning_points=int(w.sum()),
                 evaluation_protocol=PROTOCOL, evaluation_truth_version='CARE_v6',
                 empty_alarm_affiliation='benchmark')
    if kind == 'normal':
        for key in LABEL_METRICS:
            if key not in {'accuracy', 'false_alarms_per_turbine_day', 'mtbfa'}:
                label[key] = np.nan
    curves = score_metrics(y, curve_scores(prepared, floor), valid, farm, kind,
                           available=prepared['available'].any())
    curves.update(n_points=int(valid.sum()), n_positive=int((y & valid).sum()),
                  n_negative=int(((y == 0) & valid).sum()))
    return label, curves, p, w
