from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.cross_domain.metrics import LABEL_METRICS, SCORE_METRICS

KEYS = ['model', 'source_farm', 'target_farm', 'track']
POLICY = 'sequence_equal_event_support_normal_exposure_direction_equal_v1'
ALIASES = ('acc', 'accuracy_normal', 'normal_point_fpr', 'normal_fpr',
           'false_alarms_per_turbine_day_normal', 'mtbfa_normal',
           'observable_early_detection_rate')
METRICS = tuple(dict.fromkeys((*LABEL_METRICS, *SCORE_METRICS, *ALIASES)))
NORMAL_METRICS = {'accuracy', 'false_alarms_per_turbine_day', 'mtbfa'}
TIMES = {'mean_lead_time': 'lead_support_count',
         'actionable_lead_time': 'actionable_lead_support_count',
         'mean_detection_delay': 'delay_support_count'}
SPECIAL = {*NORMAL_METRICS, *TIMES, 'early_detection_rate', 'early_detection_rate_all'}
SUPPORTS = ('anomaly_event_count', 'eligible_event_count', 'early_detection_count',
            'lead_support_count', 'actionable_lead_support_count', 'delay_support_count',
            'observable_eligible_event_count', 'observable_early_detection_count',
            'normal_points', 'normal_correct_points', 'normal_alert_points',
            'normal_monitoring_days', 'false_alarm_events', 'turbine_days',
            'mtbfa_interval_sum', 'mtbfa_interval_count',
            'false_interval_sum', 'false_interval_count')
IDENTITY = ('seed', 'protocol', 'model_name', 'model_params', 'evaluation_truth_version',
            'training_data_version', 'score_timing_version', 'alarm_policy', 'threshold_method')


def _numeric(frame, name, *, required=False):
    if name not in frame:
        if required:
            raise ValueError(f'Missing required aggregation field: {name}')
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[name], errors='raise').astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError(f'Infinite {name}')
    return values


def _count(frame, name):
    values = _numeric(frame, name, required=bool(len(frame)))
    if values.isna().any() or (values < 0).any() or not np.allclose(values, np.round(values)):
        raise ValueError(f'{name} must contain defined nonnegative integer support')
    return values


def _mean(values):
    values = values.dropna()
    return float(values.mean()) if len(values) else np.nan


def _ratio(num, den):
    return float(num / den) if den > 0 else np.nan


def _weighted_time(group, metric, support, event_counts):
    values = _numeric(group, metric)
    weights = _numeric(group, support)
    missing = weights.isna()

    if missing.any() and support == 'delay_support_count':
        if not event_counts.loc[missing].eq(1).all():
            raise ValueError('Cannot infer delay support for multiple truth events')
        weights.loc[missing] = values.loc[missing].notna().astype(float)
    if weights.isna().any() or (weights < 0).any() or not np.allclose(weights, np.round(weights)):
        raise ValueError(f'Invalid {support}')
    if (weights > event_counts).any():
        raise ValueError(f'{support} exceeds original event support')
    if (weights.gt(0) & values.isna()).any() or (weights.eq(0) & values.notna()).any():
        raise ValueError(f'{metric} value does not match its support')
    if (values.dropna() < 0).any():
        raise ValueError(f'Negative {metric}')
    return _ratio(float((values.fillna(0) * weights).sum()), float(weights.sum())), int(weights.sum()), int(weights.gt(0).sum())


def _intervals(normal):
    sums, counts = [], []
    for _, row in normal.iterrows():
        count, total = row.get('false_interval_count', np.nan), row.get('false_interval_sum', np.nan)
        if pd.isna(count) != pd.isna(total):
            raise ValueError('Partial explicit interval support')
        if pd.isna(count):
            alarms = float(row['false_alarm_events'])
            count = max(alarms - 1, 0)
            mean = row.get('mtbfa', np.nan)
            if count and pd.isna(mean):
                raise ValueError('Intervals implied by alarms but MTBFA is missing')
            total = float(mean) * count if count else 0.
        if not np.isfinite(count) or not np.isfinite(total) or count < 0 or total < 0 or count != int(count):
            raise ValueError('Invalid MTBFA support')
        if count == 0 and total != 0:
            raise ValueError('Nonzero interval sum without intervals')
        sums.append(float(total)); counts.append(int(count))
    return float(sum(sums)), int(sum(counts)), int(sum(c > 0 for c in counts))


def _validate(rows):
    required = [*KEYS, 'target_file_name', 'target_asset_id', 'target_event_label']
    missing = set(required) - set(rows)
    if missing:
        raise ValueError(f'Missing identity columns: {sorted(missing)}')
    if rows.empty:
        raise ValueError('Cannot aggregate an empty evaluation table')
    if rows[required].isna().any().any():
        raise ValueError('Undefined evaluation identity')
    if not set(rows.track) <= {'label', 'score'} or not set(rows.target_event_label) <= {'normal', 'anomaly'}:
        raise ValueError('Invalid track/event label')
    if rows.duplicated([*KEYS, 'target_file_name']).any():
        raise ValueError('Duplicate sequence/operating point in direction')
    for track, other in [('score', LABEL_METRICS), ('label', SCORE_METRICS)]:
        own = rows[rows.track.eq(track)]
        for metric in other:
            if _numeric(own, metric).notna().any():
                raise ValueError(f'{track} rows contain the other track metric {metric}')
    normal = rows[rows.target_event_label.eq('normal')]
    for metric in (*LABEL_METRICS, *SCORE_METRICS):
        if metric not in NORMAL_METRICS and _numeric(normal, metric).notna().any():
            raise ValueError(f'Normal sequences contain undefined anomaly metric {metric}')


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    _validate(rows)
    records = []
    for key, group in rows.groupby(KEYS, sort=True, dropna=False):
        result = dict(zip(KEYS, key))
        for column in IDENTITY:
            if column in group:
                values = group[column].dropna().astype(str).unique()
                if len(values) > 1:
                    raise ValueError(f'Mixed {column} within one evaluation direction')
                if len(values):
                    result[column] = group[column].dropna().iloc[0]
        anomaly = group[group.target_event_label.eq('anomaly')]
        normal = group[group.target_event_label.eq('normal')]
        result.update(aggregation_policy=POLICY, time_units='sampling_points',
                      sequence_count=len(group), anomaly_sequence_count=len(anomaly),
                      normal_sequence_count=len(normal), asset_count=group.target_asset_id.nunique(),
                      anomaly_asset_count=anomaly.target_asset_id.nunique(),
                      normal_asset_count=normal.target_asset_id.nunique())
        result.update({metric: np.nan for metric in METRICS})
        result.update({metric + '_defined_sequences': 0 for metric in METRICS})
        result.update({support: np.nan for support in SUPPORTS})
        own_metrics = SCORE_METRICS if key[-1] == 'score' else LABEL_METRICS
        for metric in own_metrics:
            if metric in SPECIAL:
                continue
            values = _numeric(anomaly, metric)
            result[metric] = _mean(values)
            result[metric + '_defined_sequences'] = int(values.notna().sum())
        if key[-1] == 'score':
            records.append(result)
            continue

        events = _numeric(anomaly, 'event_count')

        events = events.fillna(1.)
        if (events < 1).any() or not np.allclose(events, np.round(events)):
            raise ValueError('Invalid original event_count in anomaly sequences')
        eligible = _count(anomaly, 'eligible_event_count')
        early = _count(anomaly, 'early_detection_count')
        if (early > eligible).any() or (eligible > events).any():
            raise ValueError('Early/eligible/original event support is inconsistent')
        result.update(anomaly_event_count=int(events.sum()), eligible_event_count=int(eligible.sum()),
                      early_detection_count=int(early.sum()),
                      early_detection_rate=_ratio(early.sum(), eligible.sum()),
                      early_detection_rate_all=_ratio(early.sum(), events.sum()),
                      early_detection_rate_defined_sequences=int(eligible.gt(0).sum()),
                      early_detection_rate_all_defined_sequences=len(anomaly))
        for metric, support in TIMES.items():
            mean, count, sequence_count = _weighted_time(anomaly, metric, support, events)
            result[metric] = mean
            result[support] = count
            result[metric + '_defined_sequences'] = sequence_count
        if result['actionable_lead_support_count'] != result['early_detection_count']:
            raise ValueError('Actionable lead support must match early detections')
        observable = _numeric(anomaly, 'observable_eligible_event_count')
        if len(observable) and observable.notna().all():
            oc = _count(anomaly, 'observable_eligible_event_count')
            oe = _numeric(anomaly, 'observable_early_detection_count')
            oe = early if oe.isna().all() else _count(anomaly, 'observable_early_detection_count')
            if (oe > oc).any() or (oc > eligible).any():
                raise ValueError('Inconsistent observable opportunity support')
            result.update(observable_eligible_event_count=int(oc.sum()),
                          observable_early_detection_count=int(oe.sum()),
                          observable_early_detection_rate=_ratio(oe.sum(), oc.sum()),
                          observable_early_detection_rate_defined_sequences=int(oc.gt(0).sum()))

        points = _count(normal, 'n_points')
        correct = _count(normal, 'correct_points')
        false_alarms = _count(normal, 'false_alarm_events')
        days = _numeric(normal, 'turbine_days', required=bool(len(normal)))
        if days.isna().any() or (days < 0).any() or (correct > points).any():
            raise ValueError('Invalid normal point/exposure support')
        if ((days == 0) != (points == 0)).any() or ((points == 0) & false_alarms.gt(0)).any():
            raise ValueError('Normal exposure and point support disagree')
        interval_sum, interval_count, interval_sequences = _intervals(normal)
        accuracy = _ratio(correct.sum(), points.sum())
        far = _ratio(false_alarms.sum(), days.sum())
        interval_mean = _ratio(interval_sum, interval_count)
        result.update(normal_points=int(points.sum()), normal_correct_points=int(correct.sum()),
                      normal_alert_points=int((points-correct).sum()),
                      normal_monitoring_days=float(days.sum()), turbine_days=float(days.sum()),
                      false_alarm_events=int(false_alarms.sum()),
                      mtbfa_interval_sum=interval_sum, mtbfa_interval_count=interval_count,
                      false_interval_sum=interval_sum, false_interval_count=interval_count)
        for metric in ('accuracy', 'accuracy_normal', 'acc'):
            result[metric] = accuracy
            result[metric + '_defined_sequences'] = int(points.gt(0).sum())
        for metric in ('normal_fpr', 'normal_point_fpr'):
            result[metric] = 1.-accuracy if np.isfinite(accuracy) else np.nan
            result[metric + '_defined_sequences'] = int(points.gt(0).sum())
        for metric in ('false_alarms_per_turbine_day', 'false_alarms_per_turbine_day_normal'):
            result[metric] = far
            result[metric + '_defined_sequences'] = int(days.gt(0).sum())
        for metric in ('mtbfa', 'mtbfa_normal'):
            result[metric] = interval_mean
            result[metric + '_defined_sequences'] = interval_sequences
        records.append(result)
    return pd.DataFrame(records)


def macro(directions: pd.DataFrame) -> pd.DataFrame:
    required = [*KEYS, *METRICS]
    missing = set(required)-set(directions)
    if missing or directions.empty:
        raise ValueError(f'Missing/empty direction summaries: {sorted(missing)}')
    if directions[KEYS].isna().any().any() or directions.duplicated(KEYS).any():
        raise ValueError('Duplicate or undefined direction identity')
    rows = []
    count_columns = ['sequence_count', 'anomaly_sequence_count', 'normal_sequence_count',
                     'asset_count', 'anomaly_asset_count', 'normal_asset_count', *SUPPORTS]
    for key, group in directions.groupby(['model', 'track'], sort=True):
        result = dict(zip(['model', 'track'], key), direction_count=len(group),
                      aggregation_policy=POLICY, time_units='sampling_points')
        for metric in METRICS:
            values = _numeric(group, metric)
            result[metric] = _mean(values)
            result[metric + '_defined_directions'] = int(values.notna().sum())
            support = metric + '_defined_sequences'
            if support in group:
                result[support] = int(group[support].sum())
        for column in count_columns:
            if column in group:

                name = column + '_direction_sum' if column.endswith('asset_count') else column
                result[name] = group[column].sum(min_count=1)
        rows.append(result)
    return pd.DataFrame(rows)
