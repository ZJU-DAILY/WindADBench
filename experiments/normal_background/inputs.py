import base64
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tsad_benchmark.common.operating_points import TRACK1_ANOMALY_RATIOS
from tsad_benchmark.evaluation.care_protocol import (
    PROTOCOL, current_truth, evaluation_mask,
)
from tsad_benchmark.evaluation.online_protocol import (
    ALARM_POLICY, THRESHOLD_METHOD, curve_scores, evaluate_threshold,
)
from tsad_benchmark.evaluation.score_timing import (
    normalize_model_name, published_score_stream,
)


def _decode(value, field, file_name):
    if not isinstance(value, str) or not value:
        raise ValueError(f'{file_name}: missing {field}')
    try:
        return pickle.loads(base64.b64decode(value.encode('utf-8'), validate=True))
    except Exception as exc:
        raise ValueError(f'{file_name}: cannot decode {field}') from exc


def _configuration(value, field):
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid {field}') from exc
    if not isinstance(parsed, dict):
        raise ValueError(f'{field} must contain an object')
    return parsed


def _truth(dataset_root, file_name, actual, row):
    farm, event = Path(file_name).stem.split('_', 1)
    if farm not in ('A', 'B', 'C') or str(row.farm_id) != farm:
        raise ValueError(f'{file_name}: inconsistent farm ID')
    if not isinstance(actual, pd.DataFrame) or list(actual.columns) != ['label']:
        raise ValueError(f'{file_name}: actual_data must retain the label DataFrame and test row IDs')
    if len(actual) != int(row.test_lens):
        raise ValueError(f'{file_name}: inconsistent test length')
    expected_ids = np.arange(int(row.train_lens), int(row.train_lens) + len(actual))
    if not np.array_equal(actual.index.to_numpy(), expected_ids):
        raise ValueError(f'{file_name}: stored test row IDs differ from the original split')
    folder = Path(dataset_root) / f'Wind Farm {farm}'
    if not (folder / 'event_info.csv').is_file():
        folder /= f'Wind Farm {farm}'
    asset = pd.read_csv(folder / 'datasets' / f'{event}.csv', sep=';',
                        usecols=['asset_id'], nrows=1).asset_id.iloc[0]
    y, status, kind = current_truth(dataset_root, file_name, expected_ids, asset)
    if not np.array_equal(actual.label.to_numpy(), y):
        raise ValueError(f'{file_name}: saved labels differ from the current dataset')
    if str(row.event_label).strip().lower() != kind:
        raise ValueError(f'{file_name}: saved event type differs from the current dataset')
    return y, evaluation_mask(farm, status), farm


def load_models(result_paths, dataset_root):
    if isinstance(result_paths, (str, Path)):
        result_paths = [result_paths]
    paths = [Path(path) for path in result_paths]
    if not paths:
        raise ValueError('At least one result CSV is required')
    required = {
        'model_name', 'model_params', 'strategy_args', 'file_name', 'farm_id',
        'event_label', 'train_lens', 'test_lens', 'typical_anomaly_ratio',
        'evaluation_protocol', 'evaluation_truth_version', 'actual_data',
        'inference_data', 'raw_score_data', 'score_release_data',
        'warning_prediction_data', 'threshold', 'threshold_method',
        'alarm_policy', 'unavailable_score_floor', 'early_detection_rate',
    }
    frames = []
    for path in paths:
        frame = pd.read_csv(path, float_precision='round_trip')
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f'{path}: missing columns {missing}')
        if frame.empty:
            raise ValueError(f'{path}: result CSV is empty')
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if not frame.evaluation_protocol.eq(PROTOCOL).all():
        raise ValueError('All result rows must use the current CARE evaluation protocol')
    if not frame.evaluation_truth_version.eq('CARE_v6').all():
        raise ValueError('All result rows must use CARE v6 truth')
    if not frame.alarm_policy.eq(ALARM_POLICY).all():
        raise ValueError('All result rows must use the current alarm policy')
    if not frame.threshold_method.eq(THRESHOLD_METHOD).all():
        raise ValueError('All result rows must use training-percentile thresholds')
    frame['_model'] = frame.model_name.map(normalize_model_name)
    frame['_ratio'] = pd.to_numeric(frame.typical_anomaly_ratio, errors='raise')
    keys = ['_model', 'file_name', '_ratio']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate model/sequence/operating-point rows across result CSVs')
    models = {}
    for model, model_frame in frame.groupby('_model', sort=False):
        configs = []
        for _, row in model_frame.iterrows():
            params = _configuration(row.model_params, 'model_params')
            strategy = _configuration(row.strategy_args, 'strategy_args')
            if (strategy.get('evaluation_protocol') != PROTOCOL or
                    strategy.get('lead_horizon_points') != 288 or
                    strategy.get('lead_delta_points') != 72 or
                    strategy.get('strategy_name') != 'unfixed_detect_score'):
                raise ValueError(f'{model}: incompatible in-farm score strategy')
            configs.append(json.dumps([params, strategy], sort_keys=True))
        if len(set(configs)) != 1:
            raise ValueError(f'{model}: cannot combine different model or strategy configurations')
        entries = []
        for file_name, group in model_frame.groupby('file_name', sort=False):
            if set(group._ratio) != set(TRACK1_ANOMALY_RATIOS):
                raise ValueError(f'{model}/{file_name}: all eight operating points are required')
            first = group.iloc[0]
            actual = _decode(first.actual_data, 'actual_data', file_name)
            y, valid, farm = _truth(dataset_root, file_name, actual, first)
            raw = np.asarray(_decode(first.raw_score_data, 'raw_score_data', file_name), dtype=float)
            release = np.asarray(_decode(first.score_release_data, 'score_release_data', file_name))
            if raw.ndim != 1 or len(raw) != len(y) or not np.isfinite(raw).all():
                raise ValueError(f'{model}/{file_name}: raw scores must be finite and aligned with truth')
            stream = published_score_stream(raw, release)
            prepared = dict(raw=raw, release=release, stream=stream, available=np.isfinite(stream))
            baseline, thresholds = {}, {}
            for _, row in group.iterrows():
                other_actual = _decode(row.actual_data, 'actual_data', file_name)
                if not isinstance(other_actual, pd.DataFrame) or not actual.equals(other_actual):
                    raise ValueError(f'{model}/{file_name}: labels differ between operating points')
                for field, expected in (('raw_score_data', raw), ('score_release_data', release)):
                    if not np.array_equal(_decode(row[field], field, file_name), expected):
                        raise ValueError(f'{model}/{file_name}: {field} differs between operating points')
                if any(row[key] != first[key] for key in ('farm_id', 'event_label', 'train_lens', 'test_lens')):
                    raise ValueError(f'{model}/{file_name}: metadata differs between operating points')
                threshold = float(row.threshold)
                if not np.isfinite(threshold):
                    raise ValueError(f'{model}/{file_name}: training threshold must be finite')
                alarms, warnings = evaluate_threshold(prepared, threshold)
                inference = _decode(row.inference_data, 'inference_data', file_name)
                if not isinstance(inference, (list, tuple)) or len(inference) != 2:
                    raise ValueError(f'{model}/{file_name}: invalid inference_data')
                if not np.array_equal(inference[0], curve_scores(prepared, float(row.unavailable_score_floor))):
                    raise ValueError(f'{model}/{file_name}: saved curve scores do not match published scores')
                if not np.array_equal(inference[1], alarms):
                    raise ValueError(f'{model}/{file_name}: saved alarms do not match the recorded threshold')
                saved_warnings = _decode(row.warning_prediction_data, 'warning_prediction_data', file_name)
                if not np.array_equal(saved_warnings, warnings):
                    raise ValueError(f'{model}/{file_name}: saved warning pulses do not match score releases')
                ratio = float(row._ratio)
                thresholds[str(ratio)] = threshold
                baseline[ratio] = dict(early_detection_rate=float(row.early_detection_rate))
            entries.append(dict(model=model, model_name=str(first.model_name), file_name=file_name,
                                farm_id=farm, y=y, valid=valid, prepared=prepared,
                                thresholds=thresholds, baseline=baseline))
        models[model] = entries
    return models
