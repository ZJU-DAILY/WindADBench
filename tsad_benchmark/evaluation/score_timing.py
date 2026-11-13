from __future__ import annotations

from collections.abc import Mapping
import operator
import re
import numpy as np


_POINT = {'cblof', 'hbos', 'lof', 'iforest', 'eif', 'knn', 'loda', 'ocsvm', 'pca', 'ae', 'dagmm'}
_BLOCK = {'anomaly_transformer', 'catch', 'dcdetector', 'duet', 'tranad', 'gpt4ts', 'unitime', 'moment', 'units', 'dada'}
_ENDPOINT = {'d3r', 'mtad_gat', 'mtgflow', 'rpcl_tcne_mts_llm', 'usad', 'chronos'}
_GLOBAL = {'gdn', 'deeppoint'}
_OTHER = {'sarad', 'torsk', 'timesnet', 'kmeans', 'lstmed', 'mscred', 'omnianomaly'}
_MODELS = _POINT | _BLOCK | _ENDPOINT | _GLOBAL | _OTHER
_ALIASES = {re.sub('[^a-z0-9]', '', name): name for name in _MODELS}
_ALIASES.update({
    'autoencoder': 'ae', 'if': 'iforest', 'isolationforest': 'iforest',
    'extendedisolationforest': 'eif', 'at': 'anomaly_transformer',
    'transad': 'tranad', 'deepoint': 'deeppoint',
    'deeppointanomalydetector': 'deeppoint', 'lstmencoderdecoder': 'lstmed',
    'pcltcnemtsllm': 'rpcl_tcne_mts_llm',
})

__all__ = ['normalize_model_name', 'timing_status', 'score_timing',
           'publication_indices', 'score_valid_mask', 'valid_calibration_scores',
           'timed_alarm_events', 'published_score_stream']


def normalize_model_name(model_name: str) -> str:
    if not isinstance(model_name, str):
        raise TypeError('model_name must be a string')
    name = model_name.strip().rsplit('.', 1)[-1]
    key = re.sub('[^a-z0-9]', '', name.lower())
    if key.endswith('model'):
        key = key[:-5]
    if key not in _ALIASES:
        raise KeyError(f'No audited score timing for model {model_name!r}')
    return _ALIASES[key]


def _integer(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be an integer, not bool')
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f'{name} must be an integer') from exc
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return value


def _params(params) -> Mapping:
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise TypeError('params must be a mapping')
    return params


def _positive(params: Mapping, key: str, alternative: str | None = None) -> int:
    if key not in params and alternative is not None:
        key = alternative
    if key not in params:
        raise ValueError(f'Score timing requires recorded parameter {key!r}')
    return _integer(params[key], key)


def timing_status(model_name: str, params=None, legacy_global: bool = False) -> dict:
    model = normalize_model_name(model_name)
    params = _params(params)
    protocol = params.get('score_protocol')
    pending = model in _GLOBAL and protocol != 'source_frozen_v1'
    return {
        'canonical_model': model,
        'status': 'pending_global_dependency' if pending else 'audited_schedule',
        'score_protocol': protocol,
        'pending_global_dependency': pending,
        'legacy_identity': bool(pending and legacy_global),
        'runtime_certified': False,
        'scope': ('offline identity only; full-evaluation-set dependence remains'
                  if pending else 'static dependency schedule; adapter and checkpoint assumptions still apply'),
    }


def score_timing(model_name: str, n: int, params=None, legacy_global: bool = False) -> np.ndarray:
    n = _integer(n, 'n', minimum=0)
    model = normalize_model_name(model_name)
    params = _params(params)
    release = np.arange(n, dtype=np.int64)
    if model in _GLOBAL:
        if params.get('score_protocol') != 'source_frozen_v1':
            if legacy_global:
                return release
            raise ValueError(f'{model}: unresolved global test dependence; use source_frozen_v1 or explicitly mark legacy_global=True')
        if model == 'gdn':
            warm = _positive(params, 'win_size') + 3
        else:
            warm = _integer(params.get('difference_warmup', 1), 'difference_warmup', minimum=0)
        release[:warm] = -1
        return release
    if model in _POINT:
        if model in {'ae', 'dagmm'}:
            length = params.get('sequence_len', params.get('sequence_length', 1))
            if _integer(length, 'sequence_len') != 1:
                raise ValueError(f'{model}: identity timing is audited only for sequence_len=1')
        return release
    if model == 'sarad':

        return release
    if model == 'torsk':
        release[:1] = -1
        return release
    if model in _ENDPOINT:
        warm = (_positive(params, 'context_length') if model == 'chronos'
                else _positive(params, 'win_size') - (model != 'mtad_gat'))
        release[:warm] = -1
        return release
    if model in _BLOCK or model == 'timesnet':
        window = _positive(params, 'win_size')
        batch = 1
        if model == 'timesnet':
            protocol = params.get('score_protocol')
            if protocol == 'independent_window_v1':
                batch = _positive(params, 'score_batch_size')
                if batch != 1:
                    raise ValueError('TimesNet independent_window_v1 requires score_batch_size=1')
            elif protocol is None:

                batch = _positive(params, 'batch_size')
            else:
                raise ValueError(f'Unsupported TimesNet score protocol: {protocol!r}')
        release = (release // (window * batch) + 1) * (window * batch) - 1

        release[(n // window) * window:] = -1
        return release
    if model in {'kmeans', 'lstmed'}:
        window = (_positive(params, 'window_size') if model == 'kmeans'
                  else _positive(params, 'sequence_len', 'sequence_length'))
        return release + window - 1
    if model == 'mscred':
        gap = _positive(params, 'gap_time')
        step = _positive(params, 'step_max')
        scales = params.get('signature_scales')
        if not isinstance(scales, (list, tuple, np.ndarray)) or len(scales) == 0:
            raise ValueError('MSCRED timing requires nonempty signature_scales')
        first = max(_integer(v, 'signature_scale') for v in scales) - 1 + (step - 1) * gap
        release = first + np.maximum(0, (release - first + gap - 1) // gap) * gap
        release[:first] = -1
        return release
    if model == 'omnianomaly':
        window = _positive(params, 'win_size')
        batch = _positive(params, 'batch_size')
        if n < window:
            return np.full(n, -1, dtype=np.int64)
        left = (window - 1) // 2
        j = np.clip(release - left, 0, n - window)

        release = (j // batch + 1) * batch + window - 2
        release[:left] = -1
        release[left + n - window + 1:] = -1
        return release
    raise KeyError(f'No score timing implementation for {model!r}')


publication_indices = score_timing


def _release_array(release) -> np.ndarray:
    raw = np.asarray(release)
    if raw.ndim != 1:
        raise ValueError('release must be a one-dimensional array')
    if raw.size == 0:
        return np.empty(0, dtype=np.int64)
    if raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
        raise ValueError('release must contain finite integer indices')
    if (raw < -1).any() or (raw != np.floor(raw)).any() or (raw >= 2**63).any():
        raise ValueError('release indices must be -1 or nonnegative int64 values')
    return raw.astype(np.int64, copy=False)


def score_valid_mask(release) -> np.ndarray:
    release = _release_array(release)
    return (release >= 0) & (release < len(release))


def _paired(scores, release):
    scores = np.asarray(scores, dtype=float)
    release = _release_array(release)
    if scores.ndim != 1 or len(scores) != len(release):
        raise ValueError('scores and release must be one-dimensional arrays of equal length')
    valid = score_valid_mask(release)
    if not np.isfinite(scores[valid]).all():
        raise ValueError('published scores must be finite; invalid padding may be non-finite')
    return scores, release, valid


def valid_calibration_scores(scores, release) -> np.ndarray:
    scores, _, valid = _paired(scores, release)
    return scores[valid]


def timed_alarm_events(scores, threshold: float, release) -> np.ndarray:
    scores, release, valid = _paired(scores, release)
    threshold = float(threshold)
    if np.isnan(threshold):
        raise ValueError('threshold cannot be NaN')
    return release[valid & (scores > threshold)]


def published_score_stream(scores, release) -> np.ndarray:
    scores, release, valid = _paired(scores, release)
    n = len(scores)
    result = np.full(n, np.nan, dtype=float)
    if not valid.any():
        return result
    batch_max = np.full(n, -np.inf, dtype=float)
    np.maximum.at(batch_max, release[valid], scores[valid])
    updated = np.isfinite(batch_max)
    latest = np.maximum.accumulate(np.where(updated, np.arange(n), -1))
    available = latest >= 0
    result[available] = batch_max[latest[available]]
    return result
