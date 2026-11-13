from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from tsad_benchmark.common.operating_points import TRACK1_ANOMALY_RATIOS
from tsad_benchmark.evaluation.online_protocol import evaluate_threshold
from .budget import BUDGETS

H, DELTA = 288, 72
SETTINGS = [(f'grid_{r:g}', 'eight_point', float(r), None) for r in TRACK1_ANOMALY_RATIOS]
SETTINGS += [('train_fixed_1pct', 'train_fixed_1pct', 1., None)]
SETTINGS += [(f'rq3_{b*100:g}pct', f'rq3_{b*100:g}pct', b*100, b) for b in BUDGETS]


def matching_candidates(donor, depth, publication_signature, candidate_mask, *, delta=DELTA):
    publication_signature = np.asarray(publication_signature, bool)
    candidate_mask = np.asarray(candidate_mask, bool)
    width = depth - delta + 1
    if width <= 0 or len(publication_signature) != width or len(candidate_mask) != width:
        raise ValueError('Window signatures must match the inclusive warning window')
    if np.any(candidate_mask & ~publication_signature):
        raise ValueError('Candidate warning offsets must be actual releases')
    key = (depth, delta, publication_signature.tobytes(), candidate_mask.tobytes())
    cache = donor.setdefault('matches', {})
    if key in cache:
        return cache[key]
    n = len(donor['y'])
    if n <= depth:
        result = np.array([], dtype=np.int64)
    else:
        releases = sliding_window_view(np.asarray(donor['publication'], bool), width)[:n-depth]
        match = np.all(releases == publication_signature, axis=1)
        offsets = np.flatnonzero(candidate_mask)
        if len(offsets):
            validity = sliding_window_view(np.asarray(donor['valid'], bool), width)[:n-depth]
            match &= np.all(validity[:, offsets], axis=1)
        result = np.flatnonzero(match)
    cache[key] = result
    return result

def window_probabilities(warnings, starts, candidate_mask):
    warnings = np.atleast_2d(np.asarray(warnings, bool))
    offsets = np.flatnonzero(candidate_mask)
    if not len(offsets):
        return np.zeros(len(warnings), dtype=float)
    starts = np.asarray(starts, int)
    if not len(starts):
        return np.full(len(warnings), np.nan)
    counts = np.zeros(len(warnings), dtype=np.int64)
    for left in range(0, len(starts), 2048):
        indices = starts[left:left+2048, None] + offsets
        counts += warnings[:, indices].any(axis=2).sum(axis=1)
    return counts / len(starts)

def publication_mask(prepared, n):
    release = np.asarray(prepared['release'])
    result = np.zeros(n, dtype=bool)
    result[np.unique(release[(release >= 0) & (release < n)])] = True
    return result

def compute_model(model, sequences, budget_lookup):
    for entry in sequences:
        y, prepared = entry['y'], entry['prepared']
        entry['publication'] = publication_mask(prepared, len(y))
        points = np.flatnonzero(y)
        entry['onset'] = int(points[0]) if len(points) else None
        entry['matches'] = {}
        entry['threshold_list'], warnings, entry['observed'] = [], [], []
        for _, _, ratio, budget in SETTINGS:
            threshold = (float(entry['thresholds'][str(float(ratio))]) if budget is None else
                         float(budget_lookup.loc[(entry['file_name'], budget), 'threshold']))
            _, releases = evaluate_threshold(prepared, threshold)
            assert not np.any(releases.astype(bool) & ~entry['publication'])
            warnings.append(releases)
            entry['threshold_list'].append(threshold)
        entry['warnings'] = np.asarray(warnings, dtype=bool)
    normals = [entry for entry in sequences if entry['onset'] is None]
    events = [entry for entry in sequences if entry['onset'] is not None and entry['onset'] >= DELTA]
    if not events:
        raise ValueError('No eligible anomaly sequences for normal-background evaluation')
    donor_rows, event_rows = [], []
    for event in events:
        onset = event['onset']
        depth = min(onset, H)
        width = depth - DELTA + 1
        left = onset - depth
        signature = event['publication'][left:onset-DELTA+1]
        candidate = signature & event['valid'][left:onset-DELTA+1]
        n_opportunity = int(candidate.sum())
        observed = event['warnings'][:, left:onset-DELTA+1][:, candidate].any(axis=1).astype(float)
        pattern = hashlib.sha256(str(depth).encode() + signature.tobytes() + candidate.tobytes()).hexdigest()[:16]
        donors = [entry for entry in normals if entry['farm_id'] == event['farm_id']]
        supported, origin_supported = [], []
        total_windows = 0
        matched_donors = 0
        for donor in donors:
            starts = matching_candidates(donor, depth, signature, candidate)
            count = len(starts)
            probabilities = window_probabilities(donor['warnings'], starts, candidate)
            if count:
                supported.append(probabilities)
                total_windows += count
                matched_donors += 1
            origin_match = bool(onset < len(donor['y']) and
                np.array_equal(donor['publication'][left:left+width], signature) and
                donor['valid'][left:left+width][candidate].all())
            origin = window_probabilities(donor['warnings'], np.array([left]) if origin_match else np.array([],int), candidate)
            if origin_match:
                origin_supported.append(origin)
            digest = hashlib.sha256(starts.astype('<i8').tobytes()).hexdigest()
            for index, (setting, policy, ratio, budget) in enumerate(SETTINGS):
                donor_rows.append(dict(model=model, farm_id=event['farm_id'], event_file=event['file_name'],
                    event_onset=onset, effective_depth=depth, window_points=width,
                    publication_opportunity_count=int(signature.sum()), opportunity_count=n_opportunity,
                    event_valid_point_count=int(event['valid'][left:left+width].sum()), pattern=pattern,
                    donor_file=donor['file_name'], setting=setting, policy=policy,
                    ratio_percent=ratio, budget_normal_point_fpr=budget,
                    donor_threshold=donor['threshold_list'][index], candidate_windows=count,
                    first_candidate_left=int(starts[0]) if count else -1,
                    last_candidate_left=int(starts[-1]) if count else -1,
                    candidate_left_sha256=digest, background_probability=probabilities[index],
                    background_status='zero_target_opportunity' if not n_opportunity else ('matched' if count else 'no_matched_donor_windows'),
                    origin_match=origin_match, origin_background_probability=origin[index]))
        if not n_opportunity:
            means = origins = np.zeros(len(SETTINGS))
        else:
            means = np.mean(supported, axis=0) if supported else np.full(len(SETTINGS), np.nan)
            origins = np.mean(origin_supported, axis=0) if origin_supported else np.full(len(SETTINGS), np.nan)
        strict_all = matched_donors == len(donors)
        for index, (setting, policy, ratio, budget) in enumerate(SETTINGS):
            if budget is not None:
                expected = float(budget_lookup.loc[(event['file_name'], budget),'early_detection_rate'])
                assert np.isclose(observed[index], expected, atol=1e-12, rtol=1e-12), (model,event['file_name'],setting)
            elif event['farm_id'] == 'A':
                expected = float(event['baseline'][float(ratio)]['early_detection_rate'])
                assert np.isclose(observed[index], expected, atol=1e-12, rtol=1e-12)
            event_rows.append(dict(model=model, farm_id=event['farm_id'], event_file=event['file_name'],
                event_onset=onset, effective_depth=depth, window_points=width,
                opportunity_count=n_opportunity, publication_opportunity_count=int(signature.sum()), pattern=pattern,
                setting=setting, policy=policy, ratio_percent=ratio, budget_normal_point_fpr=budget,
                event_threshold=event['threshold_list'][index], observed_edr=observed[index],
                background_probability=means[index], excess_over_background=observed[index]-means[index],
                origin_background_probability=origins[index], excess_over_origin=observed[index]-origins[index],
                total_normal_donors=len(donors), matched_donors=matched_donors,
                partial_donor_support=not strict_all, strict_all_donors=strict_all,
                strict_background_probability=means[index] if strict_all or not n_opportunity else np.nan,
                matched_origin_donors=len(origin_supported), candidate_windows=total_windows,
                background_status='zero_target_opportunity' if not n_opportunity else ('matched' if supported else 'no_matched_donors')))
    return pd.DataFrame(event_rows), pd.DataFrame(donor_rows)


def _complete_mean(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if len(values) and np.isfinite(values).all() else np.nan

def _event_average(group):
    result = dict(observed_edr=_complete_mean(group.observed_edr),
                  background_probability=_complete_mean(group.background_probability))
    result['excess_over_background'] = result['observed_edr']-result['background_probability']
    for name in ('origin_background_probability','strict_background_probability'):
        if name in group:
            result[name] = _complete_mean(group[name])
    if 'origin_background_probability' in result:
        result['excess_over_origin'] = result['observed_edr']-result['origin_background_probability']
    for name in ('matched_donors','total_normal_donors','candidate_windows','opportunity_count'):
        if name in group:
            result[name] = group[name].min()
    return result

def _farm_summary(group):
    result = _event_average(group)
    paired = group[np.isfinite(group.background_probability) & np.isfinite(group.observed_edr)]
    result.update(eligible_event_count=len(group), paired_event_count=len(paired),
                  missing_background_event_count=len(group)-len(paired),
                  paired_observed_edr=_complete_mean(paired.observed_edr),
                  paired_background_probability=_complete_mean(paired.background_probability))
    result['paired_excess_over_background'] = result['paired_observed_edr']-result['paired_background_probability']
    return result

def _macro_summary(group):
    fields = ['observed_edr','background_probability','excess_over_background',
              'paired_observed_edr','paired_background_probability','paired_excess_over_background',
              'origin_background_probability','excess_over_origin','strict_background_probability']
    result = {name:_complete_mean(group[name]) for name in fields if name in group}
    for name in ['eligible_event_count','paired_event_count','missing_background_event_count']:
        result[name] = int(group[name].sum())
    result['farm_count'] = len(group)
    result['paired_farm_count'] = int(group.paired_event_count.gt(0).sum())
    return result

def _group_apply(frame, keys, function):
    rows=[]
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(keys,key), **function(group)))
    return pd.DataFrame(rows)

def aggregate_events(events):
    per_event = _group_apply(events, ['model','policy','farm_id','event_file'], _event_average)
    per_farm = _group_apply(per_event, ['model','policy','farm_id'], _farm_summary)
    macro = _group_apply(per_farm, ['model','policy'], _macro_summary)
    ratio_farm = _group_apply(events, ['model','policy','setting','ratio_percent','farm_id'], _farm_summary)
    ratio_macro = _group_apply(ratio_farm, ['model','policy','setting','ratio_percent'], _macro_summary)
    return dict(per_event_setting=events,per_event=per_event,per_farm=per_farm,macro=macro,
                per_ratio_farm=ratio_farm,per_ratio_macro=ratio_macro)
