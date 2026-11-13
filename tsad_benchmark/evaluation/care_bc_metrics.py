from __future__ import annotations

import numpy as np

from tsad_benchmark.evaluation import metrics as native
from tsad_benchmark.evaluation.care_bc_range import range_vus

POLICY = "care_bc_valid_footprint_original_grid_v1"
RANGE_POLICY = "valid_mass_original_geometry_exact_250_thresholds_v1"


def _binary(values, name):
    value = np.asarray(values)
    if value.ndim != 1 or not np.isin(value, [0, 1]).all():
        raise ValueError(f"{name} must be a one-dimensional binary vector")
    return value.astype(np.int8)


def _inputs(y, predictions, valid, *, score=False):
    y = _binary(y, "labels")
    valid = _binary(valid, "valid").astype(bool)
    predictions = np.asarray(predictions, dtype=float) if score else _binary(predictions, "alarms")
    if predictions.ndim != 1 or len(y) != len(predictions) or len(y) != len(valid):
        raise ValueError("labels, predictions and valid must have equal one-dimensional lengths")
    if score and not np.isfinite(predictions).all():
        raise ValueError("scores must be finite on the original timeline")
    return y, predictions, valid


def _mean(values):
    return float(np.mean(values)) if len(values) else np.nan


def _f1(p, r):
    return np.nan if np.isnan(p) or np.isnan(r) else 2 * p * r / (p + r) if p + r else 0.0


def label_metrics(y, alarms, valid, warning_predictions=None, horizon_points=288,
                  lead_delta_points=72, points_per_day=144,
                  empty_alarm_affiliation="benchmark"):
    y, alarms, valid = _inputs(y, alarms, valid)
    if empty_alarm_affiliation not in {"benchmark", "legacy_zero"}:
        raise ValueError("Unknown empty_alarm_affiliation policy")
    horizon, delta = native._warning_parameters(horizon_points=horizon_points,
                                                lead_delta_points=lead_delta_points)
    if isinstance(points_per_day, (bool, np.bool_)) or int(points_per_day) != points_per_day or points_per_day <= 0:
        raise ValueError("points_per_day must be a positive integer")
    warning = alarms if warning_predictions is None else _binary(warning_predictions, "warning_predictions")
    if len(warning) != len(y):
        raise ValueError("warning_predictions must have the original timeline length")
    original_truth = native._events_from_binary(y)
    original_alarms = native._events_from_binary(alarms)
    prefix = np.r_[0, np.cumsum(valid, dtype=np.int64)]

    def mass(interval):
        left, right = interval
        return int(prefix[right + 1] - prefix[left]) if right >= left else 0

    def overlap(first, second):
        return mass((max(first[0], second[0]), min(first[1], second[1])))

    truth = [event for event in original_truth if mass(event)]
    predicted = [event for event in original_alarms if mass(event)]
    nt, npred = len(truth), len(predicted)
    nv = int(valid.sum())
    pos = int(np.count_nonzero(valid & (y == 1)))
    neg = nv - pos
    tp = int(np.count_nonzero(valid & (y == 1) & (alarms == 1)))
    fp = int(np.count_nonzero(valid & (y == 0) & (alarms == 1)))
    fn, tn = pos - tp, neg - fp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / pos if pos else np.nan if original_truth else 0.0
    p = p if nv else np.nan
    r = r if nv else np.nan
    out = dict.fromkeys(native.classification_metrics_label.__all__, np.nan)
    out.update(accuracy=(tp + tn) / nv if nv else np.nan,
               point_precision=p, point_recall=r, point_f1=_f1(p, r),
               correct_points=tp + tn, total_points=nv, tp=tp, fp=fp, tn=tn, fn=fn,
               raw_n_points=len(y), n_points=nv, valid_points=nv, invalid_points=len(y) - nv,
               n_positive=pos, n_negative=neg, valid_positive_points=pos, valid_negative_points=neg,
               event_count=len(original_truth), evaluable_event_count=nt,
               raw_alarm_count=len(original_alarms), evaluable_alarm_count=npred,
               normal_points=neg, normal_alert_points=fp, normal_point_fpr=fp / neg if neg else np.nan,
               alarm_positive_points=tp + fp, alarm_duration_ratio=(tp + fp) / nv if nv else np.nan,
               evaluation_policy=POLICY,
               evaluation_reason="no_valid_points" if not nv else "no_valid_positive_points" if original_truth and not pos else "ok")

    overlaps = np.array([[overlap(g, a) for a in predicted] for g in truth], dtype=float).reshape(nt, npred)
    event_r = float(np.mean(np.any(overlaps > 0, axis=1))) if nt else np.nan
    event_p = float(np.mean(np.any(overlaps > 0, axis=0))) if npred else 0.0 if nt else np.nan
    if not nt and original_truth:
        event_p = np.nan
    out.update(event_precision=event_p, event_recall=event_r, event_f1=_f1(event_p, event_r),
               miss_rate=1 - event_r if nt else np.nan)
    if nt:
        if npred:
            gmass = np.array([mass(g) for g in truth], dtype=float)
            pmass = np.array([mass(a) for a in predicted], dtype=float)
            ea_p = float(np.mean(overlaps.max(axis=0) / pmass))
            ea_r = float(np.mean(overlaps.max(axis=1) / gmass))
            gcard = np.count_nonzero(overlaps, axis=1)
            pcard = np.count_nonzero(overlaps, axis=0)

            range_r = float(np.mean(.2 * (gcard > 0) + .8 * overlaps.sum(axis=1) / gmass / np.maximum(gcard, 1)))
            range_p = float(np.mean(overlaps.sum(axis=0) / pmass / np.maximum(pcard, 1)))
        else:
            ea_p = ea_r = range_p = range_r = 0.0
        out.update(event_affiliation_f1=_f1(ea_p, ea_r), range_precision=range_p,
                   range_recall=range_r, range_f1=_f1(range_p, range_r))

    ap, ar = native._affiliation_pr(y, alarms) if len(y) else (np.nan, np.nan)
    af = 2 * ap * ar / (ap + ar) if np.isfinite(ap + ar) and ap + ar else np.nan
    if empty_alarm_affiliation == "legacy_zero" and original_truth and not alarms.any():
        ap = ar = af = 0.0
    for name, value in zip(("affiliation_precision", "affiliation_recall", "affiliation_f1"), (ap, ar, af)):
        out[name] = value
        out[name + "_unmasked_reference"] = value
    out["affiliation_policy"] = "native_unmasked_reference_" + empty_alarm_affiliation

    release_points = np.flatnonzero(valid & (warning == 1))
    state_points = np.flatnonzero(valid & (alarms == 1))
    leads, actionable, delays = [], [], []
    observable_count = 0
    eligible_count = sum(s >= delta for s, _ in original_truth)
    opportunity_points = 0
    previous_end = -1
    for start, end in original_truth:
        lower = max(0, start - horizon, previous_end + 1)
        right = min(start - 1, start - delta)
        slots = mass((lower, right)) if right >= lower else 0
        if start >= delta and slots:
            observable_count += 1
        opportunity_points += slots if start >= delta else 0
        k = int(np.searchsorted(release_points, lower))
        if k < len(release_points) and release_points[k] < start:
            lead = int(start - release_points[k])
            leads.append(lead)
            if start >= delta and lead >= delta:
                actionable.append(lead)
        k = int(np.searchsorted(state_points, start))
        if k < len(state_points) and state_points[k] <= end:
            delays.append(int(state_points[k]) - start)
        previous_end = end
    early = len(actionable)
    out.update(mean_lead_time=_mean(leads), actionable_lead_time=_mean(actionable),
               mean_detection_delay=_mean(delays), lead_support_count=len(leads),
               actionable_lead_support_count=early, delay_support_count=len(delays),
               eligible_event_count=eligible_count, early_detection_count=early,
               early_detection_rate=early / eligible_count if eligible_count else np.nan,
               early_detection_rate_all=early / len(original_truth) if original_truth else np.nan,
               observable_eligible_event_count=observable_count,
               observable_early_detection_count=early,
               observable_early_detection_rate=early / observable_count if observable_count else np.nan,
               valid_warning_opportunity_points=opportunity_points,
               lead_delta_points=delta, lead_horizon_points=horizon)

    zones = native._zone_partition(original_truth, len(y))
    false = []
    for alarm in predicted:
        assigned = any(overlap(alarm, g) for g in original_truth)
        if not assigned:
            assigned = any(overlap(alarm, (max(zone[0], start - horizon), start - 1))
                           for (start, _), zone in zip(original_truth, zones))
        if not assigned:
            false.append(alarm)
    intervals = np.diff([a[0] for a in false])
    days = nv / float(points_per_day)
    out.update(false_alarm_events=len(false), turbine_days=days,
               false_alarms_per_turbine_day=len(false) / days if days else np.nan,
               mtbfa=_mean(intervals), false_interval_sum=int(intervals.sum()),
               false_interval_count=len(intervals),
               normal_monitoring_days=days if not original_truth else np.nan)
    return out


def curves(y, scores, valid):
    y, scores, valid = _inputs(y, scores, valid, score=True)
    nv = int(valid.sum())
    pos = int(y[valid].sum())
    out = dict.fromkeys(native.classification_metrics_score.__all__, np.nan)
    out.update(curve_valid_points=nv, curve_positive_points=pos, curve_negative_points=nv - pos,
               range_curve_policy=RANGE_POLICY,
               curve_status="no_valid_points" if not nv else "no_valid_positive_points" if not pos else "ok")
    if not nv or not pos:
        return out
    out["auc_pr"] = native.auc_pr(y[valid], scores[valid])
    out["auc_roc"] = native.auc_roc(y[valid], scores[valid])
    roc, pr, vusroc, vuspr = range_vus(y, scores, valid)
    out.update(range_auc_roc=float(roc), range_auc_pr=float(pr), vus_roc=float(vusroc), vus_pr=float(vuspr))
    if pos == nv:
        out["curve_status"] = "no_valid_negative_points"
    return out
