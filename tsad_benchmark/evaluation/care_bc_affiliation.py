from __future__ import annotations

import numpy as np


POLICY = "single_event_fixed_zone_conditional_probability_continuous_v1"


def _binary(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must be a one-dimensional binary vector")
    return array.astype(bool, copy=False)


def _intervals(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return np.column_stack((np.flatnonzero(changes == 1),
                            np.flatnonzero(changes == -1))).astype(float)


class _ValidMeasure:
    def __init__(self, intervals):
        self.starts = intervals[:, 0]
        self.ends = intervals[:, 1]
        lengths = self.ends - self.starts
        self.mass = np.r_[0.0, np.cumsum(lengths)]
        self.moments = np.r_[0.0, np.cumsum(lengths * (self.starts + self.ends) / 2)]

    def _parts(self, z):
        z = np.asarray(z, dtype=float)
        complete = np.searchsorted(self.ends, z, side="right")

        starts = np.r_[self.starts, np.inf]
        partial = np.maximum(z - starts[complete], 0.0)
        return z, complete, partial

    def cdf_mass(self, z):
        _, complete, partial = self._parts(z)
        return self.mass[complete] + partial

    def primitive(self, z):
        z, complete, partial = self._parts(z)
        return z * self.mass[complete] - self.moments[complete] + partial ** 2 / 2


def _precision_integral(predicted, ground_truth, measure, total_mass):
    g_left, g_right = ground_truth[0, 0], ground_truth[-1, 1]
    center_sum = g_left + g_right
    starts, ends = predicted[:, 0], predicted[:, 1]
    integral = np.maximum(np.minimum(ends, g_right) - np.maximum(starts, g_left), 0).sum()
    h = measure.primitive

    left_end = np.minimum(ends, g_left)
    exists = starts < left_end
    a, b = starts[exists], left_end[exists]
    if len(a):
        excluded_mass_integral = h(center_sum - a) - h(center_sum - b) - h(b) + h(a)
        integral += np.sum((b - a) - excluded_mass_integral / total_mass)

    right_start = np.maximum(starts, g_right)
    exists = right_start < ends
    a, b = right_start[exists], ends[exists]
    if len(a):
        excluded_mass_integral = h(b) - h(a) - h(center_sum - a) + h(center_sum - b)
        integral += np.sum((b - a) - excluded_mass_integral / total_mass)
    return float(integral)


def _recall_pieces(predicted, ground_truth, n):
    pieces = []
    truth_index = 0
    count = len(ground_truth)

    def intersect(left, right, pivot, side):
        nonlocal truth_index
        while truth_index < count and ground_truth[truth_index, 1] <= left:
            truth_index += 1
        while truth_index < count and ground_truth[truth_index, 0] < right:
            lo = max(left, ground_truth[truth_index, 0])
            hi = min(right, ground_truth[truth_index, 1])
            if hi > lo:
                pieces.append((lo, hi, pivot, side))
            if ground_truth[truth_index, 1] <= right:
                truth_index += 1
            else:
                break

    for index, (start, end) in enumerate(predicted):
        left = (predicted[index - 1, 1] + start) / 2 if index else 0.0
        right = (end + predicted[index + 1, 0]) / 2 if index + 1 < len(predicted) else float(n)
        if start > left:
            intersect(left, start, start, -1)
        intersect(start, end, 0.0, 0)
        if right > end:
            intersect(end, right, end, 1)
    return np.asarray(pieces, dtype=float).reshape(-1, 4)


def _recall_integral(predicted, ground_truth, measure, total_mass, truth_mass, n):
    pieces = _recall_pieces(predicted, ground_truth, n)
    outside = pieces[:, 3] != 0
    if not outside.any():
        return float(truth_mass)
    a, b, pivot, side = pieces[outside].T

    inside_ball_integrals = side * (
        (measure.primitive(2 * b - pivot) - measure.primitive(2 * a - pivot)) / 2
        - measure.cdf_mass(pivot) * (b - a))
    return float(truth_mass - np.sum(inside_ball_integrals) / total_mass)


def _unit_probability(value):
    if not np.isfinite(value) or value < -1e-10 or value > 1 + 1e-10:
        raise ArithmeticError(f"Conditional Affiliation probability outside [0, 1]: {value}")
    return float(np.clip(value, 0.0, 1.0))


def label_metrics(y, alarms, valid):
    y = _binary(y, "labels")
    alarms = _binary(alarms, "alarms")
    valid = _binary(valid, "valid")
    if len(y) != len(alarms) or len(y) != len(valid):
        raise ValueError("labels, alarms and valid must have equal original-axis length")
    original_truth = _intervals(y)
    if len(original_truth) > 1:
        raise ValueError("Conditional CARE Affiliation supports at most one original truth event")

    truth = y & valid
    predictions = alarms & valid
    zone_mass = int(valid.sum())
    truth_mass = int(truth.sum())
    prediction_mass = int(predictions.sum())
    out = {
        "affiliation_precision": np.nan,
        "affiliation_recall": np.nan,
        "affiliation_f1": np.nan,
        "affiliation_policy": POLICY,
        "affiliation_n_truth_events": len(original_truth),
        "affiliation_valid_zone_points": zone_mass,
        "affiliation_valid_truth_points": truth_mass,
        "affiliation_valid_negative_points": zone_mass - truth_mass,
        "affiliation_valid_alarm_points": prediction_mass,
        "affiliation_evaluable_event_count": int(truth_mass > 0),
        "affiliation_scored_zone_points": zone_mass if truth_mass else 0,
        "affiliation_unscored_valid_alarm_points": 0 if truth_mass else prediction_mass,
        "affiliation_precision_defined": int(truth_mass > 0 and prediction_mass > 0),
        "affiliation_recall_defined": int(truth_mass > 0),
        "affiliation_empty_reason": "ok",
    }
    if not len(original_truth):
        out["affiliation_empty_reason"] = "no_truth_event"
        return out
    if not truth_mass:
        out["affiliation_empty_reason"] = "no_valid_truth"
        return out
    if not prediction_mass:
        out.update(affiliation_recall=0.0, affiliation_empty_reason="no_valid_predictions")
        return out

    measure = _ValidMeasure(_intervals(valid))
    ground_truth = _intervals(truth)
    predicted = _intervals(predictions)
    precision = _unit_probability(_precision_integral(predicted, ground_truth, measure, zone_mass) / prediction_mass)
    recall = _unit_probability(_recall_integral(predicted, ground_truth, measure, zone_mass, truth_mass, len(y)) / truth_mass)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
    out.update(affiliation_precision=precision, affiliation_recall=recall, affiliation_f1=f1)
    return out
