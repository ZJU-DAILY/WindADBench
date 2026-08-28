# -*- coding: utf-8 -*-


from typing import Any, Callable, Dict, List, Tuple

import numpy as np

try:
    from sklearn.metrics import average_precision_score as _sk_avg_precision
    from sklearn.metrics import roc_auc_score as _sk_roc_auc
    _HAS_SKLEARN = True
except Exception:
    _sk_avg_precision = None
    _sk_roc_auc = None
    _HAS_SKLEARN = False

METRICS: Dict[str, Callable] = {}
EPS = 1e-12
_trapz = getattr(np, "trapezoid", None) or np.trapz

from tsad_benchmark.evaluation.metrics.metric_utils import get_list_anomaly
from tsad_benchmark.evaluation.metrics.vus_metrics import generate_curve, metricor

_metricor = metricor()


def register_metric(name: str, fn: Callable) -> None:
    if not callable(fn):
        raise TypeError("Metric function must be callable")
    METRICS[name] = fn


# ---------- Common utilities ------------------------------------------------

def _to_1d(x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 0:
        return a.reshape(1)
    return a.reshape(-1) if a.ndim > 1 else a


def _align_labels(actual, predicted) -> Tuple[np.ndarray, np.ndarray]:
    """Equal-length 0/1 vectors from the label track (``detect_label``)."""
    y_true = _to_1d(actual).astype(float)
    y_pred = _to_1d(predicted).astype(float)
    if y_true.size != y_pred.size:
        raise ValueError("label metrics require equal-length label arrays")
    return y_true, y_pred


def _events_from_binary(y: np.ndarray) -> List[Tuple[int, int]]:
    y = _to_1d(y).astype(float)
    events: List[Tuple[int, int]] = []
    if y.size == 0:
        return events
    start = None
    for i, v in enumerate(y):
        if v > 0 and start is None:
            start = i
        elif v <= 0 and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, y.size - 1))
    return events


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _align_scores(actual, predicted) -> Tuple[np.ndarray, np.ndarray]:
    """Equal-length ground-truth labels and anomaly scores (score track)."""
    y_true = _to_1d(actual).astype(int)
    y_score = _to_1d(predicted).astype(float)
    if y_true.size != y_score.size:
        raise ValueError("score metrics require equal-length label and score arrays")
    return y_true, y_score


def _anomaly_median_window(y_true: np.ndarray) -> int:
    lengths = get_list_anomaly(y_true)
    if lengths.size == 0:
        return 100
    return max(1, int(np.median(lengths)))


# ---------- Score-track heavy-metric cache (per evaluate() call) --------------
# Range-AUC and VUS share expensive Paparrizos routines.  The Evaluator calls
# each metric function separately with the same (actual, predicted) arrays, so
# we cache by input object id within one evaluation batch and clear the cache
# when a new batch starts (see evaluator.clear_score_metric_cache).

_SCORE_HEAVY_CACHE: Dict[Tuple[int, int], Dict[str, Any]] = {}


def clear_score_metric_cache() -> None:
    """Drop cached Range-AUC / VUS results (call once per series evaluation)."""
    _SCORE_HEAVY_CACHE.clear()


def _score_heavy_bucket(actual, predicted) -> Dict[str, Any]:
    return _SCORE_HEAVY_CACHE.setdefault((id(actual), id(predicted)), {})


def _range_auc_pair(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Return (range_auc_roc, range_auc_pr) with one RangeAUC invocation."""
    positives = int(np.sum(y_true))
    if positives == 0:
        return np.nan, 0.0
    if positives == y_true.size:
        return np.nan, 1.0
    window = _anomaly_median_window(y_true)
    r_auc_roc, r_auc_pr, _, _, _ = _metricor.RangeAUC(
        labels=y_true, score=y_score, window=window, plot_ROC=True,
    )
    return float(r_auc_roc), float(r_auc_pr)


def _vus_pair(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Return (vus_roc, vus_pr) with one generate_curve invocation."""
    positives = int(np.sum(y_true))
    if positives == 0:
        return np.nan, 0.0
    if positives == y_true.size:
        return np.nan, 1.0
    window = _anomaly_median_window(y_true)
    *_, vus_roc, vus_pr = generate_curve(y_true, y_score, 2 * window)
    return float(vus_roc), float(vus_pr)


def _cached_range_auc(actual, predicted, *, pr: bool) -> float:
    bucket = _score_heavy_bucket(actual, predicted)
    if "range_auc" not in bucket:
        y_true, y_score = _align_scores(actual, predicted)
        bucket["range_auc"] = _range_auc_pair(y_true, y_score)
    r_auc_roc, r_auc_pr = bucket["range_auc"]
    return float(r_auc_pr if pr else r_auc_roc)


def _cached_vus(actual, predicted, *, pr: bool) -> float:
    bucket = _score_heavy_bucket(actual, predicted)
    if "vus" not in bucket:
        y_true, y_score = _align_scores(actual, predicted)
        bucket["vus"] = _vus_pair(y_true, y_score)
    vus_roc, vus_pr = bucket["vus"]
    return float(vus_pr if pr else vus_roc)


# ---------- Point-level (sklearn-equivalent) --------------------------------

def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, fn


def accuracy(actual, predicted, **kwargs) -> float:
    y_true, y_pred = _align_labels(actual, predicted)
    if y_true.size == 0:
        return np.nan
    return float(np.mean(y_true == y_pred))


def point_precision(actual, predicted, **kwargs) -> float:
    y_true, y_pred = _align_labels(actual, predicted)
    tp, fp, _ = _confusion(y_true, y_pred)
    denom = tp + fp
    return 0.0 if denom == 0 else float(tp / denom)


def point_recall(actual, predicted, **kwargs) -> float:
    y_true, y_pred = _align_labels(actual, predicted)
    tp, _, fn = _confusion(y_true, y_pred)
    denom = tp + fn
    return 0.0 if denom == 0 else float(tp / denom)


def point_f1(actual, predicted, **kwargs) -> float:
    p = point_precision(actual, predicted, **kwargs)
    r = point_recall(actual, predicted, **kwargs)
    denom = p + r
    return 0.0 if denom == 0 else float(2 * p * r / denom)


# ---------- Curve metrics (score track) -------------------------------------

def _weighted_pr_curve(y_cont: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score, kind="mergesort")
    yt, ys = y_cont[order], y_score[order]
    distinct = np.where(np.diff(ys))[0]
    idx = np.r_[distinct, ys.size - 1]
    tps = np.cumsum(yt)[idx]
    fps = np.cumsum(1.0 - yt)[idx]
    total_pos = float(np.sum(yt))
    prec = tps / (tps + fps + EPS)
    rec = np.zeros_like(prec) if total_pos <= 0 else tps / (total_pos + EPS)
    return np.r_[1.0, prec], np.r_[0.0, rec]


def _step_average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    if recall.size < 2:
        return 0.0
    return float(np.sum(np.diff(recall) * precision[1:]))


def _weighted_roc_curve(y_cont: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score, kind="mergesort")
    yt, ys = y_cont[order], y_score[order]
    distinct = np.where(np.diff(ys))[0]
    idx = np.r_[distinct, ys.size - 1]
    tps = np.cumsum(yt)[idx]
    fps = np.cumsum(1.0 - yt)[idx]
    total_pos = float(np.sum(yt))
    total_neg = float(np.sum(1.0 - yt))
    tpr = np.zeros_like(tps) if total_pos <= 0 else tps / (total_pos + EPS)
    fpr = np.zeros_like(fps) if total_neg <= 0 else fps / (total_neg + EPS)
    return np.r_[0.0, fpr, 1.0], np.r_[0.0, tpr, 1.0]


def auc_pr(actual, predicted, **kwargs) -> float:

    y_true, y_score = _align_scores(actual, predicted)
    if np.sum(y_true) == 0:
        return 0.0
    if _HAS_SKLEARN and _sk_avg_precision is not None:
        return float(_sk_avg_precision(y_true, y_score))
    p, r = _weighted_pr_curve(y_true.astype(float), y_score)
    return _step_average_precision(p, r)


def auc_roc(actual, predicted, **kwargs) -> float:

    y_true, y_score = _align_scores(actual, predicted)
    positives = np.sum(y_true)
    if positives == 0 or positives == y_true.size:
        return np.nan
    if _HAS_SKLEARN and _sk_roc_auc is not None:
        return float(_sk_roc_auc(y_true, y_score))
    fpr, tpr = _weighted_roc_curve(y_true.astype(float), y_score)
    return float(_trapz(tpr, fpr))


def range_auc_pr(actual, predicted, **kwargs) -> float:

    return _cached_range_auc(actual, predicted, pr=True)


def range_auc_roc(actual, predicted, **kwargs) -> float:

    return _cached_range_auc(actual, predicted, pr=False)


def vus_pr(actual, predicted, **kwargs) -> float:

    return _cached_vus(actual, predicted, pr=True)


def vus_roc(actual, predicted, **kwargs) -> float:

    return _cached_vus(actual, predicted, pr=False)


# ---------- Event-level metrics (benchmark 指标体系) --------------------------

def _matched_event_count(
    target_events: List[Tuple[int, int]],
    other_events: List[Tuple[int, int]],
) -> int:
    return sum(
        1 for target in target_events
        if any(_overlap(target, other) > 0 for other in other_events)
    )


def event_recall(actual, predicted, **kwargs) -> float:
  
    y_true, y_pred = _align_labels(actual, predicted)
    R = _events_from_binary(y_true)
    P = _events_from_binary(y_pred)
    if not R:
        return np.nan
    if not P:
        return 0.0
    return float(_matched_event_count(R, P) / len(R))


def event_precision(actual, predicted, **kwargs) -> float:

    y_true, y_pred = _align_labels(actual, predicted)
    R = _events_from_binary(y_true)
    P = _events_from_binary(y_pred)
    if not P:
        return np.nan if not R else 0.0
    if not R:
        return 0.0
    return float(_matched_event_count(P, R) / len(P))


def event_f1(actual, predicted, **kwargs) -> float:

    p = event_precision(actual, predicted, **kwargs)
    r = event_recall(actual, predicted, **kwargs)
    if np.isnan(p) or np.isnan(r):
        return np.nan
    denom = p + r
    return 0.0 if denom == 0 else float(2 * p * r / denom)


# ---------- Event overlap Affiliation (指标体系 · 幻灯片 Affiliation-F) ---------

def _event_length(event: Tuple[int, int]) -> int:
    return event[1] - event[0] + 1


def _max_overlap_fraction(
    target: Tuple[int, int],
    candidates: List[Tuple[int, int]],
) -> float:
    if not candidates:
        return 0.0
    denom = _event_length(target)
    if denom <= 0:
        return 0.0
    return float(max(_overlap(target, c) for c in candidates) / denom)


def _event_affiliation_pr(actual, predicted) -> Tuple[float, float]:

    y_true, y_pred = _align_labels(actual, predicted)
    true_events = _events_from_binary(y_true)
    pred_events = _events_from_binary(y_pred)
    if not true_events:
        return np.nan, np.nan
    if not pred_events:
        return 0.0, 0.0
    p = float(np.mean([_max_overlap_fraction(pe, true_events) for pe in pred_events]))
    r = float(np.mean([_max_overlap_fraction(ge, pred_events) for ge in true_events]))
    return p, r


def event_affiliation_f1(actual, predicted, **kwargs) -> float:

    p, r = _event_affiliation_pr(actual, predicted)
    if np.isnan(p) or np.isnan(r):
        return np.nan
    denom = p + r
    return 0.0 if denom == 0 else float(2 * p * r / denom)


# ---------- Range precision / recall / F1 (Paparrizos metricor.metric_new) ------

def _range_prf(actual, predicted) -> Tuple[float, float, float]:

    y_true, y_pred = _align_labels(actual, predicted)
    if np.sum(y_true) == 0:
        return np.nan, np.nan, np.nan
    out = _metricor.metric_new(y_true, y_pred, plot_ROC=False)
    if out is None:
        return np.nan, np.nan, np.nan
    # metric_new: [auc, P, R, F, Rrecall, ..., Rprecision, Rf, precision_at_k]
    return float(out[7]), float(out[4]), float(out[8])


def range_precision(actual, predicted, **kwargs) -> float:
    return _range_prf(actual, predicted)[0]


def range_recall(actual, predicted, **kwargs) -> float:
    return _range_prf(actual, predicted)[1]


def range_f1(actual, predicted, **kwargs) -> float:
    return _range_prf(actual, predicted)[2]


# ---------- Affiliation (Huet et al. KDD 2022, pr_from_events) ----------------

from tsad_benchmark.evaluation.metrics.affiliation.generics import convert_vector_to_events
from tsad_benchmark.evaluation.metrics.affiliation.metrics import pr_from_events


def _affiliation_pr(actual, predicted) -> Tuple[float, float]:

    y_true, y_pred = _align_labels(actual, predicted)
    n = int(y_pred.size)
    if n == 0:
        return np.nan, np.nan
    try:
        out = pr_from_events(
            convert_vector_to_events(y_pred),
            convert_vector_to_events(y_true),
            (0, n),
        )
    except ValueError:
        return np.nan, np.nan
    return float(out["precision"]), float(out["recall"])


def affiliation_precision(actual, predicted, **kwargs) -> float:
    return _affiliation_pr(actual, predicted)[0]


def affiliation_recall(actual, predicted, **kwargs) -> float:
    return _affiliation_pr(actual, predicted)[1]


def affiliation_f1(actual, predicted, **kwargs) -> float:
    p, r = _affiliation_pr(actual, predicted)
    if np.isnan(p) or np.isnan(r):
        return np.nan
    denom = p + r
    return float(2 * p * r / denom) if denom else np.nan


# ---------- Per-event zones (operational KPIs) --------------------------------

def _zone_partition(events: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:

    K = len(events)
    if K == 0:
        return []
    zones = []
    for i, (s, e) in enumerate(events):
        zl = 0 if i == 0 else (events[i - 1][1] + s) // 2 + 1
        zr = n - 1 if i == K - 1 else (e + events[i + 1][0]) // 2
        zones.append((zl, zr))
    return zones


def _first_alarm_before_event_in_zone(s: int, zone: Tuple[int, int], pts: np.ndarray) -> int:
    zl, _ = zone
    cand = pts[(pts >= zl) & (pts < s)]
    return int(cand.min()) if cand.size else -1


def _is_assigned_alarm(
    pred_event: Tuple[int, int],
    true_events: List[Tuple[int, int]],
    zones: List[Tuple[int, int]],
) -> bool:

    if any(_overlap(pred_event, te) > 0 for te in true_events):
        return True
    for (s, _), zone in zip(true_events, zones):
        pre_zone = (zone[0], s - 1)
        if pre_zone[0] <= pre_zone[1] and _overlap(pred_event, pre_zone) > 0:
            return True
    return False


def _false_alarm_events(y_true: np.ndarray, y_pred: np.ndarray) -> List[Tuple[int, int]]:

    true_events = _events_from_binary(y_true)
    pred_events = _events_from_binary(y_pred)
    if not pred_events:
        return []
    if not true_events:
        return pred_events
    zones = _zone_partition(true_events, y_true.size)
    return [
        pe for pe in pred_events
        if not _is_assigned_alarm(pe, true_events, zones)
    ]


def correct_points(actual, predicted, **kwargs) -> float:

    y_true, y_pred = _align_labels(actual, predicted)
    return float(np.sum(y_true == y_pred))


def total_points(actual, predicted, **kwargs) -> float:
 
    y_true, _ = _align_labels(actual, predicted)
    return float(y_true.size)


def false_alarm_events(actual, predicted, **kwargs) -> float:
 
    y_true, y_pred = _align_labels(actual, predicted)
    return float(len(_false_alarm_events(y_true, y_pred)))


def turbine_days(actual, predicted, **kwargs) -> float:

    points_per_day = int(kwargs.get("points_per_day", 144))
    if points_per_day <= 0:
        return np.nan
    y_true, _ = _align_labels(actual, predicted)
    return float(y_true.size / float(points_per_day))


# ---------- Operational KPIs (wind-farm) ------------------------------------

def mean_lead_time(actual, predicted, **kwargs) -> float:

    y_true, pred = _align_labels(actual, predicted)
    events = _events_from_binary(y_true)
    if not events:
        return np.nan
    pts = np.where(pred == 1)[0]
    zones = _zone_partition(events, y_true.size)
    leads = []
    for (s, _), zone in zip(events, zones):
        first = _first_alarm_before_event_in_zone(s, zone, pts)
        if first >= 0:
            leads.append(s - first)
    return float(np.mean(leads)) if leads else np.nan


def mean_detection_delay(actual, predicted, **kwargs) -> float:

    y_true, pred = _align_labels(actual, predicted)
    events = _events_from_binary(y_true)
    pts = np.where(pred == 1)[0]
    if not events:
        return np.nan
    delays = []
    for s, e in events:
        within = pts[(pts >= s) & (pts <= e)]
        if within.size:
            delays.append(int(within[0]) - s)
    return float(np.mean(delays)) if delays else np.nan


def early_detection_rate(actual, predicted, **kwargs) -> float:

    delta = int(kwargs.get("lead_delta_points", 288))
    y_true, pred = _align_labels(actual, predicted)
    events = _events_from_binary(y_true)
    if not events:
        return np.nan
    pts = np.where(pred == 1)[0]
    zones = _zone_partition(events, y_true.size)
    hits = 0
    for (s, _), zone in zip(events, zones):
        first = _first_alarm_before_event_in_zone(s, zone, pts)
        if first >= 0 and (s - first) >= delta:
            hits += 1
    return float(hits / len(events))


def false_alarms_per_turbine_day(actual, predicted, **kwargs) -> float:

    points_per_day = int(kwargs.get("points_per_day", 144))
    if points_per_day <= 0:
        return np.nan
    y_true, y_pred = _align_labels(actual, predicted)
    fa = len(_false_alarm_events(y_true, y_pred))
    days = len(y_true) / float(points_per_day)
    return float(fa / days) if days > 0 else np.nan


def mtbfa(actual, predicted, **kwargs) -> float:

    y_true, y_pred = _align_labels(actual, predicted)
    fa = sorted(pe[0] for pe in _false_alarm_events(y_true, y_pred))
    if len(fa) < 2:
        return np.nan
    return float(np.mean(np.diff(np.array(fa, dtype=float))))


# ---------- Registry + track lists ------------------------------------------

for _name, _fn in [
    ("accuracy", accuracy),
    ("point_precision", point_precision),
    ("point_recall", point_recall),
    ("point_f1", point_f1),
    ("auc_pr", auc_pr),
    ("auc_roc", auc_roc),
    ("range_auc_pr", range_auc_pr),
    ("range_auc_roc", range_auc_roc),
    ("vus_pr", vus_pr),
    ("vus_roc", vus_roc),
    ("event_precision", event_precision),
    ("event_recall", event_recall),
    ("event_f1", event_f1),
    ("event_affiliation_f1", event_affiliation_f1),
    ("range_precision", range_precision),
    ("range_recall", range_recall),
    ("range_f1", range_f1),
    ("affiliation_precision", affiliation_precision),
    ("affiliation_recall", affiliation_recall),
    ("affiliation_f1", affiliation_f1),
    ("mean_lead_time", mean_lead_time),
    ("mean_detection_delay", mean_detection_delay),
    ("early_detection_rate", early_detection_rate),
    ("false_alarms_per_turbine_day", false_alarms_per_turbine_day),
    ("mtbfa", mtbfa),
    ("correct_points", correct_points),
    ("total_points", total_points),
    ("false_alarm_events", false_alarm_events),
    ("turbine_days", turbine_days),
]:
    register_metric(_name, _fn)


class classification_metrics_score:
    __all__ = ["auc_pr", "auc_roc", "range_auc_pr", "range_auc_roc", "vus_pr", "vus_roc"]


class classification_metrics_label:
    __all__ = [
        "accuracy",
        "point_precision", "point_recall", "point_f1",
        "event_precision", "event_recall", "event_f1",
        "event_affiliation_f1",
        "range_precision", "range_recall", "range_f1",
        "affiliation_precision", "affiliation_recall", "affiliation_f1",
        "mean_lead_time", "mean_detection_delay", "early_detection_rate",
        "false_alarms_per_turbine_day", "mtbfa",
        "correct_points", "total_points", "false_alarm_events", "turbine_days",
    ]
