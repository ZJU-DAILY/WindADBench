from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from tsad_benchmark.evaluation.metrics import (
    METRICS,
    classification_metrics_label,
    classification_metrics_score,
    clear_score_metric_cache,
)


__all__ = [
    "ALARM_METRICS",
    "DEFERRED_SCORE_METRICS",
    "DETECTION_METRICS",
    "FAST_SCORE_METRICS",
    "LABEL_SUPPORTS",
    "LABEL_METRICS",
    "SCORE_METRICS",
    "UNIT_INTERVAL_METRICS",
    "compute_label_metrics",
    "compute_score_metrics",
    "lead_time_utility",
]


LABEL_SUPPORTS: Tuple[str, ...] = (
    "correct_points",
    "total_points",
    "false_alarm_events",
    "turbine_days",
)
_COMMON_LABEL_METRICS = tuple(classification_metrics_label.__all__)
LABEL_METRICS: Tuple[str, ...] = tuple(
    name for name in _COMMON_LABEL_METRICS if name not in LABEL_SUPPORTS
) + ("miss_rate", "lead_time_utility", "ltu")
SCORE_METRICS: Tuple[str, ...] = tuple(classification_metrics_score.__all__)
DEFERRED_SCORE_METRICS: Tuple[str, ...] = ("vus_pr", "vus_roc")
FAST_SCORE_METRICS: Tuple[str, ...] = tuple(
    name for name in SCORE_METRICS if name not in DEFERRED_SCORE_METRICS
)
ALARM_METRICS: Tuple[str, ...] = (
    "false_alarm_events",
    "turbine_days",
    "alarm_duration_ratio",
    "normal_monitoring_days",
)
DETECTION_METRICS: Tuple[str, ...] = (*LABEL_METRICS, *SCORE_METRICS)
UNIT_INTERVAL_METRICS: Tuple[str, ...] = tuple(
    name
    for name in DETECTION_METRICS
    if name
    not in {
        "mean_lead_time",
        "mean_detection_delay",
        "false_alarms_per_turbine_day",
        "mtbfa",
    }
)
_NORMAL_EVENT_METRICS = {
    "accuracy",
    "false_alarms_per_turbine_day",
    "mtbfa",
    *LABEL_SUPPORTS,
}
def _binary_vector(values: Iterable[int], name: str) -> np.ndarray:
    try:
        source = values if isinstance(values, np.ndarray) else list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional iterable.") from exc
    array = np.asarray(source)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only binary values.") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only 0 and 1.")
    return numeric.astype(np.int8)


def _score_vector(values: Iterable[float] | None, size: int) -> np.ndarray | None:
    if values is None:
        return None
    try:
        source = values if isinstance(values, np.ndarray) else list(values)
        scores = np.asarray(source, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("scores must be a one-dimensional numeric iterable.") from exc
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional.")
    if scores.size != size:
        raise ValueError("scores must have the same length as labels.")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite.")
    return scores


def _events(binary: np.ndarray) -> List[Tuple[int, int]]:
    padded = np.pad(binary.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _non_negative_integer(value: object, name: str, *, positive: bool) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    try:
        exact = float(value) == float(converted)
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact or converted < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer.")
    return converted


def lead_time_utility(
    labels: Iterable[int],
    predictions: Iterable[int],
    *,
    lead_delta_points: int = 288,
) -> float:
    """Mean event utility for the first alarm in [start-delta, end]."""
    y_true = _binary_vector(labels, "labels")
    y_pred = _binary_vector(predictions, "predictions")
    if y_true.size != y_pred.size:
        raise ValueError("labels and predictions must have equal lengths.")
    delta = _non_negative_integer(
        lead_delta_points, "lead_delta_points", positive=False
    )

    utilities: List[float] = []
    for start, end in _events(y_true):
        alarms = np.flatnonzero(y_pred[max(0, start - delta) : end + 1])
        if alarms.size == 0:
            utilities.append(0.0)
            continue
        first = max(0, start - delta) + int(alarms[0])
        if first <= start or end == start:
            utilities.append(1.0)
        else:
            utilities.append(float((end - first) / (end - start)))
    return float(np.mean(utilities)) if utilities else float("nan")


def _event_kind(y_true: np.ndarray, event_label: str | None) -> str:
    inferred = "anomaly" if y_true.any() else "normal"
    kind = inferred if event_label is None else str(event_label).strip().lower()
    if kind not in {"normal", "anomaly"}:
        raise ValueError("event_label must be 'normal' or 'anomaly'.")
    if kind != inferred:
        raise ValueError(f"event_label={kind!r} conflicts with point labels.")
    return kind


def _supports(y_true: np.ndarray) -> Dict[str, float]:
    return {
        "n_points": float(y_true.size),
        "n_positive": float(y_true.sum()),
        "n_negative": float(y_true.size - y_true.sum()),
    }


def compute_score_metrics(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    event_label: str | None = None,
    defer_vus: bool = False,
) -> Dict[str, float]:
    """Compute only threshold-free score metrics for one event."""
    y_true = _binary_vector(labels, "labels")
    y_score = _score_vector(scores, y_true.size)
    if y_score is None:
        raise ValueError("scores are required for the score track.")
    kind = _event_kind(y_true, event_label)
    result = _supports(y_true)
    if kind == "normal":
        result.update({name: float("nan") for name in SCORE_METRICS})
        return result

    metric_names = FAST_SCORE_METRICS if defer_vus else SCORE_METRICS
    clear_score_metric_cache()
    try:
        result.update(
            {name: float(METRICS[name](y_true, y_score)) for name in metric_names}
        )
    finally:
        clear_score_metric_cache()
    for name in DEFERRED_SCORE_METRICS:
        result.setdefault(name, float("nan"))
    return result


def compute_label_metrics(
    labels: Iterable[int],
    predictions: Iterable[int],
    *,
    event_label: str | None = None,
    points_per_day: int = 144,
    lead_delta_points: int = 288,
) -> Dict[str, float]:
    """Compute the shared benchmark label metrics plus cross-domain LTU."""
    y_true = _binary_vector(labels, "labels")
    y_pred = _binary_vector(predictions, "predictions")
    if y_true.size != y_pred.size:
        raise ValueError("labels and predictions must have equal lengths.")
    points_per_day = _non_negative_integer(
        points_per_day, "points_per_day", positive=True
    )
    lead_delta_points = _non_negative_integer(
        lead_delta_points, "lead_delta_points", positive=False
    )
    kind = _event_kind(y_true, event_label)

    shared_kwargs = {
        "points_per_day": points_per_day,
        "lead_delta_points": lead_delta_points,
    }
    clear_score_metric_cache()
    try:
        shared = {
            name: float(METRICS[name](y_true, y_pred, **shared_kwargs))
            for name in _COMMON_LABEL_METRICS
        }
    finally:
        clear_score_metric_cache()
    if kind == "normal":
        for name in _COMMON_LABEL_METRICS:
            if name not in _NORMAL_EVENT_METRICS:
                shared[name] = float("nan")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    result: Dict[str, float] = {
        **_supports(y_true),
        **shared,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "alarm_positive_points": float(y_pred.sum()),
        "alarm_duration_ratio": float(y_pred.mean()),
        "lead_delta_points": float(lead_delta_points),
        "normal_monitoring_days": (
            shared["turbine_days"] if kind == "normal" else float("nan")
        ),
    }

    if kind == "normal":
        result.update(
            miss_rate=float("nan"),
            lead_time_utility=float("nan"),
            ltu=float("nan"),
        )
        return result

    if not y_pred.any():
        for name in (
            "affiliation_precision",
            "affiliation_recall",
            "affiliation_f1",
        ):
            result[name] = 0.0

    recall = result["event_recall"]
    ltu = lead_time_utility(
        y_true, y_pred, lead_delta_points=lead_delta_points
    )
    result.update(
        {
            "miss_rate": 1.0 - recall,
            "lead_time_utility": ltu,
            "ltu": ltu,
        }
    )
    return result


def _self_test() -> None:
    normal = compute_label_metrics(
        [0] * 8,
        [0, 1, 1, 0, 0, 1, 0, 0],
        points_per_day=4,
    )
    assert normal["false_alarm_events"] == 2.0
    assert normal["alarm_duration_ratio"] == 3 / 8
    assert normal["false_alarms_per_turbine_day"] == 1.0
    assert np.isnan(normal["event_f1"])

    labels = [0, 0, 1, 1, 1, 0]
    early = compute_label_metrics(
        labels, [0, 1, 0, 1, 0, 0], lead_delta_points=2
    )
    delayed = compute_label_metrics(
        labels, [0, 0, 0, 1, 0, 0], lead_delta_points=2
    )
    missed = compute_label_metrics(
        labels, [0] * 6, lead_delta_points=2
    )
    assert early["ltu"] == 1.0 and early["miss_rate"] == 0.0
    assert early["mean_lead_time"] == 1.0
    assert early["early_detection_rate"] == 0.0
    assert delayed["ltu"] == 0.5
    assert delayed["mean_detection_delay"] == 1.0
    assert missed["ltu"] == 0.0 and missed["miss_rate"] == 1.0
    score = compute_score_metrics(labels, np.arange(6))
    assert all(name in score for name in SCORE_METRICS)


if __name__ == "__main__":
    _self_test()
