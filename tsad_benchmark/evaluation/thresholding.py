import numpy as np

from tsad_benchmark.common.operating_points import TRACK1_ANOMALY_RATIOS


TRAIN_PERCENTILE_METHOD = "train_percentile_v1"


def validate_scores(scores, expected_length: int, source: str) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim > 2 or (values.ndim == 2 and values.shape[1] != 1):
        raise ValueError(f"{source} scores must contain one score per row.")
    values = values.reshape(-1)
    if expected_length == 0 or values.size == 0:
        raise ValueError(f"{source} scores are empty.")
    if values.size != expected_length:
        raise ValueError(
            f"{source} score length {values.size} != data length {expected_length}."
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{source} scores must all be finite.")
    return values


def training_percentile_thresholds(train_scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(train_scores, dtype=float).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("training scores must be non-empty and finite.")
    return {
        str(float(ratio)): float(np.percentile(scores, 100.0 - ratio, method="linear"))
        for ratio in TRACK1_ANOMALY_RATIOS
    }
