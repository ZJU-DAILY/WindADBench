from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .artifacts import atomic_csv
from .metrics import (
    ALARM_METRICS,
    DETECTION_METRICS,
    LABEL_METRICS,
    SCORE_METRICS,
    UNIT_INTERVAL_METRICS,
)


__all__ = ["BASE_DIRECTION_COLUMNS", "aggregate_results", "write_result_tables"]


BASE_DIRECTION_COLUMNS: Tuple[str, ...] = (
    "model_name",
    "track",
    "track_variant",
    "protocol",
    "source_farm",
    "target_farm",
    "transfer_type",
)
_OPTIONAL_IDENTITY_COLUMNS = ("model_params", "seed", "run_id")
_NORMAL_ONLY_LABEL_METRICS = {
    "accuracy",
    "false_alarms_per_turbine_day",
    "mtbfa",
}


def _require(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing aggregation columns: {missing}")


def _identity_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _direction_columns(frame: pd.DataFrame) -> List[str]:
    _require(frame, BASE_DIRECTION_COLUMNS)
    columns = ["model_name"]
    columns.extend(c for c in _OPTIONAL_IDENTITY_COLUMNS if c in frame.columns)
    columns.extend(BASE_DIRECTION_COLUMNS[1:])
    return columns


def _as_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            continue
        result[column] = pd.to_numeric(result[column], errors="raise")
        finite = result[column].dropna().to_numpy(dtype=float)
        if np.isinf(finite).any():
            raise ValueError(f"{column} contains infinite values.")
    return result


def _mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="raise").dropna()
    return float(numeric.mean()) if not numeric.empty else float("nan")


def _sum(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="raise").dropna()
    return float(numeric.sum()) if not numeric.empty else 0.0


def _sum_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="raise").dropna()
    return float(numeric.sum()) if not numeric.empty else float("nan")


def _max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="raise").dropna()
    return float(numeric.max()) if not numeric.empty else float("nan")


def _group_values(keys: object, columns: Sequence[str]) -> Dict[str, object]:
    values = keys if isinstance(keys, tuple) else (keys,)
    return dict(zip(columns, values))


def _constant(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns:
        return float("nan")
    values = pd.to_numeric(group[column], errors="raise").dropna().to_numpy(float)
    if values.size == 0:
        return float("nan")
    if not np.allclose(values, values[0], rtol=1e-9, atol=1e-12):
        raise ValueError(f"{column} is not constant inside one source fit.")
    return float(values[0])


def _validate_events(
    frame: pd.DataFrame,
    direction_columns: Sequence[str],
    asset_column: str,
    event_column: str,
    label_column: str,
) -> pd.DataFrame:
    required = [
        *direction_columns,
        asset_column,
        event_column,
        label_column,
        *DETECTION_METRICS,
        "false_alarm_events",
        "turbine_days",
        "normal_monitoring_days",
        "alarm_duration_ratio",
        "alarm_positive_points",
        "n_points",
        "correct_points",
        "fit_seconds_source",
        "inference_seconds_event",
        "inference_points_event",
        "inference_peak_memory_mb",
        "inference_peak_gpu_memory_mb",
        "inference_cpu_usage_percent",
        "calibration_seconds_event",
        "calibration_checkpoint_load_seconds_event",
        "calibration_peak_memory_mb",
        "calibration_peak_gpu_memory_mb",
        "calibration_cpu_usage_percent",
        "evaluation_wall_seconds_event",
    ]
    _require(frame, required)
    if frame.empty:
        raise ValueError("Event results must be non-empty.")

    result = frame.copy()
    for column in direction_columns:
        result[column] = result[column].map(_identity_value)
        if result[column].isna().any():
            raise ValueError(f"{column} contains missing identity values.")
    for column in (asset_column, event_column):
        if result[column].isna().any():
            raise ValueError(f"{column} contains missing identity values.")
        result[column] = result[column].astype(str)

    result[label_column] = result[label_column].astype(str).str.strip().str.lower()
    invalid_labels = sorted(set(result[label_column]) - {"normal", "anomaly"})
    if invalid_labels:
        raise ValueError(f"Unexpected event labels: {invalid_labels}")
    invalid_tracks = sorted(set(result["track"]) - {"score", "label"})
    if invalid_tracks:
        raise ValueError(f"Unexpected tracks: {invalid_tracks}")

    unique_columns = [*direction_columns, asset_column, event_column]
    duplicates = result.duplicated(unique_columns, keep=False)
    if duplicates.any():
        sample = result.loc[duplicates, unique_columns].head(3).to_dict("records")
        raise ValueError(f"Duplicate event results: {sample}")

    numeric_columns = list(DETECTION_METRICS) + [
        "false_alarm_events",
        "turbine_days",
        "normal_monitoring_days",
        "alarm_duration_ratio",
        "alarm_positive_points",
        "n_points",
        "correct_points",
        "inference_seconds_event",
        "inference_seconds",
        "inference_points_event",
        "calibration_seconds_event",
        "calibration_checkpoint_load_seconds_event",
        "calibration_peak_memory_mb",
        "calibration_peak_gpu_memory_mb",
        "calibration_cpu_usage_percent",
        "inference_peak_memory_mb",
        "inference_peak_gpu_memory_mb",
        "inference_cpu_usage_percent",
        "evaluation_wall_seconds_event",
        "fit_seconds_source",
    ]
    result = _as_numeric(result, numeric_columns)

    normal = result[label_column].eq("normal")
    anomaly = ~normal
    score_track = result["track"].eq("score")
    label_track = result["track"].eq("label")
    if result.loc[score_track, list(LABEL_METRICS)].notna().any().any():
        raise ValueError("Score-track rows contain label metrics.")
    if result.loc[label_track, list(SCORE_METRICS)].notna().any().any():
        raise ValueError("Label-track rows contain score metrics.")
    score_only_nan = [
        *ALARM_METRICS,
        "tp",
        "fp",
        "tn",
        "fn",
        "correct_points",
        "alarm_positive_points",
        "lead_delta_points",
    ]
    present = [column for column in score_only_nan if column in result.columns]
    if result.loc[score_track, present].notna().any().any():
        raise ValueError("Score-track rows contain thresholded-label supports.")
    for column in UNIT_INTERVAL_METRICS:
        values = result[column].dropna()
        if ((values < -1e-12) | (values > 1.0 + 1e-12)).any():
            raise ValueError(f"{column} must be within [0, 1].")
    normal_allowed = {"accuracy", "false_alarms_per_turbine_day", "mtbfa"}
    for column in LABEL_METRICS:
        if column in normal_allowed:
            continue
        if result.loc[normal & label_track, column].notna().any():
            raise ValueError(f"Normal events must have NaN {column}.")
    if result.loc[normal & score_track, list(SCORE_METRICS)].notna().any().any():
        raise ValueError("Normal score-track events contain score metrics.")

    if anomaly.any():
        miss = result.loc[anomaly, "miss_rate"]
        recall = result.loc[anomaly, "event_recall"]
        valid = miss.notna() & recall.notna()
        if valid.any() and not np.allclose(
            miss[valid], 1.0 - recall[valid], rtol=0.0, atol=1e-12
        ):
            raise ValueError("miss_rate must equal 1 - event_recall.")
        ltu = result.loc[anomaly, "ltu"]
        utility = result.loc[anomaly, "lead_time_utility"]
        valid = ltu.notna() & utility.notna()
        if valid.any() and not np.allclose(
            ltu[valid], utility[valid], rtol=0.0, atol=1e-12
        ):
            raise ValueError("ltu and lead_time_utility disagree.")

    support_columns = [
        "false_alarm_events",
        "turbine_days",
        "alarm_positive_points",
        "n_points",
        "correct_points",
    ]
    normal_label = normal & label_track
    if label_track.any() and result.loc[label_track, support_columns].isna().any().any():
        raise ValueError("Label events have missing operational supports.")
    supports = result.loc[label_track, support_columns]
    if (supports < 0).any().any():
        raise ValueError("Operational supports must be non-negative.")
    if (result["n_points"] <= 0).any():
        raise ValueError("n_points must be positive.")
    if (
        (result.loc[label_track, "alarm_positive_points"] < 0)
        | (
            result.loc[label_track, "alarm_positive_points"]
            > result.loc[label_track, "n_points"]
        )
        | (result.loc[label_track, "correct_points"] < 0)
        | (
            result.loc[label_track, "correct_points"]
            > result.loc[label_track, "n_points"]
        )
    ).any():
        raise ValueError("Point-count supports are inconsistent.")
    integer_columns = ["alarm_positive_points", "correct_points"]
    if "inference_points_event" in result.columns:
        integer_columns.append("inference_points_event")
    if label_track.any():
        integer_columns.append("false_alarm_events")
    n_points = result["n_points"]
    if not np.allclose(n_points, np.round(n_points), rtol=0.0, atol=1e-12):
        raise ValueError("n_points must contain integer counts.")
    for column in integer_columns:
        mask = label_track
        values = result.loc[mask, column]
        if not np.allclose(values, np.round(values), rtol=0.0, atol=1e-12):
            raise ValueError(f"{column} must contain integer counts.")
    if normal_label.any() and (
        result.loc[normal_label, "normal_monitoring_days"] <= 0
    ).any():
        raise ValueError("Normal monitoring duration must be positive.")
    if normal_label.any() and (
        result.loc[normal_label, "alarm_positive_points"]
        > result.loc[normal_label, "n_points"]
    ).any():
        raise ValueError("Alarm-positive points exceed monitored points.")
    if normal_label.any() and (
        result.loc[normal_label, "correct_points"]
        > result.loc[normal_label, "n_points"]
    ).any():
        raise ValueError("Correct points exceed monitored points.")
    if label_track.any():
        expected_duration = (
            result.loc[label_track, "alarm_positive_points"]
            / result.loc[label_track, "n_points"]
        )
        if not np.allclose(
            result.loc[label_track, "alarm_duration_ratio"],
            expected_duration,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("alarm_duration_ratio disagrees with point supports.")
    if normal_label.any():
        expected_accuracy = (
            result.loc[normal_label, "correct_points"]
            / result.loc[normal_label, "n_points"]
        )
        if not np.allclose(
            result.loc[normal_label, "accuracy"],
            expected_accuracy,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("Normal accuracy disagrees with point supports.")
    for column in (
        "inference_seconds_event",
        "inference_seconds",
        "inference_points_event",
        "calibration_seconds_event",
        "calibration_checkpoint_load_seconds_event",
        "evaluation_wall_seconds_event",
        "fit_seconds_source",
    ):
        if column in result.columns and (result[column].dropna() < 0).any():
            raise ValueError(f"{column} must be non-negative.")
    return result


def _inference_column(frame: pd.DataFrame) -> str | None:
    for column in ("inference_seconds_event", "inference_seconds"):
        if column in frame.columns:
            return column
    return None


def aggregate_results(
    event_results: pd.DataFrame,
    *,
    asset_column: str = "target_asset_id",
    event_column: str = "target_event_id",
    label_column: str = "target_event_label",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate event rows to assets, then macro-average assets by direction."""
    direction_columns = _direction_columns(event_results)
    events = _validate_events(
        event_results,
        direction_columns,
        asset_column,
        event_column,
        label_column,
    )
    inference_column = _inference_column(events)
    asset_keys = [*direction_columns, asset_column]
    asset_rows: List[Dict[str, object]] = []

    for keys, group in events.groupby(asset_keys, sort=False, dropna=False):
        row = _group_values(keys, asset_keys)
        normal = group[group[label_column].eq("normal")]
        anomaly = group[group[label_column].eq("anomaly")]
        row.update(
            {
                "event_count": int(len(group)),
                "normal_event_count": int(len(normal)),
                "anomaly_event_count": int(len(anomaly)),
            }
        )
        track = str(row["track"])
        track_metrics = SCORE_METRICS if track == "score" else LABEL_METRICS
        for metric in DETECTION_METRICS:
            if track == "label" and metric in _NORMAL_ONLY_LABEL_METRICS:
                row[metric] = float("nan")
                continue
            row[metric] = (
                _mean(anomaly[metric])
                if metric in track_metrics
                else float("nan")
            )

        inference_seconds = (
            _sum_or_nan(group[inference_column])
            if inference_column
            else float("nan")
        )
        inference_points = _sum(
            group[
                "inference_points_event"
                if "inference_points_event" in group.columns
                else "n_points"
            ]
        )
        calibration_seconds = _sum_or_nan(group["calibration_seconds_event"])
        calibration_load_seconds = _sum_or_nan(
            group["calibration_checkpoint_load_seconds_event"]
        )
        evaluation_wall_seconds = _sum_or_nan(
            group["evaluation_wall_seconds_event"]
        )
        row.update(
            {
                "resource_accounting": "shared_once",
                "inference_seconds": inference_seconds,
                "inference_points": inference_points,
                "calibration_seconds": calibration_seconds,
                "calibration_checkpoint_load_seconds": calibration_load_seconds,
                "calibration_peak_memory_mb": _max(
                    group["calibration_peak_memory_mb"]
                ),
                "calibration_peak_gpu_memory_mb": _max(
                    group["calibration_peak_gpu_memory_mb"]
                ),
                "calibration_peak_cpu_usage_percent": _max(
                    group["calibration_cpu_usage_percent"]
                ),
                "inference_peak_memory_mb": _max(
                    group["inference_peak_memory_mb"]
                ),
                "inference_peak_gpu_memory_mb": _max(
                    group["inference_peak_gpu_memory_mb"]
                ),
                "inference_peak_cpu_usage_percent": _max(
                    group["inference_cpu_usage_percent"]
                ),
                "evaluation_wall_seconds": evaluation_wall_seconds,
                "throughput_points_per_second": (
                    inference_points / inference_seconds
                    if pd.notna(inference_seconds) and inference_seconds > 0
                    else float("nan")
                ),
                "end_to_end_points_per_second": (
                    inference_points / evaluation_wall_seconds
                    if pd.notna(evaluation_wall_seconds)
                    and evaluation_wall_seconds > 0
                    else float("nan")
                ),
                "fit_seconds_source": _constant(group, "fit_seconds_source"),
            }
        )

        if track == "label":
            false_alarms_all = _sum(group["false_alarm_events"])
            turbine_days_all = _sum(group["turbine_days"])
            false_alarms_normal = _sum(normal["false_alarm_events"])
            turbine_days_normal = _sum(normal["turbine_days"])
            points_all = _sum(group["n_points"])
            correct_points_all = _sum(group["correct_points"])
            normal_points = _sum(normal["n_points"])
            correct_points_normal = _sum(normal["correct_points"])
            accuracy_all = (
                correct_points_all / points_all
                if points_all > 0
                else float("nan")
            )
            accuracy_normal = (
                correct_points_normal / normal_points
                if normal_points > 0
                else float("nan")
            )
            false_alarm_rate_all = (
                false_alarms_all / turbine_days_all
                if turbine_days_all > 0
                else float("nan")
            )
            false_alarm_rate_normal = (
                false_alarms_normal / turbine_days_normal
                if turbine_days_normal > 0
                else float("nan")
            )
            mtbfa_all = _mean(group["mtbfa"])
            mtbfa_normal = _mean(normal["mtbfa"])
            row.update(
                {
                    "accuracy": accuracy_normal,
                    "accuracy_all": accuracy_all,
                    "accuracy_normal": accuracy_normal,
                    "false_alarm_events": false_alarms_normal,
                    "turbine_days": turbine_days_normal,
                    "false_alarms_per_turbine_day": false_alarm_rate_normal,
                    "mtbfa": mtbfa_normal,
                    "false_alarm_events_all": false_alarms_all,
                    "turbine_days_all": turbine_days_all,
                    "false_alarms_per_turbine_day_all": false_alarm_rate_all,
                    "mtbfa_all": mtbfa_all,
                    "false_alarm_events_normal": false_alarms_normal,
                    "turbine_days_normal": turbine_days_normal,
                    "false_alarms_per_turbine_day_normal": false_alarm_rate_normal,
                    "mtbfa_normal": mtbfa_normal,
                    "normal_monitoring_days": turbine_days_normal,
                    "normal_points": normal_points,
                }
            )
        else:
            for column in (
                "false_alarm_events",
                "turbine_days",
                "normal_monitoring_days",
                "false_alarms_per_turbine_day",
                "mtbfa",
                "accuracy_all",
                "accuracy_normal",
                "false_alarm_events_all",
                "turbine_days_all",
                "false_alarms_per_turbine_day_all",
                "mtbfa_all",
                "false_alarm_events_normal",
                "turbine_days_normal",
                "false_alarms_per_turbine_day_normal",
                "mtbfa_normal",
                "normal_points",
            ):
                row[column] = float("nan")
        asset_rows.append(row)

    per_asset = pd.DataFrame(asset_rows)
    direction_rows: List[Dict[str, object]] = []
    for keys, group in per_asset.groupby(
        direction_columns, sort=False, dropna=False
    ):
        row = _group_values(keys, direction_columns)
        row.update(
            {
                "asset_count": int(len(group)),
                "normal_asset_count": int((group["normal_event_count"] > 0).sum()),
                "anomaly_asset_count": int((group["anomaly_event_count"] > 0).sum()),
                "event_count": int(group["event_count"].sum()),
                "normal_event_count": int(group["normal_event_count"].sum()),
                "anomaly_event_count": int(group["anomaly_event_count"].sum()),
            }
        )
        track = str(row["track"])
        track_metrics = SCORE_METRICS if track == "score" else LABEL_METRICS
        for metric in DETECTION_METRICS:
            row[metric] = (
                _mean(group[metric]) if metric in track_metrics else float("nan")
            )

        inference_seconds = _sum_or_nan(group["inference_seconds"])
        inference_points = _sum(group["inference_points"])
        calibration_seconds = _sum_or_nan(group["calibration_seconds"])
        calibration_load_seconds = _sum_or_nan(
            group["calibration_checkpoint_load_seconds"]
        )
        evaluation_wall_seconds = _sum_or_nan(group["evaluation_wall_seconds"])
        row.update(
            {
                "resource_accounting": "shared_once",
                "inference_seconds": inference_seconds,
                "inference_points": inference_points,
                "calibration_seconds": calibration_seconds,
                "calibration_checkpoint_load_seconds": calibration_load_seconds,
                "calibration_peak_memory_mb": _max(
                    group["calibration_peak_memory_mb"]
                ),
                "calibration_peak_gpu_memory_mb": _max(
                    group["calibration_peak_gpu_memory_mb"]
                ),
                "calibration_peak_cpu_usage_percent": _max(
                    group["calibration_peak_cpu_usage_percent"]
                ),
                "inference_peak_memory_mb": _max(
                    group["inference_peak_memory_mb"]
                ),
                "inference_peak_gpu_memory_mb": _max(
                    group["inference_peak_gpu_memory_mb"]
                ),
                "inference_peak_cpu_usage_percent": _max(
                    group["inference_peak_cpu_usage_percent"]
                ),
                "evaluation_wall_seconds": evaluation_wall_seconds,
                "throughput_points_per_second": (
                    inference_points / inference_seconds
                    if pd.notna(inference_seconds) and inference_seconds > 0
                    else float("nan")
                ),
                "end_to_end_points_per_second": (
                    inference_points / evaluation_wall_seconds
                    if pd.notna(evaluation_wall_seconds)
                    and evaluation_wall_seconds > 0
                    else float("nan")
                ),
                "fit_seconds_source": _constant(group, "fit_seconds_source"),
            }
        )

        if track == "label":
            row.update(
                {
                    "accuracy": _mean(group["accuracy"]),
                    "accuracy_all": _mean(group["accuracy_all"]),
                    "accuracy_normal": _mean(group["accuracy_normal"]),
                    "false_alarm_events": _sum(group["false_alarm_events"]),
                    "turbine_days": _sum(group["turbine_days"]),
                    "false_alarms_per_turbine_day": _mean(
                        group["false_alarms_per_turbine_day"]
                    ),
                    "mtbfa": _mean(group["mtbfa"]),
                    "false_alarm_events_all": _sum(
                        group["false_alarm_events_all"]
                    ),
                    "turbine_days_all": _sum(group["turbine_days_all"]),
                    "false_alarms_per_turbine_day_all": _mean(
                        group["false_alarms_per_turbine_day_all"]
                    ),
                    "mtbfa_all": _mean(group["mtbfa_all"]),
                    "false_alarm_events_normal": _sum(
                        group["false_alarm_events_normal"]
                    ),
                    "turbine_days_normal": _sum(group["turbine_days_normal"]),
                    "false_alarms_per_turbine_day_normal": _mean(
                        group["false_alarms_per_turbine_day_normal"]
                    ),
                    "mtbfa_normal": _mean(group["mtbfa_normal"]),
                    "normal_monitoring_days": _sum(
                        group["normal_monitoring_days"]
                    ),
                    "normal_points": _sum(group["normal_points"]),
                }
            )
        else:
            for column in (
                "false_alarm_events",
                "turbine_days",
                "normal_monitoring_days",
                "false_alarms_per_turbine_day",
                "mtbfa",
                "accuracy_all",
                "accuracy_normal",
                "false_alarm_events_all",
                "turbine_days_all",
                "false_alarms_per_turbine_day_all",
                "mtbfa_all",
                "false_alarm_events_normal",
                "turbine_days_normal",
                "false_alarms_per_turbine_day_normal",
                "mtbfa_normal",
                "normal_points",
            ):
                row[column] = float("nan")
        direction_rows.append(row)

    return per_asset, pd.DataFrame(direction_rows)


def write_result_tables(
    run_dir: Path, per_event: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_asset, per_direction = aggregate_results(per_event)
    results_dir = Path(run_dir) / "results"
    atomic_csv(per_event, results_dir / "per_event.csv")
    atomic_csv(per_asset, results_dir / "per_turbine.csv")
    atomic_csv(per_direction, results_dir / "per_direction.csv")
    for track in ("score", "label"):
        track_dir = results_dir / track
        own_metrics = SCORE_METRICS if track == "score" else LABEL_METRICS
        other_metrics = LABEL_METRICS if track == "score" else SCORE_METRICS
        for name, frame in (
            ("per_event.csv", per_event),
            ("per_turbine.csv", per_asset),
            ("per_direction.csv", per_direction),
        ):
            track_frame = frame.loc[frame["track"].eq(track)].reset_index(
                drop=True
            )
            track_frame = track_frame.drop(
                columns=[column for column in other_metrics if column in track_frame]
            )
            empty = [
                column
                for column in track_frame.columns
                if column not in own_metrics and track_frame[column].isna().all()
            ]
            track_frame = track_frame.drop(columns=empty)
            atomic_csv(track_frame, track_dir / name)
    return per_asset, per_direction


def _self_test() -> None:
    base = {
        "model_name": "M",
        "track": "label",
        "track_variant": "fpr_label",
        "protocol": "target_normal_calibrated",
        "source_farm": "A",
        "target_farm": "B",
        "transfer_type": "cross_farm",
        "fit_seconds_source": 2.0,
    }

    def row(asset: str, event: str, label: str, value: float, **extra) -> dict:
        item = {
            **base,
            "target_asset_id": asset,
            "target_event_id": event,
            "target_event_label": label,
            "false_alarm_events": 0.0,
            "turbine_days": 1.0,
            "normal_monitoring_days": np.nan,
            "alarm_duration_ratio": 0.0,
            "alarm_positive_points": 0.0,
            "n_points": 10.0,
            "correct_points": 10.0,
            "inference_seconds_event": 1.0,
            "inference_points_event": 10.0,
            "inference_peak_memory_mb": 0.0,
            "inference_peak_gpu_memory_mb": 0.0,
            "inference_cpu_usage_percent": 0.0,
            "calibration_seconds_event": 0.0,
            "calibration_checkpoint_load_seconds_event": 0.0,
            "calibration_peak_memory_mb": 0.0,
            "calibration_peak_gpu_memory_mb": 0.0,
            "calibration_cpu_usage_percent": 0.0,
            "evaluation_wall_seconds_event": 1.0,
        }
        for metric in LABEL_METRICS:
            item[metric] = value
        for metric in SCORE_METRICS:
            item[metric] = float("nan")
        item["lead_time_utility"] = item["ltu"] = value
        item["miss_rate"] = 1.0 - item["event_recall"]
        item.update(extra)
        return item

    rows = [
        row("1", "a1", "anomaly", 1.0),
        row("1", "a2", "anomaly", 0.0),
        row("2", "a3", "anomaly", 1.0),
        row(
            "1",
            "n1",
            "normal",
            np.nan,
            accuracy=0.95,
            false_alarm_events=1.0,
            turbine_days=1.0,
            false_alarms_per_turbine_day=1.0,
            mtbfa=10.0,
            normal_monitoring_days=1.0,
            alarm_duration_ratio=0.05,
            alarm_positive_points=5.0,
            n_points=100.0,
            correct_points=95.0,
        ),
        row(
            "1",
            "n2",
            "normal",
            np.nan,
            accuracy=0.95,
            false_alarm_events=1.0,
            turbine_days=1.0,
            false_alarms_per_turbine_day=1.0,
            mtbfa=20.0,
            normal_monitoring_days=1.0,
            alarm_duration_ratio=0.05,
            alarm_positive_points=5.0,
            n_points=100.0,
            correct_points=95.0,
        ),
        row(
            "2",
            "n3",
            "normal",
            np.nan,
            accuracy=1.0,
            false_alarm_events=0.0,
            turbine_days=4.0,
            false_alarms_per_turbine_day=0.0,
            mtbfa=np.nan,
            normal_monitoring_days=4.0,
            alarm_duration_ratio=0.0,
            alarm_positive_points=0.0,
            n_points=400.0,
            correct_points=400.0,
        ),
    ]
    per_asset, per_direction = aggregate_results(pd.DataFrame(rows))
    assert np.isclose(per_direction.iloc[0]["event_f1"], 0.75)
    assert np.isclose(per_direction.iloc[0]["accuracy"], 0.975)
    assert np.isclose(
        per_direction.iloc[0]["false_alarms_per_turbine_day"], 0.5
    )
    assert np.isclose(
        per_direction.iloc[0]["false_alarms_per_turbine_day_normal"], 0.5
    )
    assert np.isclose(per_direction.iloc[0]["mtbfa"], 15.0)
    assert "alarm_duration_ratio" not in per_asset
    assert "alarm_duration_ratio" not in per_direction
    assert "aer" not in per_direction
    assert len(per_asset) == 2


if __name__ == "__main__":
    _self_test()
