"""Leakage-safe split planning and event preparation for Core-3 transfers."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .core3_contract import (
        FEATURE_COLUMNS,
        OUTPUT_COLUMNS,
        normalize_identifier,
        update_frame_digest,
    )
except ImportError:  # pragma: no cover - direct script execution
    from core3_contract import (  # type: ignore
        FEATURE_COLUMNS,
        OUTPUT_COLUMNS,
        normalize_identifier,
        update_frame_digest,
    )


REQUIRED_MANIFEST_COLUMNS = {
    "file_name",
    "farm_id",
    "event_id",
    "asset_id",
    "event_label",
    "event_start_id",
    "event_end_id",
    "derived_path",
    "train_rows",
    "prediction_rows",
    "total_rows",
}


@dataclass(frozen=True)
class MedianImputerState:
    center: pd.Series
    fitted_rows: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "features": list(FEATURE_COLUMNS),
            "center": {name: float(self.center[name]) for name in FEATURE_COLUMNS},
            "fitted_rows": int(self.fitted_rows),
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, object]) -> "MedianImputerState":
        if list(document.get("features", [])) != list(FEATURE_COLUMNS):
            raise ValueError("Imputer feature order does not match Core-3.")
        center = document.get("center")
        if not isinstance(center, Mapping):
            raise ValueError("Imputer document must contain a center mapping.")
        return cls(
            center=pd.Series({name: float(center[name]) for name in FEATURE_COLUMNS}),
            fitted_rows=int(document.get("fitted_rows", 0)),
        )


@dataclass(frozen=True)
class PreparedEvent:
    record: Mapping[str, object]
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    train_labels: np.ndarray
    test_labels: np.ndarray
    train_metadata: pd.DataFrame
    test_metadata: pd.DataFrame

    def train_features(self, imputer: MedianImputerState) -> pd.DataFrame:
        return apply_median_imputer(self.train_raw, imputer)

    def test_features(self, imputer: MedianImputerState) -> pd.DataFrame:
        return apply_median_imputer(self.test_raw, imputer)


def load_manifest(dataset_root: Path) -> pd.DataFrame:
    path = dataset_root / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Core-3 manifest not found: {path}. Run build_core3_dataset.py first."
        )
    manifest = pd.read_csv(
        path,
        dtype={"farm_id": "string", "event_id": "string", "asset_id": "string"},
    )
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Core-3 manifest is missing columns: {sorted(missing)}")
    manifest = manifest.copy()
    manifest["farm_id"] = manifest["farm_id"].astype(str).str.strip()
    manifest["event_id"] = manifest["event_id"].map(normalize_identifier)
    manifest["asset_id"] = manifest["asset_id"].map(normalize_identifier)
    manifest["event_label"] = (
        manifest["event_label"].astype(str).str.strip().str.lower()
    )
    if manifest["file_name"].duplicated().any():
        raise ValueError("Core-3 manifest contains duplicate file_name values.")
    if not set(manifest["event_label"]).issubset({"normal", "anomaly"}):
        extra = sorted(set(manifest["event_label"]) - {"normal", "anomaly"})
        raise ValueError(f"Unexpected event_label values: {extra}")
    return manifest


def resolve_derived_path(dataset_root: Path, value: object) -> Path:
    raw = str(value).strip().replace("\\", "/")
    relative = PurePosixPath(raw)
    if (
        not raw
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or (relative.parts and ":" in relative.parts[0])
    ):
        raise ValueError(f"Invalid manifest derived_path: {value!r}.")
    return dataset_root.joinpath(*relative.parts)


def normalized_holdouts(config: Mapping[str, object]) -> Dict[str, set[str]]:
    raw = config.get("holdout_assets")
    if not isinstance(raw, Mapping):
        raise ValueError("Config must contain a holdout_assets mapping.")
    result: Dict[str, set[str]] = {}
    for farm_id in ("A", "B", "C"):
        values = raw.get(farm_id)
        if not isinstance(values, list) or not values:
            raise ValueError(f"holdout_assets[{farm_id!r}] must be a non-empty list.")
        normalized = {normalize_identifier(value) for value in values}
        if "" in normalized:
            raise ValueError(f"holdout_assets[{farm_id!r}] contains an empty ID.")
        result[farm_id] = normalized
    return result


def validate_protocol(
    manifest: pd.DataFrame, config: Mapping[str, object]
) -> Dict[str, set[str]]:
    features = config.get("features")
    if list(features or []) != list(FEATURE_COLUMNS):
        raise ValueError(
            f"Cross-domain features must be exactly {list(FEATURE_COLUMNS)} in this order."
        )

    holdouts = normalized_holdouts(config)
    for farm_id in ("A", "B", "C"):
        farm = manifest.loc[manifest["farm_id"].eq(farm_id)]
        if farm.empty:
            raise ValueError(f"Manifest contains no Wind Farm {farm_id} events.")
        available = set(farm["asset_id"])
        missing = holdouts[farm_id] - available
        if missing:
            raise ValueError(
                f"Wind Farm {farm_id} holdout assets are absent: {sorted(missing)}"
            )

        target = farm.loc[farm["asset_id"].isin(holdouts[farm_id])]
        source = farm.loc[~farm["asset_id"].isin(holdouts[farm_id])]
        if source.empty or target.empty:
            raise ValueError(f"Wind Farm {farm_id} has an empty source or target split.")
        if set(source["asset_id"]) & holdouts[farm_id]:
            raise AssertionError(f"Wind Farm {farm_id} source/holdout asset leakage.")

        labels = set(target["event_label"])
        if labels != {"normal", "anomaly"}:
            raise ValueError(
                f"Wind Farm {farm_id} holdout needs both normal and anomaly events; "
                f"found {sorted(labels)}."
            )

    protocol = config.get("protocol", {})
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a mapping.")
    expected_modes = {
        "source_training": "all_non_holdout_assets_train_segments",
        "target_evaluation": "fixed_holdout_assets",
        "aggregation": "event_to_turbine_to_direction_macro",
        "temporal_start_policy": "repeat_first_prediction_point_if_required",
    }
    for key, expected in expected_modes.items():
        if protocol.get(key) != expected:
            raise ValueError(f"protocol.{key} must equal {expected!r}.")
    if protocol.get("normalization") != "model_native_source_fit_only":
        raise ValueError(
            "protocol.normalization must equal 'model_native_source_fit_only'."
        )
    imputation = protocol.get("imputation")
    expected_imputation = {
        "source_training": "source_farm_train_median",
        "strict_zero_shot": "source_farm_train_median",
        "target_normal_calibrated": "target_event_train_median",
    }
    if imputation != expected_imputation:
        raise ValueError(f"imputation must equal {expected_imputation}.")
    protocols = protocol.get("evaluation_protocols")
    allowed_protocols = {"strict_zero_shot", "target_normal_calibrated"}
    if not isinstance(protocols, list) or not protocols:
        raise ValueError("evaluation_protocols must be a non-empty list.")
    if "strict_zero_shot" not in protocols:
        raise ValueError("evaluation_protocols must contain strict_zero_shot.")
    if not set(protocols).issubset(allowed_protocols):
        raise ValueError(
            "evaluation_protocols may contain only strict_zero_shot and "
            "target_normal_calibrated."
        )
    if len(protocols) != len(set(protocols)):
        raise ValueError("evaluation_protocols contains duplicate values.")
    tracks = protocol.get("evaluation_tracks")
    expected_tracks = {"score", "label"}
    if not isinstance(tracks, list) or set(tracks) != expected_tracks:
        raise ValueError(
            "evaluation_tracks must contain score and label exactly once."
        )
    if len(tracks) != len(expected_tracks):
        raise ValueError("evaluation_tracks contains duplicate values.")
    if protocol.get("label_policy") != "protocol_fpr_threshold":
        raise ValueError(
            "protocol.label_policy must equal 'protocol_fpr_threshold'."
        )
    fpr_budget = float(protocol.get("false_positive_budget", -1.0))
    if not 0.0 <= fpr_budget < 1.0:
        raise ValueError("false_positive_budget must satisfy 0 <= value < 1.")
    if not isinstance(protocol.get("defer_score_vus", False), bool):
        raise ValueError("protocol.defer_score_vus must be boolean.")
    if int(protocol.get("points_per_day", 0)) <= 0:
        raise ValueError("points_per_day must be positive.")
    if int(protocol.get("lead_delta_points", -1)) < 0:
        raise ValueError("lead_delta_points must be non-negative.")
    return holdouts


def source_records(
    manifest: pd.DataFrame, source_farm: str, holdouts: Mapping[str, set[str]]
) -> pd.DataFrame:
    result = manifest.loc[
        manifest["farm_id"].eq(source_farm)
        & ~manifest["asset_id"].isin(holdouts[source_farm])
    ].copy()
    return result.sort_values(["asset_id", "event_id"])


def target_records(
    manifest: pd.DataFrame, target_farm: str, holdouts: Mapping[str, set[str]]
) -> pd.DataFrame:
    result = manifest.loc[
        manifest["farm_id"].eq(target_farm)
        & manifest["asset_id"].isin(holdouts[target_farm])
    ].copy()
    return result.sort_values(["asset_id", "event_id"])


def build_protocol_plan(
    manifest: pd.DataFrame,
    config: Mapping[str, object],
    source_farms: Sequence[str] = ("A", "B", "C"),
) -> pd.DataFrame:
    """Return one row per source-farm/target-event evaluation task."""

    holdouts = validate_protocol(manifest, config)
    protocol = config["protocol"]
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a mapping.")
    protocols = list(protocol["evaluation_protocols"])
    tracks = list(protocol["evaluation_tracks"])
    selected_sources = tuple(str(farm).upper() for farm in source_farms)
    if not selected_sources or len(set(selected_sources)) != len(selected_sources):
        raise ValueError("source_farms must contain unique farm IDs.")
    if not set(selected_sources).issubset({"A", "B", "C"}):
        raise ValueError("source_farms may contain only A, B, and C.")
    rows = []
    for source_farm in selected_sources:
        source = source_records(manifest, source_farm, holdouts)
        source_assets = sorted(set(source["asset_id"]))
        for target_farm in ("A", "B", "C"):
            target = target_records(manifest, target_farm, holdouts)
            for _, record in target.iterrows():
                rows.append(
                    {
                        "source_farm": source_farm,
                        "target_farm": target_farm,
                        "transfer_type": (
                            "cross_turbine"
                            if source_farm == target_farm
                            else "cross_farm"
                        ),
                        "source_asset_count": len(source_assets),
                        "source_event_count": len(source),
                        "source_assets": json.dumps(source_assets),
                        "target_asset_id": record["asset_id"],
                        "target_event_id": record["event_id"],
                        "target_file_name": record["file_name"],
                        "target_event_label": record["event_label"],
                        "target_derived_path": record["derived_path"],
                        "evaluation_protocols": json.dumps(protocols),
                        "evaluation_tracks": json.dumps(tracks),
                        "track_variants": json.dumps(
                            {"score": "continuous_score", "label": "fpr_label"}
                        ),
                        "label_policy": protocol["label_policy"],
                        "result_rows_per_model": len(protocols) * len(tracks),
                    }
                )
    plan = pd.DataFrame(rows)
    if plan.empty:
        raise ValueError("Protocol plan is empty.")
    return plan


def build_dataset_split(
    manifest: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    """Assign every physical event to its fixed source-pool or target role."""

    holdouts = validate_protocol(manifest, config)
    split = manifest[
        [
            "file_name",
            "farm_id",
            "event_id",
            "asset_id",
            "event_label",
            "derived_path",
            "train_rows",
            "prediction_rows",
            "total_rows",
        ]
    ].copy()
    split["is_holdout_asset"] = [
        asset_id in holdouts[farm_id]
        for farm_id, asset_id in zip(split["farm_id"], split["asset_id"])
    ]
    split["split_role"] = np.where(
        split["is_holdout_asset"], "fixed_target", "source_train_pool"
    )
    split["train_partition_usage"] = np.where(
        split["is_holdout_asset"],
        "target_normal_imputer_score_reference_and_same_event_context_only;"
        "unused_by_strict_zero_shot",
        "source_imputer_model_fit_and_score_reference",
    )
    split["prediction_partition_usage"] = np.where(
        split["is_holdout_asset"], "target_evaluation", "unused"
    )
    return split.sort_values(["farm_id", "asset_id", "event_id"]).reset_index(
        drop=True
    )


def direction_summary(plan: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        plan.groupby(["source_farm", "target_farm", "transfer_type"], as_index=False)
        .agg(
            source_asset_count=("source_asset_count", "first"),
            source_event_count=("source_event_count", "first"),
            target_asset_count=("target_asset_id", "nunique"),
            target_event_count=("target_event_id", "size"),
            target_normal_events=(
                "target_event_label",
                lambda values: int(pd.Series(values).eq("normal").sum()),
            ),
            target_anomaly_events=(
                "target_event_label",
                lambda values: int(pd.Series(values).eq("anomaly").sum()),
            ),
            event_result_rows_per_model=("result_rows_per_model", "sum"),
        )
        .sort_values(["source_farm", "target_farm"])
        .reset_index(drop=True)
    )
    return grouped


def _numeric_feature_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    numeric = frame.loc[:, list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    for column in FEATURE_COLUMNS:
        raw = frame[column]
        present = raw.notna() & raw.astype("string").str.strip().ne("")
        invalid = present & numeric[column].isna()
        if invalid.any():
            examples = raw.loc[invalid].astype(str).head(3).tolist()
            raise ValueError(f"{path}: non-numeric {column} values: {examples}")
    if np.isinf(numeric.to_numpy(dtype=float, copy=False)).any():
        raise ValueError(f"{path}: Core-3 features contain infinite values.")
    return numeric.astype(float)


def _labels_for_event(frame: pd.DataFrame, record: Mapping[str, object]) -> np.ndarray:
    label = str(record["event_label"]).strip().lower()
    if label == "normal":
        return np.zeros(len(frame), dtype=np.int8)
    if label != "anomaly":
        raise ValueError(f"Unexpected event_label={label!r}.")
    start = record.get("event_start_id")
    end = record.get("event_end_id")
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        raise ValueError(f"Anomaly event {record['file_name']} has no ID boundaries.")
    ids = pd.to_numeric(frame["id"], errors="raise").to_numpy(dtype=np.int64)
    start_id, end_id = int(float(start)), int(float(end))
    if start_id > end_id:
        raise ValueError(f"Invalid anomaly boundaries [{start_id}, {end_id}].")
    return ((ids >= start_id) & (ids <= end_id)).astype(np.int8)


def prepare_event(
    dataset_root: Path,
    record: Mapping[str, object],
    *,
    verify_digest: bool = True,
) -> PreparedEvent:
    path = resolve_derived_path(dataset_root, record["derived_path"])
    frame = pd.read_csv(path, sep=";", low_memory=False, float_precision="round_trip")
    if list(frame.columns) != list(OUTPUT_COLUMNS):
        raise ValueError(
            f"{path}: expected columns {list(OUTPUT_COLUMNS)}, got {list(frame.columns)}"
        )
    if len(frame) != int(record["total_rows"]):
        raise ValueError(
            f"{path}: rows={len(frame)}, manifest total_rows={record['total_rows']}"
        )
    expected_digest = record.get("content_sha256")
    if verify_digest and expected_digest is not None and not pd.isna(expected_digest):
        digest = hashlib.sha256()
        update_frame_digest(digest, frame)
        if digest.hexdigest() != str(expected_digest):
            raise ValueError(f"{path}: content digest does not match manifest.")

    split = frame["train_test"].astype(str).str.strip().replace({"prediction": "test"})
    invalid = sorted(set(split) - {"train", "test"})
    if invalid:
        raise ValueError(f"{path}: unexpected train_test values {invalid}.")
    train_mask = split.eq("train").to_numpy()
    test_mask = split.eq("test").to_numpy()
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            f"{path}: train and prediction partitions must both be non-empty."
        )
    last_train = int(np.flatnonzero(train_mask)[-1])
    first_test = int(np.flatnonzero(test_mask)[0])
    if (
        last_train >= first_test
        or not train_mask[:first_test].all()
        or not test_mask[first_test:].all()
    ):
        raise ValueError(f"{path}: train rows must precede all prediction rows.")

    labels = _labels_for_event(frame, record)
    if labels[train_mask].any():
        raise ValueError(
            f"{path}: target/source train partition contains anomaly labels (leakage)."
        )

    features = _numeric_feature_frame(frame, path)
    train_raw = features.loc[train_mask].reset_index(drop=True)
    test_raw = features.loc[test_mask].reset_index(drop=True)
    metadata_columns = ["time_stamp", "asset_id", "id", "status_type_id"]
    train_metadata = frame.loc[train_mask, metadata_columns].reset_index(drop=True)
    test_metadata = frame.loc[test_mask, metadata_columns].reset_index(drop=True)
    return PreparedEvent(
        record=dict(record),
        train_raw=train_raw.loc[:, list(FEATURE_COLUMNS)],
        test_raw=test_raw.loc[:, list(FEATURE_COLUMNS)],
        train_labels=labels[train_mask],
        test_labels=labels[test_mask],
        train_metadata=train_metadata,
        test_metadata=test_metadata,
    )


def fit_median_imputer(frames: Iterable[pd.DataFrame]) -> MedianImputerState:
    parts = []
    for frame in frames:
        if list(frame.columns) != list(FEATURE_COLUMNS):
            raise ValueError("Scaler input must use the exact Core-3 feature order.")
        if not frame.empty:
            parts.append(frame.loc[:, list(FEATURE_COLUMNS)])
    if not parts:
        raise ValueError("Cannot fit an imputer on an empty source pool.")
    pooled = pd.concat(parts, ignore_index=True)
    center = pooled.median(axis=0, skipna=True).astype(float)
    if center.isna().any():
        bad = center.index[center.isna()].tolist()
        raise ValueError(f"All imputer-fit values are missing for {bad}.")
    if not np.isfinite(center.to_numpy()).all():
        raise ValueError("Median imputer contains non-finite statistics.")
    return MedianImputerState(center=center, fitted_rows=len(pooled))


def apply_median_imputer(
    frame: pd.DataFrame, imputer: MedianImputerState
) -> pd.DataFrame:
    if list(frame.columns) != list(FEATURE_COLUMNS):
        raise ValueError("Imputer input must use the exact Core-3 feature order.")
    filled = frame.fillna(imputer.center)
    filled = filled.loc[:, list(FEATURE_COLUMNS)].astype(float)
    if not np.isfinite(filled.to_numpy(dtype=float, copy=False)).all():
        raise ValueError("Non-finite values remain after median imputation.")
    return filled


def calibrate_at_fpr(scores: Iterable[float], budget: float) -> Tuple[float, np.ndarray, float]:
    """Choose a strict-``>`` threshold whose realized FPR is at most budget."""

    values = np.asarray(list(scores), dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Calibration scores must be a non-empty finite vector.")
    if not 0.0 <= budget < 1.0:
        raise ValueError("FPR budget must satisfy 0 <= budget < 1.")
    allowed = int(math.floor(budget * len(values) + 1e-12))
    ordered = np.sort(values)
    if allowed <= 0:
        threshold = float(ordered[-1])
    else:
        threshold = float(ordered[len(values) - allowed - 1])
    predicted = values > threshold
    realized = float(predicted.mean())
    if realized > budget + 1e-12:
        raise AssertionError(
            f"Calibration produced FPR={realized}, exceeding budget={budget}."
        )
    return threshold, predicted.astype(np.int8), realized


def apply_threshold(scores: Iterable[float], threshold: float) -> np.ndarray:
    values = np.asarray(list(scores), dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Test scores must be finite.")
    return (values > float(threshold)).astype(np.int8)
