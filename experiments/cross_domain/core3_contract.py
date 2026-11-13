from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


CORE3_VERSION = "core3_v1"
FARMS: Sequence[str] = ("A", "B", "C")

META_COLUMNS: Sequence[str] = (
    "time_stamp",
    "asset_id",
    "id",
    "train_test",
    "status_type_id",
)
FEATURE_COLUMNS: Sequence[str] = (
    "wind_speed",
    "active_power",
    "rotor_speed",
)
OUTPUT_COLUMNS: Sequence[str] = (*META_COLUMNS, *FEATURE_COLUMNS)


FEATURE_MAPPING: Mapping[str, Mapping[str, Mapping[str, object]]] = {
    "A": {
        "wind_speed": {
            "sources": ["wind_speed_3_avg"],
            "aggregation": "identity",
            "unit": "m/s",
        },
        "active_power": {
            "sources": ["power_30_avg"],
            "aggregation": "identity",
            "unit": "kW",
        },
        "rotor_speed": {
            "sources": ["sensor_52_avg"],
            "aggregation": "identity",
            "unit": "rpm",
        },
    },
    "B": {
        "wind_speed": {
            "sources": ["wind_speed_61_avg"],
            "aggregation": "identity",
            "unit": "m/s",
        },
        "active_power": {
            "sources": ["power_62_avg"],
            "aggregation": "identity",
            "unit": "kW",
        },
        "rotor_speed": {
            "sources": ["sensor_25_avg"],
            "aggregation": "identity",
            "unit": "rpm",
        },
    },
    "C": {
        "wind_speed": {
            "sources": ["wind_speed_236_avg"],
            "aggregation": "identity",
            "unit": "m/s",
        },
        "active_power": {
            "sources": ["power_6_avg"],
            "aggregation": "identity",
            "unit": "kW",
        },
        "rotor_speed": {
            "sources": ["sensor_144_avg", "sensor_145_avg"],
            "aggregation": "row_mean_skipna",
            "unit": "rpm",
        },
    },
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def raw_farm_root(source_root: Path, farm_id: str) -> Path:
    farm_root = source_root / f"Wind Farm {farm_id}"
    if (farm_root / "event_info.csv").is_file():
        return farm_root
    return farm_root / f"Wind Farm {farm_id}"


def derived_farm_root(output_root: Path, farm_id: str) -> Path:
    return output_root / f"Wind Farm {farm_id}" / f"Wind Farm {farm_id}"


def normalize_identifier(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if np.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def event_sort_key(event_id: object) -> tuple[int, object]:
    text = normalize_identifier(event_id)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def source_columns_for_farm(farm_id: str) -> list[str]:
    columns = list(META_COLUMNS)
    for spec in FEATURE_MAPPING[farm_id].values():
        for column in spec["sources"]:  # type: ignore[index]
            if column not in columns:
                columns.append(str(column))
    return columns


def canonicalize_chunk(frame: pd.DataFrame, farm_id: str) -> pd.DataFrame:
    missing = set(source_columns_for_farm(farm_id)) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Wind Farm {farm_id} is missing required columns: {sorted(missing)}"
        )

    out = frame.loc[:, list(META_COLUMNS)].copy()
    for canonical_name, spec in FEATURE_MAPPING[farm_id].items():
        sources = [str(c) for c in spec["sources"]]  # type: ignore[index]
        raw_values = frame.loc[:, sources]
        numeric = raw_values.apply(pd.to_numeric, errors="coerce")

        for source in sources:
            raw_present = raw_values[source].notna() & raw_values[source].astype(
                "string"
            ).str.strip().ne("")
            invalid = raw_present & numeric[source].isna()
            if invalid.any():
                examples = raw_values.loc[invalid, source].astype(str).head(3).tolist()
                raise ValueError(
                    f"Wind Farm {farm_id} column {source!r} contains non-numeric "
                    f"values, for example {examples}."
                )
        if np.isinf(numeric.to_numpy(dtype=float, copy=False)).any():
            raise ValueError(
                f"Wind Farm {farm_id} contains infinite values in {sources}."
            )
        if spec["aggregation"] == "identity":
            out[canonical_name] = numeric.iloc[:, 0].astype(float)
        elif spec["aggregation"] == "row_mean_skipna":
            out[canonical_name] = numeric.mean(axis=1, skipna=True).astype(float)
        else:
            raise ValueError(
                f"Unsupported Core-3 aggregation: {spec['aggregation']!r}"
            )
    return out.loc[:, list(OUTPUT_COLUMNS)]


def normalized_for_digest(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(index=frame.index)
    for column in META_COLUMNS:
        normalized[column] = frame[column].astype("string").fillna("<NA>")
    for column in FEATURE_COLUMNS:
        normalized[column] = pd.to_numeric(frame[column], errors="coerce").astype(
            "float64"
        )
    return normalized.loc[:, list(OUTPUT_COLUMNS)]


def update_frame_digest(hasher: "hashlib._Hash", frame: pd.DataFrame) -> None:
    normalized = normalized_for_digest(frame)
    row_hashes = pd.util.hash_pandas_object(
        normalized, index=False, categorize=False
    ).to_numpy(dtype=np.uint64, copy=False)
    hasher.update(row_hashes.tobytes(order="C"))


def mapping_document() -> Dict[str, object]:
    return {
        "dataset_version": CORE3_VERSION,
        "delimiter": ";",
        "output_columns": list(OUTPUT_COLUMNS),
        "features": list(FEATURE_COLUMNS),
        "normalization_applied": False,
        "missing_value_imputation_applied": False,
        "mapping": FEATURE_MAPPING,
        "notes": [
            "Raw event row order and train_test values are preserved.",
            "Wind Farm C rotor_speed is the row-wise mean of two redundant channels.",
            "Missing values are imputed from allowed normal training data at runtime.",
            "Normalization is owned by each model and fitted on source training data only.",
        ],
    }


def canonical_feature_description() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sensor_name": "wind_speed",
                "statistics_type": "average",
                "description": "Canonical wind speed",
                "unit": "m/s",
                "is_angle": False,
                "is_counter": False,
            },
            {
                "sensor_name": "active_power",
                "statistics_type": "average",
                "description": "Canonical active power",
                "unit": "kW",
                "is_angle": False,
                "is_counter": False,
            },
            {
                "sensor_name": "rotor_speed",
                "statistics_type": "average",
                "description": "Canonical rotor speed",
                "unit": "rpm",
                "is_angle": False,
                "is_counter": False,
            },
        ]
    )


def ensure_exact_columns(actual: Iterable[str], expected: Sequence[str]) -> None:
    actual_list = list(actual)
    if actual_list != list(expected):
        raise ValueError(
            f"Schema mismatch. Expected {list(expected)}, got {actual_list}."
        )
