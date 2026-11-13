import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_STAT_SUFFIXES = ("_avg", "_max", "_min", "_std")
_DOMAIN_INHERIT_SUFFIXES = ("_avg", "_max", "_min")
_DEG_TO_RAD = math.pi / 180.0

DEFAULT_META_COLUMNS: Sequence[str] = (
    "time_stamp",
    "asset_id",
    "id",
    "train_test",
    "status_type",
    "status_type_id",
    "label",
)

_REQUIRED_SERIES_COLUMNS = {"train_test", "label"}

_REQUIRED_META_FIELDS = {
    "file_name", "farm_id", "event_id", "event_label",
    "train_lens",
}


class DataContractError(ValueError):
    pass


def validate_wind_series_frame(
    name: str,
    df: pd.DataFrame,
    *,
    expected_train_lens: Optional[int] = None,
    expected_prediction_lens: Optional[int] = None,
    expected_total_lens: Optional[int] = None,
) -> None:
    if df is None or df.empty:
        raise DataContractError(f"[{name}] Series is None or empty.")

    missing = _REQUIRED_SERIES_COLUMNS - set(df.columns)
    if missing:
        raise DataContractError(
            f"[{name}] Missing required columns: {sorted(missing)}"
        )

    label_col = df["label"]
    unique_labels = set(label_col.dropna().unique())
    if not unique_labels.issubset({0, 1, 0.0, 1.0}):
        raise DataContractError(
            f"[{name}] 'label' must only contain 0/1, got extra values: "
            f"{sorted(unique_labels - {0, 1, 0.0, 1.0})}"
        )

    tt = df["train_test"].astype(str).str.strip()
    invalid_tt = set(tt.unique()) - {"train", "test"}
    if invalid_tt:
        raise DataContractError(
            f"[{name}] 'train_test' contains unexpected values: {sorted(invalid_tt)}"
        )

    n_train = int((tt == "train").sum())
    n_test = int((tt == "test").sum())
    if n_train > 0 and not (tt.iloc[:n_train] == "train").all():
        raise DataContractError(
            f"[{name}] 'train_test' rows are interleaved; all 'train' rows "
            f"must precede every 'test' row (iloc-based split assumption)."
        )

    if expected_train_lens is not None:
        et = int(expected_train_lens)
        if n_train != et:
            raise DataContractError(
                f"[{name}] Metadata train_lens ({et}) does not match "
                f"the number of 'train' rows in train_test ({n_train})."
            )
    if expected_prediction_lens is not None:
        ep = int(expected_prediction_lens)
        if n_test != ep:
            raise DataContractError(
                f"[{name}] Metadata prediction_lens ({ep}) does not match "
                f"the number of 'test' rows in train_test ({n_test})."
            )
    if expected_total_lens is not None:
        etot = int(expected_total_lens)
        n = len(df)
        if n != etot:
            raise DataContractError(
                f"[{name}] Metadata total_lens ({etot}) does not match "
                f"DataFrame length ({n})."
            )

    feature_df = feature_columns_only(df)
    if feature_df.shape[1] == 0:
        raise DataContractError(
            f"[{name}] No feature columns remain after dropping metadata columns."
        )

    non_numeric = [
        c for c in feature_df.columns
        if not np.issubdtype(feature_df[c].dtype, np.number)
    ]
    if non_numeric:
        raise DataContractError(
            f"[{name}] Non-numeric feature columns detected: {non_numeric}"
        )

    finite_mask = np.isfinite(feature_df.to_numpy(dtype=float, copy=False))
    if not finite_mask.all():
        bad_cols = feature_df.columns[~finite_mask.all(axis=0)].tolist()
        raise DataContractError(
            f"[{name}] Feature columns contain non-finite values: {bad_cols}"
        )

    nan_counts = feature_df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if not nan_counts.empty:
        raise DataContractError(
            f"[{name}] Feature columns contain missing values: "
            f"{nan_counts.to_dict()}"
        )


def validate_wind_metadata_table(metadata: pd.DataFrame) -> None:
    if metadata is None or metadata.empty:
        raise DataContractError("Metadata is None or empty.")

    missing = _REQUIRED_META_FIELDS - set(metadata.columns)
    if missing:
        raise DataContractError(
            f"Metadata missing required fields: {sorted(missing)}"
        )

    for idx, row in metadata.iterrows():
        fname = row.get("file_name", idx)
        el = row.get("event_label")
        is_normal = (
            el is not None
            and not (isinstance(el, float) and pd.isna(el))
            and str(el).lower().strip() == "normal"
        )

        tl = row.get("train_lens")
        if tl is None or (isinstance(tl, float) and pd.isna(tl)):
            raise DataContractError(
                f"[{fname}] 'train_lens' is missing in metadata."
            )

        if not is_normal:
            for fld in ("event_start_id", "event_end_id"):
                v = row.get(fld)
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    raise DataContractError(
                        f"[{fname}] Non-normal event requires '{fld}' in metadata."
                    )
            sid = row.get("event_start_id")
            eid = row.get("event_end_id")
            if float(sid) > float(eid):
                raise DataContractError(
                    f"[{fname}] event_start_id ({sid}) > event_end_id ({eid})."
                )


def feature_columns_only(
    df: pd.DataFrame,
    extra_drop: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    to_drop: List[str] = [
        c for c in DEFAULT_META_COLUMNS if c in df.columns
    ]
    if extra_drop:
        to_drop += [c for c in extra_drop if c in df.columns and c not in to_drop]
    return df.drop(columns=to_drop) if to_drop else df.copy()


def load_series_covariates(folder_path: str) -> Optional[Dict]:
    if not os.path.isdir(folder_path):
        return None
    covariates = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            covariates[filename] = _load_covariate_reference(file_path)
    return covariates or None


def _load_covariate_reference(file_path: str) -> Any:
    return file_path


def _normalize_train_test_column(df: pd.DataFrame) -> pd.DataFrame:
    if "train_test" not in df.columns:
        return df
    out = df.copy()
    s = out["train_test"].astype(str).str.strip()
    out["train_test"] = s.replace({"prediction": "test"})
    return out


def _compute_row_labels(df: pd.DataFrame, event_context: Dict[str, Any]) -> pd.Series:
    el = event_context.get("event_label")
    if el is not None and not (isinstance(el, float) and pd.isna(el)):
        if str(el).lower().strip() == "normal":
            return pd.Series(0, index=df.index, dtype=int)

    if "id" not in df.columns:
        raise ValueError("Column 'id' is required to compute labels from event boundaries.")
    start_id = event_context.get("event_start_id")
    end_id = event_context.get("event_end_id")
    if (
        start_id is None
        or end_id is None
        or pd.isna(start_id)
        or pd.isna(end_id)
    ):
        raise ValueError("event_start_id and event_end_id are required for non-normal events.")

    sid = int(float(start_id))
    eid = int(float(end_id))
    if sid > eid:
        raise DataContractError(
            f"event_start_id ({sid}) > event_end_id ({eid})."
        )
    ids = df["id"].astype(int)
    id_min = int(ids.min())
    id_max = int(ids.max())
    if eid < id_min or sid > id_max:
        raise DataContractError(
            f"Event id range [{sid}, {eid}] lies outside series id range "
            f"[{id_min}, {id_max}]; resulting label would be all zeros."
        )
    return ((ids >= sid) & (ids <= eid)).astype(int)


def _strip_stat_suffix(col: str) -> str:
    for sfx in _STAT_SUFFIXES:
        if col.endswith(sfx):
            return col[: -len(sfx)]
    return col


def _strip_domain_inherit_suffix(col: str) -> Optional[str]:
    if col.endswith("_std"):
        return None
    for sfx in _DOMAIN_INHERIT_SUFFIXES:
        if col.endswith(sfx):
            return col[: -len(sfx)]
    return col


def apply_wind_domain_rules(
    df: pd.DataFrame,
    feature_desc: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if feature_desc is None or feature_desc.empty:
        return df
    fd = feature_desc.copy()
    fd.columns = [c.strip() for c in fd.columns]
    if "sensor_name" not in fd.columns:
        return df

    def _flag(col: str) -> set:
        if col not in fd.columns:
            return set()
        return set(
            fd.loc[fd[col].astype(str).str.lower() == "true", "sensor_name"]
            .astype(str).str.strip()
        )

    angles = _flag("is_angle")
    counters = _flag("is_counter")
    if not angles and not counters:
        return df

    out = df.copy()
    for col in list(out.columns):
        if col in DEFAULT_META_COLUMNS:
            continue
        base = _strip_domain_inherit_suffix(col)
        if base is None:
            continue
        if base in angles:
            rad = pd.to_numeric(out[col], errors="coerce").astype(float) * _DEG_TO_RAD
            out[col + "_sin"] = np.sin(rad)
            out[col + "_cos"] = np.cos(rad)
            out.drop(columns=[col], inplace=True)
        elif base in counters:
            x = pd.to_numeric(out[col], errors="coerce").astype(float)
            diff = x.diff()
            if len(diff) >= 2:
                diff.iloc[0] = diff.iloc[1]
            elif len(diff) == 1:
                diff.iloc[0] = 0.0
            out[col] = diff
    if "label" in out.columns:
        cols = [c for c in out.columns if c != "label"] + ["label"]
        out = out[cols]
    return out


def fill_missing_feature_values(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in DEFAULT_META_COLUMNS]
    if not feature_cols:
        return df

    out = df.copy()
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    missing = out[feature_cols].isna().any()
    cols = missing[missing].index.tolist()
    if not cols:
        return out

    filled = out[cols].ffill()
    if "train_test" in out.columns:
        train_mask = out["train_test"].astype(str).str.strip().eq("train")
    else:
        train_mask = pd.Series(True, index=out.index)
    fallback = out.loc[train_mask, cols].median(numeric_only=True).fillna(0.0)
    for col in cols:
        if filled[col].isna().any():
            filled[col] = filled[col].fillna(float(fallback.get(col, 0.0)))
    out[cols] = filled
    return out


def prepare_wind_event_frame(
    df: pd.DataFrame,
    event_context: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    out = _normalize_train_test_column(df)
    if "label" in out.columns:
        out = out.drop(columns=["label"])

    if event_context is None:
        return out

    out["label"] = _compute_row_labels(out, event_context)
    cols = [c for c in out.columns if c != "label"] + ["label"]
    return out[cols]


def read_wind_event_frame(
    path: str,
    nrows: Optional[int] = None,
    event_context: Optional[Dict[str, Any]] = None,
    standardize: bool = True,
) -> pd.DataFrame:
    data = pd.read_csv(path, sep=";", nrows=nrows if isinstance(nrows, int) else None)

    if not standardize:
        return data

    return prepare_wind_event_frame(data, event_context=event_context)


def summarize_wind_event_file(file_path: str) -> dict:
    data = read_wind_event_frame(file_path, standardize=False)
    file_name = os.path.basename(file_path)
    return {
        "file_name": file_name,
        "freq": "10min",
        "if_univariate": data.shape[1] == 1,
        "size": "user",
        "length": data.shape[0],
        "feature_dim": data.shape[1],
    }
