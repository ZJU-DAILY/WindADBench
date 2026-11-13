from typing import Dict, Optional

import pandas as pd

from tsad_benchmark.data.wind_frame_ops import (
    validate_wind_metadata_table,
    validate_wind_series_frame,
)


def _expected_length_kwargs(metadata: Optional[pd.DataFrame], name: str) -> Dict[str, int]:
    if metadata is None or metadata.empty or name not in metadata.index:
        return {}
    row = metadata.loc[name]

    def _as_int(col: str) -> Optional[int]:
        v = row.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return int(float(v))

    out: Dict[str, int] = {}
    tl = _as_int("train_lens")
    if tl is not None:
        out["expected_train_lens"] = tl
    pl = _as_int("prediction_lens")
    if pl is not None:
        out["expected_prediction_lens"] = pl
    tot = _as_int("total_lens")
    if tot is not None:
        out["expected_total_lens"] = tot
    return out


class Dataset:
    def __init__(self):
        self._metadata = None
        self._data_dict = {}
        self._covariate_dict = {}

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        return self._metadata

    def replace_data_bundle(
        self,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        covariate_dict: Optional[Dict[str, Dict]] = None,
        metadata: Optional[pd.DataFrame] = None,
    ) -> None:
        new_metadata = metadata if metadata is not None else self._metadata
        new_data_dict = data_dict if data_dict is not None else self._data_dict
        new_covariate_dict = (
            covariate_dict if covariate_dict is not None else self._covariate_dict
        )

        self._validate_bundle(new_data_dict, new_covariate_dict, new_metadata)

        self._metadata = new_metadata
        self._data_dict = new_data_dict
        self._covariate_dict = new_covariate_dict

    def _validate_bundle(
        self,
        data_dict: Dict[str, pd.DataFrame],
        covariate_dict: Dict[str, Dict],
        metadata: Optional[pd.DataFrame],
    ) -> None:
        if metadata is not None and not metadata.empty:
            validate_wind_metadata_table(metadata)
        for name, df in data_dict.items():
            validate_wind_series_frame(name, df, **_expected_length_kwargs(metadata, name))

    def merge_series_bundle(
        self,
        inc_data_dict: Dict[str, pd.DataFrame],
        inc_covariate_dict: Dict[str, Dict],
    ) -> None:
        self._validate_series_bundle(inc_data_dict, inc_covariate_dict)
        self._data_dict.update(inc_data_dict)
        self._covariate_dict.update(inc_covariate_dict)

    def _validate_series_bundle(
        self,
        inc_data_dict: Dict[str, pd.DataFrame],
        inc_covariate_dict: Dict[str, Dict],
    ) -> None:
        for name, df in inc_data_dict.items():
            validate_wind_series_frame(name, df, **_expected_length_kwargs(self._metadata, name))

    def clear_cache(self) -> None:
        self._metadata = None
        self._data_dict = {}
        self._covariate_dict = {}

    def series_frame(self, name: str) -> Optional[pd.DataFrame]:
        return self._data_dict.get(name, None)

    def covariates_for_series(self, name: str) -> Optional[Dict]:
        return self._covariate_dict.get(name, None)

    def metadata_for_series(self, name: str) -> Optional[pd.Series]:
        if self._metadata is None or name not in self._metadata.index:
            return None
        return self._metadata.loc[name]

    def has_series_frame(self, name: str) -> bool:
        return name in self._data_dict

    def has_series_metadata(self, name: str) -> bool:
        return self._metadata is not None and name in self._metadata.index

    def snapshot_state(self) -> Dict:
        return {
            "metadata": self._metadata,
            "data_dict": self._data_dict,
            "covariate_dict": self._covariate_dict,
        }

    def restore_state(self, state: Dict) -> None:
        self._metadata = state["metadata"]
        self._data_dict = state["data_dict"]
        self._covariate_dict = state["covariate_dict"]
