# -*- coding: utf-8 -*-
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import pandas as pd

from tsad_benchmark.data.series_store import Dataset
from tsad_benchmark.data.wind_frame_ops import (
    apply_wind_domain_rules,
    fill_missing_feature_values,
    load_series_covariates,
    read_wind_event_frame,
)

logger = logging.getLogger(__name__)


class DataSource:
    DATASET_CLASS = Dataset

    def __init__(
        self,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        covariate_dict: Optional[Dict[str, Dict]] = None,
        metadata: Optional[pd.DataFrame] = None,
    ):
        self._dataset = self.DATASET_CLASS()
        self._dataset.replace_data_bundle(data_dict or {}, covariate_dict or {}, metadata)

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def load_series_batch(self, series_list: List[str]) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support loading series at runtime."
        )


class LocalDataSource(DataSource):
    _INDEX_COL = "file_name"
    _DATA_FOLDER_NAME = "data"
    _COVARIATES_FOLDER_NAME = "covariates"

    def __init__(self, local_dataset_path: str, metadata_file_name: str):
        self.local_data_path = os.path.join(local_dataset_path, self._DATA_FOLDER_NAME)
        self.local_covariates_path = os.path.join(
            local_dataset_path, self._COVARIATES_FOLDER_NAME
        )
        self.metadata_path = os.path.join(local_dataset_path, metadata_file_name)
        metadata = self.refresh_metadata_index()
        super().__init__({}, {}, metadata)

    def refresh_metadata_index(self) -> pd.DataFrame:
        return self._read_metadata_table()

    def _read_metadata_table(self) -> pd.DataFrame:
        metadata = pd.read_csv(self.metadata_path)
        metadata.set_index(self._INDEX_COL, drop=False, inplace=True)
        return metadata

    def _read_series_frame(self, series_name: str) -> pd.DataFrame:
        datafile_path = os.path.join(self.local_data_path, series_name)
        return read_wind_event_frame(datafile_path)

    def _read_series_covariates(self, series_name: str) -> Optional[Dict]:
        series_name_without_extension = os.path.splitext(series_name)[0]
        covariates_folder_path = os.path.join(
            self.local_covariates_path, series_name_without_extension
        )
        return load_series_covariates(covariates_folder_path)

    def load_series_batch(self, series_list: List[str]) -> None:
        logger.info("Start loading %s series in parallel", len(series_list))

        data_dict = {}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._read_series_frame, series_name)
                for series_name in series_list
            ]
        for future, series_name in zip(futures, series_list):
            data_dict[series_name] = future.result()

        covariate_dict = {}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._read_series_covariates, series_name)
                for series_name in series_list
            ]
        for future, series_name in zip(futures, series_list):
            covariate_dict[series_name] = future.result()

        logger.info("Data loading finished.")
        self.dataset.merge_series_bundle(data_dict, covariate_dict)


class LocalWindFarmDetectDataSource(LocalDataSource):
    def __init__(
        self,
        local_dataset_path: str,
        metadata_file_name: str = "WIND_AD_META.csv",
        domain_preprocessing: bool = True,
    ):
        self._root = local_dataset_path
        self._domain_preprocessing = bool(domain_preprocessing)
        self._feature_desc_cache: Dict[str, Optional[pd.DataFrame]] = {}
        super().__init__(local_dataset_path, metadata_file_name)

    def refresh_metadata_index(self) -> pd.DataFrame:
        if os.path.exists(self.metadata_path):
            logger.info("Loading existing metadata index from %s", self.metadata_path)
            return self._repair_wind_metadata_paths(self._read_metadata_table())

        rows = []
        root = os.path.dirname(self.metadata_path)

        for farm_id in ("A", "B", "C"):
            farm_root = os.path.join(root, f"Wind Farm {farm_id}", f"Wind Farm {farm_id}")
            event_info_path = os.path.join(farm_root, "event_info.csv")
            datasets_root = os.path.join(farm_root, "datasets")
            if not os.path.exists(event_info_path) or not os.path.isdir(datasets_root):
                logger.warning("Skip wind farm %s: missing event_info or datasets folder", farm_id)
                continue

            event_info = pd.read_csv(event_info_path, sep=";")
            for _, event in event_info.iterrows():
                event_id = str(event["event_id"])
                data_file_abs_path = os.path.join(datasets_root, f"{event_id}.csv")
                if not os.path.exists(data_file_abs_path):
                    logger.warning(
                        "Skip event %s_%s: data file not found at %s",
                        farm_id,
                        event_id,
                        data_file_abs_path,
                    )
                    continue

                series_name = f"{farm_id}_{event_id}.csv"
                train_lens = None
                prediction_lens = None
                total_lens = None
                try:
                    split_col = pd.read_csv(
                        data_file_abs_path,
                        sep=";",
                        usecols=["train_test"],
                    )["train_test"].astype(str).str.strip()
                    total_lens = int(split_col.shape[0])
                    train_lens = int((split_col == "train").sum())
                    prediction_lens = int(split_col.isin(["prediction", "test"]).sum())
                except Exception as ex:
                    logger.warning(
                        "Failed to infer train/prediction lens for %s: %s",
                        data_file_abs_path,
                        ex,
                    )

                rows.append(
                    {
                        "file_name": series_name,
                        "farm_id": farm_id,
                        "event_id": int(event["event_id"]),
                        "event_label": event.get("event_label"),
                        "event_start": event.get("event_start"),
                        "event_end": event.get("event_end"),
                        "event_start_id": event.get("event_start_id"),
                        "event_end_id": event.get("event_end_id"),
                        "event_description": event.get("event_description"),
                        "raw_path": data_file_abs_path,
                        "train_lens": train_lens,
                        "prediction_lens": prediction_lens,
                        "total_lens": total_lens,
                    }
                )

        metadata = pd.DataFrame(rows)
        if metadata.empty:
            raise RuntimeError(
                "No wind-farm events found when building metadata. "
                "Please verify raw dataset directory layout."
            )

        metadata.sort_values(by=["farm_id", "event_id"], inplace=True)
        metadata.to_csv(self.metadata_path, index=False)
        metadata.set_index(self._INDEX_COL, drop=False, inplace=True)
        logger.info("Built wind-farm metadata at %s with %s rows", self.metadata_path, len(metadata))
        return metadata

    def _repair_wind_metadata_paths(self, metadata: pd.DataFrame) -> pd.DataFrame:
        metadata = metadata.copy()
        changed = False
        for idx, row in metadata.iterrows():
            farm_id = row.get("farm_id")
            event_id = row.get("event_id")
            if farm_id is None or event_id is None or pd.isna(farm_id) or pd.isna(event_id):
                continue
            expected = os.path.join(
                self._root,
                f"Wind Farm {farm_id}",
                f"Wind Farm {farm_id}",
                "datasets",
                f"{int(event_id)}.csv",
            )
            current = row.get("raw_path")
            if current is None or pd.isna(current) or not os.path.exists(str(current)):
                metadata.at[idx, "raw_path"] = expected
                changed = True

        if changed:
            metadata.to_csv(self.metadata_path, index=False)
            logger.info("Repaired stale raw_path entries in %s", self.metadata_path)
        return metadata

    def _feature_description_for_farm(self, farm_id: str) -> Optional[pd.DataFrame]:
        if farm_id in self._feature_desc_cache:
            return self._feature_desc_cache[farm_id]
        path = os.path.join(
            self._root, f"Wind Farm {farm_id}", f"Wind Farm {farm_id}", "feature_description.csv",
        )
        fd: Optional[pd.DataFrame] = None
        if os.path.exists(path):
            try:
                fd = pd.read_csv(path, sep=";")
            except Exception as exc:
                logger.warning("Failed to read feature_description for farm %s: %s", farm_id, exc)
        self._feature_desc_cache[farm_id] = fd
        return fd

    def _read_series_frame(self, series_name: str) -> pd.DataFrame:
        if self.dataset.metadata is None or series_name not in self.dataset.metadata.index:
            raise KeyError(f"Series {series_name} not found in metadata index")
        row = self.dataset.metadata.loc[series_name]
        datafile_path = row["raw_path"]
        event_context = {
            "event_label": row.get("event_label"),
            "event_start_id": row.get("event_start_id"),
            "event_end_id": row.get("event_end_id"),
        }
        data = read_wind_event_frame(datafile_path, event_context=event_context)
        if self._domain_preprocessing:
            farm_id = str(row.get("farm_id"))
            data = apply_wind_domain_rules(data, self._feature_description_for_farm(farm_id))
        data = fill_missing_feature_values(data)
        return data
