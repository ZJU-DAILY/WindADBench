# -*- coding: utf-8 -*-
import logging
import threading
from typing import Dict, Optional

import pandas as pd

from tsad_benchmark.data.series_backend import DataPoolImpl
from tsad_benchmark.data.wind_sources import DataSource

logger = logging.getLogger(__name__)


class DatasetPoolImpl(DataPoolImpl):
    def __init__(self, data_source: DataSource) -> None:
        self._data_source = data_source
        self._load_lock = threading.Lock()

    def _ensure_series_cached(self, name: str) -> None:
        if self._data_source.dataset.has_series_frame(name):
            return
        with self._load_lock:
            if not self._data_source.dataset.has_series_frame(name):
                logger.debug("Lazy-loading series: %s", name)
                self._data_source.load_series_batch([name])

    def series_frame(self, name: str) -> Optional[pd.DataFrame]:
        self._ensure_series_cached(name)
        return self._data_source.dataset.series_frame(name)

    def covariates_for_series(self, name: str) -> Optional[Dict]:
        self._ensure_series_cached(name)
        return self._data_source.dataset.covariates_for_series(name)

    def metadata_for_series(self, name: str) -> Optional[pd.Series]:
        return self._data_source.dataset.metadata_for_series(name)

    def warmup_series(self, series_list) -> None:
        unloaded = [n for n in series_list if not self._data_source.dataset.has_series_frame(n)]
        if unloaded:
            logger.info("Pre-loading %d series ...", len(unloaded))
            self._data_source.load_series_batch(unloaded)
