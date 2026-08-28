# -*- coding: utf-8 -*-
import abc
from typing import Dict, Optional

import pandas as pd


class DataPoolImpl(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def series_frame(self, name: str) -> Optional[pd.DataFrame]:
        pass

    @abc.abstractmethod
    def covariates_for_series(self, name: str) -> Optional[Dict]:
        pass

    @abc.abstractmethod
    def metadata_for_series(self, name: str) -> Optional[pd.Series]:
        pass
