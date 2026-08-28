# -*- coding: utf-8 -*-
from typing import Optional

from tsad_benchmark.data.series_backend import DataPoolImpl


class DataPool:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.pool = None
        return cls._instance

    def register_backend(self, pool: DataPoolImpl) -> None:
        self.pool = pool

    def backend(self) -> Optional[DataPoolImpl]:
        return self.pool
