# -*- coding: utf-8 -*-
import abc
import base64
import inspect
import json
import logging
import pickle
from functools import cached_property, lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tsad_benchmark.evaluation.evaluator import Evaluator


class ResultCollector:
    def __init__(self):
        self.results = []

    def append_row(self, result: Any) -> None:
        if isinstance(result, list) and result and isinstance(result[0], list):
            self.results.extend(result)
        else:
            self.results.append(result)

    def rows(self) -> List:
        return self.results

    def clear(self) -> None:
        self.results = []

    def size(self) -> int:
        return len(self.results)


class Strategy(metaclass=abc.ABCMeta):
    REQUIRED_CONFIGS = ["strategy_name"]
    DEFAULT_CONFIG_KEY = "__default__"

    def __init__(self, strategy_config: Dict, evaluator: Evaluator):
        self.strategy_config = strategy_config
        self.evaluator = evaluator
        self._validate_strategy_config()

    @abc.abstractmethod
    def run_series(self, series_name: str, model_factory: Any) -> Any:
        pass

    def config_digest(self, required_configs_only: bool = False) -> str:
        if required_configs_only:
            return json.dumps(
                {
                    k: v
                    for k, v in self.strategy_config.items()
                    if k in self.required_config_keys()
                }
            )
        return json.dumps(self.strategy_config, sort_keys=True)

    def _validate_strategy_config(self) -> None:
        provided_args = set(self.strategy_config)
        required_args = set(self.required_config_keys())
        optional_args = set(getattr(self, "OPTIONAL_CONFIGS", set()))
        missing_args = required_args - provided_args
        extra_args = provided_args - required_args - optional_args
        if missing_args:
            raise RuntimeError(f"Missing options: {', '.join(sorted(missing_args))}")
        if extra_args:
            logging.warning("Unknown options: %s", ", ".join(sorted(extra_args)))

    def make_result_collector(self) -> ResultCollector:
        return ResultCollector()

    @classmethod
    @lru_cache(maxsize=1)
    def required_config_keys(cls) -> List[str]:
        ret = []
        for super_cls in inspect.getmro(cls):
            if hasattr(super_cls, "REQUIRED_CONFIGS"):
                ret.extend(super_cls.REQUIRED_CONFIGS)
        return sorted(set(ret))

    @staticmethod
    @abc.abstractmethod
    def metric_names() -> List[str]:
        pass

    @property
    @abc.abstractmethod
    def result_columns(self) -> List[str]:
        pass

    @cached_property
    def _result_column_idx(self) -> Dict:
        return {k: i for i, k in enumerate(self.result_columns)}

    def default_row(self, **kwargs) -> List:
        ret = self.evaluator.default_result()
        ret += [np.nan] * (len(self.result_columns) - len(ret))
        for k, v in kwargs.items():
            if k not in self._result_column_idx:
                raise ValueError(f"Unknown field name {k}")
            ret[self._result_column_idx[k]] = v
        return ret

    def _pack_payload(self, data: Any) -> str:
        encoded = pickle.dumps(data)
        encoded = base64.b64encode(encoded).decode("utf-8")
        return encoded

    def _series_config_value(self, config_name: str, series_name: Optional[str]) -> Any:
        if config_name not in self.strategy_config:
            raise ValueError(f"Missing config {config_name}.")
        config_value = self.strategy_config[config_name]
        if isinstance(config_value, dict):
            if series_name not in config_value and self.DEFAULT_CONFIG_KEY not in config_value:
                raise ValueError(
                    f"Config {config_name} for series {series_name} is missing, add "
                    f"{config_name} or {self.DEFAULT_CONFIG_KEY}."
                )
            return config_value.get(series_name, config_value[self.DEFAULT_CONFIG_KEY])
        return config_value

    def _meta_value(self, meta_info: Optional[pd.Series], field: str, default: Any) -> Any:
        return meta_info[field].item() if meta_info is not None else default
