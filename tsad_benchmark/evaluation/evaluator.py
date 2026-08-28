# -*- coding: utf-8 -*-
import functools
import traceback
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd

from tsad_benchmark.evaluation.metrics import METRICS, clear_score_metric_cache


def encode_params(params):
    encoded_pairs = []
    for key, value in sorted(params.items()):
        if isinstance(value, (np.floating, float)):
            value = round(value, 3)
        encoded_pairs.append(f"{key}:{repr(value)}")
    return ";".join(encoded_pairs)


class Evaluator:
    """
    Metric evaluator used by all strategies.

    Notes:
    - Metric implementations are expected to be registered in METRICS.
    - This class is stable API; metric internals can evolve independently.
    """

    def __init__(self, metric: List[dict]):
        self.metric = metric
        self.metric_funcs = []
        self.metric_names = []

        for metric_info in self.metric:
            metric_info_copy = metric_info.copy()
            metric_name = metric_info_copy.pop("name")
            if metric_info_copy:
                metric_name += ";" + encode_params(metric_info_copy)
            self.metric_names.append(metric_name)

            metric_name_copy = metric_info.copy()
            name = metric_name_copy.pop("name")
            if name not in METRICS:
                raise KeyError(f"Metric {name} is not registered")
            fun = METRICS[name]
            self.metric_funcs.append(
                functools.partial(fun, **metric_name_copy)
                if metric_name_copy
                else fun
            )

    def evaluate(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        scaler: object = None,
        hist_data: Union[np.ndarray, pd.DataFrame] = None,
        **kwargs,
    ) -> list:
        if actual.ndim == 3:
            actual = actual.reshape(-1, actual.shape[1])
        if predicted.ndim == 3:
            predicted = predicted.reshape(-1, predicted.shape[1])

        if isinstance(hist_data, pd.DataFrame):
            hist_data_np = hist_data.values
        elif hist_data is None:
            hist_data_np = None
        else:
            hist_data_np = hist_data.reshape(-1, hist_data.shape[1])

        clear_score_metric_cache()
        return [
            m(actual, predicted, scaler=scaler, hist_data=hist_data_np, **kwargs)
            for m in self.metric_funcs
        ]

    def evaluate_with_log(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        scaler: object = None,
        hist_data: Union[np.ndarray, pd.DataFrame] = None,
        **kwargs,
    ) -> Tuple[List[Any], str]:
        if actual.ndim == 3:
            actual = actual.reshape(-1, actual.shape[1])
        if predicted.ndim == 3:
            predicted = predicted.reshape(-1, predicted.shape[1])

        if isinstance(hist_data, pd.DataFrame):
            hist_data = hist_data.values
        elif hist_data is not None and hist_data.ndim == 3:
            hist_data = hist_data.reshape(-1, hist_data.shape[1])

        clear_score_metric_cache()
        evaluate_result = []
        log_info = ""
        for m in self.metric_funcs:
            try:
                evaluate_result.append(
                    m(actual, predicted, scaler=scaler, hist_data=hist_data, **kwargs)
                )
            except Exception as e:
                evaluate_result.append(np.nan)
                log_info += f"Error in {getattr(m, '__name__', str(m))}: {traceback.format_exc()}\n{e}\n"
        return evaluate_result, log_info

    def default_result(self):
        return len(self.metric_names) * [np.nan]
