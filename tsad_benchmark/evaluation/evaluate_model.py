# -*- coding: utf-8 -*-
import functools
import json
import logging
import os
import traceback
from dataclasses import dataclass, field
from typing import Callable, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

from tsad_benchmark.data.series_registry import DataPool
from tsad_benchmark.evaluation.evaluator import Evaluator
from tsad_benchmark.evaluation.metrics import classification_metrics_score
from tsad_benchmark.evaluation.strategy import STRATEGY
from tsad_benchmark.evaluation.strategy.constants import FieldNames
from tsad_benchmark.evaluation.strategy.strategy import Strategy

logger = logging.getLogger(__name__)

# Wind-farm metadata columns to join into every result row.
_WIND_META_COLS = [
    FieldNames.FARM_ID,
    FieldNames.EVENT_ID,
    FieldNames.EVENT_LABEL,
    FieldNames.EVENT_DESCRIPTION,
    FieldNames.TRAIN_LENS,
    FieldNames.TEST_LENS,
    FieldNames.TOTAL_LENS,
    FieldNames.ANOMALY_SPAN_LEN,
]

_NORMAL_EVENT_METRICS = {
    "accuracy",
    "false_alarms_per_turbine_day",
    "mtbfa",
    "correct_points",
    "total_points",
    "false_alarm_events",
    "turbine_days",
}

_DEFERRED_SCORE_VUS_METRICS = ["vus_pr", "vus_roc"]


def _row_strategy_args(row: pd.Series) -> dict:
    raw = row.get(FieldNames.STRATEGY_ARGS)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _safe_execute(fn: Callable, args: Tuple, default_row: Callable):
    try:
        return fn(*args)
    except Exception as e:
        log = f"{traceback.format_exc()}\n{e}"
        return default_row(**{FieldNames.LOG_INFO: log})


def _normalise_metrics(metric_config, strategy_class) -> List[dict]:
    if metric_config == "all":
        metric = list(strategy_class.metric_names())
    elif isinstance(metric_config, (str, dict)):
        metric = [metric_config]
    else:
        metric = list(metric_config)
    return [{"name": m} if isinstance(m, str) else dict(m) for m in metric]


def _is_score_strategy(strategy_class) -> bool:
    names = set(strategy_class.metric_names())
    return set(classification_metrics_score.__all__).issubset(names)


def _split_deferred_score_vus(
    metric: List[dict],
    strategy_class,
    evaluation_config: dict,
) -> Tuple[List[dict], List[str]]:
    if not evaluation_config.get("defer_score_vus", False):
        return metric, []
    if not _is_score_strategy(strategy_class):
        return metric, []

    selected = {m.get("name") for m in metric}
    deferred = [m for m in _DEFERRED_SCORE_VUS_METRICS if m in selected]
    if not deferred:
        return metric, []
    immediate = [m for m in metric if m.get("name") not in set(deferred)]
    return immediate, deferred


def _insert_deferred_metric_columns(result_df: pd.DataFrame, strategy: Strategy) -> None:
    deferred = list(getattr(strategy, "deferred_metric_names", []) or [])
    if not deferred:
        return
    insert_at = (
        result_df.columns.get_loc(FieldNames.MODEL_PARAMS)
        if FieldNames.MODEL_PARAMS in result_df.columns
        else len(result_df.columns)
    )
    for metric_name in deferred:
        if metric_name in result_df.columns:
            continue
        result_df.insert(insert_at, metric_name, np.nan)
        insert_at += 1


# ---------------------------------------------------------------------------
# Backend abstraction (sync now, parallel-ready later)
# ---------------------------------------------------------------------------

class _SyncTaskResult:
    def __init__(self, fn: Callable):
        self._fn = fn

    def result(self):
        return self._fn()


class _SyncBackend:
    """
    Synchronous execution backend.

    Replace ``schedule`` with a thread / process pool submission to enable
    parallel evaluation without changing any call-sites.
    """

    def schedule(self, fn: Callable, args: Tuple) -> _SyncTaskResult:
        return _SyncTaskResult(lambda: fn(*args))


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    strategy: Strategy
    result_list: List
    model_factory: object
    series_list: List[str]
    #: Flush a DataFrame to disk every *batch_size* collected series.
    batch_size: int = 100_000

    def collect(self) -> Generator[pd.DataFrame, None, None]:
        """
        Iterate over scheduled tasks, build DataFrames in batches and yield them.

        Each yielded DataFrame has all columns defined by ``FieldNames.all_fields()``
        plus any metric columns produced by the strategy.
        """
        collector = self.strategy.make_result_collector()
        strict_errors = bool(self.strategy.strategy_config.get("strict_errors", True))
        for i, result in enumerate(self.result_list):
            if strict_errors:
                collector.append_row(result.result())
            else:
                collector.append_row(
                    _safe_execute(
                        result.result,
                        (),
                        functools.partial(
                            self.strategy.default_row,
                            **{FieldNames.FILE_NAME: self.series_list[i]},
                        ),
                    )
                )
            if collector.size() >= self.batch_size:
                yield build_result_df(collector.rows(), self.model_factory, self.strategy)
                collector.clear()

        if collector.size() > 0:
            yield build_result_df(collector.rows(), self.model_factory, self.strategy)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def eval_model(
    model_factory,
    series_list: list,
    evaluation_config: dict,
) -> EvalResult:
    """
    Build an :class:`EvalResult` that lazily executes the evaluation.

    :param model_factory:      A callable (or class) that returns a model instance.
                               Must expose ``model_name`` and ``model_hyper_params``
                               as class/instance attributes.
    :param series_list:        Names of series to evaluate (must be registered in DataPool).
    :param evaluation_config:  Dict with keys:
                               - ``strategy_args``: dict, must include ``strategy_name``.
                               - ``metrics``: list of metric dicts, ``"all"``, or a single name.
    :return: :class:`EvalResult`
    """
    strategy_class = STRATEGY.get(evaluation_config["strategy_args"]["strategy_name"])
    if strategy_class is None:
        raise RuntimeError(
            f"Unknown strategy: {evaluation_config['strategy_args']['strategy_name']}. "
            f"Available: {sorted(STRATEGY)}"
        )

    metric = _normalise_metrics(evaluation_config.get("metrics", "all"), strategy_class)
    invalid_metrics = [
        m["name"] for m in metric if m.get("name") not in strategy_class.metric_names()
    ]
    if invalid_metrics:
        raise RuntimeError(f"Metrics not accepted by {strategy_class.__name__}: {invalid_metrics}")

    metric, deferred_metric_names = _split_deferred_score_vus(
        metric,
        strategy_class,
        evaluation_config,
    )
    evaluator = Evaluator(metric)
    strategy = strategy_class(evaluation_config["strategy_args"], evaluator)
    strategy.deferred_metric_names = deferred_metric_names

    backend = _SyncBackend()
    result_list = [
        backend.schedule(strategy.run_series, (series_name, model_factory))
        for series_name in series_list
    ]
    return EvalResult(strategy, result_list, model_factory, series_list)


def build_result_df(
    result_list: List,
    model_factory,
    strategy: Strategy,
) -> pd.DataFrame:
    """
    Convert raw result rows into a fully annotated DataFrame.

    Steps:
    1. Build base DataFrame from strategy field names.
    2. Inject model / strategy meta columns.
    3. Join wind-farm metadata (farm_id, event_id, event_label, etc.) from DataPool.
    4. Validate all required FieldNames are present.
    """
    result_df = pd.DataFrame(result_list, columns=strategy.result_columns)
    _insert_deferred_metric_columns(result_df, strategy)

    # Inject model / strategy identity columns.
    if FieldNames.MODEL_PARAMS not in result_df.columns:
        result_df.insert(
            0,
            FieldNames.MODEL_PARAMS,
            json.dumps(getattr(model_factory, "model_hyper_params", {}), sort_keys=True),
        )
    else:
        model_params = result_df.pop(FieldNames.MODEL_PARAMS)
        result_df.insert(0, FieldNames.MODEL_PARAMS, model_params)
    result_df.insert(0, FieldNames.STRATEGY_ARGS, strategy.config_digest())
    result_df.insert(0, FieldNames.MODEL_NAME, getattr(model_factory, "model_name", "unknown"))

    # Join wind-farm metadata columns from DataPool.
    _inject_wind_meta(result_df)
    _mask_normal_event_metrics(result_df)

    missing_fields = set(FieldNames.all_fields()) - set(result_df.columns)
    if missing_fields:
        raise ValueError(f"Missing required fields: {sorted(missing_fields)}")
    return result_df


def _mask_normal_event_metrics(result_df: pd.DataFrame) -> None:
    """Keep only metrics that are meaningful on normal events.

    Detection, event, score-curve, and early-warning metrics require at least
    one ground-truth anomaly.  Normal wind-farm events are retained for
    point-level accuracy and false-alarm robustness only.
    """
    if FieldNames.EVENT_LABEL not in result_df.columns:
        return

    normal_mask = (
        result_df[FieldNames.EVENT_LABEL]
        .astype(str)
        .str.lower()
        .str.strip()
        .eq("normal")
    )
    if not normal_mask.any():
        return

    from tsad_benchmark.evaluation.metrics import (
        classification_metrics_label,
        classification_metrics_score,
    )

    metric_names = set(classification_metrics_label.__all__) | set(
        classification_metrics_score.__all__
    )
    for col in result_df.columns:
        base_name = str(col).split(";", 1)[0]
        if base_name in metric_names and base_name not in _NORMAL_EVENT_METRICS:
            result_df.loc[normal_mask, col] = np.nan


def _inject_wind_meta(result_df: pd.DataFrame) -> None:
    """
    For every row in *result_df*, look up wind-farm meta fields from DataPool
    and add them as columns (in-place).  Missing values are filled with None.

    Derived columns:
    - train_lens / test_lens describe the segment actually evaluated by
      the active strategy.  For ``unfixed_detect_*`` this is the dataset's
      original train / prediction split; for ``fixed_detect_*`` and
      ``all_detect_*`` it is recomputed from strategy_args.
    - anomaly_span_len = event_end_id - event_start_id + 1  (None for normal)
    """
    pool = DataPool().backend()

    for col in _WIND_META_COLS:
        if col not in result_df.columns:
            result_df[col] = None

    if pool is None or FieldNames.FILE_NAME not in result_df.columns:
        return

    for idx, row in result_df.iterrows():
        file_name = row.get(FieldNames.FILE_NAME)
        if not file_name:
            continue
        meta = pool.metadata_for_series(str(file_name))
        if meta is None:
            continue

        for col in (FieldNames.FARM_ID, FieldNames.EVENT_ID,
                    FieldNames.EVENT_LABEL, FieldNames.EVENT_DESCRIPTION):
            if col in meta.index:
                result_df.at[idx, col] = meta[col]

        if "train_lens" in meta.index:
            result_df.at[idx, FieldNames.TRAIN_LENS] = meta["train_lens"]
        if "prediction_lens" in meta.index:
            result_df.at[idx, FieldNames.TEST_LENS] = meta["prediction_lens"]
        if "total_lens" in meta.index:
            result_df.at[idx, FieldNames.TOTAL_LENS] = meta["total_lens"]

        strategy_args = _row_strategy_args(row)
        strategy_name = str(strategy_args.get("strategy_name", "")).lower()
        try:
            total = meta.get("total_lens")
            if total is not None and not (isinstance(total, float) and pd.isna(total)):
                total_i = int(float(total))
                if strategy_name.startswith("fixed_detect"):
                    split_ratio = float(strategy_args["train_test_split"])
                    split_idx = max(0, min(int(total_i * split_ratio), total_i))
                    result_df.at[idx, FieldNames.TRAIN_LENS] = split_idx
                    result_df.at[idx, FieldNames.TEST_LENS] = total_i - split_idx
                elif strategy_name.startswith("all_detect"):
                    result_df.at[idx, FieldNames.TRAIN_LENS] = total_i
                    result_df.at[idx, FieldNames.TEST_LENS] = total_i
        except (KeyError, TypeError, ValueError):
            pass

        try:
            sid = meta.get("event_start_id")
            eid = meta.get("event_end_id")
            el = meta.get("event_label")
            is_normal = (
                el is not None
                and not (isinstance(el, float) and pd.isna(el))
                and str(el).lower().strip() == "normal"
            )
            if not is_normal and sid is not None and eid is not None:
                if not (isinstance(sid, float) and pd.isna(sid)):
                    result_df.at[idx, FieldNames.ANOMALY_SPAN_LEN] = (
                        int(float(eid)) - int(float(sid)) + 1
                    )
        except (TypeError, ValueError):
            pass


def save_result(
    result: EvalResult,
    output_path: str,
    append: bool = False,
) -> None:
    """
    Collect all batches from *result* and write to *output_path* (CSV).

    :param result:      EvalResult returned by :func:`eval_model`.
    :param output_path: Destination file path (will be created/overwritten).
    :param append:      If True and the file already exists, append without header.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    first_batch = True
    for df in result.collect():
        if first_batch:
            if append and os.path.exists(output_path):
                mode, write_header = "a", False
            else:
                mode, write_header = "w", True
            first_batch = False
        else:
            mode, write_header = "a", False
        df.to_csv(output_path, mode=mode, header=write_header, index=False)
    logger.info("Results saved to %s", output_path)
