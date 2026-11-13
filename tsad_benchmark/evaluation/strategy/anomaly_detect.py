import logging
import gc
import json
import os
import time
import traceback
from typing import Any, List

import numpy as np
import pandas as pd
import threading

try:

    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None

try:

    import torch as _torch  # type: ignore
except Exception:  # pragma: no cover
    _torch = None

from tsad_benchmark.common.random_utils import DEFAULT_SEED, fix_random_seed, preserve_random_state
from tsad_benchmark.common.operating_points import TRACK1_ANOMALY_RATIOS
from tsad_benchmark.data.series_registry import DataPool
from tsad_benchmark.data.wind_frame_ops import feature_columns_only
from tsad_benchmark.evaluation.metrics import (
    classification_metrics_label,
    classification_metrics_score,
)
from tsad_benchmark.evaluation.strategy.constants import FieldNames
from tsad_benchmark.evaluation.strategy.strategy import Strategy
from tsad_benchmark.evaluation.evaluator import Evaluator
from tsad_benchmark.models import AnomalyModelBase
from tsad_benchmark.evaluation.thresholding import (
    validate_scores,
)
from tsad_benchmark.evaluation.online_protocol import (
    ALARM_POLICY, THRESHOLD_METHOD as TRAIN_PERCENTILE_METHOD,
    WARNING_HORIZON_POINTS, WARNING_DELTA_POINTS,
    prepare_scores, calibrate_scores, evaluate_threshold, curve_scores,
)


class AnomalyDetect(Strategy):
    OPTIONAL_CONFIGS = {
        "strict_length",
        "strict_errors",
        "seed",
        "lead_horizon_points",
        "lead_delta_points",
        "evaluation_protocol",
    }

    USES_TRAINING_THRESHOLD = False

    def config_digest(self, required_configs_only: bool = False) -> str:
        config = json.loads(super().config_digest(required_configs_only))
        config.update(
            lead_horizon_points=self.strategy_config.get('lead_horizon_points', WARNING_HORIZON_POINTS),
            lead_delta_points=self.strategy_config.get('lead_delta_points', WARNING_DELTA_POINTS),
        )
        if self.USES_TRAINING_THRESHOLD:
            config.update(
                threshold_method=TRAIN_PERCENTILE_METHOD,
                anomaly_ratios=list(TRACK1_ANOMALY_RATIOS),
                alarm_policy=ALARM_POLICY,
            )
        return json.dumps(config, sort_keys=True)

    def _fail_fast(self) -> bool:
        return bool(self.strategy_config.get("strict_errors", True))

    class _ResourceMonitor:
        def __init__(self, interval_sec: float = 0.05):
            self.interval_sec = interval_sec
            self._running = False
            self._thread = None
            self._stop_event = threading.Event()
            self.peak_gpu_bytes = 0
            self._cpu_time_start = None
            self._cpu_time_end = None
            self._wall_start = None
            self._wall_end = None
            self._nvml_handle = None
            self._torch_cuda = False
            self._pid = os.getpid()

        @staticmethod
        def _nvml_pid_memory(handle, pid: int) -> int:
            used = 0
            for getter in (
                getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None),
                getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None),
            ):
                if getter is None:
                    continue
                try:
                    for proc in getter(handle):
                        if int(getattr(proc, "pid", -1)) == pid:
                            value = int(getattr(proc, "usedGpuMemory", 0) or 0)

                            if 0 <= value < (1 << 63):
                                used = max(used, value)
                except Exception:
                    pass
            return used

        @staticmethod
        def _torch_cuda_available() -> bool:
            if _torch is None:
                return False
            try:
                return bool(_torch.cuda.is_available())
            except Exception:
                return False

        def start(self):
            if self._torch_cuda_available():
                try:
                    _torch.cuda.synchronize()
                    _torch.cuda.reset_peak_memory_stats()
                    self._torch_cuda = True
                except Exception:
                    self._torch_cuda = False

            if not self._torch_cuda and pynvml is not None:
                try:
                    pynvml.nvmlInit()
                    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                    gpu_id = (visible.split(",")[0].strip() if visible is not None
                              else os.environ.get("TSAD_GPU_INDEX", "0"))
                    if gpu_id.startswith(("GPU-", "MIG-")):
                        self._nvml_handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_id)
                    else:
                        self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_id))
                except Exception:
                    self._nvml_handle = None
                    try:
                        pynvml.nvmlShutdown()
                    except Exception:
                        pass

            if self._nvml_handle is not None:
                self._running = True

                def _sample():
                    while self._running:
                        if self._nvml_handle is not None:
                            try:
                                used = self._nvml_pid_memory(self._nvml_handle, self._pid)
                                if used > self.peak_gpu_bytes:
                                    self.peak_gpu_bytes = used
                            except Exception:
                                pass
                        self._stop_event.wait(self.interval_sec)

                self._thread = threading.Thread(target=_sample, daemon=True)
                self._thread.start()

            self._cpu_time_start = time.process_time()
            self._wall_start = time.perf_counter()

        def stop(self):
            self._wall_end = time.perf_counter()
            self._cpu_time_end = time.process_time()
            self._running = False
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)

            if self._torch_cuda:
                try:
                    _torch.cuda.synchronize()
                    self.peak_gpu_bytes = int(_torch.cuda.max_memory_allocated())
                except Exception:
                    pass

            if pynvml is not None and self._nvml_handle is not None:
                try:
                    self.peak_gpu_bytes = max(
                        self.peak_gpu_bytes,
                        self._nvml_pid_memory(self._nvml_handle, self._pid),
                    )
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

        def peak_memory_mb(self):
            return np.nan

        def peak_gpu_memory_mb(self):
            if self.peak_gpu_bytes <= 0:
                return np.nan
            return float(self.peak_gpu_bytes / (1024.0 * 1024.0))

        def cpu_usage_percent(self):
            if self._cpu_time_start is None or self._cpu_time_end is None:
                return np.nan
            wall = max((self._wall_end or 0) - (self._wall_start or 0), 1e-9)
            cpu_delta = max(self._cpu_time_end - self._cpu_time_start, 0.0)
            ncpu = os.cpu_count()
            ncpu = max(int(ncpu or 1), 1)
            return float((cpu_delta / wall) * 100.0 / ncpu)

    @staticmethod
    def _synchronize_torch_cuda() -> None:
        if _torch is None:
            return
        try:
            if _torch.cuda.is_available():
                _torch.cuda.synchronize()
        except Exception:
            pass

    @staticmethod
    def _model_scalar_stat(model, name: str) -> float:
        try:
            fn = getattr(model, name, None)
            if not callable(fn):
                return np.nan
            value = float(fn())
            not_applicable = (
                name == "estimate_n_params"
                and getattr(fn, "__func__", None) is AnomalyModelBase.estimate_n_params
            )
            if not np.isfinite(value) or value <= 0:
                if not not_applicable:
                    logging.getLogger(__name__).warning(
                        "[%s] %s returned %s; cost statistic is missing.",
                        model.model_name, name, value,
                    )
                return np.nan
            return value
        except Exception:
            logging.getLogger(__name__).warning(
                "[%s] %s failed; cost statistic is missing.",
                getattr(model, "model_name", type(model).__name__), name, exc_info=True,
            )
            return np.nan

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    @classmethod
    def _effective_model_params(cls, model, model_factory: Any) -> str:
        params = getattr(model, "model_hyper_params", None)
        if not isinstance(params, dict) or not params:
            params = getattr(model_factory, "model_hyper_params", {})
        try:
            return json.dumps(params or {}, sort_keys=True, default=cls._json_default)
        except Exception:
            return json.dumps({}, sort_keys=True)

    def _score_output(self, test_data: pd.DataFrame):
        cap = self.model.capability
        if cap.has_score_output():
            return self.model.detect_score(test_data)
        raise RuntimeError(
            f"[{self.model.model_name}] declares no score output. "
            "Implement detect_score or use a label-track strategy."
        )

    def _label_output(self, test_data: pd.DataFrame, test_label: pd.DataFrame = None):
        if not self.model.capability.has_label_output():
            raise RuntimeError(
                f"[{self.model.model_name}] declares no label output. "
                "Implement detect_label or use a score-track strategy."
            )
        return self.model.detect_label(test_data)

    def _threshold_output(self, test_data: pd.DataFrame):
        scores = validate_scores(self._score_output(test_data), len(test_data), "test")
        self._prepared_test = prepare_scores(
            scores, self.model.model_name, self.model.model_hyper_params,
        )
        predictions, self._warning_maps = {}, {}
        for ratio, threshold in self._label_thresholds.items():
            predictions[ratio], self._warning_maps[ratio] = evaluate_threshold(self._prepared_test, threshold)
        return predictions, curve_scores(self._prepared_test, getattr(self, '_score_floor', -1e30))

    def _calibrate_labels(self, train_data, train_label):
        with preserve_random_state():
            scores = validate_scores(
                self._score_output(train_data), len(train_data), "training"
            )
        self._prepared_train = prepare_scores(
            scores, self.model.model_name, self.model.model_hyper_params,
        )
        self._label_thresholds, self._score_floor, self._calibration_valid_rows = calibrate_scores(
            self._prepared_train, train_label,
        )
        return scores

    def run_series(self, series_name: str, model_factory: Any) -> Any:
        gc.collect()
        seed = self.strategy_config.get("seed", DEFAULT_SEED)
        fix_random_seed(None if seed is None else int(seed))
        model = model_factory()
        try:
            self.model = model
            train_data, train_label, test_data, test_label = self.make_train_test_view(series_name)
            if self.USES_TRAINING_THRESHOLD and len(train_data) == 0:
                raise ValueError("training data is empty; train-only calibration requires training scores.")

            fit_monitor = self._ResourceMonitor()
            fit_monitor.start()
            start_fit_time = time.perf_counter()
            try:
                model.fit(train_data, train_label)
                if self.USES_TRAINING_THRESHOLD:
                    train_scores = self._calibrate_labels(train_data, train_label)
            finally:
                self._synchronize_torch_cuda()
                end_fit_time = time.perf_counter()
                fit_monitor.stop()
            model_params = self._effective_model_params(model, model_factory)
            if self.USES_TRAINING_THRESHOLD:

                packed_train_scores = self._pack_payload(train_scores)
                packed_train_labels = self._pack_payload(train_label)

            infer_monitor = self._ResourceMonitor()
            infer_monitor.start()
            start_inference_time = time.perf_counter()
            try:
                detect_out = self.run_inference(test_data, test_label=test_label)
            finally:
                self._synchronize_torch_cuda()
                end_inference_time = time.perf_counter()
                infer_monitor.stop()

            fit_peak_memory = fit_monitor.peak_memory_mb()
            infer_peak_memory = infer_monitor.peak_memory_mb()
            fit_cpu_usage = fit_monitor.cpu_usage_percent()
            infer_cpu_usage = infer_monitor.cpu_usage_percent()
            fit_gpu_peak_memory = fit_monitor.peak_gpu_memory_mb()
            infer_gpu_peak_memory = infer_monitor.peak_gpu_memory_mb()

            flops = np.nan
            n_params = self._model_scalar_stat(model, "estimate_n_params")
            model_size_mb = self._model_scalar_stat(model, "estimate_model_size_mb")
            predict_map, aux_map = self._to_prediction_maps(detect_out)

            actual = test_label.to_numpy().reshape(-1).astype(float)
            care = self.strategy_config.get('evaluation_protocol')
            if care:
                from tsad_benchmark.evaluation import care_protocol
                if care != care_protocol.PROTOCOL or not self.USES_TRAINING_THRESHOLD:
                    raise ValueError('CARE evaluation requires the current score protocol.')
                if (self.strategy_config.get('lead_horizon_points', 288),
                        self.strategy_config.get('lead_delta_points', 72)) != (288, 72):
                    raise ValueError('The current CARE protocol requires H=288 and delta=72.')
                backend = DataPool().backend()
                meta = backend.metadata_for_series(series_name)
                frame = backend.series_frame(series_name).iloc[int(meta['train_lens']):]
                if len(frame) != len(actual) or not np.array_equal(frame['label'], actual):
                    raise ValueError('CARE evaluation requires the original train/test split.')
                farm = str(meta['farm_id'])
                valid = care_protocol.evaluation_mask(farm, frame['status_type_id'].to_numpy())
                score_metrics = care_protocol.score_metrics(
                    actual, next(iter(aux_map.values())), valid, farm, str(meta['event_label']),
                    available=self._prepared_test['available'].any(),
                )
                care_support = care_protocol.timing_support(self._prepared_test, actual, valid)
            elif self.USES_TRAINING_THRESHOLD:
                score_evaluator = Evaluator([
                    metric for metric in self.evaluator.metric
                    if metric["name"] in classification_metrics_score.__all__
                ])
                label_evaluator = Evaluator([
                    metric for metric in self.evaluator.metric
                    if metric["name"] in classification_metrics_label.__all__
                ])
                if self._prepared_test['available'].any():
                    score_values, score_log = score_evaluator.evaluate_with_log(
                        actual=actual, predicted=next(iter(aux_map.values())),
                    )
                else:
                    score_values, score_log = [np.nan] * len(score_evaluator.metric_names), ''
                score_metrics = dict(zip(score_evaluator.metric_names, score_values))
            results = []
            for ratio, pred in predict_map.items():
                pred_arr = np.asarray(pred).reshape(-1).astype(float)
                aux_arr = np.asarray(aux_map.get(ratio, np.array([]))).reshape(-1)
                strict = self.strategy_config.get("strict_length", True)
                pred_arr, aux_arr = self._match_result_length(
                    pred_arr, aux_arr, len(actual),
                    strict=bool(strict), series_name=series_name,
                )

                if care:
                    measured = care_protocol.in_farm_labels(
                        actual, pred_arr, self._warning_maps[ratio], valid, farm,
                    )
                    measured.update(score_metrics)
                    measured.update(care_support)
                    metric_values = [measured[name] for name in self.evaluator.metric_names]
                    extra = {key: value for key, value in measured.items()
                             if key not in self.evaluator.metric_names}
                    payload = [aux_arr, pred_arr]
                    log_info = ''
                elif self.USES_TRAINING_THRESHOLD:
                    label_values, label_log = label_evaluator.evaluate_with_log(
                        actual=actual, predicted=pred_arr,
                        warning_predictions=self._warning_maps[ratio],
                        lead_horizon_points=self.strategy_config.get('lead_horizon_points', WARNING_HORIZON_POINTS),
                        lead_delta_points=self.strategy_config.get('lead_delta_points', WARNING_DELTA_POINTS),
                    )
                    metrics = dict(score_metrics)
                    metrics.update(zip(label_evaluator.metric_names, label_values))
                    metric_values = [metrics[name] for name in self.evaluator.metric_names]
                    log_info = score_log + label_log
                    payload = [aux_arr, pred_arr]
                else:
                    metric_values, log_info = self.evaluator.evaluate_with_log(
                        actual=actual, predicted=pred_arr,
                        lead_horizon_points=self.strategy_config.get('lead_horizon_points', WARNING_HORIZON_POINTS),
                        lead_delta_points=self.strategy_config.get('lead_delta_points', WARNING_DELTA_POINTS),
                    )
                    payload = [pred_arr, aux_arr]
                if self._fail_fast() and log_info:
                    raise RuntimeError(
                        f"Metric evaluation failed for series {series_name}:\n"
                        f"{log_info}"
                    )
                results.append(
                    metric_values
                    + [
                        model_params,
                        series_name,
                        end_fit_time - start_fit_time,
                        end_inference_time - start_inference_time,
                        flops,
                        n_params,
                        model_size_mb,
                        fit_peak_memory,
                        infer_peak_memory,
                        fit_cpu_usage,
                        infer_cpu_usage,
                        fit_gpu_peak_memory,
                        infer_gpu_peak_memory,
                        ratio,
                        self._pack_payload(test_label),
                        self._pack_payload(payload),
                        *([
                            packed_train_scores,
                            packed_train_labels,
                            self._label_thresholds[ratio],
                            TRAIN_PERCENTILE_METHOD,
                            self._pack_payload(self._prepared_test['raw']),
                            self._pack_payload(self._prepared_test['release']),
                            self._pack_payload(self._warning_maps[ratio]),
                            ALARM_POLICY,
                            self._prepared_test['status']['status'],
                            float(self._prepared_test['available'].mean()),
                            self._calibration_valid_rows,
                            self._score_floor,
                        ] if self.USES_TRAINING_THRESHOLD else []),
                        *([json.dumps(extra, default=self._json_default)] if care else []),
                        log_info,
                    ]
                )
            return results
        except Exception as e:
            if self._fail_fast():
                raise
            log = f"The error series is: {series_name}\n{traceback.format_exc()}\n{e}"
            return [
                self.default_row(
                    **{
                        FieldNames.MODEL_PARAMS: self._effective_model_params(model, model_factory),
                        FieldNames.FILE_NAME: series_name,
                        FieldNames.LOG_INFO: log,
                    }
                )
            ]
        finally:

            self.model = None

    @staticmethod
    def _split_at_row(data: pd.DataFrame, idx: int):
        idx = max(0, min(int(idx), len(data)))
        return data.iloc[:idx].copy(), data.iloc[idx:].copy()

    @staticmethod
    def _match_result_length(
        pred: np.ndarray,
        aux: np.ndarray,
        target_len: int,
        strict: bool = True,
        series_name: str = "",
    ):
        _logger = logging.getLogger(__name__)

        if len(pred) != target_len:
            msg = (
                f"[{series_name}] Model output length ({len(pred)}) "
                f"!= expected ({target_len})."
            )
            if strict:
                raise ValueError(msg + " Set strict_length=false to auto-fix.")
            _logger.warning("%s Auto-aligning (pad/truncate).", msg)
            if len(pred) < target_len:
                pred = np.pad(pred, (0, target_len - len(pred)),
                              mode="constant", constant_values=0)
            else:
                pred = pred[:target_len]

        if len(aux) == 0:
            aux = np.zeros(target_len, dtype=float)
        elif len(aux) != target_len:
            if len(aux) < target_len:
                aux = np.pad(aux, (0, target_len - len(aux)),
                             mode="constant", constant_values=0)
            else:
                aux = aux[:target_len]
        return pred, aux

    @staticmethod
    def _to_prediction_maps(detect_out):
        if isinstance(detect_out, tuple):
            pred_out, aux_out = detect_out
        else:
            pred_out, aux_out = detect_out, None

        if isinstance(pred_out, dict):
            predict_map = {str(k): np.asarray(v) for k, v in pred_out.items()}
            if isinstance(aux_out, dict):
                aux_map = {str(k): np.asarray(v) for k, v in aux_out.items()}
            else:
                aux_map = {str(k): np.asarray(aux_out if aux_out is not None else np.array([])) for k in predict_map}
        else:
            key = "None"
            predict_map = {key: np.asarray(pred_out)}
            aux_map = {key: np.asarray(aux_out if aux_out is not None else np.array([]))}
        return predict_map, aux_map

    def make_train_test_view(self, series_name: str):
        raise NotImplementedError

    def run_inference(self, test_data: pd.DataFrame, test_label: pd.DataFrame = None):
        raise NotImplementedError

    @staticmethod
    def metric_names():
        raise NotImplementedError

    @property
    def result_columns(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.MODEL_PARAMS,
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.FLOPS,
            FieldNames.N_PARAMS,
            FieldNames.MODEL_SIZE_MB,
            FieldNames.FIT_PEAK_MEMORY,
            FieldNames.INFER_PEAK_MEMORY,
            FieldNames.FIT_CPU_USAGE,
            FieldNames.INFER_CPU_USAGE,
            FieldNames.FIT_GPU_PEAK_MEMORY,
            FieldNames.INFER_GPU_PEAK_MEMORY,
            FieldNames.ANOMALY_RATIO,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            *([
                FieldNames.TRAIN_SCORE_DATA,
                FieldNames.TRAIN_ACTUAL_DATA,
                FieldNames.THRESHOLD,
                FieldNames.THRESHOLD_METHOD,
                'raw_score_data',
                'score_release_data',
                'warning_prediction_data',
                'alarm_policy',
                'score_status',
                'score_availability_fraction',
                'calibration_valid_rows',
                'unavailable_score_floor',
            ] if self.USES_TRAINING_THRESHOLD else []),
            *(['care_evaluation_details'] if self.strategy_config.get('evaluation_protocol') else []),
            FieldNames.LOG_INFO,
        ]


class FixedDetectScore(AnomalyDetect):
    USES_TRAINING_THRESHOLD = True
    REQUIRED_CONFIGS = ["train_test_split"]

    def make_train_test_view(self, series_name):
        data = DataPool().backend().series_frame(series_name)
        if "label" not in data.columns:
            raise ValueError("Series must contain `label` column for anomaly detection.")
        split_ratio = float(self._series_config_value("train_test_split", series_name))
        split_idx = int(len(data) * split_ratio)
        train, test = self._split_at_row(data, split_idx)
        train_label = train.loc[:, ["label"]]
        test_label = test.loc[:, ["label"]]
        train_data = feature_columns_only(train)
        test_data = feature_columns_only(test)
        return train_data, train_label, test_data, test_label

    def run_inference(self, test_data, test_label=None):
        return self._threshold_output(test_data)

    @staticmethod
    def metric_names():
        return classification_metrics_score.__all__ + classification_metrics_label.__all__


class FixedDetectLabel(FixedDetectScore):
    USES_TRAINING_THRESHOLD = False

    def run_inference(self, test_data, test_label=None):
        return self._label_output(test_data, test_label=test_label)

    @staticmethod
    def metric_names():
        return classification_metrics_label.__all__


class UnFixedDetectScore(AnomalyDetect):
    USES_TRAINING_THRESHOLD = True

    def make_train_test_view(self, series_name):
        data = DataPool().backend().series_frame(series_name)
        meta_info = DataPool().backend().metadata_for_series(series_name)
        if "label" not in data.columns:
            raise ValueError("Series must contain `label` column for anomaly detection.")
        if meta_info is None or "train_lens" not in meta_info:
            raise ValueError("UnFixed strategy requires `train_lens` in metadata.")

        split_idx = int(meta_info["train_lens"])
        data = data.reset_index(drop=True)
        train, test = self._split_at_row(data, split_idx)
        train_label = train.loc[:, ["label"]]
        test_label = test.loc[:, ["label"]]
        train_data = feature_columns_only(train)
        test_data = feature_columns_only(test)
        return train_data, train_label, test_data, test_label

    def run_inference(self, test_data, test_label=None):
        return self._threshold_output(test_data)

    @staticmethod
    def metric_names():
        return classification_metrics_score.__all__ + classification_metrics_label.__all__


class UnFixedDetectLabel(UnFixedDetectScore):
    USES_TRAINING_THRESHOLD = False

    def run_inference(self, test_data, test_label=None):
        return self._label_output(test_data, test_label=test_label)

    @staticmethod
    def metric_names():
        return classification_metrics_label.__all__


_ALL_STRATEGY_WARNED = False


def _warn_full_series_strategy() -> None:
    global _ALL_STRATEGY_WARNED
    if _ALL_STRATEGY_WARNED:
        return
    _ALL_STRATEGY_WARNED = True
    logging.getLogger(__name__).warning(
        "all_detect_* is transductive (train == test, including anomaly region). "
        "Use unfixed_detect_* for fair comparison; all_detect_* is provided only "
        "as a transductive baseline reference."
    )


class AllDetectScore(AnomalyDetect):
    def __init__(self, *args, **kwargs):
        _warn_full_series_strategy()
        super().__init__(*args, **kwargs)

    def make_train_test_view(self, series_name):
        data = DataPool().backend().series_frame(series_name)
        if "label" not in data.columns:
            raise ValueError("Series must contain `label` column for anomaly detection.")
        test_label = data.loc[:, ["label"]]
        feature_data = feature_columns_only(data)
        return feature_data, None, feature_data, test_label

    def run_inference(self, test_data, test_label=None):
        return self._score_output(test_data)

    @staticmethod
    def metric_names():
        return classification_metrics_score.__all__


class AllDetectLabel(AllDetectScore):
    def run_inference(self, test_data, test_label=None):
        return self._label_output(test_data, test_label=test_label)

    @staticmethod
    def metric_names():
        return classification_metrics_label.__all__
