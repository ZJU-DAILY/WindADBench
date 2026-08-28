# -*- coding: utf-8 -*-
import logging
import json
import os
import time
import traceback
from typing import Any, List

import numpy as np
import pandas as pd
import threading

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    # Optional: NVIDIA GPU memory sampling (VRAM peak).
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pynvml = None

try:
    # Preferred: process-isolated CUDA allocator peak (PyTorch).
    import torch as _torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _torch = None

from tsad_benchmark.common.random_utils import DEFAULT_SEED, fix_random_seed
from tsad_benchmark.data.series_registry import DataPool
from tsad_benchmark.data.wind_frame_ops import feature_columns_only
from tsad_benchmark.evaluation.metrics import (
    classification_metrics_label,
    classification_metrics_score,
)
from tsad_benchmark.evaluation.strategy.constants import FieldNames
from tsad_benchmark.evaluation.strategy.strategy import Strategy


class AnomalyDetect(Strategy):

    OPTIONAL_CONFIGS = {
        "strict_length",
        "strict_errors",
        "seed",
    }

    def _fail_fast(self) -> bool:
        return bool(self.strategy_config.get("strict_errors", True))

    # ------------------------------------------------------------------
    # Resource monitor (memory & CPU sampling during fit / inference)
    # ------------------------------------------------------------------

    class _ResourceMonitor:
        def __init__(self, interval_sec: float = 0.05):
            self.interval_sec = interval_sec
            self._running = False
            self._thread = None
            self.peak_rss_bytes = 0
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
                            used += int(getattr(proc, "usedGpuMemory", 0) or 0)
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
            self._wall_start = time.perf_counter()

            # Preferred GPU path: PyTorch's process-isolated allocator peak.
            if self._torch_cuda_available():
                try:
                    _torch.cuda.synchronize()
                    _torch.cuda.reset_peak_memory_stats()
                    self._torch_cuda = True
                except Exception:
                    self._torch_cuda = False

            # NVML fallback (used when torch.cuda is unavailable).
            if not self._torch_cuda and pynvml is not None:
                try:
                    pynvml.nvmlInit()
                    gpu_index = int(os.environ.get("TSAD_GPU_INDEX", "0"))
                    self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                except Exception:
                    self._nvml_handle = None

            if psutil is not None:
                proc = psutil.Process()
                cpu_t = proc.cpu_times()
                self._cpu_time_start = float(cpu_t.user + cpu_t.system)
                try:
                    self.peak_rss_bytes = int(proc.memory_info().rss)
                except Exception:
                    pass
                self._running = True

                def _sample():
                    p = psutil.Process()
                    while self._running:
                        try:
                            rss = int(p.memory_info().rss)
                            if rss > self.peak_rss_bytes:
                                self.peak_rss_bytes = rss
                        except Exception:
                            pass
                        if self._nvml_handle is not None:
                            try:
                                used = self._nvml_pid_memory(self._nvml_handle, self._pid)
                                if used > self.peak_gpu_bytes:
                                    self.peak_gpu_bytes = used
                            except Exception:
                                pass
                        time.sleep(self.interval_sec)

                self._thread = threading.Thread(target=_sample, daemon=True)
                self._thread.start()
            else:
                if self._nvml_handle is not None:
                    try:
                        self.peak_gpu_bytes = self._nvml_pid_memory(self._nvml_handle, self._pid)
                    except Exception:
                        pass

        def stop(self):
            self._wall_end = time.perf_counter()
            if psutil is not None:
                self._running = False
                if self._thread is not None:
                    self._thread.join(timeout=1.0)
                proc = psutil.Process()
                cpu_t = proc.cpu_times()
                self._cpu_time_end = float(cpu_t.user + cpu_t.system)
                try:
                    rss = int(proc.memory_info().rss)
                    if rss > self.peak_rss_bytes:
                        self.peak_rss_bytes = rss
                except Exception:
                    pass

            if self._torch_cuda:
                try:
                    _torch.cuda.synchronize()
                    self.peak_gpu_bytes = int(_torch.cuda.max_memory_allocated())
                except Exception:
                    pass

            if pynvml is not None and self._nvml_handle is not None:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

        def peak_memory_mb(self):
            if self.peak_rss_bytes <= 0:
                return np.nan
            return float(self.peak_rss_bytes / (1024.0 * 1024.0))

        def peak_gpu_memory_mb(self):
            if self.peak_gpu_bytes <= 0:
                return np.nan
            return float(self.peak_gpu_bytes / (1024.0 * 1024.0))

        def cpu_usage_percent(self):
            if self._cpu_time_start is None or self._cpu_time_end is None:
                return np.nan
            wall = max((self._wall_end or 0) - (self._wall_start or 0), 1e-9)
            cpu_delta = max(self._cpu_time_end - self._cpu_time_start, 0.0)
            ncpu = psutil.cpu_count(logical=True) if psutil is not None else 1
            ncpu = max(int(ncpu or 1), 1)
            return float((cpu_delta / wall) * 100.0 / ncpu)

    # ------------------------------------------------------------------
    # FLOPs estimation hook
    # ------------------------------------------------------------------

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
    def _model_flops(model, train_data, test_data):
        try:
            if hasattr(model, "estimate_flops"):
                return float(model.estimate_flops(train_data=train_data, test_data=test_data))
        except Exception:
            pass
        return np.nan

    @staticmethod
    def _model_scalar_stat(model, name: str) -> float:
        """Invoke ``model.<name>()`` returning a float, ``nan`` on failure."""
        try:
            fn = getattr(model, name, None)
            if not callable(fn):
                return np.nan
            return float(fn())
        except Exception:
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

    # ------------------------------------------------------------------
    # Capability-aware detection helpers
    # ------------------------------------------------------------------

    def _score_output(self, test_data: pd.DataFrame):
   
        cap = self.model.capability
        if cap.has_score_output():
            return self.model.detect_score(test_data)
        raise RuntimeError(
            f"[{self.model.model_name}] declares no score output. "
            "Use a *DetectLabel strategy or implement detect_score on it."
        )

    def _label_output(self, test_data: pd.DataFrame, test_label: pd.DataFrame = None):

        cap = self.model.capability
        if not cap.has_label_output():
            raise RuntimeError(
                f"[{self.model.model_name}] declares no label output. "
                "Use a *DetectScore strategy (fixed_detect_score / "
                "unfixed_detect_score / all_detect_score) for this model, "
                "or implement detect_label on it."
            )
        try:
            return self.model.detect_label(test_data, test_label=test_label)
        except TypeError as exc:
            if "test_label" not in str(exc):
                raise
            return self.model.detect_label(test_data)

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    def run_series(self, series_name: str, model_factory: Any) -> Any:
        seed = self.strategy_config.get("seed", DEFAULT_SEED)
        fix_random_seed(None if seed is None else int(seed))
        model = model_factory()
        try:
            self.model = model
            train_data, train_label, test_data, test_label = self.make_train_test_view(series_name)

            # --- fit phase ---
            fit_monitor = self._ResourceMonitor()
            fit_monitor.start()
            start_fit_time = time.perf_counter()
            try:
                model.fit(train_data, train_label)
            finally:
                self._synchronize_torch_cuda()
                end_fit_time = time.perf_counter()
                fit_monitor.stop()
            model_params = self._effective_model_params(model, model_factory)

            # --- inference phase ---
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

            flops = self._model_flops(model, train_data, test_data)
            n_params = self._model_scalar_stat(model, "estimate_n_params")
            model_size_mb = self._model_scalar_stat(model, "estimate_model_size_mb")
            predict_map, aux_map = self._to_prediction_maps(detect_out)

            actual = test_label.to_numpy().reshape(-1).astype(float)
            results = []
            for ratio, pred in predict_map.items():
                pred_arr = np.asarray(pred).reshape(-1).astype(float)
                aux_arr = np.asarray(aux_map.get(ratio, np.array([]))).reshape(-1)
                strict = self.strategy_config.get("strict_length", True)
                pred_arr, aux_arr = self._match_result_length(
                    pred_arr, aux_arr, len(actual),
                    strict=bool(strict), series_name=series_name,
                )

                metric_values, log_info = self.evaluator.evaluate_with_log(
                    actual=actual,
                    predicted=pred_arr,
                )
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
                        self._pack_payload([pred_arr, aux_arr]),
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

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Abstract interface (subclasses must implement)
    # ------------------------------------------------------------------

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
            FieldNames.LOG_INFO,
        ]


# ======================================================================
# Fixed-ratio split strategies
# ======================================================================

class FixedDetectScore(AnomalyDetect):
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
        return self._score_output(test_data)

    @staticmethod
    def metric_names():
        return classification_metrics_score.__all__


class FixedDetectLabel(FixedDetectScore):
    def run_inference(self, test_data, test_label=None):
        return self._label_output(test_data, test_label=test_label)

    @staticmethod
    def metric_names():
        return classification_metrics_label.__all__


# ======================================================================
# Metadata-driven split strategies (train_lens from meta)
# ======================================================================

class UnFixedDetectScore(AnomalyDetect):
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
        return self._score_output(test_data)

    @staticmethod
    def metric_names():
        return classification_metrics_score.__all__


class UnFixedDetectLabel(UnFixedDetectScore):
    def run_inference(self, test_data, test_label=None):
        return self._label_output(test_data, test_label=test_label)

    @staticmethod
    def metric_names():
        return classification_metrics_label.__all__


# ======================================================================
# Full-data strategies (train == test, transductive — NOT recommended)
# ======================================================================

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
