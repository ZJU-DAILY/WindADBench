# -*- coding: utf-8 -*-
from __future__ import absolute_import

from typing import List


class FieldNames:
    MODEL_NAME = "model_name"
    FILE_NAME = "file_name"
    MODEL_PARAMS = "model_params"
    STRATEGY_ARGS = "strategy_args"
    FIT_TIME = "fit_time"
    INFERENCE_TIME = "inference_time"
    ACTUAL_DATA = "actual_data"
    INFERENCE_DATA = "inference_data"
    LOG_INFO = "log_info"
    ANOMALY_RATIO = "typical_anomaly_ratio"
    FLOPS = "flops"
    N_PARAMS = "n_params"
    MODEL_SIZE_MB = "model_size_mb"
    FIT_PEAK_MEMORY = "fit_peak_memory_mb"
    INFER_PEAK_MEMORY = "inference_peak_memory_mb"
    FIT_CPU_USAGE = "fit_cpu_usage_percent"
    INFER_CPU_USAGE = "inference_cpu_usage_percent"
    FIT_GPU_PEAK_MEMORY = "fit_gpu_peak_memory_mb"
    INFER_GPU_PEAK_MEMORY = "inference_gpu_peak_memory_mb"

    # Wind-farm specific metadata fields
    FARM_ID = "farm_id"
    EVENT_ID = "event_id"
    EVENT_LABEL = "event_label"
    EVENT_DESCRIPTION = "event_description"
    TRAIN_LENS = "train_lens"
    TEST_LENS = "test_lens"
    TOTAL_LENS = "total_lens"
    ANOMALY_SPAN_LEN = "anomaly_span_len"

    # Meta fields added by build_result_df; not produced by strategy.execute
    _META_INJECT_FIELDS = (
        "farm_id",
        "event_id",
        "event_label",
        "event_description",
        "train_lens",
        "test_lens",
        "total_lens",
        "anomaly_span_len",
        "model_name",
        "strategy_args",
        "model_params",
    )

    @classmethod
    def all_fields(cls) -> List[str]:
        return [
            cls.MODEL_NAME,
            cls.FILE_NAME,
            cls.FARM_ID,
            cls.EVENT_ID,
            cls.EVENT_LABEL,
            cls.EVENT_DESCRIPTION,
            cls.TRAIN_LENS,
            cls.TEST_LENS,
            cls.TOTAL_LENS,
            cls.ANOMALY_SPAN_LEN,
            cls.MODEL_PARAMS,
            cls.STRATEGY_ARGS,
            cls.FIT_TIME,
            cls.INFERENCE_TIME,
            cls.FLOPS,
            cls.N_PARAMS,
            cls.MODEL_SIZE_MB,
            cls.FIT_PEAK_MEMORY,
            cls.INFER_PEAK_MEMORY,
            cls.FIT_CPU_USAGE,
            cls.INFER_CPU_USAGE,
            cls.FIT_GPU_PEAK_MEMORY,
            cls.INFER_GPU_PEAK_MEMORY,
            cls.ANOMALY_RATIO,
            cls.ACTUAL_DATA,
            cls.INFERENCE_DATA,
            cls.LOG_INFO,
        ]
