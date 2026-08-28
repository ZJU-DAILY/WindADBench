# -*- coding: utf-8 -*-
from tsad_benchmark.evaluation.strategy.anomaly_detect import (
    AllDetectLabel,
    AllDetectScore,
    FixedDetectLabel,
    FixedDetectScore,
    UnFixedDetectLabel,
    UnFixedDetectScore,
)


STRATEGY = {
    "fixed_detect_score": FixedDetectScore,
    "fixed_detect_label": FixedDetectLabel,
    "unfixed_detect_score": UnFixedDetectScore,
    "unfixed_detect_label": UnFixedDetectLabel,
    "all_detect_score": AllDetectScore,
    "all_detect_label": AllDetectLabel,
}
